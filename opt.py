import inspect

import torch

from metrics import compute_metrics
from utils import save_eval_images, save_sample_images
from logger import MetricLogger, SmoothedValue


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _forward_with_optional_intermediates(model, inputs, loss_fn):
    raw_model = _unwrap_model(model)
    wants_intermediates = "return_intermediate" in inspect.signature(
        raw_model.forward
    ).parameters and getattr(loss_fn, "intermediate_enabled", False)
    if wants_intermediates:
        return model(inputs, return_intermediate=True)
    return model(inputs), None


def _compute_loss(loss_fn, model, pred_l, targets, intermediates):
    params = inspect.signature(loss_fn.forward).parameters
    if "model" in params or "intermediates" in params:
        kwargs = {}
        if "model" in params and getattr(loss_fn, "fourier_enabled", False):
            kwargs["model"] = _unwrap_model(model)
        if "intermediates" in params and intermediates is not None:
            kwargs["intermediates"] = intermediates
        return loss_fn(pred_l, targets, **kwargs)
    return loss_fn(pred_l, targets)


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    epoch,
    loss_fn,
    print_freq=10,
    log_dir="logs",
):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        inputs = batch["inputs"].to(args.device)
        targets = batch["targets"].to(args.device)

        pred_l, intermediates = _forward_with_optional_intermediates(
            model, inputs, loss_fn
        )

        loss_dict = _compute_loss(loss_fn, model, pred_l, targets, intermediates)
        total_loss = loss_dict["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
            metric_logger.update(lr=float(optimizer.param_groups[0]["lr"]))
        for loss_name, loss_value in loss_dict.items():
            metric_logger.update(**{f"{loss_name}_loss": loss_value.item()})

        if batch_idx % (print_freq * 5) == 0:
            save_sample_images(
                inputs, pred_l, targets, batch_idx, epoch, args.output_dir
            )

    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args,
    data_loader,
    model,
    epoch,
    loss_fn,
    print_freq=100,
    results_path=None,
    log_dir="logs",
):
    """Evaluate model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    header = f"Test: [{epoch}]"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            inputs = batch["inputs"].to(args.device)
            targets = batch["targets"].to(args.device)
            filenames = batch["filenames"]

            pred_l, intermediates = _forward_with_optional_intermediates(
                model, inputs, loss_fn
            )
            pred_l = pred_l.clamp(0.0, 1.0)
            if intermediates is not None:
                intermediates = [inter.clamp(0.0, 1.0) for inter in intermediates]

            loss_dict = _compute_loss(loss_fn, model, pred_l, targets, intermediates)
            for loss_name, loss_value in loss_dict.items():
                metric_logger.update(**{f"{loss_name}_loss": loss_value.item()})

            metrics = compute_metrics(targets, pred_l, args.device)
            for metric_name, metric_value in metrics.items():
                metric_logger.update(**{f"{metric_name}": metric_value})

            if args.save_images:
                save_eval_images(inputs, pred_l, targets, filenames, args.output_dir)

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
