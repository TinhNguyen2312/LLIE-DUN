import argparse
import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import yaml

from net import Model
from opt import evaluate_fn
import utils


class PairFolderDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.low_dir = self.root / "low"
        self.light_dir = self.root / "light"

        if not self.low_dir.exists() or not self.light_dir.exists():
            raise FileNotFoundError(
                f"Expected subfolders 'low' and 'light' in {self.root}"
            )

        exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
        )
        low_names = {
            p.name for p in self.low_dir.iterdir() if p.is_file() and p.suffix in exts
        }
        light_names = {
            p.name for p in self.light_dir.iterdir() if p.is_file() and p.suffix in exts
        }
        self.file_names = sorted(list(low_names & light_names))

        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.file_names[idx]
        low_path = self.low_dir / name
        light_path = self.light_dir / name

        low_img = Image.open(low_path).convert("RGB")
        light_img = Image.open(light_path).convert("RGB")

        low_t = self.to_tensor(low_img)
        light_t = self.to_tensor(light_img)

        return {
            "input": low_t,
            "target": light_t,
            "filename": name,
            "idx": idx,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        item = batch[0]
        return {
            "inputs": item["input"].unsqueeze(0),
            "targets": item["target"].unsqueeze(0),
            "filenames": [item["filename"]],
            "indices": [item["idx"]],
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Model on a folder with 'low' and 'light' subfolders"
    )
    parser.add_argument(
        "--input", required=True, help="Folder containing 'low' and 'light'"
    )
    parser.add_argument("--resume", required=True, help="Path to checkpoint .pth")
    parser.add_argument(
        "--cfg", type=str, default="", help="Optional YAML config for model/eval"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--save_images", action="store_true", help="Save evaluation images"
    )
    parser.add_argument("--out", default="outputs/eval_folder", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device
        if args.device in ("cpu", "cuda")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = None
    if args.cfg:
        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            print(f"[WARN] cfg not found: {cfg_path}. Using defaults.")
        else:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

    if cfg and "model" in cfg:
        model = Model(**cfg["model"]).to(device)
    else:
        model = Model().to(device)

    from fvcore.nn import flop_count_table, FlopCountAnalysis

    flops = FlopCountAnalysis(model, inputs=(torch.randn(1, 3, 512, 512).to(device),))
    print(flop_count_table(flops))

    dataset = PairFolderDataset(Path(args.input))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PairFolderDataset.collate_fn,
        pin_memory=True,
    )

    ret, missing, unexpected, _ = utils.load_pretrained_flexibly(
        model, args.resume, device="cpu", strict=False
    )
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    class EvalArgs:
        pass

    shim = EvalArgs()
    shim.device = device
    shim.output_dir = args.out
    save_images_cfg = False
    if cfg and isinstance(cfg.get("evaluation", {}), dict):
        save_images_cfg = bool(cfg["evaluation"].get("save_images", False))
    shim.save_images = args.save_images or save_images_cfg

    os.makedirs(args.out, exist_ok=True)

    class IdentityLoss:
        def __call__(self, pred, target):
            return {"total": (pred - target).abs().mean()}

    loss_fn = IdentityLoss()

    evaluate_fn(
        args=shim,
        data_loader=dataloader,
        model=model,
        epoch=0,
        loss_fn=loss_fn,
        print_freq=50,
        results_path=None,
        log_dir=os.path.join(args.out, "logs"),
    )

    print(f"Evaluation completed. Outputs at: {args.out}")


if __name__ == "__main__":
    main()
