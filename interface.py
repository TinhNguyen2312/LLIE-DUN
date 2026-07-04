import torch
import os
import argparse
import yaml
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings

from net import Model
import utils

warnings.filterwarnings("ignore")


def load_model(pretrained_path, config, device):
    print(f"Loading model from {pretrained_path}")

    model = Model(**config["model"])
    model = model.to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state_dict = utils.extract_model_state_dict(checkpoint)

        aligned, missing, unexpected = utils.align_and_filter_state_dict(
            model, state_dict
        )
        if aligned:
            model.load_state_dict(aligned, strict=False)
            print(
                f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        else:
            print("Warning: No compatible weights loaded from checkpoint")

    model.eval()
    return model


def predict_images(model, input_dir, output_dir, device):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    image_files = []
    for ext in exts:
        image_files.extend(input_dir.glob(f"**/*{ext}"))
    image_files = sorted([f for f in image_files if f.is_file()])

    print(f"Found {len(image_files)} images in {input_dir}")
    transform = transforms.ToTensor()

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing"):
            output_path = output_dir / img_path.name
            if output_path.exists():
                continue

            image = Image.open(img_path).convert("RGB").resize((512, 512))
            tensor = transform(image).unsqueeze(0).to(device)

            prediction = model(tensor)
            prediction = prediction.clamp(0.0, 1.0)

            pred_tensor = prediction[0].detach().cpu()
            pred_numpy = pred_tensor.permute(1, 2, 0).numpy() * 255.0
            pred_numpy = pred_numpy.clip(0, 255).astype("uint8")
            pred_image = Image.fromarray(pred_numpy)

            pred_image.save(output_path)

    print(f"Saved {len(image_files)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Low-Light Image Enhancement Inference "
    )
    parser.add_argument("-i", required=True, help="Directory containing input images")
    parser.add_argument("-m", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("-o", required=True, help="Directory to save enhanced images")
    parser.add_argument("-c", default="./configs/lol.yaml", help="Path to config file")
    parser.add_argument("-d", default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.c):
        raise FileNotFoundError(f"Config file not found: {args.c}")

    with open(args.c, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(args.d if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.m, config, device)

    predict_images(model, args.i, args.o, device)


if __name__ == "__main__":
    main()
