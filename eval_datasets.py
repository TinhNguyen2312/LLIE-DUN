import argparse
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from net import Model
from utils import load_pretrained_flexibly, save_img


def list_images_multi_ext(root_dir: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP"]
    files = []

    for ext in exts:
        files.extend(Path(root_dir).glob(f"**/{ext}"))

    files = sorted(list({str(p) for p in files}))
    return files


def pil_to_tensor_rgb01(pil_img: Image.Image) -> torch.Tensor:
    return T.ToTensor()(pil_img.convert("RGB").resize((256, 256))).unsqueeze(0)


def run_on_dataset(
    dataset_name: str,
    input_dir: Path,
    save_root: Path,
    model: torch.nn.Module,
    device: torch.device,
):
    if not input_dir.exists():
        print(f"[WARN] Dataset '{dataset_name}' missing at {input_dir}. Skipping.")
        return {}

    out_dir = save_root / f"{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_images_multi_ext(str(input_dir))

    if not files:
        for sub in "low":
            trial_dir = input_dir / sub
            if trial_dir.exists():
                files = list_images_multi_ext(str(trial_dir))
                if files:
                    break

    if not files:
        print(f"[WARN] No images found for dataset '{dataset_name}' at {input_dir}.")
        return {}

    print(f"[INFO] Found {len(files)} images in {dataset_name}")

    model.eval()
    with torch.no_grad():
        for fp in tqdm(files, desc=f"[{dataset_name}]"):
            base = Path(fp).name
            out_path = out_dir / base

            if out_path.exists():
                continue

            try:
                pil = Image.open(fp).convert("RGB")

                x = pil_to_tensor_rgb01(pil).to(device)

                out = model(x)

                save_img(out[0], str(out_path))

            except Exception as e:
                print(f"[ERROR] Failed to process {fp}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình Model trên nhiều dataset"
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="./datasets",
        help="Đường dẫn đến thư mục chứa datasets",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_eval",
        help="Đường dẫn đến thư mục lưu kết quả",
    )
    parser.add_argument("--save_images", action="store_true", help="Lưu ảnh kết quả")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device để chạy model (cuda/cpu/auto)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requested = [
        ("ExDark", datasets_root / "ExDark"),
        ("DICM", datasets_root / "DICM"),
        ("FiveK", datasets_root / "FiveK"),
        ("LIME", datasets_root / "LIME"),
        ("LOL", datasets_root / "LOL"),
        ("MEF", datasets_root / "MEF"),
        ("SICE", datasets_root / "SICE/low"),
        ("VV", datasets_root / "VV"),
        ("SDSD", datasets_root / "SDSD"),
    ]

    for name, path in requested:
        model = Model(
            in_ch=3,
            c=[32, 64, 128, 256],
            enc_blocks=[2, 2, 3, 3],
            dec_blocks=[2, 2, 3, 3],
            asp_patch=8,
            asp_K=16,
        )
        model = model.to(device)
        weight_path = "cp/lol-blur.pth"
        load_pretrained_flexibly(model, weight_path, device="cpu", strict=False)

        run_on_dataset(
            dataset_name=name,
            input_dir=path,
            save_root=output_root,
            model=model,
            device=device,
        )


if __name__ == "__main__":
    main()
