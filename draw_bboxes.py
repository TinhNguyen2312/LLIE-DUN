import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw bounding boxes on an image using a simple label file "
            "(format: <category> <x> <y> <w> <h> ...)"
        )
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--labels",
        "-l",
        default="objectDetection/lb.txt",
        help="Path to label file (default: objectDetection/lb.txt)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output image path. Defaults to <input_name>_annotated.png",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=26,
        help="Font size for category labels (default: 46)",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="Path to a .ttf/.ttc font to use for labels (optional)",
    )
    return parser.parse_args()


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except Exception:  # pragma: no cover
        print(
            "This script requires Pillow. Install it with: pip install pillow",
            file=sys.stderr,
        )
        raise


def load_labels(labels_path: str) -> Tuple[List[Dict], List[str]]:
    """Load labels from a whitespace-separated file."""
    annotations: List[Dict] = []
    categories: List[str] = []

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                parts = stripped.split()
                if len(parts) < 5:
                    print(
                        f"Skipping line (needs at least 5 columns): {stripped}",
                        file=sys.stderr,
                    )
                    continue

                cat_name = parts[0]
                try:
                    x, y, w, h = map(float, parts[1:5])
                except Exception:
                    print(
                        f"Skipping line (bad bbox numbers): {stripped}",
                        file=sys.stderr,
                    )
                    continue

                annotations.append({"category": cat_name, "bbox": [x, y, w, h]})
                if cat_name not in categories:
                    categories.append(cat_name)
    except FileNotFoundError:
        print(f"Label file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    return annotations, categories


# def filter_annotations_by_image_id(
#     annotations: List[Dict], image_id: int
# ) -> List[Dict]:
#     return [a for a in annotations if int(a.get("image_id", -1)) == int(image_id)]


def compute_output_path(input_image_path: str, out_path: Optional[str]) -> str:
    if out_path:
        return out_path
    root, ext = os.path.splitext(input_image_path)
    return f"res_{root.split('/')[-1]}.png"


def get_palette() -> List[Tuple[int, int, int]]:
    # Distinct colors for categories; repeats modulo length
    return [
        (0, 112, 243),  # blue
        (0, 200, 83),  # green
        (255, 193, 7),  # amber
        (233, 30, 99),  # pink
        (156, 39, 176),  # purple
        (255, 87, 34),  # deep orange
        (0, 188, 212),  # cyan
        (121, 85, 72),  # brown
        (63, 81, 181),  # indigo
        (255, 0, 0),  # red
    ]


def draw_bboxes(
    image_path: str,
    annotations: List[Dict],
    out_path: str,
    font_size: int,
    font_path: Optional[str],
    cat_colors: Dict[str, Tuple[int, int, int]],
) -> None:
    require_pillow()
    from PIL import Image, ImageDraw, ImageFont

    # Fixed target dimensions
    target_width = 1600
    target_height = 1063

    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size

    # Resize to target dimensions
    image = image.resize((target_width, target_height), Image.LANCZOS)

    draw = ImageDraw.Draw(image)

    # Load TrueType font with requested size; fallback to defaults
    font: Optional[ImageFont.ImageFont] = None
    # Try user-provided font path first
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except Exception:
            font = None
    # Try a bundled/common font
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            font = None
    # Try a common Windows font path
    if font is None and os.name == "nt":
        for candidate in (
            r"C:\\Windows\\Fonts\\arial.ttf",
            r"C:\\Windows\\Fonts\\segoeui.ttf",
            r"C:\\Windows\\Fonts\\tahoma.ttf",
        ):
            try:
                font = ImageFont.truetype(candidate, size=font_size)
                break
            except Exception:
                continue
    # Final fallback to default (non-scalable)
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x, y, w, h = map(float, bbox)
        except Exception:
            continue

        cat_name = str(ann.get("category", "unknown"))

        # Format: [x, y, w, h] where x=left, y=top, w=width, h=height
        x1_orig = x
        y1_orig = y
        x2_orig = x + w
        y2_orig = y + h

        color = cat_colors.get(cat_name, (255, 255, 255))

        # Draw rectangle
        draw.rectangle([(x1_orig, y1_orig), (x2_orig, y2_orig)], outline=color, width=3)

        # Label background box for readability
        label_text = cat_name
        try:
            # Newer Pillow: textbbox provides accurate metrics
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            # Fallback if textbbox not available
            text_w, text_h = draw.textsize(label_text, font=font)  # type: ignore

        # Clamp label box to remain valid inside image bounds (x2 >= x1, y2 >= y1)
        safe_w = max(1, target_width)
        safe_h = max(1, target_height)
        label_x1 = max(0, min(int(x1_orig), safe_w - 1))
        label_y1 = max(0, min(int(y1_orig - text_h - 4), safe_h - 1))
        label_x2 = min(safe_w - 1, label_x1 + text_w + 8)
        label_y2 = min(safe_h - 1, label_y1 + text_h + 4)
        if label_x2 < label_x1:
            label_x2 = label_x1
        if label_y2 < label_y1:
            label_y2 = label_y1
        # Ensure at least 1px height/width for the filled rectangle
        if label_x2 == label_x1 and label_x1 < safe_w - 1:
            label_x2 += 1
        if label_y2 == label_y1 and label_y1 < safe_h - 1:
            label_y2 += 1

        # Filled rectangle behind text
        draw.rectangle([(label_x1, label_y1), (label_x2, label_y2)], fill=color)
        # Text in white
        text_pos = (
            label_x1 + 4,
            label_y1 + max(0, (label_y2 - label_y1 - text_h) // 2),
        )
        draw.text(text_pos, label_text, fill=(255, 255, 255), font=font)

    image.save(out_path)


def main() -> None:
    args = parse_args()

    annotations, categories = load_labels(args.labels)
    if not annotations:
        print("No annotations found in label file.", file=sys.stderr)
        sys.exit(1)

    palette = get_palette()
    cat_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    out_path = compute_output_path(args.image, args.out)
    draw_bboxes(
        image_path=args.image,
        annotations=annotations,
        out_path=out_path,
        font_size=int(args.font_size),
        font_path=args.font_path,
        cat_colors=cat_colors,
    )

    print(f"Saved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
