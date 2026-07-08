import cv2
import numpy as np
import os
import argparse


# def draw_dashed_line(img, p1, p2, color, thickness=2, dash_length=12):
#     dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
#     if dist == 0:
#         return
#     for i in range(0, dist, dash_length * 2):
#         start_x = int(p1[0] + (p2[0] - p1[0]) * i / dist)
#         start_y = int(p1[1] + (p2[1] - p1[1]) * i / dist)
#         end_x = int(p1[0] + (p2[0] - p1[0]) * (i + dash_length) / dist)
#         end_y = int(p1[1] + (p2[1] - p1[1]) * (i + dash_length) / dist)
#         cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


def compose_with_two_zooms(
    image,
    bbox1,
    bbox2,
    colors=((0, 0, 255), (0, 255, 0)),
    margin=12,
):
    """Vẽ 2 bbox và tạo hai ảnh zoom ở hàng dưới cùng của ảnh đầu ra.

    bbox: [x, y, w, h]
    Trả về ảnh mới đã ghép.
    """
    h, w = image.shape[:2]
    canvas = image.copy()

    # Vẽ bbox lên ảnh gốc
    (x1, y1, w1, h1) = [int(v) for v in bbox1]
    (x2, y2, w2, h2) = [int(v) for v in bbox2]
    cv2.rectangle(canvas, (x1, y1), (x1 + w1, y1 + h1), colors[0], 2)
    cv2.rectangle(canvas, (x2, y2), (x2 + w2, y2 + h2), colors[1], 2)

    # Tạo crop và resize sao cho: height_zoom1 + margin + height_zoom2 = height ảnh gốc
    crop1 = image[max(0, y1) : y1 + h1, max(0, x1) : x1 + w1]
    crop2 = image[max(0, y2) : y2 + h2, max(0, x2) : x2 + w2]
    if crop1.size == 0 or crop2.size == 0:
        return canvas
    available_h = max(0, h - margin)
    target_h1 = available_h // 2
    target_h2 = available_h - target_h1
    # Giữ tỉ lệ khung hình theo chiều cao đặt trước
    target_w1 = int(max(1, crop1.shape[1]) * (target_h1 / max(1, crop1.shape[0])))
    target_w2 = int(max(1, crop2.shape[1]) * (target_h2 / max(1, crop2.shape[0])))

    # Resize chất lượng cao + sharpen nhẹ để giảm vỡ nét
    def upscale_and_sharpen(src, w, h, amount=0.6):
        resized = cv2.resize(src, (w, h), interpolation=cv2.INTER_LANCZOS4)
        blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(resized, 1.0 + amount, blurred, -amount, 0)
        return sharp

    zoom1 = upscale_and_sharpen(crop1, target_w1, target_h1, amount=0.6)
    zoom2 = upscale_and_sharpen(crop2, target_w2, target_h2, amount=0.6)

    # Kích thước canvas mới (thêm cột bên phải đặt 2 zoom)
    sidebar_w = max(target_w1, target_w2) + margin * 2
    out_h = h
    out_w = w + sidebar_w
    out = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    out[:h, :w] = canvas

    # Tọa độ đặt hai zoom ở bên phải (xếp dọc)
    z1_x = w + margin
    z1_y = 2
    z2_x = w + margin
    z2_y = target_h1 + margin - 2

    out[z1_y : z1_y + target_h1, z1_x : z1_x + target_w1] = zoom1
    out[z2_y : z2_y + target_h2, z2_x : z2_x + target_w2] = zoom2

    # Khung quanh zooms
    cv2.rectangle(
        out,
        (z1_x - 2, z1_y - 2),
        (z1_x + target_w1 + 2, z1_y + target_h1 + 2),
        colors[0],
        2,
    )
    cv2.rectangle(
        out,
        (z2_x - 2, z2_y - 2),
        (z2_x + target_w2 + 2, z2_y + target_h2 + 2),
        colors[1],
        2,
    )

    # Nối dashed line từ bbox sang khung zoom tương ứng (giữ comment nếu không cần)
    # draw_dashed_line(out, (x1 + w1, y1 + h1 // 2), (z1_x, z1_y), colors[0], 2, 10)
    # draw_dashed_line(out, (x2 + w2, y2 + h2 // 2), (z2_x, z2_y), colors[1], 2, 10)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Vẽ 2 bbox và tạo 2 ảnh zoom cho từng ảnh trong file hoặc thư mục"
    )
    parser.add_argument("input", help="Đường dẫn ảnh hoặc thư mục chứa ảnh")
    parser.add_argument("--outdir", default="zoom_outputs", help="Thư mục lưu kết quả")
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument(
        "--bbox1",
        nargs=4,
        type=int,
        metavar=("x", "y", "w", "h"),
        help="BBox 1 dạng 4 số: x y w h (mặc định dùng preset)",
    )
    parser.add_argument(
        "--bbox2",
        nargs=4,
        type=int,
        metavar=("x", "y", "w", "h"),
        help="BBox 2 dạng 4 số: x y w h (mặc định dùng preset)",
    )

    args = parser.parse_args()

    in_path = args.input
    os.makedirs(args.outdir, exist_ok=True)

    # preset nếu không truyền bbox
    bbox1 = args.bbox1 if args.bbox1 is not None else [280, 90, 30, 45]
    bbox2 = args.bbox2 if args.bbox2 is not None else [180, 90, 30, 45]

    def process_one(path):
        img = cv2.imread(path)
        if img is None:
            print(f"[!] Không đọc được ảnh: {path}")
            return
        out = compose_with_two_zooms(img, bbox1=bbox1, bbox2=bbox2, margin=args.margin)
        name = os.path.splitext(os.path.basename(path))[0] + "_zoom.jpg"
        save_path = os.path.join(args.outdir, name)
        cv2.imwrite(save_path, out)
        print(f"[✓] Saved: {save_path}")

    if os.path.isdir(in_path):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [
            os.path.join(in_path, f)
            for f in sorted(os.listdir(in_path))
            if f.lower().endswith(exts)
        ]
        if not files:
            print(f"[!] Không tìm thấy ảnh trong thư mục: {in_path}")
            return
        for fp in files:
            process_one(fp)
    else:
        process_one(in_path)


if __name__ == "__main__":
    main()
