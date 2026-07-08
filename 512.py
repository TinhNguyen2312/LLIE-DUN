import os
import glob
from PIL import Image
from tqdm import tqdm
import sys


def resize_smart(src_dir, target_size=(512, 512)):
    tif_files = glob.glob(os.path.join(src_dir, "**", "*.*"), recursive=True)
    print(f"Tìm thấy {len(tif_files)} file")

    for file_path in tqdm(tif_files):
        try:
            with Image.open(file_path) as img:
                if img.size != target_size:
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    img_resized.save(file_path)

        except Exception as e:
            print(f"Lỗi {file_path}: {e}")


if __name__ == "__main__":
    src = sys.argv[1]
    resize_smart(src)
