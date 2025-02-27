import os
from pathlib import Path

import msgspec
from tqdm import tqdm


class ImageData(msgspec.Struct):
    path: str
    label: str


def create_simple_dataset(datas: list[ImageData], image_root: str, base_dir: None):
    with open("rec_gt_valid.txt", "w", encoding="utf-8") as out_file:
        for data in tqdm(datas, desc="LMDB 데이터셋 생성중"):
            image_path = os.path.join(image_root, data.path)
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
            except Exception as e:
                continue
            if base_dir:
                result_path = Path(base_dir) / data.path
                result_path = result_path.as_posix()
            else:
                result_path = data.path
            result_label = data.label

            out_file.write(f"{result_path}\t{result_label}\n")
