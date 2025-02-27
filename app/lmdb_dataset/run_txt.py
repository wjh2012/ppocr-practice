import os
from pathlib import Path

from app.data_reader.txt_label_parser import TxtLabelParser
from app.lmdb_dataset.lmdb_impl import LMDBRepository


def run_with_txt_label_dir(txt_path, sub_dir):
    reader = TxtLabelParser(txt_path)
    formatted_data = reader.parse()

    sub_dir = Path(sub_dir)

    for data in formatted_data:
        data.path = sub_dir / data.path  # .as_posix()

    return formatted_data


def run_on_data_directory(
    data_dir, lmdb_size: int, lmdb_base_path: str, train_ratio: float = 0.8
):
    all_data = []

    for sub_name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, sub_name)
        if os.path.isdir(sub_dir):
            label_file = None
            for fname in ["labels.txt", "label.txt"]:
                candidate = os.path.join(sub_dir, fname)
                if os.path.exists(candidate):
                    label_file = candidate
                    break
            if label_file is None:
                print(f"{sub_dir} 에 label 파일이 없습니다. 건너뜁니다.")
                continue

            print(f"{sub_dir} 의 label 파일: {label_file}")
            data = run_with_txt_label_dir(label_file, sub_name)
            all_data.extend(data)

    print(f"전체 데이터 개수: {len(all_data)}")

    total_samples = len(all_data)
    train_count = int(total_samples * train_ratio)
    train_data = all_data[:train_count]
    val_data = all_data[train_count:]

    train_lmdb_path = os.path.join(lmdb_base_path, "lmdb_train")
    val_lmdb_path = os.path.join(lmdb_base_path, "lmdb_val")

    map_size = lmdb_size * 1024 * 1024 * 1024

    train_repo = LMDBRepository(train_lmdb_path)
    val_repo = LMDBRepository(val_lmdb_path)

    print("학습 데이터 LMDB 생성 시작...")
    train_repo.create_lmdb(
        map_size=map_size,
        datas=train_data,
        image_root=data_dir,
    )

    print("검증 데이터 LMDB 생성 시작...")
    val_repo.create_lmdb(
        map_size=map_size,
        datas=val_data,
        image_root=data_dir,
    )


if __name__ == "__main__":
    data_dir = r"C:\Users\WONJANGHO\Desktop\datas\train"
    lmdb_size = 10
    lmdb_base_path = r"C:\Users\WONJANGHO\Desktop\lmdb"

    run_on_data_directory(data_dir, lmdb_size, lmdb_base_path)
