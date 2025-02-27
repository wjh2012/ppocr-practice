import os
import argparse
from app.data_reader.json_label_parser import JsonLabelParser
from app.lmdb_dataset.lmdb_impl import LMDBRepository


def run_with_json_label(
    lmdb_size: int,
    json_path: str,
    data_path: str,
    lmdb_base_path: str,
    train_ratio: float = 0.8,
):
    reader = JsonLabelParser(json_path)
    parsed_label_data = reader.parse()
    formatted_data = reader.convert_to_formatted_dto(parsed_label_data)

    total_samples = len(formatted_data)
    train_count = int(total_samples * train_ratio)
    train_data = formatted_data[:train_count]
    val_data = formatted_data[train_count:]

    train_lmdb_path = os.path.join(lmdb_base_path, "lmdb_train")
    val_lmdb_path = os.path.join(lmdb_base_path, "lmdb_val")

    map_size = lmdb_size * 1024 * 1024 * 1024

    train_repo = LMDBRepository(train_lmdb_path)
    val_repo = LMDBRepository(val_lmdb_path)

    print("학습 데이터 LMDB 생성 시작...")
    train_repo.create_lmdb(
        map_size=map_size,
        datas=train_data,
        image_root=data_path,
    )

    print("검증 데이터 LMDB 생성 시작...")
    val_repo.create_lmdb(
        map_size=map_size,
        datas=val_data,
        image_root=data_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSON 라벨 데이터를 LMDB로 변환하는 스크립트"
    )
    parser.add_argument(
        "--lmdb_size",
        type=int,
        default=10,
        help="lmdb 크기",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\printed_data_info.json",
        help="JSON 파일 경로",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\01_printed_word_images\word",
        help="이미지 루트 경로",
    )
    parser.add_argument(
        "--lmdb_base_path",
        type=str,
        default=r"C:\Users\WONJANGHO\Desktop\lmdb",
        help="LMDB 저장 기본 경로",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="학습 데이터 비율 (기본값: 0.8)",
    )

    args = parser.parse_args()

    run_with_json_label(
        lmdb_size=args.lmdb_size,
        json_path=args.json_path,
        data_path=args.data_path,
        lmdb_base_path=args.lmdb_base_path,
        train_ratio=args.train_ratio,
    )
