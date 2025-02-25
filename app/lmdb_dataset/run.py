import os
from pathlib import Path

from app.data_reader.json_label_parser import JsonLabelParser
from app.data_reader.txt_label_parser import TxtLabelParser
from app.lmdb_dataset.lmdb_impl import LMDBRepository


def run_with_json_label():
    json_path = r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\printed_data_info.json"

    reader = JsonLabelParser(json_path)
    parsed_label_data = reader.parse()
    formatted_data = reader.convert_to_formatted_dto(parsed_label_data)

    print(f"데이터 길이: {len(parsed_label_data.images)}")
    print(formatted_data[0].path)
    print(formatted_data[0].label)

    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\lmdb_train"
    image_root_path = r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\01_printed_sentence_images\sentence"

    map_size = 10 * 1024 * 1024 * 1024
    repo = LMDBRepository(output_lmdb_path)
    repo.create_lmdb(
        map_size=map_size,
        datas=formatted_data,
        image_root=image_root_path,
    )


def run_with_txt_label(txt_path, output_lmdb_path, image_root_path):
    reader = TxtLabelParser(txt_path)
    formatted_data = reader.parse()

    print(f"데이터 길이: {len(formatted_data)}")
    print(formatted_data[0].path)
    print(formatted_data[0].label)

    map_size = 1 * 1024 * 1024 * 1024

    repo = LMDBRepository(output_lmdb_path)
    repo.create_lmdb(
        map_size=map_size,
        datas=formatted_data,
        image_root=image_root_path,
    )


def run_with_txt_label_dir(txt_path, sub_dir):
    reader = TxtLabelParser(txt_path)
    formatted_data = reader.parse()

    sub_dir = Path(sub_dir)

    for data in formatted_data:
        data.path = sub_dir / data.path  # .as_posix()

    return formatted_data


def run_on_data_directory(data_dir, output_lmdb_path):
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
    print(all_data)
    map_size = 1 * 1024 * 1024 * 1024
    repo = LMDBRepository(output_lmdb_path)

    repo.create_lmdb(map_size=map_size, datas=all_data, image_root=data_dir)


if __name__ == "__main__":
    data_dir = r"C:\Users\WONJANGHO\Desktop\data"
    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\AI\micr_eval"

    run_on_data_directory(data_dir, output_lmdb_path)
