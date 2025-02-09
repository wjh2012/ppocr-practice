from app.data_reader.json_label_parser import JsonLabelParser
from app.data_reader.txt_label_parser import TxtLabelParser
from app.lmdb_impl import LMDBRepository


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
    repo.read_lmdb(top_n=10)


def run_with_txt_label():
    txt_path = r"C:\Users\WONJANGHO\Desktop\out\labels.txt"
    reader = TxtLabelParser(txt_path)
    formatted_data = reader.parse()

    print(f"데이터 길이: {len(formatted_data)}")
    print(formatted_data[0].path)
    print(formatted_data[0].label)

    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\lmdb_mcir_train"
    image_root_path = r"C:\Users\WONJANGHO\Desktop\out"
    map_size = 1 * 1024 * 1024 * 1024

    repo = LMDBRepository(output_lmdb_path)
    repo.create_lmdb(
        map_size=map_size,
        datas=formatted_data,
        image_root=image_root_path,
    )
    repo.read_lmdb(top_n=10)


if __name__ == "__main__":
    run_with_txt_label()
