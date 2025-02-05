from app.data_reader.label_parser import JsonLabelParser
from app.lmdb_maker import LMDBMaker


def run():
    json_path = r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\printed_data_info.json"

    reader = JsonLabelParser(json_path)
    parsed_label_data = reader.parse_json()

    print(f"데이터 길이: {len(parsed_label_data.images)}")

    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\lmdb_train"
    image_root_path = r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\01_printed_sentence_images\sentence"

    map_size = 10 * 1024 * 1024 * 1024
    lmdb_maker = LMDBMaker(
        output_lmdb_path, parsed_label_data, image_root_path, map_size
    )
    lmdb_maker.create_lmdb_dataset()


if __name__ == "__main__":
    run()
