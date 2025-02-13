from app.data_reader.txt_label_parser import TxtLabelParser
from app.simple_dataset.simple_impl import create_simple_dataset


def run_with_txt_label(txt_path, image_root, base_dir=None):
    reader = TxtLabelParser(txt_path)
    formatted_data = reader.parse()

    print(f"데이터 길이: {len(formatted_data)}")
    print(formatted_data[0].path)
    print(formatted_data[0].label)

    create_simple_dataset(
        datas=formatted_data, image_root=image_root, base_dir=base_dir
    )


if __name__ == "__main__":
    image_root_path = r"C:\Users\WONJANGHO\Desktop\eval"
    base_dir_path = "train_data/rec/valid"
    txt_path = r"C:\Users\WONJANGHO\Desktop\eval\labels.txt"

    run_with_txt_label(txt_path=txt_path, image_root=image_root_path)
