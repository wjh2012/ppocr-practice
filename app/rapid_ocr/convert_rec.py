from paddleocr_convert import PaddleOCRModelConvert


def main():
    converter = PaddleOCRModelConvert()
    save_dir = "rec_models"
    model_path = r"C:\Users\WONJANGHO\Desktop\AI\inference\micr_model.tar"
    txt_path = "micr_dict.txt"
    converter(model_path, save_dir, txt_path=txt_path)


if __name__ == "__main__":
    main()
