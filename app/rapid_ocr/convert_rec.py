from paddleocr_convert import PaddleOCRModelConvert


def main():
    converter = PaddleOCRModelConvert()
    save_dir = "rec_models"
    model_path = (
        r"C:\Users\WONJANGHO\Desktop\AI\inference\rec\kor\korean_PP-OCRv3_rec_infer.tar"
    )
    txt_path = r"C:\Users\WONJANGHO\Desktop\AI\inference\rec\kor\korean_dict.txt"
    converter(model_path, save_dir, txt_path=txt_path)


if __name__ == "__main__":
    main()
