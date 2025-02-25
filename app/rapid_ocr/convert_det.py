from paddleocr_convert import PaddleOCRModelConvert


def main():
    converter = PaddleOCRModelConvert()
    save_dir = "det_models"
    model_path = r"C:\Users\WONJANGHO\Desktop\AI\inference\en_PP-OCRv3_det_infer.tar"
    converter(model_path, save_dir)


if __name__ == "__main__":
    main()
