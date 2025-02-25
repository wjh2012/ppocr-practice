from rapidocr_onnxruntime import RapidOCR


def main():
    model_path = "./rec_models/micr_model/micr_model.onnx"
    txt_path = "micr_dict.txt"

    engine = RapidOCR(
        rec_model_path=model_path, rec_keys_path=txt_path, use_det=False, use_cls=False
    )

    img = "test_image/fit.jpg"
    result, elapse = engine(img)
    print(result)
    print(elapse)


if __name__ == "__main__":
    main()
