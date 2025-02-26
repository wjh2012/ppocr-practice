from rapidocr_onnxruntime import RapidOCR


def main():
    model_path = "./rec_models/micr_v2/micr_v2.onnx"
    txt_path = r"C:\Users\WONJANGHO\Desktop\AI\model\inference\rec\250226\micr_dict.txt"

    engine = RapidOCR(
        rec_model_path=model_path, rec_keys_path=txt_path, use_det=False, use_cls=False
    )

    img_path = "test_image/check1.jpg"
    result, elapse = engine(img_path, box_thresh=0.8, text_score=0.8)
    print(result)
    print(elapse)


if __name__ == "__main__":
    main()
