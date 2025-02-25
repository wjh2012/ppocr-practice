import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def draw_results(image_path, results, output_path="output.jpg"):
    # 이미지 로드
    image = cv2.imread(image_path)

    for result in results:
        box, text, conf = result
        box = np.array(box, dtype=np.int32)  # 박스 좌표 변환

        # 박스 그리기
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=1)

        # 텍스트 그리기
        x, y = box[0]  # 좌상단 좌표
        cv2.putText(
            image,
            f"{text} ({conf:.2f})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # 결과 저장
    cv2.imwrite(output_path, image)
    print(f"결과가 {output_path}로 저장되었습니다.")


def main():
    rec_model_path = "./rec_models/micr_model/micr_model.onnx"
    det_model_path = "./det_models/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.onnx"
    txt_path = r"C:\Users\WONJANGHO\Desktop\AI\inference\micr_dict.txt"

    engine = RapidOCR(
        use_det=True,
        use_cls=False,
        use_rec=True,
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        rec_keys_path=txt_path,
    )

    img_path = "test_image/check1.jpg"
    result, elapse = engine(img_path)
    print(elapse)
    draw_results(img_path, result)


if __name__ == "__main__":
    main()
