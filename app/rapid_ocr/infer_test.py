import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont
from rapidocr_onnxruntime import RapidOCR


def draw_results_kr(image_path, results, output_path="output.jpg"):
    # 이미지 로드
    image = cv2.imread(image_path)

    # 박스 그리기 (draw_results와 동일하게 cv2로 그리기, 두께 1)
    for result in results:
        box, text, conf = result
        box = np.array(box, dtype=np.int32)  # 박스 좌표 변환
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=1)

    # cv2로 그려진 박스가 포함된 이미지를 PIL로 변환 (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # 한글 폰트 설정 (시스템에 있는 TTF 폰트 경로로 변경)
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows: 맑은고딕
    font = ImageFont.truetype(font_path, 20)

    # 텍스트 그리기 (draw_results와 좌표 동일하게, y-5)
    for result in results:
        box, text, conf = result
        box = np.array(box, dtype=np.int32)
        x, y = box[0]  # 좌상단 좌표
        draw.text((x, y - 5), f"{text} ({conf:.2f})", font=font, fill=(255, 0, 0))

    # PIL 이미지를 다시 BGR로 변환 후 저장
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)
    print(f"결과가 {output_path}로 저장되었습니다.")


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
    det_model_path = "./det_models/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.onnx"

    rec_model_path = "./rec_models/micr_v2/micr_v2.onnx"
    txt_path = r"C:\Users\WONJANGHO\Desktop\AI\inference\rec\250226\micr_dict.txt"
    # rec_model_path = (
    #     "./rec_models/korean_PP-OCRv3_rec_infer/korean_PP-OCRv3_rec_infer.onnx"
    # )
    txt_path = r"C:\Users\WONJANGHO\Desktop\AI\model\inference\rec\250226\micr_dict.txt"

    engine = RapidOCR(
        use_det=True,
        use_cls=False,
        use_rec=True,
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        rec_keys_path=txt_path,
    )

    img_path = "test_image/check1.jpg"
    result, elapse = engine(img_path, box_thresh=0.5, text_score=0.92)
    print(elapse)
    # draw_results_kr(img_path, result)
    draw_results(img_path, result)


if __name__ == "__main__":
    main()
