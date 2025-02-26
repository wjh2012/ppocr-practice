import cv2

from rapid_layout import RapidLayout, VisLayout


def main():
    layout_engine = RapidLayout(model_type="doclayout_docstructbench", conf_thres=0.1)
    img_path = "test_image/check2.jpg"
    img = cv2.imread(img_path)

    boxes, scores, class_names, elapse = layout_engine(img)
    ploted_img = VisLayout.draw_detections(img, boxes, scores, class_names)
    for _ in range(len(boxes)):
        print(boxes)
        print(scores)
        print(class_names)
    if ploted_img is not None:
        cv2.imwrite("layout_out.jpg", ploted_img)


if __name__ == "__main__":
    main()
