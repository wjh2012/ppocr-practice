import msgspec

"""
example = {
    "info": {
        "name": "한글 손글씨 데이터",
        "description": "Korean OCR Data Set (Printed Text, Normal)",
        "data_created": "2019-10-11",
    },
    "images": [
        {
            "id": "00000000",
            "width": 98,
            "height": 124,
            "file_name": "00000000.png",
            "date_captured": "2019-10-03 18:18:14",
        },
        {
            "id": "00000001",
            "width": 91,
            "height": 128,
            "file_name": "00000001.png",
            "date_captured": "2019-10-03 18:11:32",
        },  # ...
    ],
    "annotations": [
        {
            "attributes": {
                "font": "나눔고딕 코딩",
                "type": "글자(음절)",
                "is_aug": false,
            },
            "id": "00009229",
            "image_id": "00009229",
            "text": "넧",
        },
        {
            "attributes": {"font": "하나", "type": "글자(음절)", "is_aug": false},
            "id": "00009230",
            "image_id": "00009230",
            "text": "쀹",
        },  # ...
    ],
}
"""


class Info(msgspec.Struct):
    name: str
    description: str


class Image(msgspec.Struct):
    id: str
    width: int
    height: int
    file_name: str


class Attribute(msgspec.Struct):
    font: str
    type: str
    is_aug: bool


class Annotation(msgspec.Struct):
    attributes: Attribute
    id: str
    image_id: str
    text: str


class PrintedDataInfo(msgspec.Struct):
    info: Info
    images: list[Image]
    annotations: list[Annotation]


class JsonLabelParser:
    def __init__(self, label_json_path):
        self.label_json_path = label_json_path
        self.decoder = msgspec.json.Decoder(PrintedDataInfo)

    def parse_json(self):
        with open(self.label_json_path, "r", encoding="utf-8") as f:
            label_data = self.decoder.decode(f.read())

        image_id_set = {image.id for image in label_data.images}
        annotation_image_id_set = {
            annotation.image_id for annotation in label_data.annotations
        }

        matched_image_ids = image_id_set & annotation_image_id_set
        matched_images = [
            image for image in label_data.images if image.id in matched_image_ids
        ]
        matched_annotations = [
            annotation
            for annotation in label_data.annotations
            if annotation.image_id in matched_image_ids
        ]

        print(
            f"총 이미지 개수: {len(label_data.images)} → 매칭된 이미지 개수: {len(matched_images)}"
        )
        print(
            f"총 어노테이션 개수: {len(label_data.annotations)} → 매칭된 어노테이션 개수: {len(matched_annotations)}"
        )

        return PrintedDataInfo(
            info=label_data.info, images=matched_images, annotations=matched_annotations
        )


# 예제 실행
if __name__ == "__main__":
    json_path = r"D:\ml\한국어글자체이미지\02.인쇄체_230721_add\printed_data_info.json"

    reader = JsonLabelParser(json_path)
    parsed_data = reader.parse_json()

    print(f"데이터 길이: {len(parsed_data.images)}")

    if parsed_data.annotations:
        first_annotation = parsed_data.annotations[0]
        print(
            f"첫 번째 이미지 이름: {first_annotation.image_id}, 속성: {first_annotation.attributes}, 텍스트: {first_annotation.text}"
        )

    if parsed_data.images:
        first_image = parsed_data.images[0]
        print(
            f"첫 번째 이미지 파일: {first_image.file_name}, 크기: {first_image.width}x{first_image.height}"
        )
