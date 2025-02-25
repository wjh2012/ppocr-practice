import msgspec


class ImageData(msgspec.Struct):
    path: str
    label: str


class TxtLabelParser:
    def __init__(self, txt_label_path):
        self.txt_label_path = txt_label_path

    def parse(self):
        try:
            with open(self.txt_label_path, "r", encoding="utf-8") as f:
                txt_data = f.read().splitlines()  # 줄 단위 리스트로 변환
                result = []

                for line in txt_data:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) < 2:
                        print(f"잘못된 형식의 라인 무시: {line}")
                        continue

                    image_path = parts[0]
                    label = parts[1]

                    result.append(ImageData(path=image_path, label=label))

                return result

        except Exception as e:
            print(f"txt 읽기 중 오류 발생: {e}")


if __name__ == "__main__":
    txt_path = r"C:\Users\WONJANGHO\Desktop\train\labels.txt"

    reader = TxtLabelParser(txt_path)
    parsed_data = reader.parse()

    print(f"데이터 길이: {len(parsed_data)}")

    if parsed_data[0]:
        first_path = parsed_data[0].path
        first_label = parsed_data[0].label
        print(f"첫 번째 이미지 이름: {first_path}, 텍스트: {first_label}")
