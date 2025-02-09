import os
import lmdb
import msgspec
from tqdm import tqdm


class ImageData(msgspec.Struct):
    path: str
    label: str


class LMDBRepository:
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path

    def create_lmdb(self, map_size, datas: list[ImageData], image_root):
        try:
            env = lmdb.open(self.lmdb_path, map_size=map_size)
            with env.begin(write=True) as txn:
                valid_samples = 0  # 실제 저장된(유효한) 샘플 개수
                for data in tqdm(datas, desc="LMDB 데이터셋 생성중"):
                    image_path = os.path.join(image_root, data.path)
                    try:
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                    except Exception as e:
                        continue

                    valid_samples += 1
                    image_key = f"image-{valid_samples:09d}".encode("utf-8")
                    label_key = f"label-{valid_samples:09d}".encode("utf-8")

                    txn.put(image_key, image_data)
                    txn.put(label_key, data.label.encode("utf-8"))

                txn.put(
                    "num-samples".encode("utf-8"), str(valid_samples).encode("utf-8")
                )
            env.close()
            print("✅ LMDB 데이터셋 생성 완료!")
        except Exception as e:
            print(f"데이터 삽입 중 오류 발생: {e}")

    def read_lmdb(self, top_n: int):
        try:
            env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
            labels = []
            with env.begin() as txn:
                # 1부터 top_n까지 "label-%09d" 형식의 key로 레이블 읽기
                for i in range(1, top_n + 1):
                    label_key = f"label-{i:09d}".encode("utf-8")
                    label_value = txn.get(label_key)
                    if label_value is None:
                        break
                    labels.append(label_value.decode("utf-8"))
            env.close()
            return labels
        except Exception as e:
            print(f"LMDB 읽기 중 오류 발생: {e}")
            return []


def write_test():
    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\lmdb_train"
    test_image_root_path = r"C:\Users\WONJANGHO\Desktop\test"
    test_data = [
        ImageData(path="1.jpg", label="테스트라벨1입니다"),
        ImageData(path="2.jpg", label="테스트라벨2입니다"),
        ImageData(path="3.jpg", label="테스트라벨3입니다"),
        ImageData(path="4.jpg", label="테스트라벨4입니다"),
        ImageData(path="5.jpg", label="테스트라벨5입니다"),
        ImageData(path="6.jpg", label="테스트라벨6입니다"),
        ImageData(path="7.jpg", label="테스트라벨7입니다"),
        ImageData(path="8.jpg", label="테스트라벨8입니다"),
        ImageData(path="9.jpg", label="테스트라벨9입니다"),
        ImageData(path="10.jpg", label="테스트라벨10입니다"),
    ]

    repo = LMDBRepository(output_lmdb_path)
    repo.create_lmdb(
        map_size=10 * 1024 * 1024 * 1024,  # 10GB
        datas=test_data,
        image_root=test_image_root_path,
    )


def read_test(top_n=5):
    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\lmdb_train"
    repo = LMDBRepository(output_lmdb_path)

    read_data = repo.read_lmdb(top_n=top_n)
    print(f"상위 {top_n}개의 레이블:")
    for label in read_data:
        print(label)


if __name__ == "__main__":
    read_test()
