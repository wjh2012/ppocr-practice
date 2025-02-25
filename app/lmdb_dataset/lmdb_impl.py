import os
import lmdb
import msgspec
from tqdm import tqdm
import numpy as np
import cv2


class ImageData(msgspec.Struct):
    path: str
    label: str


class LMDBRepository:
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path

    def create_lmdb(
        self,
        map_size: int,
        datas: list[ImageData],
        image_root: str,
        batch_size: int = 100,
    ):
        """LMDB 데이터셋을 생성하는 함수 (배치 처리 최적화)"""
        try:
            env = lmdb.open(self.lmdb_path, map_size=map_size)
            with env.begin(write=True) as txn:
                cache = {}  # 배치 저장을 위한 캐시
                valid_samples = 0  # 실제 저장된(유효한) 샘플 개수

                for data in tqdm(datas, desc="LMDB 데이터셋 생성중"):
                    image_path = os.path.join(image_root, data.path)
                    try:
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                    except Exception as e:
                        print(f"이미지 읽기 에러 ({image_path}): {e}")
                        continue

                    valid_samples += 1
                    image_key = "image-%09d".encode() % valid_samples
                    label_key = "label-%09d".encode() % valid_samples

                    cache[image_key] = image_data
                    cache[label_key] = data.label.encode("utf-8")

                    # 일정 개수마다 배치 저장
                    if valid_samples % batch_size == 0:
                        self._write_cache(txn, cache)
                        cache = {}  # 캐시 초기화

                # 남은 데이터 저장
                if cache:
                    self._write_cache(txn, cache)

                txn.put(
                    "num-samples".encode("utf-8"), str(valid_samples).encode("utf-8")
                )

            env.close()
            print(f"✅ LMDB 데이터셋 생성 완료! (총 {valid_samples}개 샘플)")
        except Exception as e:
            print(f"데이터 삽입 중 오류 발생: {e}")

    def _write_cache(self, txn, cache):
        for k, v in cache.items():
            txn.put(k, v)


# 테스트 함수
def write_test():
    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\micr_train"
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
        batch_size=1000,  # 배치 크기 지정
    )


def check_lmdb_keys(lmdb_path=r"C:\Users\WONJANGHO\Desktop\micr_train", index=1):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        # 확인할 키 이름 생성
        image_key = f"image-{index:09d}"
        label_key = f"label-{index:09d}"

        # 해당 키의 데이터 읽기
        image = txn.get(image_key.encode())
        label = txn.get(label_key.encode())

        # 키 존재 여부 출력
        if image is None:
            print(f"{image_key} 키가 존재하지 않습니다.")
        else:
            print(f"{image_key} 키가 존재합니다.")

        if label is None:
            print(f"{label_key} 키가 존재하지 않습니다.")
        else:
            print(f"{label_key} 키가 존재합니다.")

        # 전체 샘플 수 확인 (num-samples 키)
        num_samples = txn.get("num-samples".encode())
        if num_samples is None:
            print("num-samples 키가 존재하지 않습니다.")
        else:
            num_samples = int(num_samples.decode())
            print(f"총 샘플 수: {num_samples}")

    env.close()


def check_lmdb_keys2(lmdb_path=r"C:\Users\WONJANGHO\Desktop\micr_train", index=1):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()

        print("=== 상위 10개 키-값 ===")
        count = 0
        for key, value in cursor:
            print(f"Key: {key.decode()}, Value Length: {len(value)} bytes")
            count += 1
            if count >= 10:
                break

        print("\n=== 특정 키 확인 ===")
        # 확인할 키 이름 생성
        image_key = f"image-{index:09d}"
        label_key = f"label-{index:09d}"

        # 해당 키의 데이터 읽기
        image = txn.get(image_key.encode())
        label = txn.get(label_key.encode())

        # 키 존재 여부 출력
        if image is None:
            print(f"{image_key} 키가 존재하지 않습니다.")
        else:
            print(f"{image_key} 키가 존재합니다.")

        if label is None:
            print(f"{label_key} 키가 존재하지 않습니다.")
        else:
            print(f"{label_key} 키가 존재합니다.")

        # 전체 샘플 수 확인 (num-samples 키)
        num_samples = txn.get("num-samples".encode())
        if num_samples is None:
            print("num-samples 키가 존재하지 않습니다.")
        else:
            num_samples = int(num_samples.decode())
            print(f"총 샘플 수: {num_samples}")

    env.close()


def check_first_image_channel(lmdb_path=r"C:\Users\WONJANGHO\Desktop\micr_train"):
    # LMDB 환경 열기 (읽기 전용)
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        # 첫 번째 이미지의 키 생성 (인덱스는 1부터 시작)
        image_key = "image-000000001".encode("utf-8")
        image_bytes = txn.get(image_key)
        if image_bytes is None:
            print("첫 번째 이미지 키가 존재하지 않습니다.")
            return

        # 바이트 데이터를 NumPy 배열로 변환
        image_np = np.frombuffer(image_bytes, np.uint8)
        # OpenCV로 이미지 디코딩 (기본적으로 BGR 컬러 이미지)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            print("이미지 디코딩에 실패했습니다.")
            return

        # 이미지 shape 출력 (예: (높이, 너비, 채널))
        print("첫 번째 이미지의 shape:", image.shape)
        # 채널 수 출력 (shape가 3차원인 경우)
        if len(image.shape) == 3:
            print("채널 수:", image.shape[2])
        else:
            print("채널 수 정보를 확인할 수 없습니다.")

    env.close()


if __name__ == "__main__":
    output_lmdb_path = r"C:\Users\WONJANGHO\Desktop\AI\micr_eval"
    check_lmdb_keys(lmdb_path=output_lmdb_path, index=40)
