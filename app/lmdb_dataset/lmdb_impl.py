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
        # LMDB 저장 디렉토리가 없으면 생성
        os.makedirs(self.lmdb_path, exist_ok=True)

    def create_lmdb(
        self,
        map_size: int,
        datas: list[ImageData],
        image_root: str,
        batch_size: int = 100,
    ):
        try:
            env = lmdb.open(
                self.lmdb_path, map_size=map_size, writemap=True, sync=False
            )

            # 기존 num-samples 값 확인 (최초 실행 시 0)
            with env.begin(write=True) as txn:
                num_samples_bytes = txn.get("num-samples".encode("utf-8"))
                if num_samples_bytes is not None:
                    valid_samples = int(num_samples_bytes.decode("utf-8"))
                    print(f"기존 LMDB의 샘플 수: {valid_samples}")
                else:
                    valid_samples = 0

            cache = {}
            batch_count = 0

            for data in tqdm(datas, desc="LMDB 데이터셋 생성중"):
                image_path = os.path.join(image_root, data.path)
                try:
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                except Exception as e:
                    # 이미지 파일 읽기 실패 시 건너뜁니다.
                    continue

                valid_samples += 1
                batch_count += 1

                image_key = "image-%09d".encode("utf-8") % valid_samples
                label_key = "label-%09d".encode("utf-8") % valid_samples

                cache[image_key] = image_data
                cache[label_key] = data.label.encode("utf-8")

                # 배치가 완료되면 별도의 트랜잭션으로 커밋
                if batch_count == batch_size:
                    with env.begin(write=True) as txn:
                        self._write_cache(txn, cache)
                        txn.put(
                            "num-samples".encode("utf-8"),
                            str(valid_samples).encode("utf-8"),
                        )
                    cache = {}
                    batch_count = 0

            # 남은 데이터 커밋
            if cache:
                with env.begin(write=True) as txn:
                    self._write_cache(txn, cache)
                    txn.put(
                        "num-samples".encode("utf-8"),
                        str(valid_samples).encode("utf-8"),
                    )

            env.close()
            print(f"✅ LMDB 데이터셋 생성 완료! (총 {valid_samples}개 샘플)")
        except Exception as e:
            print(f"데이터 삽입 중 오류 발생: {e}")

    def _write_cache(self, txn, cache):
        for k, v in cache.items():
            txn.put(k, v)
