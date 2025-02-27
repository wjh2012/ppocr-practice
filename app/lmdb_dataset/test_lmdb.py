import argparse

import lmdb


def check_lmdb_keys_all(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()

        # 상위 10개 키-값 출력
        print("=== 상위 10개 키-값 ===")
        count = 0
        for key, value in cursor:
            try:
                key_str = key.decode()
            except Exception as e:
                key_str = str(key)
            print(f"Key: {key_str}, Value Length: {len(value)} bytes")
            count += 1
            if count >= 10:
                break

        # 하위 10개 키-값 출력 (역순으로 탐색)
        print("\n=== 하위 10개 키-값 ===")
        count = 0
        if cursor.last():
            while True:
                try:
                    key_str = cursor.key().decode()
                except Exception as e:
                    key_str = str(cursor.key())
                print(f"Key: {key_str}, Value Length: {len(cursor.value())} bytes")
                count += 1
                if count >= 10:
                    break
                if not cursor.prev():
                    break
        else:
            print("데이터베이스에 키가 존재하지 않습니다.")

        num_samples = txn.get("num-samples".encode("utf-8"))
        if num_samples is None:
            print("num-samples 키가 존재하지 않습니다.")
        else:
            try:
                num_samples = int(num_samples.decode())
                print(f"총 샘플 수: {num_samples}")
            except Exception as e:
                print("num-samples 값을 정수로 변환하는데 실패했습니다.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LMDB 테스트 스크립트")
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default=r"C:\Users\WONJANGHO\Desktop\lmdb",
        help="LMDB 저장 기본 경로",
    )
    args = parser.parse_args()

    check_lmdb_keys_all(lmdb_path=args.lmdb_path)
