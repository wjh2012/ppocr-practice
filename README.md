```shell
# 생성 스크립트
python app/lmdb_dataset/run_json.py --lmdb_size 200 --json_path /home/taehwa/ocr_train_dataset/kor_font/printed/printed_data_info.json --data_path /home/taehwa/ocr_train_dataset/kor_font/printed/extracted --lmdb_base_path /home/taehwa/data_test
python app/lmdb_dataset/run_json.py --lmdb_size 200 --json_path /home/taehwa/ocr_train_dataset/kor_font/printed_aug/augmentation_data_info.json --data_path /home/taehwa/ocr_train_dataset/kor_font/printed_aug/extracted --lmdb_base_path /home/taehwa/data_test

# 테스트 스크립트
python app/lmdb_dataset/test_lmdb.py --lmdb_path /home/taehwa/data_test/lmdb_train
python app/lmdb_dataset/test_lmdb.py --lmdb_path /home/taehwa/data_test/lmdb_val
```
