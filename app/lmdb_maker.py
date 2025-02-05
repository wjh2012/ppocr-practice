import os
import lmdb
from PIL import Image
from tqdm import tqdm

from app.data_reader.label_parser import PrintedDataInfo


class LMDBMaker:
    def __init__(
        self,
        output_path,
        labels: PrintedDataInfo,
        image_root,
        map_size=10 * 1024 * 1024 * 1024,
        check_valid=True,
        batch_size=1000,
    ):
        self.output_path = output_path
        self.image_root = image_root
        self.map_size = int(map_size)
        self.check_valid = check_valid
        self.labels = labels
        self.batch_size = batch_size

        self.image_dict = {image.id: image for image in self.labels.images}

    def create_lmdb_dataset(self):
        env = lmdb.open(self.output_path, map_size=self.map_size, writemap=True)
        num_samples = len(self.labels.annotations)

        with env.begin(write=True) as txn:
            txn.put("num-samples".encode(), str(num_samples).encode())

        with tqdm(total=num_samples, desc="LMDB 생성 중") as pbar:
            batch = []
            for idx, annotation in enumerate(self.labels.annotations):
                batch.append((idx, annotation))
                if len(batch) >= self.batch_size:
                    self.write_batch(env, batch)
                    pbar.update(len(batch))
                    batch.clear()

            if batch:
                self.write_batch(env, batch)
                pbar.update(len(batch))

        print("✅ LMDB 데이터셋 생성 완료!")
        env.close()

    def write_batch(self, env, batch):
        with env.begin(write=True) as txn:
            for idx, annotation in batch:
                image = self.image_dict.get(annotation.image_id)
                if image is None:
                    continue

                image_path = os.path.join(self.image_root, image.file_name)
                if not os.path.exists(image_path) or (
                    self.check_valid and not self.check_image_is_valid(image_path)
                ):
                    continue

                image_key = f"image-{idx+1:07d}".encode()
                label_key = f"label-{idx+1:07d}".encode()

                with open(image_path, "rb") as f:
                    image_bin = f.read()

                txn.put(image_key, image_bin)
                txn.put(label_key, annotation.text.encode("utf-8"))

    @staticmethod
    def check_image_is_valid(image_path):
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception:
            return False
