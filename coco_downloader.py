import requests
from tqdm import tqdm
from pycocotools.coco import COCO

import os
import json
import zipfile
from pathlib import Path


class COCOCatDogDownloader:
    def __init__(self):
        self.DOG_LABEL = 0
        self.CAT_LABEL = 1
        self.DATA_FOLDER = Path("cat_dog_images")

    def download_and_prepare_dataset(self, num_instances_per_class=2500):
        if os.path.exists("annotations"):
            print("Annotations already exist, skipping download")
        else:
            self._download_coco_annotations()

        dataset = self._load_cats_and_dogs(
            './annotations/instances_train_val_2017.json', num_instances_per_class)

        self._download_coco_images(dataset)

    def _download_coco_annotations(self):
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        response = requests.get(annotations_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open("annotations_trainval2017.zip", "wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        with zipfile.ZipFile("annotations_trainval2017.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

        os.remove("annotations_trainval2017.zip")

        all_data = dict()

        with open('./annotations/instances_train2017.json', 'r') as f:
            train_data = json.load(f)

        train_data['images'] = list(
            map(lambda data: {**data, 'split': 'train'}, train_data['images']))
        all_data['images'] = train_data['images']
        all_data['annotations'] = train_data['annotations']
        all_data['categories'] = train_data['categories']
        del train_data

        with open('./annotations/instances_val2017.json', 'r') as f:
            val_data = json.load(f)

        val_data['images'] = list(
            map(lambda data: {**data, 'split': 'val'}, val_data['images']))
        all_data['images'] += val_data['images']
        all_data['annotations'] += val_data['annotations']
        del val_data

        with open("./annotations/instances_train_val_2017.json", 'w') as f:
            json.dump(all_data, f, indent=2)

        os.remove("./annotations/instances_train2017.json")
        os.remove("./annotations/instances_val2017.json")

    def _load_cats_and_dogs(self, annotation_file, num_instances_per_class=2500):
        coco = COCO(annotation_file)
        cat_names = ['cat']
        dog_names = ['dog']

        cat_ids = coco.getCatIds(catNms=cat_names)
        dog_ids = coco.getCatIds(catNms=dog_names)

        print(f"Cat category ID: {cat_ids}")
        print(f"Dog category ID: {dog_ids}")

        cat_img_ids = set(coco.getImgIds(catIds=cat_ids))
        dog_img_ids = set(coco.getImgIds(catIds=dog_ids))

        pure_cat_imgs = list(cat_img_ids - dog_img_ids)
        pure_dog_imgs = list(dog_img_ids - cat_img_ids)
        mixed_imgs = list(cat_img_ids & dog_img_ids)

        print(f"Found {len(pure_cat_imgs)} pure cat images")
        print(f"Found {len(pure_dog_imgs)} pure dog images")
        print(f"Found {len(mixed_imgs)} mixed images")

        dataset = []

        for img_id in mixed_imgs:
            img_info = coco.loadImgs(img_id)[0]
            cat_anns = coco.loadAnns(
                coco.getAnnIds(imgIds=img_id, catIds=cat_ids))
            dog_anns = coco.loadAnns(
                coco.getAnnIds(imgIds=img_id, catIds=dog_ids))

            bboxes = []
            labels = []

            for ann in cat_anns:
                bboxes.append(ann['bbox'])
                labels.append(self.CAT_LABEL)

            for ann in dog_anns:
                bboxes.append(ann['bbox'])
                labels.append(self.DOG_LABEL)

            dataset.append({
                'image_id': img_id,
                'filename': img_info['file_name'],
                'split': img_info['split'],  # type: ignore
                'width': img_info['width'],
                'height': img_info['height'],
                'bboxes': bboxes,
                'labels': labels
            })

        cat_dataset = []
        for img_id in pure_cat_imgs:
            img_info = coco.loadImgs(img_id)[0]
            cat_anns = coco.loadAnns(
                coco.getAnnIds(imgIds=img_id, catIds=cat_ids))

            bboxes = []
            labels = []

            for ann in cat_anns:
                bboxes.append(ann['bbox'])
                labels.append(self.CAT_LABEL)

            cat_dataset.append({
                'image_id': img_id,
                'filename': img_info['file_name'],
                'split': img_info['split'],  # type: ignore
                'width': img_info['width'],
                'height': img_info['height'],
                'bboxes': bboxes,
                'labels': labels
            })

        cat_dataset.sort(key=lambda x: len(x['labels']), reverse=True)
        dataset.extend(cat_dataset[:num_instances_per_class])

        dog_dataset = []
        for img_id in pure_dog_imgs:
            img_info = coco.loadImgs(img_id)[0]
            dog_anns = coco.loadAnns(
                coco.getAnnIds(imgIds=img_id, catIds=dog_ids))

            bboxes = []
            labels = []

            for ann in dog_anns:
                bboxes.append(ann['bbox'])
                labels.append(self.DOG_LABEL)

            dog_dataset.append({
                'image_id': img_id,
                'filename': img_info['file_name'],
                'split': img_info['split'],  # type: ignore
                'width': img_info['width'],
                'height': img_info['height'],
                'bboxes': bboxes,
                'labels': labels
            })

        dog_dataset.sort(key=lambda x: len(x['labels']), reverse=True)
        dataset.extend(dog_dataset[:num_instances_per_class])

        print(f"Selected {len(mixed_imgs)} mixed images (all)")
        print(
            f"Selected {min(len(cat_dataset), num_instances_per_class)} pure cat images")
        print(
            f"Selected {min(len(dog_dataset), num_instances_per_class)} pure dog images")

        return dataset

    def _download_coco_images(self, dataset, base_url="http://images.cocodataset.org/"):
        self.DATA_FOLDER.mkdir(exist_ok=True)

        annotations_data = []

        for data in tqdm(dataset):
            filename = data['filename']
            image_path = self.DATA_FOLDER / filename

            if image_path.exists():
                annotations_data.append(data)
                continue

            if data['split'] == "train":
                image_url = base_url + "train2017/" + filename
            elif data['split'] == 'val':
                image_url = base_url + "val2017/" + filename

            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()

                with open(image_path, 'wb') as f:
                    f.write(response.content)

                annotations_data.append(data)

            except requests.exceptions.RequestException as e:
                print(f"Couldn't download image: {filename}, error: {e}")
                continue
            except Exception as e:
                print(f"Failed to save image: {filename}, error: {e}")
                continue

        annotations_file = self.DATA_FOLDER / "cat_dog_annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations_data, f, indent=2)
