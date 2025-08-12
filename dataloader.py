import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import json
from pathlib import Path
from yolo_helpers import remove_bg


class CatDogDataset(Dataset):
    def __init__(self, annotations_file, images_dir=None, transform=None, target_size=(224, 224), use_albumentations=True, use_bg_removal=False, bg_removal_p=0.25):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        if images_dir is None:
            self.images_dir = Path(annotations_file).parent
        else:
            self.images_dir = Path(images_dir)

        self.transform = transform
        self.target_size = target_size
        self.use_albumentations = use_albumentations
        self.use_bg_removal = use_bg_removal
        self.bg_removal_p = bg_removal_p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = self.images_dir / item['filename']
        image = Image.open(image_path).convert('RGB')

        orig_width, orig_height = image.size

        if self.target_size:
            image = image.resize(self.target_size)
            target_width, target_height = self.target_size
        else:
            target_width, target_height = orig_width, orig_height

        boxes = np.array(item['bboxes'], dtype=np.float32)
        labels = np.array(item['labels'], dtype=np.int64)

        if len(boxes) > 0 and self.target_size:
            scale_x = target_width / orig_width
            scale_y = target_height / orig_height

            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

        if len(boxes) > 0:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        image_np = np.array(image)
        
        if self.use_bg_removal and len(boxes) > 0 and np.random.random() < self.bg_removal_p:
            bboxes_normalized = []
            for bbox in boxes:
                x_min, y_min, x_max, y_max = bbox
                x_min_norm = max(0, min(1, x_min / target_width))
                y_min_norm = max(0, min(1, y_min / target_height))
                x_max_norm = max(0, min(1, x_max / target_width))
                y_max_norm = max(0, min(1, y_max / target_height))
                if x_max_norm > x_min_norm and y_max_norm > y_min_norm:
                    bboxes_normalized.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            
            if bboxes_normalized:
                image_np = apply_bg_removal(image_np, bboxes_normalized)

        if self.use_albumentations and self.transform:
            bboxes_albumentations = []
            class_labels = []

            if len(boxes) > 0:
                for i, bbox in enumerate(boxes):
                    x_min, y_min, x_max, y_max = bbox
                    x_min_norm = max(0, min(1, x_min / target_width))
                    y_min_norm = max(0, min(1, y_min / target_height))
                    x_max_norm = max(0, min(1, x_max / target_width))
                    y_max_norm = max(0, min(1, y_max / target_height))

                    if x_max_norm > x_min_norm and y_max_norm > y_min_norm:
                        bboxes_albumentations.append(
                            [x_min_norm, y_min_norm, x_max_norm, y_max_norm])
                        class_labels.append(labels[i])

            transformed = self.transform(
                image=image_np,
                bboxes=bboxes_albumentations,
                class_labels=class_labels
            )

            image = transformed['image']

            if len(transformed['bboxes']) > 0:
                transformed_boxes = np.array(transformed['bboxes'])
                transformed_boxes[:, 0] *= target_width
                transformed_boxes[:, 1] *= target_height
                transformed_boxes[:, 2] *= target_width
                transformed_boxes[:, 3] *= target_height
                boxes = torch.tensor(transformed_boxes, dtype=torch.float32)
                labels = torch.tensor(
                    transformed['class_labels'], dtype=torch.int64)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
        else:
            if self.transform:
                image = self.transform(Image.fromarray(image_np))
            else:
                image = Image.fromarray(image_np)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(item['image_id']),
            'filename': item['filename']
        }

        return image, target


def apply_bg_removal(image, bboxes):
    if len(bboxes) == 0:
        return image

    height, width = image.shape[:2]
    pixel_bboxes = []
    for bbox in bboxes:
        x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox
        x_min = x_min_norm * width
        y_min = y_min_norm * height
        w = (x_max_norm - x_min_norm) * width
        h = (y_max_norm - y_min_norm) * height
        pixel_bboxes.append([x_min, y_min, w, h])

    _, bg_removed_pil = remove_bg(image, pixel_bboxes)

    return np.array(bg_removed_pil)



def get_train_augmentations(target_size=(224, 224), p=0.5):
    transforms_list = [
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
        ], p=p),

        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=p * 0.2),

        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=p * 0.7
        ),

        A.RandomSizedBBoxSafeCrop(
            height=target_size[0],
            width=target_size[1],
            erosion_rate=0.2,
            p=p * 0.3
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(transforms_list, bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))


def get_val_augmentations():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))


def _stratified_split(data, val_ratio=0.15, random_state=42):
    CAT_LABEL = 1
    DOG_LABEL = 0

    pure_cat = []
    pure_dog = []
    mixed = []

    for i, item in enumerate(data):
        labels = item['labels']
        has_cat = CAT_LABEL in labels
        has_dog = DOG_LABEL in labels

        if has_cat and has_dog:
            mixed.append(i)
        elif has_cat:
            pure_cat.append(i)
        elif has_dog:
            pure_dog.append(i)

    np.random.seed(random_state)

    mixed_val_size = max(1, int(len(mixed) * val_ratio))
    mixed_val_indices = np.random.choice(mixed, mixed_val_size, replace=False)
    mixed_train_indices = [i for i in mixed if i not in mixed_val_indices]

    cat_val_size = max(1, int(len(pure_cat) * val_ratio))
    cat_val_indices = np.random.choice(
        pure_cat, cat_val_size, replace=False) if len(pure_cat) > 0 else []
    cat_train_indices = [i for i in pure_cat if i not in cat_val_indices]

    dog_val_size = max(1, int(len(pure_dog) * val_ratio))
    dog_val_indices = np.random.choice(
        pure_dog, dog_val_size, replace=False) if len(pure_dog) > 0 else []
    dog_train_indices = [i for i in pure_dog if i not in dog_val_indices]

    train_indices = mixed_train_indices + cat_train_indices + dog_train_indices
    val_indices = list(mixed_val_indices) + \
        list(cat_val_indices) + list(dog_val_indices)

    print(f"\nSplit Results:")
    print(f"Training set: {len(train_indices)} images")
    print(f"- Mixed: {len(mixed_train_indices)}")
    print(f"- Pure cat: {len(cat_train_indices)}")
    print(f"- Pure dog: {len(dog_train_indices)}")

    print(f"Validation set: {len(val_indices)} images")
    print(f"- Mixed: {len(mixed_val_indices)}")
    print(f"- Pure cat: {len(cat_val_indices)}")
    print(f"- Pure dog: {len(dog_val_indices)}")

    return train_indices, val_indices


def create_split_datasets(annotations_file, val_ratio=0.15, random_state=42):
    with open(annotations_file, 'r') as f:
        full_data = json.load(f)

    if val_ratio > 0:
        train_indices, val_indices = _stratified_split(
            full_data, val_ratio, random_state)

        train_data = [full_data[i] for i in train_indices]
        val_data = [full_data[i] for i in val_indices]

        data_folder = Path(annotations_file).parent

        train_annotations_file = data_folder / "train_annotations.json"
        val_annotations_file = data_folder / "val_annotations.json"

        with open(train_annotations_file, 'w') as f:
            json.dump(train_data, f, indent=2, sort_keys=True)

        with open(val_annotations_file, 'w') as f:
            json.dump(val_data, f, indent=2, sort_keys=True)

        print(f"\nSaved:")
        print(f"- Training annotations: {train_annotations_file}")
        print(f"- Validation annotations: {val_annotations_file}")

    else:
        train_annotations_file = annotations_file
        val_annotations_file = "./test_annotations.json"

    return train_annotations_file, val_annotations_file


def _collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


def create_dataloaders(train_annotations_file, val_annotations_file, images_dir=None,
                       batch_size=8, num_workers=0, train_transform=None, val_transform=None,
                       target_size=(224, 224), use_albumentations=True, augmentation_strength=0.5,
                       use_bg_removal=True, bg_removal_p=0.25):
    if use_albumentations:
        if train_transform is None:
            train_transform = get_train_augmentations(
                target_size, p=augmentation_strength)

        if val_transform is None:
            val_transform = get_val_augmentations()
    else:
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

    train_dataset = CatDogDataset(
        train_annotations_file, images_dir, train_transform, target_size, use_albumentations, use_bg_removal, bg_removal_p)
    val_dataset = CatDogDataset(
        val_annotations_file, images_dir, val_transform, target_size, use_albumentations, False, 0.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn
    )

    return train_loader, val_loader
