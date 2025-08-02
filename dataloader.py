import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import json
from pathlib import Path


class CatDogDataset(Dataset):
    def __init__(self, annotations_file, images_dir=None, transform=None, target_size=(224, 224)):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        if images_dir is None:
            self.images_dir = Path(annotations_file).parent
        else:
            self.images_dir = Path(images_dir)

        self.transform = transform
        self.target_size = target_size

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

        if self.transform:
            image = self.transform(image)

        boxes = torch.tensor(item['bboxes'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.int64)

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

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(item['image_id']),
            'filename': item['filename']
        }

        return image, target


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

    return train_annotations_file, val_annotations_file


def _collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


def create_dataloaders(train_annotations_file, val_annotations_file, images_dir=None,
                       batch_size=8, num_workers=0, train_transform=None, val_transform=None,
                       target_size=(224, 224)):
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
        train_annotations_file, images_dir, train_transform, target_size)
    val_dataset = CatDogDataset(
        val_annotations_file, images_dir, val_transform, target_size)

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
