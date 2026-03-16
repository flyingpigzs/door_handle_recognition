import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class DoorHandleDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_files=None, transforms=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms

        if image_files is None:
            valid_exts = {".jpg", ".jpeg", ".png"}
            self.image_files = sorted(
                [p.name for p in self.image_dir.iterdir() if p.suffix.lower() in valid_exts]
            )
        else:
            self.image_files = sorted(image_files)

    def __len__(self):
        return len(self.image_files)

    def _read_yolo_label(self, label_path, img_width, img_height):
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            return boxes, labels

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)

            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id) + 1)  # background=0, object classes start from 1

        return boxes, labels

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = self.image_dir / image_name
        label_path = self.label_dir / f"{Path(image_name).stem}.txt"

        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        boxes, labels = self._read_yolo_label(label_path, img_width, img_height)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        image = torch.from_numpy(
            __import__("numpy").array(image, dtype="float32") / 255.0
        ).permute(2, 0, 1)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target