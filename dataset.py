import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset


class DoorHandleDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_files=None, transforms=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms

        valid_exts = {".jpg", ".jpeg", ".png"}

        if image_files is None:
            self.image_files = sorted(
                [p.name for p in self.image_dir.iterdir() if p.suffix.lower() in valid_exts]
            )
        else:
            self.image_files = sorted(image_files)

        self._check_missing_labels_only()

    def _check_missing_labels_only(self):
        image_stems = {Path(name).stem for name in self.image_files}
        label_stems = {p.stem for p in self.label_dir.glob("*.txt")}

        missing_labels = sorted(image_stems - label_stems)

        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            warnings.warn(
                f"{len(missing_labels)} image(s) do not have matching label files. "
                f"They will be treated as images with no objects. Examples: {preview}"
            )

    def __len__(self):
        return len(self.image_files)

    def _read_yolo_label(self, label_path, img_width, img_height):
        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                warnings.warn(
                    f"Invalid label format in {label_path.name} at line {line_num}: '{line.strip()}'. "
                    f"This line will be skipped."
                )
                continue

            try:
                class_id, x_center, y_center, width, height = map(float, parts)
            except ValueError:
                warnings.warn(
                    f"Non-numeric label values in {label_path.name} at line {line_num}: '{line.strip()}'. "
                    f"This line will be skipped."
                )
                continue

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

            if xmax <= xmin or ymax <= ymin:
                warnings.warn(
                    f"Invalid box in {label_path.name} at line {line_num}. "
                    f"This box will be skipped."
                )
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(class_id) + 1)  # background = 0, object classes start from 1

        return boxes, labels

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = self.image_dir / image_name
        label_path = self.label_dir / f"{Path(image_name).stem}.txt"

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        img_width, img_height = image.size

        boxes, labels = self._read_yolo_label(label_path, img_width, img_height)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        if self.transforms is not None:
            try:
                image, target = self.transforms(image, target)
            except TypeError:
                image = self.transforms(image)

        return image, target