import csv
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from args import get_args
from augmentations import get_conservative_train_transforms
from dataset import DoorHandleDataset
from model import get_model
from trainer import train_one_epoch, validate_one_epoch, collate_fn


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(image_dir, train_split=0.8, seed=42):
    image_dir = Path(image_dir)
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([p.name for p in image_dir.iterdir() if p.suffix.lower() in valid_exts])

    random.seed(seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    return train_files, val_files


def save_loss_history(history, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for row in history:
            writer.writerow([row["epoch"], row["train_loss"], row["val_loss"]])

    print(f"Loss history saved to {save_path}")


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    image_dir = os.path.join(args.data_dir, args.image_dir)
    label_dir = os.path.join(args.data_dir, args.label_dir)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = outputs_dir / args.save_path
    loss_csv_path = outputs_dir / "loss_history.csv"

    train_files, val_files = split_dataset(image_dir, args.train_split, args.seed)

    print(f"Total images: {len(train_files) + len(val_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    train_transforms = (
        get_conservative_train_transforms() if args.augmentation == "conservative" else None
    )
    print(f"Training augmentation: {args.augmentation}")

    train_dataset = DoorHandleDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_files=train_files,
        transforms=train_transforms,
    )
    val_dataset = DoorHandleDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_files=val_files,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = get_model(args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device, epoch)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")

    save_loss_history(history, loss_csv_path)

    print("Training finished.")


if __name__ == "__main__":
    main()