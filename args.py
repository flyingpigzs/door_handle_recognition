import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for door handle detection")

    parser.add_argument("--data-dir", type=str, default="data", help="Root directory of the dataset")
    parser.add_argument("--image-dir", type=str, default="images", help="Image folder inside data-dir")
    parser.add_argument("--label-dir", type=str, default="labels", help="Label folder inside data-dir")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")

    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")

    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes including background")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--save-path", type=str, default="fasterrcnn_door_handle.pth", help="Model save path")

    parser.add_argument(
        "--augmentation",
        type=str,
        default="conservative",
        choices=["none", "conservative"],
        help="Training augmentation: none, or conservative (ColorJitter + random horizontal flip with bbox sync)",
    )

    return parser.parse_args()