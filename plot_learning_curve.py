import argparse
import csv

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Plot learning curve from CSV file")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/loss_history.csv",
        help="Path to loss history CSV file",
    )
    return parser.parse_args()


def load_loss_history(csv_path):
    epochs = []
    train_losses = []
    val_losses = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    return epochs, train_losses, val_losses


def main():
    args = get_args()

    epochs, train_losses, val_losses = load_loss_history(args.csv)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()