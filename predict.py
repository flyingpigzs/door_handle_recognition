import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageOps
from torchvision.transforms import functional as F

from model import get_model


def get_args():
    parser = argparse.ArgumentParser(description="Batch prediction for door handle detection")

    parser.add_argument("--model", type=str, default="outputs/fasterrcnn_door_handle.pth", help="Path to trained model")
    parser.add_argument("--input", type=str, default="test_images", help="Input folder containing test images")
    parser.add_argument("--output", type=str, default="prediction_results", help="Output folder for prediction results")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for filtering predictions")
    parser.add_argument("--topk", type=int, default=1, help="Keep top-k predictions after threshold filtering")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--show-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw probability scores on output images (disable with --no-show-scores)",
    )

    return parser.parse_args()


def load_model(model_path, num_classes=2, device="cpu"):
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def predict_image(model, image_path, device, score_threshold=0.3, keep_top_k=1):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])[0]

    boxes = predictions["boxes"].cpu()
    labels = predictions["labels"].cpu()
    scores = predictions["scores"].cpu()

    filtered = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            filtered.append((box.tolist(), label.item(), score.item()))

    filtered.sort(key=lambda x: x[2], reverse=True)

    if keep_top_k is not None and keep_top_k > 0:
        filtered = filtered[:keep_top_k]

    filtered_boxes = [x[0] for x in filtered]
    filtered_labels = [x[1] for x in filtered]
    filtered_scores = [x[2] for x in filtered]

    return image, filtered_boxes, filtered_labels, filtered_scores


def draw_predictions(image, boxes, labels, scores, class_names=None, show_scores=True):
    if class_names is None:
        class_names = {1: "door_handle"}

    image = image.copy()
    draw = ImageDraw.Draw(image)

    font = None
    if show_scores:
        try:
            # Use a larger TrueType font when available for readability.
            font_size = 120  # larger for readability
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
        except Exception:
            font = None

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        if not show_scores:
            continue

        label_name = class_names.get(label, str(label))
        text = f"{label_name}: {score:.2f}"

        text_x = xmin
        # Place text above the box when possible, otherwise inside the box.
        text_y = ymin - 140
        if text_y < 0:
            text_y = ymin + 2

        # Draw a filled background for readability.
        if font is not None:
            left, top, right, bottom = draw.textbbox((text_x, text_y), text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((text_x, text_y), text)
        pad = 6
        draw.rectangle([left - pad, top - pad, right + pad, bottom + pad], fill="red")

        if font is not None:
            draw.text((text_x, text_y), text, fill="white", font=font)
        else:
            draw.text((text_x, text_y), text, fill="white")

    return image


def get_image_files(input_dir):
    valid_exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts])


def main():
    args = get_args()

    model_path = Path(args.model)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    score_threshold = args.threshold
    keep_top_k = args.topk
    device = args.device
    num_classes = 2

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(input_dir)

    if not image_files:
        raise FileNotFoundError(f"No image files found in: {input_dir}")

    model, device = load_model(
        model_path=model_path,
        num_classes=num_classes,
        device=device,
    )

    print(f"Found {len(image_files)} image(s) in {input_dir}")
    print(f"Saving prediction results to {output_dir}")
    print(
        f"threshold={score_threshold}, topk={keep_top_k}, device={device}, "
        f"show_scores={args.show_scores}"
    )

    for image_path in image_files:
        image, boxes, labels, scores = predict_image(
            model=model,
            image_path=image_path,
            device=device,
            score_threshold=score_threshold,
            keep_top_k=keep_top_k,
        )

        result_image = draw_predictions(
            image, boxes, labels, scores, show_scores=args.show_scores
        )

        output_path = output_dir / image_path.name
        result_image.save(output_path)

        print(f"{image_path.name}: detected {len(boxes)} object(s), saved to {output_path}")

    print("Prediction finished.")


if __name__ == "__main__":
    main()