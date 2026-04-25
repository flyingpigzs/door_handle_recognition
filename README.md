# Door Handle Detection with Faster R-CNN

This project was created for the course **Image Recognition and Labeling**.

The goal of the project is to train an object detection model to detect **door handles** in images using:

- Custom dataset (self-captured images)
- Annotation with CVAT
- YOLO annotation format
- Faster R-CNN model (PyTorch / Torchvision)

---

## Project Structure

```bash
├── args.py
├── main.py
├── dataset.py
├── model.py
├── trainer.py
├── predict.py
├── plot_learning_curve.py
│
├── data/
│   ├── images/
│   └── labels/
│
├── outputs/
│   ├── fasterrcnn_door_handle.pth
│   └── loss_history.csv
│
├── README.md
└── .gitignore
```

### File descriptions

| File                   | Description                                      |
|------------------------|--------------------------------------------------|
| args.py                | Defines training arguments and hyperparameters   |
| main.py                | Entry point, creates loaders and starts training |
| dataset.py             | Loads images and YOLO labels                     |
| model.py               | Defines Faster R-CNN model                       |
| trainer.py             | Training and validation loops                    |
| predict.py             | Run inference on test images                     |
| plot_learning_curve.py | Plot training / validation loss curve            |

---

## Dataset

Images were captured manually (~100 images).

Annotation was done using **CVAT**.

The dataset is automatically split into training and validation sets inside `main.py`.

Export format: 

```Ultralytics YOLO (Detection)```

Folder structure:

```bash
data/
images/
labels/
```

Each label file: 

```class x_center y_center width height```

YOLO format is converted to Faster R-CNN format inside `dataset.py`.

---

## Classes

Only one object class: 

```door_handle```

For Faster R-CNN:

```
num_classes = 2
0 = background
1 = door_handle
```

---

## Model

We use: torchvision.models.detection.fasterrcnn_resnet50_fpn

The classifier head is replaced to match our number of classes.

Defined in: 

```model.py```

---

## Training

Run training:

```python main.py –data-dir data –epochs 10 –batch-size 2 –num-classes 2```

Example:

```python main.py –epochs 15 –batch-size 4```

During training:

- Training loss and validation loss are recorded
- Best model is saved automatically
- Results are stored in the outputs folder

Outputs:

outputs/
  fasterrcnn_door_handle.pth
  loss_history.csv

---

## Learning Curve

After training, loss history is saved as:

```outputs/loss_history.csv```

To plot the learning curve:

```python plot_learning_curve.py```

or

```python plot_learning_curve.py --csv outputs/loss_history.csv```

## Prediction

To run inference on new images, place images inside:

test_images/

Then run:

```python predict.py```

Optional parameters:

```python predict.py --threshold 0.3 --topk 1```

Arguments:

--threshold : confidence threshold  
--topk : keep only top-k detections per image  

Prediction results will be saved to:

```prediction_results/```

## Requirements

Recommended Python version:

```Python 3.10+```

Install dependencies:

```pip install -r requirements.txt```

## Outputs

Training results are saved in:

```outputs/```

Files:

fasterrcnn_door_handle.pth  → best model  
loss_history.csv            → loss per epoch  

Prediction results are saved in:

```prediction_results/```