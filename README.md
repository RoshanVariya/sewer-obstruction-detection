# Sewer Obstruction Detection

Computer-vision object detection for a sewer inspection robot project developed for ROBOfest Gujarat.

The included trained Ultralytics checkpoint loads successfully in the developer's local environment and has been tested on two example images. The checkpoint reports two classes:

- `tree_root`
- `obstacle`

> **Status:** Initial inference verified. This repository does not yet include a full quantitative evaluation. Two successful predictions do not establish model accuracy or field reliability.

## Demo results observed during initial testing

| Test image | Console result |
|---|---|
| `01000249.jpg` | 1 detection reported as `tree_root` |
| `00001060.jpg` | 1 detection reported as `obstacle` |

These are observations from two individual runs, not a benchmark. The source images and annotated outputs are not included in this repository because they have not been supplied here.

## Repository contents

```text
sewer-obstruction-detection/
├── models/
│   ├── best.pt
│   └── MODEL_CARD.md
├── src/
│   ├── predict.py
│   └── batch_predict.py
├── docs/
│   ├── dataset.md
│   ├── model_details.md
│   └── evaluation.md
├── examples/
│   ├── input/
│   └── predictions/
├── results/
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick start

### 1. Prerequisites

Use a Python version supported by the PyTorch and Ultralytics versions available for your operating system. The original developer environment used Python 3.14.6, but compatibility can vary by package release.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can call the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Predict one image

```powershell
python src/predict.py --source "C:\path\to\sewer_image.jpg"
```

By default, the script loads `models/best.pt`, prints the task and class names, and saves annotated predictions in an Ultralytics `runs/detect/predict...` folder.

Set a confidence threshold if needed:

```powershell
python src/predict.py --source "C:\path\to\sewer_image.jpg" --conf 0.25
```

### 4. Predict a folder of images

```powershell
python src/batch_predict.py --source "C:\path\to\sewer_images"
```

You can also provide a video file or other source supported by Ultralytics to `predict.py`.

## Model classes: an important limitation

The checkpoint's class mapping is:

```text
0: tree_root
1: obstacle
```

`obstacle` is a general category. Do not interpret it as separate detection classes for plastic, sludge, stones, or other specific materials unless the dataset and training annotations confirm that distinction.

## Dataset, training, and evaluation

The dataset source, redistribution permissions, train/validation/test split, exact model architecture, training configuration, and quantitative metrics still need to be filled in from the original project records. See:

- [`docs/dataset.md`](docs/dataset.md)
- [`docs/model_details.md`](docs/model_details.md)
- [`docs/evaluation.md`](docs/evaluation.md)

No precision, recall, mAP, or accuracy figures are claimed here because verified metrics have not been provided.
