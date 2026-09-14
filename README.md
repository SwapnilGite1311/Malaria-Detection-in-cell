# 🔬 Malaria Smear Analyzer

Finds red blood cells in a microscope image of a blood smear, checks each cell for the
malaria parasite, and reports the **parasitemia** (the percentage of infected cells).
Grad-CAM heatmaps show where the model looked. A sidebar model picker switches between
4 trained models (MobileNetV3-Small by default) to compare their decisions.

> Student project for learning purposes. Not a medical device.

## Results

A cell is called infected when the model is at least 80% sure. The two tables below
are for ResNet18; see the model comparison for all 4 models.

**Kaggle test set** (2,609 unseen single-cell images, *P. falciparum*)

| Metric | Score |
|---|---|
| Accuracy | 95.98% |
| Sensitivity (infected cells caught) | 93.04% |
| Specificity (healthy cells cleared) | 98.68% |

**Real smear photos** (40 photos, 1,869 expert-labelled cells, *P. vivax*, from
[BBBC041](https://bbbc.broadinstitute.org/BBBC041))

| Metric | Score |
|---|---|
| Expert-labelled cells found | 72.9% |
| Boxes that were real cells | 74.7% |
| Sensitivity (on found cells) | 95.9% |
| Specificity (on found cells) | 98.4% |
| Average parasitemia error per photo | 4.2 percentage points |

**Model comparison** (full table in `PROJECT_NOTES.md`, Task 9)

| Model | Parameters | Kaggle acc | Real-photo parasitemia error | CPU per cell |
|---|---|---|---|---|
| Simple CNN (from scratch) | 0.39 M | 94.0% | 29.3 pts | 44 ms |
| MobileNetV3-Small | 1.52 M | 96.2% | 1.9 pts | 8.5 ms |
| EfficientNet-B0 | 4.01 M | 96.6% | 2.0 pts | 63 ms |
| ResNet18 | 11.18 M | 96.0% | 4.2 pts | 85 ms |

MobileNetV3-Small is the app's default model.

## Setup

1. Download the [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
   and unzip it so that `cell_images/Parasitized` and `cell_images/Uninfected` exist.
2. Create the environment and install the libraries:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch from https://pytorch.org first.

## Run

```bash
.venv/Scripts/python.exe -m src.dataset    # check the data and the split
.venv/Scripts/python.exe -m src.train      # fine-tune the model (~12 min on an RTX 3050)
.venv/Scripts/python.exe -m src.evaluate   # score it on the test set
.venv/Scripts/python.exe -m src.smear      # build and analyse demo smears
.venv/Scripts/streamlit.exe run app.py     # open the web app

.venv/Scripts/python.exe -m scripts.fetch_real_smears   # download 40 real photos (~10 MB)
.venv/Scripts/python.exe -m src.real_eval               # score the system on them
.venv/Scripts/python.exe -m pytest -v                   # run the 31 automated tests (~70 s)

.venv/Scripts/python.exe -m src.train --model mobilenet_v3_small   # train another model
.venv/Scripts/python.exe -m src.compare_models                     # compare all trained models

.venv/Scripts/python.exe -m scripts.build_space                    # build the online demo folder
.venv/Scripts/python.exe -m scripts.upload_space --space USER/NAME  # upload it (after: hf auth login)
```

Real smear photos: BBBC041 P. vivax Malaria Blood Smear Dataset by Jane Hung,
Broad Bioimage Benchmark Collection, licensed CC BY-NC-SA 3.0.

## Project structure

```
app.py              Streamlit web app
src/config.py       All settings (paths, image size, epochs...)
src/dataset.py      Load images, split by source photo, augmentation
src/model.py        ResNet18 with a 2-class final layer
src/train.py        Training loop, saves the best model
src/evaluate.py     Test-set metrics and confusion matrix
src/smear.py        Cell detection (OpenCV), per-cell classification, parasitemia
src/gradcam.py      Grad-CAM heatmaps
src/real_eval.py    Score the system on real photos against expert labels
scripts/            Download the real smear photo sample
tests/              27 automated tests
sample_smears/      Demo smears built from unseen test cells
real_smears/        Real photos + expert labels (downloaded, not committed)
outputs/            Charts, metrics and analysed images
models/             Trained weights (not committed)
PROJECT_NOTES.md    How everything works, explained simply
```
