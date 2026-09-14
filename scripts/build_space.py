"""Build a folder ready to upload as a free Hugging Face Space (the online demo).

Run:  python -m scripts.build_space
Output: deploy/hf_space/  (upload the CONTENTS of this folder to your Space)

A Space runs our app inside a "Docker container": a small Linux computer described
by the Dockerfile below. Free Spaces have no GPU, so we install the smaller
CPU-only version of PyTorch.
"""
import json
import shutil

from src import config
from src.model import MODEL_NAMES, model_path

SPACE_DIR = config.PROJECT_DIR / "deploy" / "hf_space"

FILES_TO_COPY = [
    "app.py",
    "src/__init__.py",
    "src/config.py",
    "src/dataset.py",
    "src/model.py",
    "src/gradcam.py",
    "src/smear.py",
    "outputs/test_results.json",
    "outputs/comparison/model_comparison.json",  # the app's sidebar shows these scores
]

REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.7.1
torchvision==0.22.1
numpy==2.2.6
pillow==11.2.1
matplotlib==3.10.6
opencv-python-headless==5.0.0.93
streamlit==1.63.0
grad-cam==1.5.7
"""

DOCKERFILE = """\
FROM python:3.11-slim

# OpenCV needs this system library
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as a normal user (id 1000), not as administrator
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install libraries first, so rebuilding after a code change is quick
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""

STREAMLIT_CONFIG = """\
[browser]
gatherUsageStats = false

[server]
headless = true
"""

SPACE_README = """\
---
title: Malaria Smear Analyzer
emoji: 🔬
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Finds red blood cells and flags malaria parasites
---

# 🔬 Malaria Smear Analyzer

Upload a microscope image of a thin blood smear. The app finds each red blood cell,
checks it for the malaria parasite, and reports the parasitemia (percentage of
infected cells). Grad-CAM heatmaps show where the model looked.

Pick between 4 trained models in the sidebar (MobileNetV3-Small, EfficientNet-B0,
ResNet18, and a simple CNN without pretraining) to compare their decisions on the same image.

Try one of the built-in sample smears if you don't have an image.

> Student project for learning purposes. **Not a medical device.** Do not use it for diagnosis.

Training data: Malaria Cell Images Dataset (NIH / Lister Hill National Center for
Biomedical Communications), via Kaggle. Runs on a free CPU, so analysis takes a few seconds.
"""


def main():
    if SPACE_DIR.exists():
        shutil.rmtree(SPACE_DIR)  # start clean so deleted files don't linger

    for relative in FILES_TO_COPY:
        target = SPACE_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.PROJECT_DIR / relative, target)

    # Every trained model, so the online model picker offers the same choices
    for name in MODEL_NAMES:
        if model_path(name).exists():
            target = SPACE_DIR / model_path(name).relative_to(config.PROJECT_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path(name), target)

    smear_target = SPACE_DIR / "sample_smears"
    smear_target.mkdir(parents=True)
    for smear in sorted((config.PROJECT_DIR / "sample_smears").glob("*.png")):
        shutil.copy2(smear, smear_target / smear.name)

    (SPACE_DIR / "requirements.txt").write_text(REQUIREMENTS)
    (SPACE_DIR / "Dockerfile").write_text(DOCKERFILE)
    (SPACE_DIR / ".streamlit").mkdir()
    (SPACE_DIR / ".streamlit" / "config.toml").write_text(STREAMLIT_CONFIG)
    (SPACE_DIR / "README.md").write_text(SPACE_README, encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in SPACE_DIR.rglob("*") if p.is_file()) / 1e6
    print(f"Built {SPACE_DIR} ({total_mb:.1f} MB)")
    for path in sorted(SPACE_DIR.rglob("*")):
        if path.is_file():
            print("  ", path.relative_to(SPACE_DIR))


if __name__ == "__main__":
    main()
