"""Automated checks for the whole system: data, model, cell detection, Grad-CAM and app.

Run:  python -m pytest -v
"""
import io

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import src.smear as smear_module
from src import config
from src.dataset import eval_transform, list_images, source_photo, split_samples
from src.smear import BACKGROUND_COLOR, analyze_smear, detect_cells, make_synthetic_smear

needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                 reason="train the model first: python -m src.train")


# ---------- Shared setup (fixtures run once and are reused by every test) ----------

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model(device):
    from src.model import load_trained_model
    return load_trained_model(device)


@pytest.fixture(scope="session")
def splits():
    return split_samples(list_images())


@pytest.fixture(scope="session")
def smear_1(splits):
    return make_synthetic_smear(splits["test"], 0.25, seed=1)


def found_vs_truth(image, truth):
    """Return (cells detected, cells actually placed)."""
    return len(detect_cells(image)), len(truth)


# ---------- Data ----------

def test_all_images_are_used(splits):
    assert sum(len(s) for s in splits.values()) == 27558


def test_no_photo_is_shared_between_sets(splits):
    photos = {name: {source_photo(p) for p, _ in s} for name, s in splits.items()}
    assert not photos["train"] & photos["val"]
    assert not photos["train"] & photos["test"]
    assert not photos["val"] & photos["test"]


def test_split_is_the_same_every_run(splits):
    assert split_samples(list_images()) == splits


# ---------- Model ----------

@needs_model
def test_model_accuracy_on_test_cells(model, device, splits):
    sample = splits["test"][::10]  # every 10th test cell keeps the test quick
    correct = 0
    for start in range(0, len(sample), 64):
        chunk = sample[start:start + 64]
        batch = torch.stack([eval_transform(Image.open(p).convert("RGB")) for p, _ in chunk])
        with torch.no_grad():
            predictions = model(batch.to(device)).argmax(dim=1).cpu().tolist()
        correct += sum(pred == label for pred, (_, label) in zip(predictions, chunk))
    assert correct / len(sample) > 0.93


# ---------- Cell detection ----------

@pytest.mark.parametrize("image", [
    np.full((900, 1400, 3), BACKGROUND_COLOR, np.uint8),
    np.full((600, 800, 3), 255, np.uint8),
    np.zeros((600, 800, 3), np.uint8),
    np.full((20, 20, 3), 200, np.uint8),
], ids=["blank slide", "pure white", "pure black", "tiny image"])
def test_empty_images_have_no_cells(image):
    assert detect_cells(image) == []


def test_camera_noise_is_not_cells():
    rng = np.random.default_rng(0)
    slide = np.full((900, 1400, 3), BACKGROUND_COLOR, np.float32)
    noisy = np.clip(slide + rng.normal(0, 8, slide.shape), 0, 255).astype(np.uint8)
    assert detect_cells(noisy) == []


@needs_model
@pytest.mark.parametrize("seed, infected_fraction", [(1, 0.25), (2, 0.10), (3, 0.0)])
def test_demo_smears_are_counted_correctly(model, device, splits, seed, infected_fraction):
    image, truth = make_synthetic_smear(splits["test"], infected_fraction, seed=seed)
    _, summary, _ = analyze_smear(image, model, device)
    assert summary["total_cells"] == len(truth)
    assert abs(summary["infected_cells"] - sum(t[4] for t in truth)) <= 1


def test_jpeg_compression(smear_1):
    image, truth = smear_1
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=70)
    found, expected = found_vs_truth(np.array(Image.open(buffer)), truth)
    assert found == expected


def test_dark_background(smear_1):
    image, truth = smear_1
    dark = image.copy()
    dark[(image == np.array(BACKGROUND_COLOR, np.uint8)).all(axis=2)] = (40, 30, 40)
    found, expected = found_vs_truth(dark, truth)
    assert abs(found - expected) <= 1


def test_uneven_lighting(smear_1):
    image, truth = smear_1
    darker_on_left = np.linspace(0.55, 1.0, image.shape[1])[None, :, None]
    found, expected = found_vs_truth((image * darker_on_left).astype(np.uint8), truth)
    assert abs(found - expected) <= 1


@pytest.mark.parametrize("scale", [0.5, 3.0])
def test_different_image_sizes(smear_1, scale):
    image, truth = smear_1
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, expected = found_vs_truth(resized, truth)
    assert abs(found - expected) <= 2


def test_touching_cells_are_separated(monkeypatch, splits):
    monkeypatch.setattr(smear_module, "GAP", -25)  # negative gap = cells overlap
    image, truth = make_synthetic_smear(splits["test"], 0.25, n_cells=30, seed=7)
    found, expected = found_vs_truth(image, truth)
    assert abs(found - expected) <= 0.15 * expected


# ---------- Grad-CAM ----------

@pytest.mark.parametrize("name", ["simple_cnn", "mobilenet_v3_small", "efficientnet_b0", "resnet18"])
def test_gradcam_heatmap_for_every_model(device, splits, name):
    from src.gradcam import gradcam_overlay
    from src.model import load_trained_model, model_path
    if not model_path(name).exists():
        pytest.skip(f"train it first: python -m src.train --model {name}")
    model = load_trained_model(device, name)
    path, label = splits["test"][0]
    overlay = gradcam_overlay(model, np.array(Image.open(path).convert("RGB")), label)
    assert overlay.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8


# ---------- Web app ----------

def test_bad_uploads_do_not_crash():
    from app import read_image
    assert read_image(io.BytesIO(b"this is not an image")) is None
    assert read_image(io.BytesIO(b"")) is None


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_unusual_image_formats_load_as_rgb(mode):
    from app import read_image
    buffer = io.BytesIO()
    Image.new(mode, (50, 40)).save(buffer, "PNG")
    assert read_image(buffer).shape == (40, 50, 3)


@needs_model
def test_csv_report(model, device, smear_1):
    from app import build_csv_report
    results, summary, _ = analyze_smear(smear_1[0], model, device)
    rows = build_csv_report(results, summary, "resnet18").strip().splitlines()
    assert rows[0] == "model,ResNet18"
    assert rows[-1].startswith(str(len(results)))  # one row per cell, numbered from 1


@needs_model
def test_single_cell_prediction(model, device, splits):
    from app import predict_single_cell
    path, _ = splits["test"][0]
    label, probability = predict_single_cell(np.array(Image.open(path).convert("RGB")),
                                             model, device)
    assert 0.0 <= probability <= 1.0
    assert label == int(probability >= config.INFECTED_THRESHOLD)


@needs_model
@pytest.mark.skipif(not (config.PROJECT_DIR / "real_smears" / "labels.json").exists(),
                    reason="download real photos first: python -m scripts.fetch_real_smears")
def test_real_smear_photos(model, device):
    """40 real microscope photos labelled by experts (BBBC041). Scores measured on
    2026-09-14 were recall 73%, precision 73%, sensitivity 96%, specificity 98%;
    the bars below leave some margin so small changes don't fail the test."""
    import json
    import src.smear as smear_module
    from src.real_eval import REAL_DIR, evaluate
    smear_module.STAIN_NORMALIZE = True
    entries = json.loads((REAL_DIR / "labels.json").read_text())
    report, _ = evaluate(entries, model, device)
    assert report["detection_recall"] >= 0.65
    assert report["detection_precision"] >= 0.65
    assert report["sensitivity"] >= 0.85
    assert report["specificity"] >= 0.95


@needs_model
def test_app_analyses_a_sample_smear():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(config.PROJECT_DIR / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception
    app.selectbox(key="sample").select("smear_1.png").run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Cells found"] == "23"


@needs_model
def test_app_model_picker():
    from streamlit.testing.v1 import AppTest
    from app import available_models
    app = AppTest.from_file(str(config.PROJECT_DIR / "app.py"), default_timeout=300)
    app.run()
    picker = app.selectbox(key="model")
    assert picker.value == (config.DEFAULT_MODEL if config.DEFAULT_MODEL in available_models()
                            else available_models()[0])

    app.selectbox(key="sample").select("smear_1.png").run()
    for name in available_models():  # every model must analyse the smear without errors
        app.selectbox(key="model").select(name).run()
        assert not app.exception, name
        assert app.selectbox(key="model").value == name
        assert {m.label: m.value for m in app.metric}["Cells found"] == "23"
