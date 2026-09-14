"""Task 9: compare every trained model on the same tests.

Train the models first, e.g.:  python -m src.train --model mobilenet_v3_small
Run:                           python -m src.compare_models
"""
import json
import re
import time

import torch

import src.smear as smear_module
from src import config
from src.dataset import get_dataloaders
from src.model import MODEL_NAMES, load_trained_model, model_path
from src.real_eval import REAL_DIR, evaluate

RESULTS_DIR = config.OUTPUTS_DIR / "comparison"
PRETTY_NAMES = {"simple_cnn": "Simple CNN (from scratch)", "mobilenet_v3_small": "MobileNetV3-Small",
                "efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet18 (used in app)"}


def test_probabilities(model, loader, device):
    """Infection probability and true label for every Kaggle test cell."""
    probabilities, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            probabilities.append(torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu())
            labels.append(y)
    return torch.cat(probabilities), torch.cat(labels).bool()


def scores(probabilities, labels, threshold):
    predicted = probabilities >= threshold
    return {
        "accuracy": (predicted == labels).float().mean().item(),
        "sensitivity": (predicted & labels).sum().item() / labels.sum().item(),
        "specificity": (~predicted & ~labels).sum().item() / (~labels).sum().item(),
    }


def milliseconds_per_cell(model, device, batch_size, repeats=20):
    """Average time to classify one cell. The GPU is timed in batches, the CPU one at a time."""
    model = model.to(device).eval()
    batch = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    with torch.no_grad():
        for _ in range(3):  # warm-up: the first runs are slow while things get set up
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()  # GPU work runs in the background; wait for it before timing
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / (repeats * batch_size) * 1000


def training_minutes(name):
    """Read training time from the saved history, or from the ResNet18 training log."""
    suffix = "" if name == "resnet18" else f"_{name}"
    history_file = config.OUTPUTS_DIR / f"training_history{suffix}.json"
    if history_file.exists():
        history = json.loads(history_file.read_text())
        if "training_minutes" in history:
            return history["training_minutes"]
    log_file = config.OUTPUTS_DIR / "train_log.txt"
    if name == "resnet18" and log_file.exists():
        return sum(int(s) for s in re.findall(r"\| (\d+)s", log_file.read_text())) / 60
    return None


def main():
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    test_loader = get_dataloaders()["test"]
    real_entries = None
    if (REAL_DIR / "labels.json").exists():
        real_entries = json.loads((REAL_DIR / "labels.json").read_text())
        smear_module.STAIN_NORMALIZE = True

    rows = []
    for name in MODEL_NAMES:
        if not model_path(name).exists():
            print(f"Skipping {name}: not trained yet")
            continue
        print(f"Testing {name}...", flush=True)
        model = load_trained_model(gpu, name)
        probabilities, labels = test_probabilities(model, test_loader, gpu)

        row = {
            "model": name,
            "parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
            "file_mb": model_path(name).stat().st_size / 1e6,
            "training_minutes": training_minutes(name),
            "kaggle_at_50": scores(probabilities, labels, 0.5),
            "kaggle_at_80": scores(probabilities, labels, config.INFECTED_THRESHOLD),
            "gpu_ms_per_cell": milliseconds_per_cell(model, gpu, batch_size=64),
        }
        if real_entries:
            report, _ = evaluate(real_entries, model, gpu)
            row["real"] = {key: report[key] for key in
                           ("sensitivity", "specificity", "mean_parasitemia_error")}
        row["cpu_ms_per_cell"] = milliseconds_per_cell(model, cpu, batch_size=1)
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(rows, indent=2))

    # A Markdown table, easy to paste into notes or slides
    lines = [
        "| Model | Params (M) | File (MB) | Train (min) | Kaggle acc | Kaggle sens | Kaggle spec "
        "| Real sens | Real spec | Real parasitemia error | CPU ms/cell | GPU ms/cell |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        k, real = r["kaggle_at_80"], r.get("real", {})
        pct = lambda v: f"{v:.1%}" if v is not None else "n/a"
        lines.append(
            f"| {PRETTY_NAMES[r['model']]} | {r['parameters_millions']:.2f} | {r['file_mb']:.1f} "
            f"| {r['training_minutes']:.0f} | {pct(k['accuracy'])} | {pct(k['sensitivity'])} "
            f"| {pct(k['specificity'])} | {pct(real.get('sensitivity'))} "
            f"| {pct(real.get('specificity'))} | {pct(real.get('mean_parasitemia_error'))} "
            f"| {r['cpu_ms_per_cell']:.1f} | {r['gpu_ms_per_cell']:.2f} |"
        )
    table = "\n".join(lines)
    (RESULTS_DIR / "model_comparison.md").write_text(
        f"Scores use the app's rule: infected at {config.INFECTED_THRESHOLD:.0%} or more.\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
