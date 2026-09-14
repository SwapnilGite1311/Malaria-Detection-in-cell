"""Task 3: the final exam. Score the trained model on test cells it has never seen.

Run:  python -m src.evaluate
"""
import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from src import config
from src.dataset import get_dataloaders
from src.model import load_trained_model


def predict_all(model, loader, device):
    """Return true labels and predicted labels for every image in the loader.

    Uses the same rule as the app: infected only if the model is at least
    INFECTED_THRESHOLD sure, so these scores describe what users actually see.
    """
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            infection_probability = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            predicted = (infection_probability >= config.INFECTED_THRESHOLD).long()
            predicted_labels.extend(predicted.cpu().tolist())
            true_labels.extend(labels.tolist())
    return true_labels, predicted_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(device)
    y_true, y_pred = predict_all(model, get_dataloaders()["test"], device)

    # Confusion matrix layout: rows = the truth, columns = what the model said
    # [[healthy called healthy,  healthy called infected ],
    #  [infected called healthy, infected called infected]]
    (true_neg, false_pos), (false_neg, true_pos) = confusion_matrix(y_true, y_pred)

    results = {
        "test_cells": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        # Sensitivity: of all truly infected cells, how many did we catch?
        "sensitivity": true_pos / (true_pos + false_neg),
        # Specificity: of all truly healthy cells, how many did we correctly clear?
        "specificity": true_neg / (true_neg + false_pos),
        "missed_infections": int(false_neg),
        "false_alarms": int(false_pos),
    }

    print(classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4))
    for name, value in results.items():
        print(f"{name:>18}: {value:.2%}" if isinstance(value, float) else f"{name:>18}: {value}")

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(config.OUTPUTS_DIR / "test_results.json", "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=config.CLASSES, cmap="Blues", colorbar=False)
    display.ax_.set_title(f"Test set: {results['accuracy']:.2%} accuracy")
    display.figure_.tight_layout()
    display.figure_.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=100)
    print(f"Saved results to {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
