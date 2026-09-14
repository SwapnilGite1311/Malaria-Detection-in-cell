"""Task 2: fine-tune the pretrained model on our malaria cell images.

Run:  python -m src.train
"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from src import config
from src.dataset import get_dataloaders
from src.model import MODEL_NAMES, build_model, model_path


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    """Go through a whole dataset once. Trains if an optimizer is given, else only measures."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, correct, seen = 0.0, 0, 0
    # no_grad() skips the bookkeeping needed for learning, making evaluation faster
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # autocast uses 16-bit numbers where safe: faster, and fits a 4 GB GPU
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                outputs = model(images)          # 1. the model makes a guess
                loss = loss_fn(outputs, labels)  # 2. measure how wrong the guess is

            if is_training:
                optimizer.zero_grad()            # 3. clear the previous step's adjustments
                scaler.scale(loss).backward()    # 4. work out how each weight caused the error
                scaler.step(optimizer)           # 5. nudge the weights to reduce the error
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / seen, correct / seen


def plot_history(history, path):
    """Draw loss and accuracy per epoch, to show the model really learned."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set(title="Loss (lower is better)", xlabel="Epoch")

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="validation")
    ax_acc.set(title="Accuracy (higher is better)", xlabel="Epoch")

    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(model_name="resnet18"):
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    print(f"Model: {model_name}")

    loaders = get_dataloaders()
    model = build_model(model_name, pretrained=True).to(device)

    # A model trained from scratch has no pretrained knowledge to protect,
    # so it can take bigger learning steps
    learning_rate = (config.LEARNING_RATE_FROM_SCRATCH if model_name == "simple_cnn"
                     else config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    save_path = model_path(model_name)
    suffix = "" if model_name == "resnet18" else f"_{model_name}"  # keep ResNet18's files as they were
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    training_start = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], loss_fn, device,
                                          optimizer, scaler)
        val_loss, val_acc = run_epoch(model, loaders["val"], loss_fn, device)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        # Keep only the version that did best on validation data
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved = "  <- best so far, saved"

        print(f"Epoch {epoch}/{config.EPOCHS} | "
              f"train acc {train_acc:.2%} loss {train_loss:.3f} | "
              f"val acc {val_acc:.2%} loss {val_loss:.3f} | "
              f"{time.time() - start:.0f}s{saved}", flush=True)

    plot_history(history, config.OUTPUTS_DIR / f"training_curves{suffix}.png")
    history["training_minutes"] = (time.time() - training_start) / 60
    history["best_val_acc"] = best_val_acc
    with open(config.OUTPUTS_DIR / f"training_history{suffix}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":  # required on Windows when DataLoader uses num_workers
    parser = argparse.ArgumentParser(description="Fine-tune a model on the malaria cells")
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    main(parser.parse_args().model)
