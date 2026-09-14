"""Task 1: find the cell images, split them fairly, and serve them to PyTorch.

Run this file directly to check the data:  python -m src.dataset
"""
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


def list_images():
    """Return a list of (image_path, label) pairs for every cell image."""
    samples = []
    for label, class_name in enumerate(config.CLASSES):
        folder = config.DATA_DIR / class_name
        # glob("*.png") skips junk files like Thumbs.db that Windows creates
        for path in sorted(folder.glob("*.png")):
            samples.append((path, label))
    return samples


def source_photo(path):
    """Each cell was cut out of a bigger microscope photo.

    'C100P61ThinF_IMG_20150918_144104_cell_162.png' -> 'C100P61ThinF_IMG_20150918_144104'
    """
    return path.stem.split("_cell_")[0]


def split_samples(samples):
    """Split into train / validation / test sets *by source photo*.

    Cells cut from the same photo look very alike. If some went into training
    and others into testing, the test would be too easy and the accuracy fake.
    So every photo's cells go into exactly one set.
    """
    groups = defaultdict(list)
    for path, label in samples:
        groups[source_photo(path)].append((path, label))

    photo_names = sorted(groups)
    random.Random(config.SEED).shuffle(photo_names)

    n_train = int(len(photo_names) * config.TRAIN_SPLIT)
    n_val = int(len(photo_names) * config.VAL_SPLIT)
    split_names = {
        "train": photo_names[:n_train],
        "val": photo_names[n_train:n_train + n_val],
        "test": photo_names[n_train + n_val:],
    }
    # A list comprehension: flatten the cells of every photo in the split
    return {split: [cell for name in names for cell in groups[name]]
            for split, names in split_names.items()}


# Training images get random flips and turns ("augmentation") so the model
# learns what a parasite looks like, not which way up the cell happened to be.
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

# Validation/test images are never changed randomly, so scores are repeatable
eval_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


class CellDataset(Dataset):
    """PyTorch asks a Dataset two things: how many items, and 'give me item i'."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def get_dataloaders():
    """A DataLoader hands the model shuffled batches of 64 images at a time."""
    splits = split_samples(list_images())
    loaders = {}
    for split, samples in splits.items():
        is_train = split == "train"
        dataset = CellDataset(samples, train_transform if is_train else eval_transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=is_train,
            # 2 background processes load images in parallel. Each one loads its
            # own copy of PyTorch (~1.5 GB), so more would overflow 8 GB of RAM.
            num_workers=2,
            pin_memory=True,  # speeds up copying batches to the GPU
        )
    return loaders


def save_sample_grid(samples, path, count=12):
    """Save a picture of random cells with their labels, as a sanity check."""
    picks = random.Random(config.SEED).sample(samples, count)
    fig, axes = plt.subplots(2, count // 2, figsize=(count, 4.5))
    for ax, (image_path, label) in zip(axes.flat, picks):
        ax.imshow(Image.open(image_path))
        ax.set_title(config.CLASSES[label], color="red" if label else "green")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    samples = list_images()
    print(f"Found {len(samples)} images in {config.DATA_DIR}")

    splits = split_samples(samples)
    for split, split_samples_ in splits.items():
        infected = sum(label for _, label in split_samples_)
        print(f"{split:>5}: {len(split_samples_):>6} cells  "
              f"({infected} infected, {len(split_samples_) - infected} healthy)")

    # Prove that no microscope photo is shared between two sets
    photos = {split: {source_photo(p) for p, _ in s} for split, s in splits.items()}
    overlap = (photos["train"] & photos["val"]) | (photos["train"] & photos["test"]) \
        | (photos["val"] & photos["test"])
    print(f"Photos shared between sets: {len(overlap)} (should be 0)")

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    save_sample_grid(samples, config.OUTPUTS_DIR / "sample_cells.png")
    print(f"Saved a sample picture to {config.OUTPUTS_DIR / 'sample_cells.png'}")
