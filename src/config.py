"""All project settings live here, so no other file has hidden 'magic numbers'."""
from pathlib import Path

# Folders (Path objects join with "/" and work on Windows, Mac and Linux)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "cell_images"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# The position in this list is the label number: 0 = healthy, 1 = infected
CLASSES = ["Uninfected", "Parasitized"]

# 224x224 is the image size the pretrained model originally learned on
IMAGE_SIZE = 224
BATCH_SIZE = 64

# Training: one "epoch" = the model sees every training image once
EPOCHS = 5
LEARNING_RATE = 1e-4  # small steps, so we gently adjust what the model already knows
LEARNING_RATE_FROM_SCRATCH = 1e-3  # the simple CNN starts knowing nothing
MODEL_PATH = MODELS_DIR / "best_model.pth"

# Decisions: a cell is called infected only when the model is at least 80% sure.
# Healthy cells far outnumber infected ones, so even a few false alarms inflate
# the parasitemia %. Cells between 20% and 80% are flagged for a human to check.
INFECTED_THRESHOLD = 0.8
REVIEW_ABOVE = 0.2

# A fixed seed makes the "random" split identical every time we run the code
SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # the remaining 10% becomes the test set

# Pretrained ImageNet models expect colours scaled with these exact numbers
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
