"""The AI models. The app uses ResNet18; the others exist for the model comparison."""
import torch
from torch import nn
from torchvision import models

from src import config

MODEL_NAMES = ["simple_cnn", "mobilenet_v3_small", "efficientnet_b0", "resnet18"]

# Names shown in the app's model picker
MODEL_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small (recommended)",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
    "simple_cnn": "Simple CNN (no pretraining)",
}


class SimpleCNN(nn.Module):
    """A small network trained from scratch (no pretraining), as a baseline.

    Four blocks of: convolution (finds patterns) -> batch norm (keeps numbers stable)
    -> ReLU (keeps only positive signals) -> max pool (halves the image size).
    """

    def __init__(self, num_classes):
        super().__init__()
        blocks = []
        channels_in = 3
        for channels_out in (32, 64, 128, 256):
            blocks += [
                nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels_in = channels_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)  # average each of the 256 pattern maps to one number
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_model(name="resnet18", pretrained=True):
    """Create a model with a final layer for our 2 classes.

    pretrained=True downloads ImageNet weights (only the first time).
    The app uses pretrained=False because it loads our own trained weights instead.
    """
    num_classes = len(config.CLASSES)

    if name == "simple_cnn":
        return SimpleCNN(num_classes)

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # The last layer answered "which of 1000 ImageNet objects?"; now "Uninfected or Parasitized?"
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}")


def model_path(name):
    """ResNet18 is the app's main model; the others get their own files."""
    return config.MODEL_PATH if name == "resnet18" else config.MODELS_DIR / f"{name}.pth"


def load_trained_model(device, name="resnet18"):
    """Load a trained model, ready for predictions."""
    model = build_model(name, pretrained=False)
    model.load_state_dict(torch.load(model_path(name), map_location=device))
    model.to(device)
    model.eval()  # switch off training-only behaviour
    return model
