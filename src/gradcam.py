"""Task 5: Grad-CAM heatmaps, showing WHERE in the cell the model looked.

Red/yellow areas mattered most for the decision; blue areas barely mattered.
"""
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models

from src import config
from src.dataset import eval_transform
from src.model import SimpleCNN


def target_layer(model):
    """The last image-reading layer of each model type.

    That layer still keeps a rough map of positions, while already understanding
    "parasite-like" shapes, which is exactly what a heatmap needs.
    """
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    if isinstance(model, (models.MobileNetV3, models.EfficientNet)):
        return model.features[-1]
    if isinstance(model, SimpleCNN):
        return model.features[-2]  # the last ReLU, before the final max pool shrinks the map
    raise TypeError(f"No Grad-CAM layer defined for {type(model).__name__}")


def gradcam_overlay(model, cell_rgb, target_class):
    """Return the cell image with a heatmap for `target_class` blended on top."""
    device = next(model.parameters()).device
    input_tensor = eval_transform(Image.fromarray(cell_rgb)).unsqueeze(0).to(device)

    with GradCAM(model=model, target_layers=[target_layer(model)]) as cam:
        heatmap = cam(input_tensor=input_tensor,
                      targets=[ClassifierOutputTarget(target_class)])[0]

    resized = Image.fromarray(cell_rgb).resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    background = np.asarray(resized, dtype=np.float32) / 255.0  # colours as 0..1
    return show_cam_on_image(background, heatmap, use_rgb=True)
