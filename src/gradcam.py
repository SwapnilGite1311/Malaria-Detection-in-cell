"""Task 5: Grad-CAM heatmaps, showing WHERE in the cell the model looked.

Red/yellow areas mattered most for the decision; blue areas barely mattered.
"""
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src import config
from src.dataset import eval_transform


def gradcam_overlay(model, cell_rgb, target_class):
    """Return the cell image with a heatmap for `target_class` blended on top."""
    device = next(model.parameters()).device
    input_tensor = eval_transform(Image.fromarray(cell_rgb)).unsqueeze(0).to(device)

    # layer4 is ResNet18's last block of image-reading layers: it still keeps
    # a rough map of positions, while already understanding "parasite-like" shapes
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        heatmap = cam(input_tensor=input_tensor,
                      targets=[ClassifierOutputTarget(target_class)])[0]

    resized = Image.fromarray(cell_rgb).resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    background = np.asarray(resized, dtype=np.float32) / 255.0  # colours as 0..1
    return show_cam_on_image(background, heatmap, use_rgb=True)
