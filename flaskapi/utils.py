import io
import os
import timm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from utils.utils_train import best_or_last_checkpoint, load_checkpoint

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_model(config):

    # load model
    model = timm.create_model(config.MODEL, pretrained=True, num_classes=config.NCLASSES, in_chans=1)

    # checkpoint path
    model_checkpoints_path = os.path.join(config.CHECKPOINTS_PATH, config.MODEL)

    # check if there are any checkpoints
    checkpoint_path = best_or_last_checkpoint(path=model_checkpoints_path)

    checkpoint = load_checkpoint(path=checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model


def process_image(config, image):

    preprocess = transforms.Compose(
                                    [
                                        transforms.Resize(config.IMAGE_SIZE + 1),
                                        transforms.CenterCrop(config.IMAGE_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        transforms.Grayscale(num_output_channels=1)
                                    ]
                                )

    return preprocess(image).unsqueeze(0)


def save_grad_cam(config, model, input_tensor: torch.Tensor, image: Image, target, use_cuda: bool):

    preprocess = transforms.Compose(
                                    [
                                        transforms.Resize(config.IMAGE_SIZE + 1),
                                        transforms.CenterCrop(config.IMAGE_SIZE),
                                        transforms.ToTensor(),
                                    ]
                                )

    image = preprocess(image).permute(1, 2, 0)
    image = np.float32(image)

    target_layers = [model.bn2]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    targets = [ClassifierOutputTarget(target)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

    name = f"gradcam/{target}.jpeg"
    visualization = Image.fromarray(visualization)
    # visualization.save(name)

    return visualization

