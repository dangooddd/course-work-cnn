#!/usr/bin/env python

import torch
from PIL import Image
from argparse import ArgumentParser
from utils import (
    get_model,
    get_eval_transforms,
    get_last_layer,
    VGG_MODELS,
    RESNET_MODELS,
    ALEXNET_MODELS,
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def logits_to_class(logits):
    idx_to_class = [
        "butterfly",
        "cat",
        "chicken",
        "cow",
        "dog",
        "elephant",
        "horse",
        "sheep",
        "spider",
        "squirrel",
    ]

    predicted_class = torch.argmax(logits).item()
    return idx_to_class[predicted_class]


@torch.no_grad()
def run_model(model, device, image):
    model.eval()
    transform = get_eval_transforms()
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    return logits_to_class(logits), logits


def apply_gradcam(model, device, image, layer):
    model.eval()
    transform = get_eval_transforms()
    input = transform(image).unsqueeze(0).to(device)
    target_layers = [layer]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None
    grayscale_cam = cam(input, targets=targets, aug_smooth=True)[0, :]
    rgb_img = input.squeeze(0).permute((1, 2, 0)).cpu().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)


if __name__ == "__main__":
    parser = ArgumentParser(prog="model")

    parser.add_argument(
        "filename",
        help="File to classify",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device: cpu or gpu(cuda)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="cvgg13",
        choices=VGG_MODELS + RESNET_MODELS + ALEXNET_MODELS,
        help="Model to use",
    )

    args = parser.parse_args()
    image = Image.open(args.filename)
    model = get_model(args.model).to(args.device)
    model.load_state_dict(torch.load(f"data/weights/{args.model}.pt"))

    prediction = run_model(model, args.device, image)[0]
    print(prediction)

    layer = get_last_layer(model, args.model)
    gradcam_image = apply_gradcam(model, args.device, image, layer)
    gradcam_image.show()
