import torch
import torch.nn as nn
import torchvision.models as models


def resnet18(num_classes=10, weights=None):
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def resnet50(num_classes=10, weights=None):
    model = models.resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
