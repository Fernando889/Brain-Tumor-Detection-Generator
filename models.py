from torch import nn
import torchvision
from utils import set_seed


def create_effnetb0(out_features: int):

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    models = torchvision.models.efficientnet_b0(weights=weights)

    for params in models.features.parameters():
        params.requires_grad = False

    set_seed()

    models.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=out_features)
    )

    models.name = "effnet_b0"
    print(f"Created New Models: {models.name}")
    return models


def create_effnetb2(out_features: int):

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    models = torchvision.models.efficientnet_b2(weights=weights)

    for params in models.features.parameters():
        params.requires_grad = False

    set_seed()

    models.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=out_features)
    )

    models.name = "effnet_b2"
    print(f"Created New Models: {models.name}")
    return models
