VGG_MODELS = ["vgg13", "cvgg13", "vgg16"]
RESNET_MODELS = ["resnet18", "resnet18_pretrained", "resnet50"]
ALEXNET_MODELS = ["alexnet_lrn", "alexnet_bn"]


def get_model(model_name):
    from models.vgg import CVGG13, VGG13, VGG16
    from models.alexnet import AlexNetLRN, AlexNetBN
    from models.resnet import resnet18, resnet50

    match model_name:
        case "vgg13":
            return VGG13()
        case "vgg16":
            return VGG16()
        case "cvgg13":
            return CVGG13()
        case "alexnet_lrn":
            return AlexNetLRN()
        case "alexnet_bn":
            return AlexNetBN()
        case "resnet18":
            return resnet18()
        case "resnet18_pretrained":
            return resnet18(weights="DEFAULT")
        case "resnet50":
            return resnet50()


def get_last_layer(model, model_name):
    if model_name in VGG_MODELS:
        return model.features[-1]
    elif model_name in RESNET_MODELS:
        return model.layer4[-1]


def get_transforms():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
        ]
    )


def get_eval_transforms():
    from torchvision import transforms

    return transforms.Compose(
        [
            get_transforms(),
            transforms.ToTensor(),
        ]
    )
