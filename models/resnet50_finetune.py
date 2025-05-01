from torchvision.models import resnet50
import torch.nn as nn

def get_model(num_classes=43, pretrained=True, finetune=False):
    model = resnet50(pretrained=pretrained)
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
