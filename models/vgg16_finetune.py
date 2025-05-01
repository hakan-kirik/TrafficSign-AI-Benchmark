from torchvision.models import vgg16
import torch.nn as nn

def get_model(num_classes=43, pretrained=True, finetune=False):
    model = vgg16(pretrained=pretrained)

    # Eğer ince ayar değilse: tüm katmanları dondur
    if not finetune:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False


    model.classifier[6] = nn.Sequential(
        nn.Linear(model.classifier[6].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    return model
