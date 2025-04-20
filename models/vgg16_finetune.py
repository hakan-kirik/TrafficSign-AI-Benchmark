from torchvision.models import vgg16
import torch.nn as nn

def get_model(num_classes=43, pretrained=True, finetune=True):
    model = vgg16(pretrained=pretrained)

    # Eğer ince ayar değilse: tüm katmanları dondur
    if not finetune:
        for param in model.features.parameters():
            param.requires_grad = False

    # Son katmanı değiştir (43 sınıflı çıkış için)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model
