from timm import create_model
import torch.nn as nn

def get_model(num_classes=43, pretrained=True, finetune=False):
    # Pretrained ViT modelini yükle
    model = create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=1000  # Başlangıçta ImageNet çıkışlı
    )

    if not finetune:
        for param in model.parameters():
            param.requires_grad = False

    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    for param in model.head.parameters():
        param.requires_grad = True

    return model
