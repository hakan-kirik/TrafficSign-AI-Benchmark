from timm import create_model

def get_model(num_classes=43, pretrained=True):
    model = create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    return model
