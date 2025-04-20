from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, image_size=64, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_data = datasets.ImageFolder(f'{data_dir}/train', transform=transform)
    test_data = datasets.ImageFolder(f'{data_dir}/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
