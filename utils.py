from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def get_dataloaders(data_dir, image_size=64, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(f'{data_dir}/train', transform=transform)
    test_data = datasets.ImageFolder(f'{data_dir}/test', transform=transform)

    # Train veri seti için sınıf dağılımını hesapla
    targets = np.array(train_data.targets)
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts  # Az sınıfa büyük ağırlık

    # Her örnek için ağırlık belirle
    sample_weights = class_weights[targets]

    # Sampler oluştur
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Aynı veriden tekrar seçebilirsin
    )

    # Train loader sampler ile
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)

    # Test loader normal
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
