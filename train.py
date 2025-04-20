import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from utils import get_dataloaders
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device='cpu', model_name="model"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = running_loss / total
        train_acc = 100 * correct / total
        print(f" Epoch {epoch+1}: Train Accuracy: {train_acc:.2f}%  Loss: {running_loss:.4f}  Avg Loss: {avg_loss:.4f}")

        # ---------- VALIDASYON ----------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f" Validation Accuracy: {val_acc:.2f}%")

        # ---------- MODEL KAYIT ----------
        last_path = f"checkpoints/{model_name}_last.pt"
        torch.save(model.state_dict(), last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = f"checkpoints/{model_name}_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f" En iyi model güncellendi: {best_path}")


# CLI kullanımı
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="cnn or resnet50")
    args = parser.parse_args()

    from models import cnn_model, resnet50_finetune,vgg16_finetune

    train_loader, test_loader = get_dataloaders("./data/GTSRB", image_size=64)

    if args.model == "cnn":
        model = cnn_model.SimpleCNN()
    elif args.model == "resnet50":
        model = resnet50_finetune.get_model()
    elif args.model == "vgg16":
        model = vgg16_finetune.get_model()
    else:
        model = cnn_model.SimpleCNN()
        #raise ValueError("Model not found!")

    train_model(model, train_loader, test_loader, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu',model_name=args.model)
