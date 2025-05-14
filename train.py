import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from utils import get_dataloaders
from models import efficientnet,vit
import os
from sklearn.metrics import classification_report

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
def get_hyperparams(model_name):
    if model_name == "vit":
        return {"lr": 3e-5, "optimizer": "AdamW", "weight_decay": 0.01}
    elif model_name == "resnet50":
        return {"lr": 1e-3, "optimizer": "AdamW", "weight_decay": 1e-4}
    elif model_name == "efficientnet":
        return {"lr": 1e-3, "optimizer": "Adam", "weight_decay": 1e-5}
    elif model_name == "vgg16":
        return {"lr": 1e-3, "optimizer": "SGD", "weight_decay": 5e-4}
    elif model_name == "cnn":
        return {"lr": 1e-3, "optimizer": "Adam", "weight_decay": 1e-4}
    else:
        return {"lr": 1e-3, "optimizer": "Adam", "weight_decay": 0}

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu', model_name="model"):
    model.to(device)
    hyperparams = get_hyperparams(model_name)
    lr = hyperparams["lr"]
    weight_decay = hyperparams["weight_decay"]

    if hyperparams["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif hyperparams["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    early_stop_counter = 0
    patience = 3
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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        print(f" Validation Accuracy: {val_acc:.2f}%")
        #print(classification_report(all_labels, all_preds))

        # ---------- MODEL KAYIT ----------
        last_path = f"checkpoints/{model_name}_last.pt"
        torch.save(model.state_dict(), last_path)
        best_path = f"checkpoints/{model_name}_best.pt"
        best_best_path = f"checkpoints/{model_name}_best_best.pt"

        if os.path.exists(best_best_path):
            previous_best_state = torch.load(best_best_path, map_location=device)
            model.load_state_dict(previous_best_state)
            model.eval()
            prev_val_correct = 0
            prev_val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    prev_val_total += labels.size(0)
                    prev_val_correct += (predicted == labels).sum().item()
            prev_val_acc = 100 * prev_val_correct / prev_val_total
            model.load_state_dict(torch.load(last_path, map_location=device))
        else:
            prev_val_acc = 0.0

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f" En iyi model guncellendi: {best_path}")
            if val_acc > prev_val_acc:
                torch.save(model.state_dict(), best_best_path)
                print(f"  Mevcut model onceki en iyiden de iyi! Kaydedildi: {best_best_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Erken durdurma: Model daha fazla gelismiyor.")
                break

        scheduler.step()


# CLI kullanımı
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="cnn, resnet50, vgg16, efficientnet, vit")
    args = parser.parse_args()

    from models import cnn_model, resnet50_finetune,vgg16_finetune

    train_loader, test_loader = get_dataloaders("./data/GTSRB", image_size=64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == "cnn":
        model = cnn_model.SimpleCNN()
    elif args.model == "resnet50":
        model = resnet50_finetune.get_model()
    elif args.model == "vgg16":
        model = vgg16_finetune.get_model()
    elif args.model == "efficientnet":
        model = efficientnet.get_model()
    elif args.model == "vit":
        train_loader, test_loader = get_dataloaders("./data/GTSRB", image_size=224)
        model= vit.get_model(finetune=True)
    else:
        model =None
    if model is not None:
        train_model(model, train_loader, test_loader, num_epochs=20, device=device,model_name=args.model)
    else:
        models = ["resnet50", "vgg16"]
        for model_name in models:
            print(f"\n================ Training {model_name} ================")
            if model_name == "vit":
                train_loader, test_loader = get_dataloaders("./data/GTSRB", image_size=224)
                model = vit.get_model()
            else:
                train_loader, test_loader = get_dataloaders("./data/GTSRB", image_size=64)
                if model_name == "cnn":
                    model = cnn_model.SimpleCNN()
                elif model_name == "resnet50":
                    model = resnet50_finetune.get_model(finetune=True)
                elif model_name == "vgg16":
                    model = vgg16_finetune.get_model(finetune=True)
                elif model_name == "efficientnet":
                    model = efficientnet.get_model()
            train_model(model, train_loader, test_loader, num_epochs=20, device=device, model_name=model_name)