import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse
import pandas as pd
from utils import get_dataloaders
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, output_dict=False)
    matrix = confusion_matrix(all_labels, all_preds)
    return acc, f1, report, matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="cnn, resnet50")
    parser.add_argument("--weight", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--num_classes", type=int, default=43)
    args = parser.parse_args()

    #  Model seçimi
    from models import cnn_model, resnet50_finetune,vgg16_finetune

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, test_loader = get_dataloaders("data/GTSRB", image_size=64)

    if args.model == "cnn":
        model = cnn_model.SimpleCNN(num_classes=args.num_classes)
    elif args.model == "resnet50":
        model = resnet50_finetune.get_model(num_classes=args.num_classes)
    elif args.model == "vgg16":
        model = vgg16_finetune.get_model(num_classes=args.num_classes)
    else:
        raise ValueError("Unknown model")

    #  Model yükle
    model.load_state_dict(torch.load(args.weight, map_location=device))

    #  Değerlendirme
    acc, f1, report, matrix = evaluate_model(model, test_loader, device, num_classes=args.num_classes)

    print("\n Evaluation Results")
    print(f" Accuracy: {acc:.2%}")
    print(f" F1 Score: {f1:.4f}")
    print("\n Classification Report:\n")
    print(report)

    #  CSV'ye sonuç yaz
    os.makedirs("results", exist_ok=True)
    results_path = "results/results.csv"
    new_line = f"{args.model},{os.path.basename(args.weight)},{acc:.4f},{f1:.4f}\n"

    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("Model,Checkpoint,Accuracy,F1\n")

    with open(results_path, "a") as f:
        f.write(new_line)

    #  Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, fmt="d", cmap="Blues")
    plt.title(f"{args.model.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/{args.model}_confusion_matrix.png")
    plt.show()
