import pandas as pd
import matplotlib.pyplot as plt
import os

# Dosyayı oku
results_path = "results/results.csv"
if not os.path.exists(results_path):
    print("'results.csv' bulunamadı. Lütfen önce evaluate.py ile sonuç üretin.")
    exit()

df = pd.read_csv(results_path)

# En iyi modeli bul
best_row = df.loc[df['Accuracy'].idxmax()]
best_model = best_row['Model']
best_acc = best_row['Accuracy']
best_f1 = best_row['F1']

print("\nEn İyi Model")
print(f"Model: {best_model}")
print(f"Accuracy: {best_acc:.2%}")
print(f"F1 Score: {best_f1:.4f}")

# Grafikler klasörü oluştur
os.makedirs("charts", exist_ok=True)

# Bar chart - Accuracy
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['Accuracy'], color='skyblue')
for i, v in enumerate(df['Accuracy']):
    plt.text(i, v + 0.002, f"{v:.2%}", ha='center', fontsize=10)

bars[df['Accuracy'].idxmax()].set_color('limegreen')  # En iyi model yeşil

plt.title("Model Karşılaştırması - Accuracy")
plt.ylabel("Doğruluk (Accuracy)")
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("charts/accuracy_comparison.png")
plt.show()

# Bar chart - F1 Score
plt.figure(figsize=(10, 6))
bars2 = plt.bar(df['Model'], df['F1'], color='salmon')
for i, v in enumerate(df['F1']):
    plt.text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=10)

bars2[df['F1'].idxmax()].set_color('limegreen')

plt.title("Model Karşılaştırması - F1 Skoru")
plt.ylabel("F1 Score")
plt.ylim(0.9, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("charts/f1_comparison.png")
plt.show()

# Tablo olarak da gösterelim
print("\n Tüm Sonuçlar:\n")
print(df.sort_values(by='Accuracy', ascending=False).to_string(index=False))
