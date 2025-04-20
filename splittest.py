import os
import pandas as pd
import shutil

test_images_dir = './data/GTSRB/Test'  # Bu dizinde test resimleri bulunur
test_labels_path = './data/GTSRB/Test.csv'  # Bu csv'de resim isimleri ve etiketler olur

df = pd.read_csv(test_labels_path)
output_dir = './data/GTSRB/test'

for _, row in df.iterrows():
    filename = row['Path']
    filename = os.path.basename(filename)
    label = str(row['ClassId'])
    src = os.path.join(test_images_dir, filename)
    dst_dir = os.path.join(output_dir, label)

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dst_dir, filename))

print(" Test verisi başarıyla sınıf klasörlerine ayrıldı.")
