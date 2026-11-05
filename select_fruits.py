import pandas as pd
import shutil
from pathlib import Path

SOURCE = Path("D:/Fruit_CBIR_NEW/images")
DEST = Path("D:/CBIR_Selected")
CSV = Path("D:/Fruit_CBIR_NEW/metadata.csv")

fruits = ["apple", "banana", "orange", "strawberry"]

df = pd.read_csv(CSV)
df = df[df["fruit_type"].isin(fruits)]

copied = 0
missing = []

for _, row in df.iterrows():
    fruit = row["fruit_type"]
    label = row["label"]
    filename = row["filename"]

    source_file = SOURCE / filename

    # Etiketler doğrulanana kadar sayısal isim kullanıyoruz.
    target_folder = DEST / fruit / f"label_{label}"
    target_folder.mkdir(parents=True, exist_ok=True)

    if source_file.exists():
        shutil.copy2(source_file, target_folder / filename)
        copied += 1
    else:
        missing.append(filename)

print(f"Kopyalanan görsel: {copied}")
print(f"Eksik görsel: {len(missing)}")

if missing:
    print("Eksik dosyalar:", missing[:20])

if copied == 1600 and not missing:
    print("BAŞARILI: 1600 görsel ayrıldı!")