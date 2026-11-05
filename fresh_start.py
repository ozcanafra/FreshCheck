import os

# Hugging Face önbelleği de D diskinde olacak
os.environ["HF_HOME"] = "D:/Fruit_CBIR_NEW/hf_home"

from datasets import load_dataset
from pathlib import Path
from PIL import Image
from collections import Counter
import csv

BASE = Path("D:/Fruit_CBIR_NEW")
CACHE = BASE / "cache"
IMAGES = BASE / "images"

CACHE.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

print("Ham veri seti yükleniyor...", flush=True)

ds = load_dataset(
    "Project-AgML/fresh_rotten_fruit_classification",
    "raw",
    split="train",
    cache_dir=str(CACHE)
)

print(f"Veri seti: {len(ds)} görsel", flush=True)

if len(ds) != 3200:
    raise RuntimeError("Beklenen 3200 görsel bulunamadı!")

# CBIR için etiketleri ayrı CSV dosyasında sakla
metadata_path = BASE / "metadata.csv"

with open(metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "fruit_type"])

    for i, row in enumerate(ds):
        filename = f"fruit_{i:05d}.png"
        destination = IMAGES / filename

        # Önce geçici dosyaya yaz, sonra tamamlanmış
        # görseli nihai dosya adıyla kaydet
        temporary = IMAGES / f"fruit_{i:05d}.tmp"

        image = row["image"]
        image.save(temporary, format="PNG")

        with Image.open(temporary) as check:
            check.verify()

        temporary.replace(destination)

        writer.writerow([
            filename,
            row["label"],
            row["fruit_type"]
        ])

        if (i + 1) % 100 == 0:
            print(f"Kaydedildi: {i + 1}/3200", flush=True)

# Son doğrulama
files = sorted(IMAGES.glob("fruit_*.png"))

print("\nSONUÇ")
print("Beklenen görsel:", len(ds))
print("Kaydedilen görsel:", len(files))
print("Metadata:", metadata_path)

if len(files) == 3200:
    print("BAŞARILI: 3200 görsel kaydedildi!")
else:
    print("UYARI: Eksik görseller var!")