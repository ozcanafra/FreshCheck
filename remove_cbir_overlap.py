from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import imagehash
import shutil
import csv

# CBIR_Raw_Clean'e dokunulmaz, sadece okunur.
CBIR = Path("D:/CBIR_Raw_Clean")
DATASET = Path("D:/Unified_Full_Clean")
REPORT = Path("D:/Unified_Full_Clean_Report")

# Eşleşenler silinmez, buraya taşınır (geri almak için geri taşıyın).
MOVED = REPORT / "cbir_overlap"

# pHash farkı ve küçültülmüş görüntü piksel farkı bu değerlerin
# altındaysa: aynı fotoğraf (yeniden boyutlanmış/kırpılmış) veya
# aynı meyvenin aynı seriden karesi.
MAX_PHASH = 8
MAX_PIXEL = 22


def measure(root):
    data = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
        data[path.relative_to(root).as_posix()] = (
            imagehash.phash(img).hash.flatten(),
            np.asarray(img.resize((32, 32))).astype(np.float32),
        )
    return data


print("CBIR_Raw_Clean ölçülüyor...")
cbir = measure(CBIR)
print("Unified_Dataset_Clean ölçülüyor...")
dataset = measure(DATASET)

keys = list(dataset)
hashes = np.stack([dataset[k][0] for k in keys])
thumbs = np.stack([dataset[k][1] for k in keys])

matches = {}

for cbir_key, (phash, thumb) in cbir.items():
    distance = (hashes != phash).sum(1)
    for j in np.where(distance <= MAX_PHASH)[0]:
        # Düz zeminde farklı meyveler de eşleşebilir, sadece aynı sınıfa bak
        if keys[j].split("/")[:2] != cbir_key.split("/")[:2]:
            continue

        pixel = float(np.abs(thumbs[j] - thumb).mean())
        if pixel <= MAX_PIXEL:
            matches.setdefault(keys[j], []).append((cbir_key, int(distance[j]), round(pixel, 1)))

for key in matches:
    dst = MOVED / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(DATASET / key), str(dst))

with open(REPORT / "cbir_overlap.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["unified_dosya", "cbir_dosya", "phash_mesafesi", "piksel_farkı"])
    for key in sorted(matches):
        for cbir_key, distance, pixel in matches[key]:
            writer.writerow([key, cbir_key, distance, pixel])

matched_cbir = {c for v in matches.values() for c, _, _ in v}

print("\n--- SONUÇ ---")
print("CBIR fotoğrafı:", len(cbir))
print("Unified'da karşılığı bulunan CBIR fotoğrafı:", len(matched_cbir))
print("Unified'dan çıkarılan fotoğraf:", len(matches))
print("Unified'da kalan fotoğraf:", len(keys) - len(matches))
print("Çıkarılanlar:", MOVED)
