from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import imagehash
import shutil
import csv

# İki zip (-001 ve -002) aynı klasöre açıldı, kaynak tam veri setidir.
# Kaynak klasöre dokunulmaz, sadece okunur.
SOURCE = Path("D:/Unified_Dataset_Source/Unified_Dataset")
TARGET = Path("D:/Unified_Full_Raw")
REPORT = Path("D:/Unified_Full_Raw_Report")

# Kaggle "fresh and rotten" setinden gelen döndürülmüş (augmente)
# PNG'lerin boyutları; köşelerde yansıtılmış dolgu izleri var.
AUGMENTED_PNG_SIZES = {(320, 258), (480, 322)}

# Aynı yönde pHash farkı ve küçültülmüş görüntü piksel farkı bu
# değerlerin altındaysa: aynı karenin tekrarı.
DUP_PHASH = 4
DUP_PIXEL = 3

if TARGET.exists():
    raise RuntimeError(f"{TARGET} zaten var, üzerine yazılmaz. Önce silin veya taşıyın.")

REPORT.mkdir(parents=True, exist_ok=True)

files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
print(f"Toplam fotoğraf: {len(files)}")

info = {}
rejected = {}

# 1. Augmente olanları ayır, kalanları ölç
for i, path in enumerate(files, 1):
    key = path.relative_to(SOURCE).as_posix()

    if "augmented" in path.parts:
        rejected[key] = "augmented klasörü"
        continue

    try:
        with Image.open(path) as img:
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        rejected[key] = f"okunamadı: {e}"
        continue

    if path.suffix.lower() == ".png" and img.size in AUGMENTED_PNG_SIZES:
        rejected[key] = f"döndürülmüş augmente görüntü {img.size}"
        continue

    info[key] = {
        "size": img.size,
        "phash": imagehash.phash(img),
        "thumb": np.asarray(img.resize((64, 64))).astype(float),
    }

    if i % 500 == 0:
        print(f"İncelendi: {i}/{len(files)}")

# 2. Tekrar eden kareler: her grupta en yüksek çözünürlüklü kalır
keys = list(info)
duplicates = []

for a_index, a in enumerate(keys):
    for b in keys[a_index + 1:]:
        if a.split("/")[0] != b.split("/")[0]:
            continue

        distance = info[a]["phash"] - info[b]["phash"]
        if distance > DUP_PHASH:
            continue

        pixel = np.abs(info[a]["thumb"] - info[b]["thumb"]).mean()
        if pixel > DUP_PIXEL:
            continue

        duplicates.append((a, b, distance, round(float(pixel), 1)))

for a, b, distance, pixel in duplicates:
    if a in rejected or b in rejected:
        continue

    area_a = info[a]["size"][0] * info[a]["size"][1]
    area_b = info[b]["size"][0] * info[b]["size"][1]
    keep, drop = (a, b) if area_a >= area_b else (b, a)

    if keep.split("/")[:2] != drop.split("/")[:2]:
        # Farklı sınıflarda aynı kare: etiket çelişkisi, ikisi de çıkar
        rejected[keep] = f"etiket çelişkisi: {drop} ile aynı"
        rejected[drop] = f"etiket çelişkisi: {keep} ile aynı"
    else:
        rejected[drop] = f"tekrar kare: {keep} ile aynı"

# 3. Ham fotoğrafları birebir (değiştirmeden) kopyala
counts = {}
source_counts = {}

for path in files:
    key = path.relative_to(SOURCE).as_posix()
    fruit, label = key.split("/")[:2]
    source_counts[(fruit, label)] = source_counts.get((fruit, label), 0) + 1
    counts.setdefault((fruit, label), 0)

    if key in rejected:
        continue

    dst = TARGET / fruit / label
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst / path.name)
    counts[(fruit, label)] += 1

# 4. Raporlar
with open(REPORT / "rejected.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["dosya", "neden"])
    writer.writerows(sorted(rejected.items()))

with open(REPORT / "duplicate_pairs.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["fotoğraf_1", "fotoğraf_2", "phash_mesafesi", "piksel_farkı"])
    writer.writerows(duplicates)

with open(REPORT / "summary.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["meyve", "etiket", "kaynak", "ham", "çıkarılan"])
    for (fruit, label), source_count in sorted(source_counts.items()):
        clean = counts[(fruit, label)]
        writer.writerow([fruit, label, source_count, clean, source_count - clean])

print("\n--- SONUÇ ---")
for (fruit, label), count in sorted(counts.items()):
    print(f"{fruit}/{label}: {count}")

print("Ham fotoğraf:", sum(counts.values()))
print("Çıkarılan fotoğraf:", len(rejected))
print("Ham veri seti:", TARGET)
print("Rapor klasörü:", REPORT)
print("Kaynak klasör değiştirilmedi.")
