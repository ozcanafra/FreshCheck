from pathlib import Path
from PIL import Image
import numpy as np
import imagehash
import shutil
import csv

# Kaynak klasöre dokunulmaz, sadece okunur.
SOURCE = Path("D:/CBIR_Clean")
TARGET = Path("D:/CBIR_Raw_Clean")
REPORT = Path("D:/CBIR_Raw_Clean_Report")

fruits = ["apple", "banana", "orange", "strawberry"]
labels = ["fresh", "rotten"]

# Kenar bölgesinde açık renkli düz arka plan oranı bunun altındaysa
# ve kenar renkleri çok değişkense: kadrajı dolduran pazar / yığın fotoğrafı.
MIN_BACKGROUND = 0.5
MAX_BORDER_STD = 57

# Aynı yönde pHash farkı ve küçültülmüş görüntü piksel farkı bu
# değerlerin altındaysa: aynı karenin tekrarı.
DUP_PHASH = 6
DUP_PIXEL = 22

MIN_SIZE = 224

# Otomatik testle ayrılamayan (açık renkli) elma pazar yığınları,
# görsel kontrolle belirlendi.
MANUAL_EXCLUDE = {
    "apple/fresh/fruit_00136.png": "pazar/yığın fotoğrafı (elle)",
    "apple/fresh/fruit_00141.png": "pazar/yığın fotoğrafı (elle)",
    "apple/fresh/fruit_00142.png": "pazar/yığın fotoğrafı (elle)",
    "apple/fresh/fruit_00148.png": "pazar/yığın fotoğrafı (elle)",
    "apple/fresh/fruit_00149.png": "pazar/yığın fotoğrafı (elle)",
    "apple/fresh/fruit_00150.png": "pazar/yığın fotoğrafı (elle)",
    "apple/fresh/fruit_00151.png": "pazar/yığın fotoğrafı (elle)",
}

if TARGET.exists():
    raise RuntimeError(f"{TARGET} zaten var, üzerine yazılmaz. Önce silin veya taşıyın.")

REPORT.mkdir(parents=True, exist_ok=True)

files = sorted(
    f for fruit in fruits for label in labels
    for f in (SOURCE / fruit / label).glob("*.png")
)
print(f"Toplam fotoğraf: {len(files)}")

info = {}
rejected = {}

# 1. Her fotoğrafı ölç
for i, path in enumerate(files, 1):
    key = path.relative_to(SOURCE).as_posix()

    try:
        with Image.open(path) as img:
            img.load()
            img = img.convert("RGB")
    except Exception as e:
        rejected[key] = f"okunamadı: {e}"
        continue

    small = img.resize((256, 256))
    rgb = np.asarray(small).astype(float)
    hsv = np.asarray(small.convert("HSV")).astype(float)

    border = np.zeros((256, 256), bool)
    border[:20] = border[-20:] = True
    border[:, :20] = border[:, -20:] = True

    bright = hsv[..., 2][border] > 140
    low_sat = hsv[..., 1][border] < 70

    info[key] = {
        "size": img.size,
        "background": float((bright & low_sat).mean()),
        "border_std": float(rgb[border].std()),
        "phash": imagehash.phash(img),
        "thumb": np.asarray(img.resize((48, 48))).astype(float),
    }

    if i % 100 == 0:
        print(f"İncelendi: {i}/{len(files)}")

# 2. Tek tek fotoğraf kontrolleri
for key, d in info.items():
    if key in MANUAL_EXCLUDE:
        rejected[key] = MANUAL_EXCLUDE[key]
    elif min(d["size"]) < MIN_SIZE:
        rejected[key] = f"düşük çözünürlük {d['size']}"
    elif d["background"] < MIN_BACKGROUND and d["border_std"] >= MAX_BORDER_STD:
        rejected[key] = "pazar/yığın fotoğrafı (arka plan yok)"

# 3. Tekrar eden kareler: her grupta en yüksek çözünürlüklü kalır
keys = [k for k in info if k not in rejected]
duplicates = []

for a_index, a in enumerate(keys):
    for b in keys[a_index + 1:]:
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

# 4. Temiz fotoğrafları birebir (değiştirmeden) kopyala
counts = {}

for path in files:
    key = path.relative_to(SOURCE).as_posix()
    fruit, label = key.split("/")[:2]
    counts.setdefault((fruit, label), 0)

    if key in rejected:
        continue

    dst = TARGET / fruit / label
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst / path.name)
    counts[(fruit, label)] += 1

# 5. Raporlar
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
    writer.writerow(["meyve", "etiket", "kaynak", "temiz", "çıkarılan"])
    for fruit in fruits:
        for label in labels:
            source_count = len(list((SOURCE / fruit / label).glob("*.png")))
            clean = counts.get((fruit, label), 0)
            writer.writerow([fruit, label, source_count, clean, source_count - clean])

print("\n--- SONUÇ ---")
for (fruit, label), count in counts.items():
    print(f"{fruit}/{label}: {count}")

print("Temiz fotoğraf:", sum(counts.values()))
print("Çıkarılan fotoğraf:", len(rejected))
print("Temiz veri seti:", TARGET)
print("Rapor klasörü:", REPORT)
print("Kaynak klasör değiştirilmedi.")
