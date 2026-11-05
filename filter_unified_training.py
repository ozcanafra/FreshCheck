from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import imagehash
import shutil
import csv

# Ham veri setine dokunulmaz, sadece okunur.
SOURCE = Path("D:/Unified_Full_Raw")
TARGET = Path("D:/Unified_Full_Clean")
REPORT = Path("D:/Unified_Full_Clean_Report")

# Görsel kontrolle belirlenen sorunlu fotoğraflar (filigran, yığın,
# yanlış meyve, döndürülmüş augmente vb.), nedenleriyle birlikte.
MANUAL_EXCLUDE = Path(__file__).with_name("manual_exclude.csv")

# Kısa kenarı bundan küçük olanlar eğitimde aşırı büyütülür, bulanıklaşır.
MIN_SIZE = 150

# dHash farkı ve küçültülmüş görüntü piksel farkı bu değerlerin
# altındaysa: aynı meyvenin seri çekimdeki neredeyse aynı karesi.
# Düz zeminde dHash tek başına farklı meyveleri de eşleştirir,
# bu yüzden ikisi birlikte aranır.
DUP_DHASH = 8
DUP_PIXEL = 5

if TARGET.exists():
    raise RuntimeError(f"{TARGET} zaten var, üzerine yazılmaz. Önce silin veya taşıyın.")

REPORT.mkdir(parents=True, exist_ok=True)

with open(MANUAL_EXCLUDE, encoding="utf-8-sig") as f:
    manual = {row["dosya"]: row["neden"] for row in csv.DictReader(f)}

files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
print(f"Toplam fotoğraf: {len(files)}")

missing = set(manual) - {p.relative_to(SOURCE).as_posix() for p in files}
if missing:
    raise RuntimeError(f"Listede olup kaynakta olmayan dosyalar: {sorted(missing)[:5]}")

info = {}
rejected = {}

# 1. Elle belirlenenler ve tek tek fotoğraf kontrolleri
for i, path in enumerate(files, 1):
    key = path.relative_to(SOURCE).as_posix()

    if key in manual:
        rejected[key] = manual[key]
        continue

    try:
        with Image.open(path) as img:
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        rejected[key] = f"okunamadı: {e}"
        continue

    if min(img.size) < MIN_SIZE:
        rejected[key] = f"düşük çözünürlük {img.size}"
        continue

    info[key] = {
        "size": img.size,
        "dhash": imagehash.dhash(img),
        "thumb": np.asarray(img.resize((64, 64))).astype(float),
    }

    if i % 500 == 0:
        print(f"İncelendi: {i}/{len(files)}")

# 2. Neredeyse aynı kareler: her grupta en yüksek çözünürlüklü kalır
keys = list(info)
duplicates = []

for a_index, a in enumerate(keys):
    for b in keys[a_index + 1:]:
        if a.split("/")[0] != b.split("/")[0]:
            continue

        distance = info[a]["dhash"] - info[b]["dhash"]
        if distance > DUP_DHASH:
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

# 3. Temiz fotoğrafları birebir (değiştirmeden) kopyala
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
    writer.writerow(["fotoğraf_1", "fotoğraf_2", "dhash_mesafesi", "piksel_farkı"])
    writer.writerows(duplicates)

with open(REPORT / "summary.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["meyve", "etiket", "kaynak", "temiz", "çıkarılan"])
    for (fruit, label), source_count in sorted(source_counts.items()):
        clean = counts[(fruit, label)]
        writer.writerow([fruit, label, source_count, clean, source_count - clean])

print("\n--- SONUÇ ---")
for (fruit, label), count in sorted(counts.items()):
    print(f"{fruit}/{label}: {count}")

print("Temiz fotoğraf:", sum(counts.values()))
print("Çıkarılan fotoğraf:", len(rejected))
print("Eğitim veri seti:", TARGET)
print("Rapor klasörü:", REPORT)
print("Kaynak klasör değiştirilmedi.")
