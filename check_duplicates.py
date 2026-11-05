from pathlib import Path
from collections import defaultdict
from itertools import combinations
from PIL import Image
import hashlib
import imagehash
import csv

BASE = Path("D:/CBIR_Selected")
REPORT = Path("D:/CBIR_Duplicate_Report")
REPORT.mkdir(parents=True, exist_ok=True)

files = sorted(BASE.rglob("*.png"))

print(f"Toplam fotoğraf: {len(files)}")

sha_groups = defaultdict(list)
phashes = {}
errors = []

# 1. Dosyaları tara
for i, path in enumerate(files, 1):
    try:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        sha_groups[sha].append(path)

        with Image.open(path) as img:
            phashes[path] = imagehash.phash(
                img.convert("RGB")
            )

    except Exception as e:
        errors.append((str(path), str(e)))

    if i % 100 == 0:
        print(f"İncelendi: {i}/{len(files)}")

# 2. Birebir aynı dosyalar
exact = []

for group in sha_groups.values():
    if len(group) > 1:
        for a, b in combinations(group, 2):
            exact.append((str(a), str(b)))

# 3. Görsel olarak benzer fotoğraflar
similar = []

items = list(phashes.items())

for i in range(len(items)):
    path1, hash1 = items[i]

    for j in range(i + 1, len(items)):
        path2, hash2 = items[j]

        distance = hash1 - hash2

        if distance <= 5:
            similar.append((
                str(path1),
                str(path2),
                distance
            ))

# 4. Raporları kaydet
with open(REPORT / "exact_duplicates.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["fotoğraf_1", "fotoğraf_2"])
    writer.writerows(exact)

with open(REPORT / "similar_images.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow([
        "fotoğraf_1",
        "fotoğraf_2",
        "phash_mesafesi"
    ])
    writer.writerows(sorted(similar, key=lambda x: x[2]))

with open(REPORT / "errors.csv",
          "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)
    writer.writerow(["dosya", "hata"])
    writer.writerows(errors)

print("\n--- ANALİZ SONUCU ---")
print("Toplam fotoğraf:", len(files))
print("Birebir aynı dosya çiftleri:", len(exact))
print("Benzer fotoğraf çiftleri:", len(similar))
print("Okunamayan dosya:", len(errors))

print("\nRapor klasörü:", REPORT)
print("Hiçbir fotoğraf silinmedi.")