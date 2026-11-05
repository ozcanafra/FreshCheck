from pathlib import Path
import shutil

BASE = Path("D:/CBIR_Selected")
fruits = ["apple", "banana", "orange", "strawberry"]

for fruit in fruits:
    folder = BASE / fruit

    for old, new in [("label_0", "fresh"), ("label_1", "rotten")]:
        source = folder / old
        target = folder / new

        if source.exists():
            target.mkdir(exist_ok=True)

            for file in source.glob("*.png"):
                shutil.copy2(file, target / file.name)

            print(f"{fruit}: {old} -> {new}")

print("\n--- SON KONTROL ---")

total = 0

for fruit in fruits:
    for label in ["fresh", "rotten"]:
        folder = BASE / fruit / label
        count = len(list(folder.glob("*.png")))
        total += count

        print(f"{fruit}/{label}: {count}")

print(f"\nToplam: {total}/1600")

if total == 1600:
    print("BAŞARILI: 1600 fotoğraf hazır!")
else:
    print("UYARI: Eksik fotoğraflar var!")