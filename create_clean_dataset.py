from pathlib import Path
import shutil
import hashlib

SOURCE = Path("D:/CBIR_Selected")
TARGET = Path("D:/CBIR_Clean")

fruits = ["apple", "banana", "orange", "strawberry"]
labels = ["fresh", "rotten"]

total = 0
duplicates = 0

for fruit in fruits:
    for label in labels:
        src = SOURCE / fruit / label
        dst = TARGET / fruit / label
        dst.mkdir(parents=True, exist_ok=True)

        seen = set()
        copied = 0

        for file in sorted(src.glob("*.png")):
            file_hash = hashlib.sha256(file.read_bytes()).hexdigest()

            if file_hash in seen:
                duplicates += 1
                print(f"Tekrar atlandi: {file.name}")
                continue

            seen.add(file_hash)
            shutil.copy2(file, dst / file.name)
            copied += 1
            total += 1

        print(f"{fruit}/{label}: {copied}")

print("\n--- SONUC ---")
print("Temiz veri seti:", total)
print("Atlanan tekrar:", duplicates)
print("Konum:", TARGET)