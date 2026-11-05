from datasets import load_dataset
from pathlib import Path

ds = load_dataset(
    "Project-AgML/fresh_rotten_fruit_classification",
    "raw",
    split="train",
    cache_dir="D:/hf_cache"
)

output = Path("D:/CBIR_Fruits")
output.mkdir(parents=True, exist_ok=True)

for i, row in enumerate(ds):
    filename = output / f"fruit_{i:05d}.png"

    if filename.exists():
        continue

    row["image"].save(filename)

    if (i + 1) % 100 == 0:
        print(f"İşlenen: {i + 1}/{len(ds)}")

print("Tamamlandı!")
print("Toplam dosya:", len(list(output.glob("fruit_*.png"))))