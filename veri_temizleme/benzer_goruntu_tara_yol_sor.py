from pathlib import Path
import csv
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

SIMILARITY_THRESHOLD = 0.97
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "valid", "validation", "test"}
# Bu klasörlerin içindeki görüntüler taranmaz (küçük harfle yaz).
# Karantina klasörü eklendi: temizlikten sonra tekrar tarama yaparsan
# taşınan dosyalar yeniden çıkmasın.
EXCLUDED_DIRS = {"removed_dups", "augmented", "_karantina_kopya_adaylari"}


def get_class_and_split(path, root):
    parts = path.relative_to(root).parts
    for i, part in enumerate(parts):
        if part.lower() in SPLIT_NAMES:
            return (parts[i + 1] if i + 1 < len(parts) else "UNKNOWN", part)
    return path.parent.name, "UNKNOWN"


def main():
    raw_path = input("Veri setinin klasör yolunu yapıştırıp Enter'a bas: ").strip().strip('"')
    dataset_dir = Path(raw_path).expanduser()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"\nHATA: Klasör bulunamadı:\n{dataset_dir}")
        input("\nKapatmak için Enter'a bas...")
        return

    paths = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(x.lower() in EXCLUDED_DIRS for x in p.relative_to(dataset_dir).parts)
    )
    if not paths:
        print("Bu klasörde desteklenen görüntü bulunamadı.")
        input("\nKapatmak için Enter'a bas...")
        return

    print(f"\nBulunan görüntü: {len(paths)}")
    print("MobileNet hazırlanıyor. İlk çalıştırmada model ağırlıkları indirilebilir...")
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().to("cpu")
    preprocess = weights.transforms()

    vectors, good_paths, classes, splits = [], [], [], []
    skipped = 0

    with torch.inference_mode():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths, tensors = [], []
            for p in paths[start:start + BATCH_SIZE]:
                try:
                    with Image.open(p) as im:
                        tensors.append(preprocess(im.convert("RGB")))
                    batch_paths.append(p)
                except (UnidentifiedImageError, OSError, ValueError) as e:
                    skipped += 1
                    print(f"Atlandı: {p} ({e})")

            if tensors:
                feats = model(torch.stack(tensors)).float()
                vectors.append(F.normalize(feats, p=2, dim=1).cpu())
                good_paths.extend(batch_paths)
                for p in batch_paths:
                    cls, split = get_class_and_split(p, dataset_dir)
                    classes.append(cls)
                    splits.append(split)

            print(f"Özellik çıkarımı: {min(start + BATCH_SIZE, len(paths))}/{len(paths)}")

    if not vectors:
        print("Hiçbir görüntü işlenemedi.")
        input("\nKapatmak için Enter'a bas...")
        return

    features = torch.cat(vectors)
    print(f"\nİşlenen: {len(good_paths)} | Okunamayan: {skipped}")
    print("Benzerlikler hesaplanıyor...")

    results = []
    block_size = 128
    for start in range(0, len(good_paths), block_size):
        end = min(start + block_size, len(good_paths))
        sims = features[start:end] @ features.T

        for offset in range(end - start):
            i = start + offset
            for j in torch.where(sims[offset] >= SIMILARITY_THRESHOLD)[0].tolist():
                if j <= i or classes[i].casefold() != classes[j].casefold():
                    continue
                results.append({
                    "similarity": round(float(sims[offset, j]), 6),
                    "class": classes[i],
                    "image_1_split": splits[i],
                    "image_1": str(good_paths[i]),
                    "image_2_split": splits[j],
                    "image_2": str(good_paths[j]),
                    "same_split": splits[i].casefold() == splits[j].casefold()
                })

        print(f"Benzerlik taraması: {end}/{len(good_paths)}")

    results.sort(key=lambda r: r["similarity"], reverse=True)
    report = dataset_dir / "benzer_goruntuler_raporu.csv"
    fields = [
        "similarity", "class", "image_1_split", "image_1",
        "image_2_split", "image_2", "same_split"
    ]
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    cross = sum(not r["same_split"] for r in results)
    print("\nTARAMA TAMAMLANDI")
    print(f"Benzer çift sayısı: {len(results)}")
    print(f"Farklı train/valid/test bölümlerindeki benzer çift: {cross}")
    print(f"Rapor kaydedildi: {report}")
    print("Hiçbir görüntü silinmedi veya taşınmadı.")
    print("Not: Benzerlik kesin kopya demek değildir; raporu kontrol et.")
    input("\nKapatmak için Enter'a bas...")


if __name__ == "__main__":
    main()
