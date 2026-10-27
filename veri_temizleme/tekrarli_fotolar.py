
from pathlib import Path
import hashlib
from collections import defaultdict

# BURAYI kendi veri setinin klasör yoluyla değiştir
VERI_KLASORU = Path(r"C:\Users\berce\Downloads\FreshFruit")

UZANTILAR = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

dosyalar = defaultdict(list)

for dosya in VERI_KLASORU.rglob("*"):
    if not dosya.is_file() or dosya.suffix.lower() not in UZANTILAR:
        continue

    try:
        sha256 = hashlib.sha256(dosya.read_bytes()).hexdigest()
        dosyalar[sha256].append(dosya)
    except OSError as e:
        print(f"Okunamadı: {dosya} ({e})")

tekrar_gruplari = [
    grup for grup in dosyalar.values()
    if len(grup) > 1
]

rapor = VERI_KLASORU / "tekrar_raporu.txt"

with rapor.open("w", encoding="utf-8") as f:
    for i, grup in enumerate(tekrar_gruplari, start=1):
        f.write(f"\nGrup {i}:\n")
        for dosya in grup:
            f.write(f"{dosya}\n")

print(f"İncelenen benzersiz dosya grubu: {len(dosyalar)}")
print(f"Birebir tekrar grubu sayısı: {len(tekrar_gruplari)}")
print(f"Rapor kaydedildi: {rapor}")
print("Hiçbir fotoğraf silinmedi.")
