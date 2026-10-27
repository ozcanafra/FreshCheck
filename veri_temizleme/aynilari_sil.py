
from pathlib import Path

# Önceki kodun oluşturduğu rapor
RAPOR = Path(
    r"C:\Users\berce\Downloads\kaggle daki\tekrar_raporu.txt"
)

if not RAPOR.exists():
    print(f"Rapor bulunamadı: {RAPOR}")
    raise SystemExit

# Raporu oku ve tekrar gruplarını ayır
gruplar = []
mevcut_grup = []

for satir in RAPOR.read_text(encoding="utf-8").splitlines():
    satir = satir.strip()

    if satir.startswith("Grup "):
        if mevcut_grup:
            gruplar.append(mevcut_grup)
        mevcut_grup = []

    elif satir:
        yol = Path(satir)
        if yol.is_file():
            mevcut_grup.append(yol)

if mevcut_grup:
    gruplar.append(mevcut_grup)

# Her gruptaki ilk dosyayı koru
silinecekler = []

for i, grup in enumerate(gruplar, start=1):
    if len(grup) < 2:
        continue

    print(f"\nGrup {i}")
    print(f"KORUNACAK: {grup[0]}")

    for dosya in grup[1:]:
        print(f"SİLİNECEK: {dosya}")
        silinecekler.append(dosya)

print(f"\nTekrar grubu: {len(gruplar)}")
print(f"Silinmesi planlanan dosya: {len(silinecekler)}")

onay = input(
    "\nBu dosyaları kalıcı olarak silmek için EVET yaz: "
)

if onay == "EVET":
    silinen = 0

    for dosya in silinecekler:
        try:
            dosya.unlink()
            silinen += 1
            print(f"Silindi: {dosya}")
        except OSError as e:
            print(f"Silinemedi: {dosya} - {e}")

    print(f"\nToplam silinen dosya: {silinen}")
else:
    print("İşlem iptal edildi. Dosyalar silinmedi.")
