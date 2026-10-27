import json
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

# Bu betik dosyaları kalıcı olarak silmez.
# Önce temizleme planı oluşturur; onay verirsen kopya adaylarını
# _KARANTINA_KOPYA_ADAYLARI klasörüne taşır.
# Yalnızca HTML'de "Kopya" işaretlediğin çiftler işleme alınır.

# Dosya adında bu ipuçlarından biri varsa (veya yolda "augment" geçen
# klasör varsa) görüntü augmented sayılır.
AUG_HINTS = ("aug", "rot", "flip", "crop", "zoom", "bright", "noise", "shear")

# Split önceliği yalnızca çözünürlük ve dosya boyutu eşitse kullanılır.
SPLIT_PRIORITY = {"test": 0, "valid": 1, "validation": 1, "val": 1, "train": 2}
EVAL_SPLITS = {"test", "valid", "validation", "val"}


def main():
    raw = input(
        "Veri seti klasörünün yolunu yapıştır "
        "(CSV ve JSON dosyalarının bulunduğu klasör): "
    ).strip().strip('"')
    BASE = Path(raw).expanduser()

    if not BASE.is_dir():
        print(f"Klasör bulunamadı: {BASE}")
        input("Kapatmak için Enter...")
        return

    ORB_CSV = BASE / "benzer_goruntuler_orb_raporu.csv"
    JSON_FILE = BASE / "benzer_goruntu_secimleri.json"
    PLAN_CSV = BASE / "kopya_temizleme_plani.csv"
    QUARANTINE = BASE / "_KARANTINA_KOPYA_ADAYLARI"

    for p in (ORB_CSV, JSON_FILE):
        if not p.exists():
            print(f"Dosya bulunamadı: {p}")
            print("CSV ve JSON dosyalarını bu klasöre koy.")
            input("Kapatmak için Enter...")
            return

    df = pd.read_csv(ORB_CSV)
    with JSON_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    selections = data.get("selections", {})

    # JSON ile CSV'nin aynı taramaya ait olduğunu doğrula.
    json_pairs = data.get("pairs")
    if json_pairs:
        mismatch = []
        for pid in selections:
            try:
                i = int(pid) - 1
                expected = [
                    str(df.iloc[i]["image_1"]),
                    str(df.iloc[i]["image_2"]),
                ]
            except (ValueError, IndexError):
                mismatch.append(pid)
                continue

            if i < 0 or json_pairs.get(pid) != expected:
                mismatch.append(pid)

        if mismatch:
            print(
                f"\nDURDURULDU: {len(mismatch)} seçimin dosya yolları "
                "CSV ile uyuşmuyor."
            )
            print("JSON ve CSV farklı taramalardan olabilir. Hiçbir şey yapılmadı.")
            print("Örnek numaralar:", ", ".join(mismatch[:10]))
            input("Kapatmak için Enter...")
            return
    else:
        print("\nUYARI: JSON içinde çift yolları yok.")
        print("CSV tekrar üretildiyse seçimler yanlış çiftlere uygulanabilir.")
        if input("Yine de devam etmek için DEVAM yaz: ").strip() != "DEVAM":
            print("Hiçbir şey yapılmadı.")
            input("Kapatmak için Enter...")
            return

    def rel_parts(path_text):
        p = Path(path_text)
        try:
            return p.relative_to(BASE).parts
        except ValueError:
            return p.parts

    def split_of(path_text):
        for part in rel_parts(path_text):
            if part.lower() in SPLIT_PRIORITY:
                return part.lower()
        return "unknown"

    pixel_cache = {}
    file_size_cache = {}

    def pixels_of(path_text):
        if path_text not in pixel_cache:
            try:
                with Image.open(path_text) as im:
                    pixel_cache[path_text] = im.width * im.height
            except Exception:
                pixel_cache[path_text] = 0
        return pixel_cache[path_text]

    def file_size_of(path_text):
        if path_text not in file_size_cache:
            try:
                file_size_cache[path_text] = Path(path_text).stat().st_size
            except OSError:
                file_size_cache[path_text] = 0
        return file_size_cache[path_text]

    def dimensions_of(path_text):
        try:
            with Image.open(path_text) as im:
                return f"{im.width}x{im.height}"
        except Exception:
            return ""

    def is_aug(path_text):
        parts = [x.lower() for x in rel_parts(path_text)]
        name = parts[-1] if parts else ""
        if any(hint in name for hint in AUG_HINTS):
            return True
        return any("augment" in part for part in parts[:-1])

    # Yalnızca HTML'de "copy" seçilmiş çiftlerden bağlantı grafiği oluştur.
    graph = defaultdict(set)
    pair_rows = []

    for idx, row in df.iterrows():
        pair_id = str(idx + 1)
        decision = selections.get(pair_id, "unselected")
        a, b = str(row["image_1"]), str(row["image_2"])

        if decision == "copy":
            graph[a].add(b)
            graph[b].add(a)
            pair_rows.append((pair_id, a, b))

    if not pair_rows:
        print("\n'Kopya' olarak işaretlenmiş çift yok. Yapılacak bir şey yok.")
        input("Kapatmak için Enter...")
        return

    # Kopya çiftlerinden bağlantılı grupları oluştur.
    visited = set()
    components = []

    for node in list(graph):
        if node in visited:
            continue

        stack = [node]
        comp = set()

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(graph[cur] - visited)

        components.append(comp)

    def keeper_key(path_text):
        # EN ÖNEMLİ ÖLÇÜT: en yüksek çözünürlük (genişlik x yükseklik).
        # Çözünürlük eşitse dosya boyutu büyük olan korunur.
        # Bunlar da eşitse augmented olmadığı tahmin edilen tercih edilir.
        # Sonraki eşitliklerde split, bağlantı sayısı ve dosya yolu kullanılır.
        return (
            -pixels_of(path_text),
            -file_size_of(path_text),
            is_aug(path_text),
            SPLIT_PRIORITY.get(split_of(path_text), 3),
            -len(graph[path_text]),
            path_text.lower(),
        )

    plan = []

    for group_num, comp in enumerate(components, start=1):
        existing = [p for p in comp if Path(p).is_file()]

        if len(existing) < 2:
            for image_path in sorted(comp):
                plan.append({
                    "group": group_num,
                    "decision": "SKIP_MISSING_FILES",
                    "image_path": image_path,
                    "keep_image": "",
                    "resolution": dimensions_of(image_path),
                    "pixels": pixels_of(image_path),
                    "file_size_bytes": file_size_of(image_path),
                    "split": split_of(image_path),
                    "is_augmented_guess": is_aug(image_path),
                    "connections_in_copy_pairs": len(graph[image_path]),
                    "exists_on_this_PC": Path(image_path).is_file(),
                    "warning": "Grupta karşılaştırılabilir en az iki mevcut dosya yok",
                })
            continue

        keeper = sorted(existing, key=keeper_key)[0]

        for image_path in sorted(comp):
            exists = Path(image_path).is_file()
            decision = "KEEP" if image_path == keeper else "MOVE_TO_QUARANTINE"
            warning = ""

            if not exists:
                decision = "SKIP_MISSING_FILES"
                warning = "Dosya bu konumda bulunamadı"
            elif decision == "MOVE_TO_QUARANTINE" and split_of(image_path) in EVAL_SPLITS:
                warning = "test/valid içinden taşınıyor; özellikle kontrol et"

            plan.append({
                "group": group_num,
                "decision": decision,
                "image_path": image_path,
                "keep_image": keeper,
                "resolution": dimensions_of(image_path),
                "pixels": pixels_of(image_path),
                "file_size_bytes": file_size_of(image_path),
                "split": split_of(image_path),
                "is_augmented_guess": is_aug(image_path),
                "connections_in_copy_pairs": len(graph[image_path]),
                "exists_on_this_PC": exists,
                "warning": warning,
            })

    if not plan:
        print("\nTemizleme planı oluşturulamadı.")
        input("Kapatmak için Enter...")
        return

    plan_df = pd.DataFrame(plan)
    plan_df.to_csv(PLAN_CSV, index=False, encoding="utf-8-sig")

    move_rows = plan_df[plan_df["decision"] == "MOVE_TO_QUARANTINE"]
    missing = plan_df[~plan_df["exists_on_this_PC"]]
    eval_moves = move_rows[move_rows["warning"].str.contains("test/valid", na=False)]
    keep_aug = plan_df[
        (plan_df["decision"] == "KEEP") & (plan_df["is_augmented_guess"])
    ]

    print(f"\nKopya diye işaretlenen çift sayısı: {len(pair_rows)}")
    print(f"Bağlantılı görüntü grubu sayısı: {len(components)}")
    print(f"Korunacak görüntü sayısı: {(plan_df['decision'] == 'KEEP').sum()}")
    print(f"Karantinaya taşınması önerilen görüntü sayısı: {len(move_rows)}")
    print(f"Bu bilgisayarda yolu bulunamayan görüntü sayısı: {len(missing)}")
    print(f"UYARI - test/valid içinden taşınacak görüntü sayısı: {len(eval_moves)}")
    print(f"UYARI - korunan ama augmented gibi görünen görüntü sayısı: {len(keep_aug)}")
    print(f"Plan dosyası: {PLAN_CSV}")
    print("\nKoruma önceliği: yüksek çözünürlük > büyük dosya boyutu > diğer ölçütler.")
    print("Bu işlem kalıcı silme değildir; seçilen dosyalar karantinaya taşınır.")
    print("TAŞI yazmadan önce plan CSV'sindeki KEEP/MOVE satırlarını kontrol et.")
    print("İptal etmek için Enter'a bas.")

    answer = input(
        "\nKontrol ettikten sonra dosyaları karantinaya taşımak için TAŞI yaz: "
    ).strip()

    if answer != "TAŞI":
        print("Hiçbir görsel taşınmadı. Plan oluşturuldu.")
        input("Kapatmak için Enter...")
        return

    moved = 0
    skipped = 0

    for _, row in move_rows.iterrows():
        src = Path(row["image_path"])

        if not src.is_file():
            skipped += 1
            continue

        try:
            rel = src.relative_to(BASE)
        except ValueError:
            rel = Path(src.parent.name) / src.name

        dest = QUARANTINE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        n = 1
        original_dest = dest
        while dest.exists():
            dest = original_dest.with_name(
                f"{original_dest.stem}_copy{n}{original_dest.suffix}"
            )
            n += 1

        shutil.move(str(src), str(dest))
        moved += 1

    print(f"\nTamamlandı. Karantinaya taşınan: {moved}; bulunamadığı için atlanan: {skipped}")
    print(f"Karantina klasörü: {QUARANTINE}")
    print("Karantina klasörünü kontrol etmeden silme.")
    input("Kapatmak için Enter...")


if __name__ == "__main__":
    main()
