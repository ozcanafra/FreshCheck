#!/usr/bin/env python3
"""
name_split_augmented_v4.py  -  ADIM 1: SADECE İSİMLE augmented ayırma (hash yok, çok hızlı)

Dosya adında (veya klasör adında) augmentation anahtar kelimesi geçen görüntüleri bulur:
    rotated_by_15_..., saltandpepper_..., translation_..., vertical_flip_..., Banana__Healthy_augmented_... gibi.

İKİ ÇALIŞMA ŞEKLİ
  --in-place  : Her dosyanın KENDİ KLASÖRÜNDE  augmented/  alt klasörü oluşturulur ve dosya oraya TAŞINIR.
                  apple/fresh/rotated_by_15_X.png  ->  apple/fresh/augmented/rotated_by_15_X.png
                Her taşıma name_split_moves.csv dosyasına yazılır; --undo ile geri alınabilir.
  (varsayılan): out-dir/<meyve>/<durum>/augmented/ altına KOPYALAR (veya --move ile taşır).

Varsayılan DRY-RUN: --apply verilmedikçe hiçbir dosya kopyalanmaz/taşınmaz.
Zaten  augmented  adlı klasörün içinde duran dosyalara dokunulmaz (tekrar çalıştırmak güvenlidir).

Kullanım:
  python name_split_augmented_v4.py KLASOR --in-place                    # sadece rapor
  python name_split_augmented_v4.py KLASOR --in-place --apply            # klasör içinde augmented/ oluştur ve taşı
  python name_split_augmented_v4.py --undo name_split_moves.csv          # geri al
  python name_split_augmented_v4.py KLASOR --apply --out-dir C:\\fruit_step1   # eski davranış: ayrı klasöre kopyala
"""
import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AUG_DIR = "augmented"

# Dosya/klasör adı augmentation anahtar kelimeleri (gerekirse --extra ile ekle)
AUG_WORDS = (r"rotated_by|rotat|saltandpepper|salt_and_pepper|translation|translate|"
             r"vertical_flip|horizontal_flip|flip|mirror|augmented|augmentation|augment|"
             r"(?<![a-z])aug(?![a-z])|noise|shear|zoom|bright|brightness|contrast|"
             r"blur|blurred|crop|cropped|shift|cutout")
# orijinal adını bulmak için atılacak ön ekler (örn. rotated_by_15_X.png -> X.png)
BASE_RE = re.compile(r"^(rotated_by_\d+|saltandpepper|salt_and_pepper|translation|vertical_flip|horizontal_flip)_", re.I)

FRUIT_KEYS = {"apple": "apple", "banana": "banana", "orange": "orange",
              "pomegranate": "pomegranate", "strawberr": "strawberry", "grape": "grape",
              "mango": "mango", "guava": "guava", "jujube": "jujube"}
ROTTEN = re.compile(r"(rotten|bad|stale|spoil|bozuk)", re.I)
FRESH = re.compile(r"(fresh|healthy|good|taze)", re.I)


def guess_fs(parts):
    """En derin parçadan yukarı doğru (meyve, durum) arar."""
    fruit = state = None
    for part in reversed(parts):
        t = part.lower()
        if fruit is None:
            fruit = next((v for k, v in FRUIT_KEYS.items() if k in t), None)
        if state is None:
            r, f = bool(ROTTEN.search(t)), bool(FRESH.search(t))
            if r != f:
                state = "rotten" if r else "fresh"
    return fruit, state


def class_for(root_name, rel):
    """Önce klasör yolu; klasör eksik bırakırsa dosya adından tamamla (dosya adı klasörü ezmez)."""
    fruit, state = guess_fs((root_name,) + rel[:-1])
    if fruit is None or state is None:
        f2, s2 = guess_fs((rel[-1],))
        fruit, state = fruit or f2, state or s2
    return f"{fruit}_{state}" if fruit and state else "?"


def undo(log):
    n = miss = 0
    with open(log, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            src, dst = Path(row["old_path"]), Path(row["new_path"])
            if dst.exists() and not src.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
                n += 1
                try:
                    dst.parent.rmdir()          # boş kaldıysa augmented klasörünü kaldır
                except OSError:
                    pass
            else:
                miss += 1
    print(f"Geri alındı: {n} dosya | atlanan (yeni yerinde yok ya da eski yerinde zaten var): {miss}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="Bir veya birden fazla veri seti klasörü")
    ap.add_argument("--in-place", action="store_true", help="Her dosyanın kendi klasöründe augmented/ oluştur ve oraya TAŞI")
    ap.add_argument("--extra", default="", help="Ek anahtar kelimeler, regex: ornek  'blurred|cutout'")
    ap.add_argument("--apply", action="store_true", help="Gerçekten kopyala/taşı (yoksa sadece rapor)")
    ap.add_argument("--move", action="store_true", help="out-dir modunda: kopyalamak yerine taşı")
    ap.add_argument("--out-dir", default="dataset_step1_names", help="(--in-place yokken) hedef klasör")
    ap.add_argument("--report", default="name_split_report.csv")
    ap.add_argument("--moves-log", default="name_split_moves.csv", help="Taşımaların kaydı (geri alma için)")
    ap.add_argument("--undo", metavar="LOG", help="Bir taşıma kaydını geri al ve çık")
    a = ap.parse_args()

    sys.stdout.reconfigure(errors="replace")
    if a.undo:
        undo(a.undo)
        return
    if not a.roots:
        sys.exit("En az bir klasör verin (ya da --undo LOG).")

    words = AUG_WORDS + (f"|{a.extra}" if a.extra else "")
    aug_re = re.compile(f"({words})", re.I)

    roots = [Path(r).resolve() for r in a.roots]
    for r in roots:
        if not r.is_dir():
            sys.exit(f"Klasör bulunamadı: {r}")
    out_dir = Path(a.out_dir).resolve()
    if a.apply and not a.in_place and out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} boş değil. Boş/yeni bir --out-dir kullanın.")

    items = []   # (path, root_index, rel_parts)
    for ri, r in enumerate(roots):
        for p in sorted(r.rglob("*")):
            if p.suffix.lower() in EXTS and p.is_file() and (a.in_place or out_dir not in p.parents):
                items.append((p, ri, p.relative_to(r).parts))
        print(f"  [{ri}] {r.name}: {sum(1 for x in items if x[1] == ri)} görüntü")
    if not items:
        sys.exit("Görüntü bulunamadı.")
    print(f"{len(items)} görüntü bulundu.")
    existing = {str(p).lower() for p, _, _ in items}

    rows, used = [], set()
    cls_cnt, kw_cnt, orig_cnt, unmatched_words = Counter(), Counter(), Counter(), Counter()
    for p, ri, rel in items:
        cls = class_for(roots[ri].name, rel)
        where = kw = ""
        m = aug_re.search(rel[-1])
        if m:
            where, kw = "name", m.group(1).lower()
        else:
            for d in rel[:-1]:
                m = aug_re.search(d)
                if m:
                    where, kw = "folder", m.group(1).lower()
                    break
        base, found, action, dest = "", "", "", ""
        if where:
            label = "augmented"
            in_aug = p.parent.name.lower() == AUG_DIR
            mb = BASE_RE.match(rel[-1])
            if mb:
                base = rel[-1][mb.end():]
                dirs = [p.parent] + ([p.parent.parent] if in_aug else [])
                found = "yes" if any(str(d / base).lower() in existing for d in dirs) else "no"
            else:
                found = "n/a"
            if a.in_place:
                if in_aug:
                    action, dest = "skip_already_in_augmented", str(p)
                else:
                    action, dest = "move", str(p.parent / AUG_DIR / p.name)
            else:
                if cls == "?":
                    folder = out_dir / "_unknown" / AUG_DIR
                else:
                    fr, st = cls.split("_")
                    folder = out_dir / fr / st / AUG_DIR
                d = folder / p.name
                c = 0
                while d in used:
                    c += 1
                    d = folder / f"{p.stem}__{ri}_{c}{p.suffix}"
                used.add(d)
                action, dest = ("move" if a.move else "copy"), str(d)
            kw_cnt[kw] += 1
            orig_cnt[found] += 1
        else:
            label = "unmatched"
            unmatched_words[re.split(r"[\d_ \-.]+", p.stem)[0].lower() or "(bos)"] += 1
        cls_cnt[(cls, label)] += 1
        rows.append([cls, label, where, kw, base, found, action, str(p), dest])

    with open(a.report, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["class_guess", "label", "matched_in", "keyword", "original_name",
                    "original_in_folder", "action", "path", "dest"])
        w.writerows(rows)

    n_aug = sum(1 for r in rows if r[1] == "augmented")
    n_act = sum(1 for r in rows if r[6] in ("move", "copy"))
    classes = sorted({c for c, _ in cls_cnt})
    print(f"\n{'sınıf':22s} {'augmented':>10s} {'unmatched':>10s}")
    for c in classes:
        print(f"{c:22s} {cls_cnt[(c, 'augmented')]:10d} {cls_cnt[(c, 'unmatched')]:10d}")
    print(f"{'TOPLAM':22s} {n_aug:10d} {len(rows) - n_aug:10d}")
    print("\nEşleşen anahtar kelimeler:")
    for k, v in kw_cnt.most_common():
        print(f"  {k:20s} {v}")
    print("\nOrijinali aynı klasörde bulunan (ön ek atılmış adla):",
          f"yes={orig_cnt['yes']}  no={orig_cnt['no']}  n/a={orig_cnt['n/a']}")
    print("\nEşleşmeyen dosyaların en sık başlangıç kelimeleri (kaçırılan bir augmentation adı var mı bak):")
    for k, v in unmatched_words.most_common(15):
        print(f"  {k:20s} {v}")
    if a.in_place:
        dirs = {Path(r[8]).parent for r in rows if r[6] == "move"}
        print(f"\n{n_act} dosya taşınacak, {len(dirs)} klasörde '{AUG_DIR}' alt klasörü oluşacak/kullanılacak.")
        skipped = sum(1 for r in rows if r[6] == "skip_already_in_augmented")
        if skipped:
            print(f"{skipped} dosya zaten '{AUG_DIR}' klasöründe, atlandı.")
    print(f"Rapor: {a.report}")

    if a.apply:
        mover = shutil.move if (a.in_place or a.move) else shutil.copy2
        moved = []
        for r in rows:
            if r[6] in ("move", "copy"):
                d = Path(r[8])
                if d.exists():
                    print(f"  ATLANDI (hedef zaten var): {d}")
                    continue
                d.parent.mkdir(parents=True, exist_ok=True)
                mover(r[7], str(d))
                if r[6] == "move":
                    moved.append((r[7], str(d)))
        if moved:
            with open(a.moves_log, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["old_path", "new_path"])
                w.writerows(moved)
            print(f"Taşıma kaydı: {a.moves_log}  (geri almak için: --undo {a.moves_log})")
        print(f"{'Kopyalandı' if (not a.in_place and not a.move) else 'Taşındı'}: {n_act} dosya")
    else:
        print("DRY-RUN: hiçbir dosya kopyalanmadı/taşınmadı. Uygulamak için --apply ekleyin.")


if __name__ == "__main__":
    main()
