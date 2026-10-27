#!/usr/bin/env python3
"""
organize_fruit_dataset_hashonly.py  -  İSİMSİZ sürüm: dosya adlarına HİÇ bakmaz, sadece pHash benzerliğine göre ayırır

Kurallar
--------
1) SINIF KORUMASI: Sınıf (meyve_durum) klasör yolundan tahmin edilir (sadece sınıf için; augmented kararı için değil).
   Benzerlik karşılaştırması yalnızca aynı sınıfın içinde yapılır. Meyve veya durumu bilinmeyenler hiç karşılaştırılmaz
   ve _unknown/review'a gider.
2) Sadece görsel benzerlik (256 bit pHash, 8 yön: 4 döndürme x ayna):
   - Benzer görüntülerden oluşan kümede en yüksek çözünürlüklü (eşitse en büyük dosya) -> RAW, diğerleri -> AUGMENTED
   - Benzeri olmayan görüntüler -> RAW
   Dikkat: hash 90 derecenin katlarını ve aynalamayı yakalar; 15 derece gibi dönmeleri, crop ve zoom'u yakalayamaz.
   Küme içinde hangisinin "gerçek orijinal" olduğu çözünürlük/boyuta göre tahmin edilir, yanılabilir (rapordaki cluster
   sütunuyla kontrol edin).
3) Hiçbir dosya silinmez. Varsayılan DRY-RUN. --apply kopyalar, --apply --move taşır.

Çıktı:  out-dir/apple/fresh/raw/  augmented/  review/     (+ out-dir/_unknown/...)

Kurulum:  pip install pillow numpy scipy

Akış (isimli sürümün hash önbelleğini yeniden kullanabilirsiniz: --cache kaggle_cache.npz):
  python organize_fruit_dataset_hashonly.py KLASOR --sweep --cache kaggle_cache.npz
  python organize_fruit_dataset_hashonly.py KLASOR --threshold 20 --preview 30 --cache kaggle_cache.npz \
         --report kaggle_hashonly_report.csv
"""
import argparse
import csv
import os
import random
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HASH = 16                      # 16x16 = 256 bit
IMG = HASH * 4
L = HASH * HASH
CACHE_VERSION = f"phash-dihedral-v3-h{HASH}-i{IMG}"   # isimli sürümle aynı: önbellek paylaşılabilir

# anahtar kelime -> klasör adı (sadece SINIF tahmini için; augmented kararında kullanılmaz)
FRUIT_KEYS = {"apple": "apple", "banana": "banana", "orange": "orange",
              "pomegranate": "pomegranate", "strawberr": "strawberry", "grape": "grape",
              "mango": "mango", "guava": "guava", "jujube": "jujube"}
ROTTEN = re.compile(r"(rotten|bad|stale|spoil|bozuk)", re.I)
FRESH = re.compile(r"(fresh|healthy|good|taze)", re.I)


def guess_class(parts):
    """parts: (kök_adı, alt klasörler..., dosya_adı). En derin parçadan yukarı doğru aranır."""
    fruit = state = None
    for part in reversed(parts):
        t = part.lower()
        if fruit is None:
            fruit = next((v for k, v in FRUIT_KEYS.items() if k in t), None)
        if state is None:
            r, f = bool(ROTTEN.search(t)), bool(FRESH.search(t))
            if r != f:
                state = "rotten" if r else "fresh"
    return f"{fruit}_{state}" if fruit and state else None


# --------------------------------------------------------------------------- hash
def dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def phash_bits(a):
    d = dctn(a.astype(np.float64))[:HASH, :HASH]
    return (d > np.median(d)).flatten()


def process(path):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.alpha_composite(im)
                im = bg
            w, h = im.size
            g = im.convert("L").resize((IMG, IMG), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
        bits = np.stack([phash_bits(v) for v in dihedral(a)])
        return str(path), w, h, os.path.getsize(path), np.packbits(bits, axis=1)
    except Exception as e:
        return str(path), None, None, None, str(e)


# --------------------------------------------------------------------------- pairs
def find_pairs(packed, max_thr, chunk=64):
    n = len(packed)
    bits = np.unpackbits(packed, axis=2).astype(np.float32) * 2 - 1
    A = np.ascontiguousarray(bits[:, 0, :])
    B = np.ascontiguousarray(bits.reshape(n * 8, L))
    I, J, D = [], [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dist = ((L - A[s:e] @ B.T) / 2).reshape(e - s, n, 8).min(axis=2)
        ii, jj = np.nonzero(dist <= max_thr)
        gi = ii + s
        m = jj > gi
        I.append(gi[m]); J.append(jj[m]); D.append(dist[ii[m], jj[m]].astype(int))
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def pooled_pairs(packed, pools, max_thr):
    """Çiftler sadece aynı sınıf havuzunun içinde aranır."""
    I, J, D = [], [], []
    total = 0
    for cls, idx in sorted(pools.items()):
        if cls == "?" or len(idx) < 2:      # sınıfı bilinmeyenler ASLA karşılaştırılmaz
            continue
        idx = np.asarray(idx)
        total += len(idx) ** 2 // 2
        print(f"  [{cls}] {len(idx)} görüntü", file=sys.stderr)
        i, j, d = find_pairs(packed[idx], max_thr)
        I.append(idx[i]); J.append(idx[j]); D.append(d)
    print(f"  toplam ~{total:,} çift karşılaştırıldı (sınıf havuzları içinde)", file=sys.stderr)
    if not I:
        e = np.array([], dtype=int)
        return e, e, e
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def cluster(n, I, J, D, thr):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, d in zip(I, J, D):
        if d <= thr:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def pair_dist(packed, i, j):
    a = np.unpackbits(packed[i, 0]).astype(int)
    return int(min((a != np.unpackbits(packed[j, k]).astype(int)).sum() for k in range(8)))


# --------------------------------------------------------------------------- preview
def make_preview(items, files, packed, label, out_dir, n, thumb=140):
    """items: [(küme, tutulan)]  - en şüpheli (uzaklığı büyük) yarısı + rastgele yarısı."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = sorted(((max(pair_dist(packed, k, i) for i in g if i != k), g, k) for g, k in items),
                    key=lambda x: -x[0])
    half = max(1, n // 2)
    pick = scored[:half]
    random.seed(0)
    pick += random.sample(scored[half:], min(len(scored) - half, n - half)) if len(scored) > half else []
    per_sheet, cols = 8, 6
    for sh in range(0, len(pick), per_sheet):
        part = pick[sh:sh + per_sheet]
        sheet = Image.new("RGB", (cols * thumb, len(part) * (thumb + 16)), "white")
        dr = ImageDraw.Draw(sheet)
        for r, (_, g, k) in enumerate(part):
            order = [k] + [i for i in g if i != k][:cols - 1]
            for c, i in enumerate(order):
                try:
                    with Image.open(files[i]["path"]) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.thumbnail((thumb - 6, thumb - 6))
                except Exception:
                    im = Image.new("RGB", (thumb - 6, thumb - 6), "gray")
                x, y = c * thumb, r * (thumb + 16)
                color = (0, 160, 0) if i == k else (210, 0, 0)
                dr.rectangle([x, y, x + thumb - 2, y + thumb - 2], outline=color, width=3)
                sheet.paste(im, (x + 3, y + 3))
                txt = "RAW" if i == k else f"AUG d={pair_dist(packed, k, i)}"
                dr.text((x + 4, y + thumb), txt, fill=color)
        sheet.save(out_dir / f"preview_{sh // per_sheet + 1:02d}.png")
    print(f"Önizleme PNG'leri: {out_dir}  (yeşil=raw, kırmızı=augmented sayılan, d=Hamming uzaklığı)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="Bir veya birden fazla veri seti klasörü")
    ap.add_argument("--threshold", type=int, default=20, help="Hamming eşiği (256 bit üzerinden). Varsayılan 20")
    ap.add_argument("--sweep", action="store_true", help="Farklı eşiklerde küme sayısı tablosu yazdır ve çık")
    ap.add_argument("--apply", action="store_true",
                    help="Yalnızca AUGMENTED etiketli dosyaları kendi klasörlerindeki augmented/ altına taşı")
    ap.add_argument("--report", default="organize_hashonly_report.csv")
    ap.add_argument("--preview", type=int, default=0, help="N küme için karşılaştırma PNG'si üret")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--cache", default="phash_cache2.npz")
    a = ap.parse_args()

    sys.stdout.reconfigure(errors="replace")
    roots = [Path(r).resolve() for r in a.roots]
    for r in roots:
        if not r.is_dir():
            sys.exit(f"Klasör bulunamadı: {r}")
    paths, owner, sig = [], {}, []
    for ri, r in enumerate(roots):
        for p in sorted(r.rglob("*")):
            if "augmented" in [part.lower() for part in p.parts]:
                continue
            if p.suffix.lower() in EXTS:
                st = p.stat()
                paths.append(p)
                sig.append(f"{p}|{st.st_size}|{st.st_mtime_ns}")
                owner[str(p)] = ri
        print(f"  [{ri}] {r.name}: {sum(1 for v in owner.values() if v == ri)} görüntü")
    if not paths:
        sys.exit("Görüntü bulunamadı.")
    print(f"{len(paths)} görüntü bulundu.")

    # --- hash (önbellekli)
    cache = Path(a.cache)
    files, packed = None, None
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if "sig" in z and "meta" in z and str(z["meta"]) == CACHE_VERSION and list(z["sig"]) == sig:
                files = [dict(path=str(p), w=int(w), h=int(h), size=int(s))
                         for p, w, h, s in zip(z["paths"], z["w"], z["h"], z["size"])]
                packed = z["packed"]
                print("Hash önbellekten okundu (yol, boyut ve değişiklik zamanı eşleşti).")
            else:
                print("Önbellek geçersiz (dosyalar veya algoritma değişmiş), hash yeniden hesaplanıyor.")
        except Exception as e:
            print(f"Önbellek okunamadı ({e}), hash yeniden hesaplanıyor.")
    if packed is None:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(process, paths, chunksize=32))
        for r in (r for r in res if r[1] is None):
            print(f"  ATLANDI (bozuk): {r[0]}  -> {r[4]}")
        good = [r for r in res if r[1] is not None]
        files = [dict(path=r[0], w=r[1], h=r[2], size=r[3]) for r in good]
        packed = np.stack([r[4] for r in good])
        np.savez(cache, sig=np.array(sig), meta=np.array(CACHE_VERSION), paths=np.array([f["path"] for f in files]),
                 w=[f["w"] for f in files], h=[f["h"] for f in files],
                 size=[f["size"] for f in files], packed=packed)
    n = len(files)

    for f in files:
        ri = owner[f["path"]]
        rel = Path(f["path"]).relative_to(roots[ri]).parts
        f["src"] = f"{ri}_{roots[ri].name}"
        f["cls"] = guess_class((roots[ri].name,) + rel) or "?"

    pools = {}
    for i, f in enumerate(files):
        pools.setdefault(f["cls"], []).append(i)
    print("\nSınıf dağılımı:")
    for c, idx in sorted(pools.items()):
        print(f"  {c:22s} {len(idx):6d}")
    if "?" in pools:
        print(f"  UYARI: {len(pools['?'])} görüntünün meyve/durum sınıfı tahmin edilemedi. Bunlar HİÇ karşılaştırılmaz -> _unknown/review.")

    # --- çiftler (sınıf havuzları içinde)
    max_thr = max(a.threshold, 40) if a.sweep else a.threshold
    print(f"\nÇift karşılaştırma (8 yön, eşik<={max_thr}) ...")
    I, J, D = pooled_pairs(packed, pools, max_thr)

    if a.sweep:
        print("\n eşik | küme sayısı | augmented sayılacak dosya")
        for t in range(4, 41, 4):
            gs = cluster(n, I, J, D, t)
            print(f"  {t:3d} | {len(gs):11d} | {sum(len(g) - 1 for g in gs):8d}")
        print("\nEşik arttıkça sayı aniden zıplıyorsa sonrası yanlış pozitif olabilir; --preview 30 ile kontrol edin.")
        return

    groups = cluster(n, I, J, D, a.threshold)

    def keep_key(i):
        f = files[i]
        return (-(f["w"] * f["h"]), -f["size"], f["path"])

    label, evid, cid_of, keeper_of, prev_items = [None] * n, [""] * n, [0] * n, {}, []
    for cid, g in enumerate(groups, 1):
        k = min(g, key=keep_key)
        keeper_of[cid] = k
        for i in g:
            cid_of[i] = cid
            if i == k:
                label[i], evid[i] = "raw", "cluster_keeper_hash_only"
            else:
                label[i], evid[i] = "augmented", "similar_to_raw_hash_only"
        prev_items.append((g, k))
    for i in range(n):
        if label[i] is None:
            if files[i]["cls"] == "?":
                label[i], evid[i] = "review", "unknown_class_not_compared"
            else:
                label[i], evid[i] = "raw", "unique"

    # --- Hedef: AUGMENTED dosyalar kaynak klasörlerinin altındaki augmented/ içine.
    # RAW dosyaların hedefi kendi mevcut yoludur; onlara dokunulmaz.
    used = set()

    def dest_for(i):
        src_path = Path(files[i]["path"])
        if label[i] != "augmented":
            return src_path

        d = src_path.parent / "augmented"
        name = src_path.name
        cand, c = d / name, 0
        while cand in used or cand.exists():
            c += 1
            cand = d / f"{src_path.stem}__hashaug_{c}{src_path.suffix}"
        used.add(cand)
        return cand

    dests = [dest_for(i) for i in range(n)]

    with open(a.report, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster", "class_guess", "label", "evidence", "source", "dist_to_keeper", "size", "path", "dest"])
        for i in range(n):
            k = keeper_of.get(cid_of[i])
            dist = pair_dist(packed, k, i) if (k is not None and k != i) else ""
            w.writerow([cid_of[i], files[i]["cls"], label[i], evid[i], files[i]["src"], dist,
                        f'{files[i]["w"]}x{files[i]["h"]}', files[i]["path"], dests[i]])

    cnt = Counter((files[i]["cls"], label[i]) for i in range(n))
    print(f"\nEşik={a.threshold} (SADECE HASH): {len(groups)} küme")
    print(f"  {'sınıf':22s} {'raw':>9s} {'augmented':>10s} {'review':>8s}")
    for c in sorted(pools):
        print(f"  {c:22s} {cnt[(c, 'raw')]:9d} {cnt[(c, 'augmented')]:10d} {cnt[(c, 'review')]:8d}")
    tot = Counter(label)
    print(f"  {'TOPLAM':22s} {tot['raw']:9d} {tot['augmented']:10d} {tot['review']:8d}")
    print(f"En büyük 5 küme: {sorted((len(g) for g in groups), reverse=True)[:5]}  (çok büyükse eşik yüksek olabilir)")
    print(f"Rapor: {a.report}")

    if a.preview:
        if prev_items:
            make_preview(prev_items, files, packed, label, Path("organize_hashonly_preview"), a.preview)
        else:
            print("Önizlenecek küme yok.")

    if a.apply:
        aug_indices = [i for i in range(n) if label[i] == "augmented"]
        print(f"Taşınacak AUGMENTED görüntü: {len(aug_indices)}")
        print("RAW ve diğer görüntüler yerinde bırakılacak.")

        # Tüm hedefler önceden hesaplandı ve çakışmalar farklı adlarla çözüldü.
        moved = 0
        for i in aug_indices:
            src_path = Path(files[i]["path"])
            dest_path = dests[i]
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            moved += 1

        print(f"Taşındı: {moved} AUGMENTED görüntü -> kendi sınıf klasörlerindeki augmented/ klasörleri")
        # Kaynak yolları değiştiği için eski hash önbelleğini kaldır.
        if cache.exists():
            cache.unlink()
    else:
        print("DRY-RUN: hiçbir dosya taşınmadı. Uygulamak için --apply ekleyin.")


if __name__ == "__main__":
    main()
