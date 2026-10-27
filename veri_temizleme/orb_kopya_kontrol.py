from pathlib import Path
import csv
import base64
import io
import html
import json
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_gray(path):
    # cv2.imread Windows'ta Türkçe/Unicode karakterli yolları okuyamaz;
    # bu yüzden dosyayı numpy ile okuyup imdecode kullanıyoruz.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if image is None:
        return None
    h, w = image.shape
    scale = min(1.0, 1000 / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def compare_images(path1, path2):
    a, b = load_gray(path1), load_gray(path2)
    if a is None or b is None:
        return 0, 0, 0.0, "Görüntü okunamadı"

    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    if des1 is None or des2 is None:
        return 0, 0, 0.0, "Yetersiz ayrıntı"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return len(good), 0, 0.0, "Zayıf eşleşme - kontrol et"

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    except cv2.error:
        mask = None

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    ratio = inliers / len(good) if good else 0.0
    if inliers >= 18 and ratio >= 0.45:
        label = "Güçlü ortak ayrıntı - kopya adayı"
    elif inliers >= 8 and ratio >= 0.25:
        label = "Belirsiz - elle kontrol et"
    else:
        label = "Zayıf eşleşme - farklı olabilir"
    return len(good), inliers, ratio, label


def tier_of(label):
    """HTML süzgeci için kademe. ORB karar vermez, sadece etiketler."""
    if label.startswith("Güçlü"):
        return "strong"
    if label.startswith("Zayıf"):
        return "weak"
    # Belirsiz, Yetersiz ayrıntı, Görüntü okunamadı -> mutlaka elle bak
    return "unsure"


def image_uri(path_text):
    try:
        with Image.open(path_text) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((280, 230))
            canvas = Image.new("RGB", (280, 230), "white")
            canvas.paste(im, ((280 - im.width) // 2, (230 - im.height) // 2))
            buf = io.BytesIO()
            canvas.save(buf, format="JPEG", quality=76)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def main():
    raw = input("benzer_goruntuler_raporu.csv dosyasının yolunu gir: ").strip().strip('"')
    csv_path = Path(raw).expanduser()
    if not csv_path.is_file():
        print(f"HATA: CSV bulunamadı:\n{csv_path}")
        input("Kapatmak için Enter...")
        return

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV boş.")
        input("Kapatmak için Enter...")
        return

    if len(rows) > 1000:
        print(f"UYARI: {len(rows)} çift var. HTML çok büyük olabilir (yüzlerce MB) ve "
              "tarayıcı yavaşlayabilir. Gerekirse CSV'yi parçalara bölmeyi düşün.")

    # Aynı CSV -> aynı kimlik -> seçimler korunur.
    # CSV değişirse (yeni tarama, farklı eşik) kimlik değişir ve eski seçimler
    # yanlış çiftlere uygulanmaz.
    run_id = hashlib.sha1(
        "\n".join(f"{r.get('image_1', '')}|{r.get('image_2', '')}" for r in rows).encode("utf-8")
    ).hexdigest()[:12]

    results, cards = [], []
    tier_counts = {"strong": 0, "unsure": 0, "weak": 0}
    for idx, row in enumerate(rows, 1):
        p1, p2 = row.get("image_1", ""), row.get("image_2", "")
        good, inliers, ratio, label = compare_images(p1, p2)
        tier = tier_of(label)
        tier_counts[tier] += 1
        result = dict(row)
        result.update({
            "orb_good_matches": good,
            "orb_inliers": inliers,
            "orb_inlier_ratio": round(ratio, 4),
            "orb_assessment": label
        })
        results.append(result)

        sim = html.escape(str(row.get("similarity", "")))
        cls = html.escape(str(row.get("class", "")))
        safe_label = html.escape(label)
        safe_p1, safe_p2 = html.escape(p1), html.escape(p2)
        uri1, uri2 = image_uri(p1), image_uri(p2)
        pic1 = f'<img src="{uri1}" alt="Görüntü 1">' if uri1 else '<p>Görüntü açılamadı</p>'
        pic2 = f'<img src="{uri2}" alt="Görüntü 2">' if uri2 else '<p>Görüntü açılamadı</p>'

        cards.append(f"""
        <article class="card" data-id="{idx}" data-tier="{tier}" data-a="{safe_p1}" data-b="{safe_p2}">
          <h2>Eşleşme {idx} · MobileNet: {sim}</h2>
          <p><b>Sınıf:</b> {cls}</p>
          <p><b>ORB ön değerlendirmesi:</b> {safe_label}</p>
          <p><b>İyi eşleşme:</b> {good} · <b>Geometrik ortak nokta:</b> {inliers} · <b>Oran:</b> {ratio:.3f}</p>
          <div class="images">
            <div>{pic1}<p class="path">{safe_p1}</p></div>
            <div>{pic2}<p class="path">{safe_p2}</p></div>
          </div>
          <div class="choices">
            <label><input type="radio" name="pair_{idx}" value="copy"> Kopya / türetilmiş görüntü</label>
            <label><input type="radio" name="pair_{idx}" value="different"> Farklı fotoğraf</label>
            <label><input type="radio" name="pair_{idx}" value="unsure"> Emin değilim</label>
          </div>
          <small class="saved-status" aria-live="polite">Seçim yapılmadı</small>
        </article>
        """)

    out_csv = csv_path.parent / "benzer_goruntuler_orb_raporu.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    out_html = csv_path.parent / "benzer_goruntuler_orb_kontrol.html"
    page = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benzer Görüntü Kontrolü</title>
<style>
body{{font-family:Arial,sans-serif;margin:20px;background:#f3f4f6;color:#222}}
header{{position:sticky;top:0;z-index:5;background:#fff;padding:12px;border:1px solid #ddd;border-radius:10px;margin-bottom:16px}}
.card{{background:white;border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0;max-width:1100px;break-inside:avoid}}
.card h2{{font-size:19px}} .images{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.images img{{display:block;width:100%;max-width:280px;height:230px;object-fit:contain;background:#eee}}
.path{{font-size:12px;overflow-wrap:anywhere;color:#444}}
.choices{{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0}}
.choices label{{cursor:pointer}} .saved-status{{color:#176b37}}
button{{padding:9px 12px;margin:4px;cursor:pointer}}
button.active{{background:#222;color:#fff}}
.notice{{background:#fff3cd;padding:12px;border-radius:8px;line-height:1.5}}
@media(max-width:650px){{.images{{grid-template-columns:1fr}}}}
</style></head><body>
<header>
  <b>İncelenen: {len(rows)} eşleşme</b> · <span id="progress">0/{len(rows)} seçim</span><br>
  <button id="export">Seçimleri JSON olarak indir / kaydet</button>
  <button id="clear">Tüm seçimleri temizle</button><br>
  <span>Göster:</span>
  <button class="flt" data-t="all">Hepsi ({len(rows)})</button>
  <button class="flt" data-t="unsure">Belirsiz ({tier_counts['unsure']})</button>
  <button class="flt" data-t="strong">Güçlü ({tier_counts['strong']})</button>
  <button class="flt" data-t="weak">Zayıf ({tier_counts['weak']})</button>
  <button class="flt" data-t="todo">Seçilmemiş</button>
  <p class="notice">Seçimlerin bu tarayıcıda otomatik kaydedilir. Başka bilgisayara taşımak veya yedeklemek için JSON indirme düğmesini kullan. Hiçbir fotoğraf silinmez veya taşınmaz. ORB etiketi sadece yardımcıdır; "güçlü" olanlar dahil her çifti sen karar verirsin.</p>
</header>
<h1>ORB Benzer Görüntü Kontrolü</h1>
<p>ORB değerlendirmesi yardımcı bilgidir; kesin kopya kararı değildir.</p>
{"".join(cards)}
<script>
const RUN_ID = "{run_id}";
const STORE_KEY = "fruit_duplicate_review_v2_" + RUN_ID;
const total = {len(rows)};
let selections = {{}};
try {{ selections = JSON.parse(localStorage.getItem(STORE_KEY) || "{{}}"); }} catch (e) {{ selections = {{}}; }}

// Varsayılan: her çift "Kopya" seçili başlasın.
// Daha önce kullanıcı tarafından verilmiş seçimleri koru.
document.querySelectorAll(".card").forEach(card => {{
  const id = card.dataset.id;
  if (!selections[id]) selections[id] = "copy";
}});
localStorage.setItem(STORE_KEY, JSON.stringify(selections));

function updateProgress() {{
  const count = Object.keys(selections).filter(k => selections[k]).length;
  document.getElementById("progress").textContent = count + "/" + total + " seçim";
}}
function restore() {{
  document.querySelectorAll(".card").forEach(card => {{
    const id = card.dataset.id;
    const value = selections[id];
    if (value) {{
      const radio = card.querySelector('input[value="' + value + '"]');
      if (radio) radio.checked = true;
      card.querySelector(".saved-status").textContent = "Seçim kaydedildi";
    }}
  }});
  updateProgress();
}}
function applyFilter(t) {{
  document.querySelectorAll(".card").forEach(c => {{
    const done = !!selections[c.dataset.id];
    const show = t === "all" || c.dataset.tier === t || (t === "todo" && !done);
    c.style.display = show ? "" : "none";
  }});
  document.querySelectorAll(".flt").forEach(b => b.classList.toggle("active", b.dataset.t === t));
}}
document.querySelectorAll(".flt").forEach(b =>
  b.addEventListener("click", () => applyFilter(b.dataset.t)));
document.querySelectorAll(".card input[type=radio]").forEach(radio => {{
  radio.addEventListener("change", event => {{
    const card = event.target.closest(".card");
    selections[card.dataset.id] = event.target.value;
    localStorage.setItem(STORE_KEY, JSON.stringify(selections));
    card.querySelector(".saved-status").textContent = "Seçim kaydedildi";
    updateProgress();
  }});
}});
document.getElementById("export").addEventListener("click", () => {{
  // Seçimlerle birlikte her çiftin dosya yollarını da kaydediyoruz;
  // betik 3 bunları CSV ile karşılaştırıp kayma olup olmadığını kontrol eder.
  const pairs = {{}};
  document.querySelectorAll(".card").forEach(c => {{ pairs[c.dataset.id] = [c.dataset.a, c.dataset.b]; }});
  const payload = {{
    created_at: new Date().toISOString(),
    run_id: RUN_ID,
    source_html: location.pathname,
    selections: selections,
    pairs: pairs
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type:"application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "benzer_goruntu_secimleri.json";
  a.click();
  URL.revokeObjectURL(url);
}});
document.getElementById("clear").addEventListener("click", () => {{
  if (!confirm("Tüm seçimleri temizlemek istiyor musun?")) return;
  selections = {{}};
  localStorage.removeItem(STORE_KEY);
  document.querySelectorAll(".card input[type=radio]").forEach(x => x.checked = false);
  document.querySelectorAll(".saved-status").forEach(x => x.textContent = "Seçim yapılmadı");
  updateProgress();
}});
restore();
applyFilter("all");
</script>
</body></html>"""
    out_html.write_text(page, encoding="utf-8")

    print("\nTAMAMLANDI")
    print(f"HTML raporu: {out_html}")
    print(f"CSV raporu: {out_csv}")
    print(f"Kademeler -> güçlü: {tier_counts['strong']}, belirsiz: {tier_counts['unsure']}, zayıf: {tier_counts['weak']}")
    print("HTML'yi çift tıklayarak tarayıcıda aç.")
    print("Seçimler tarayıcıda otomatik kaydolur. Yedek için HTML'deki JSON düğmesini kullan.")
    print("JSON dosyasını (benzer_goruntu_secimleri.json) CSV ile aynı klasöre koy.")
    print("Fotoğraflar silinmedi veya taşınmadı.")
    input("Kapatmak için Enter...")


if __name__ == "__main__":
    main()
