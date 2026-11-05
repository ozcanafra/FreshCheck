# Veri Temizleme

Taze/çürük meyve sınıflandırması için veri setlerini hazırlayan script'ler.
Hiçbir script kaynak klasörü değiştirmez; sonuçlar yeni klasöre kopyalanır,
çıkarılan her dosyanın nedeni rapor CSV'lerine yazılır.

## 1. CBIR veri seti (test seti)

Kaynak: Hugging Face `Project-AgML/fresh_rotten_fruit_classification` (3200 görsel)

| Sıra | Script | Ne yapar |
|---|---|---|
| 1 | `fresh_start.py` | Veri setini indirir, görselleri ve `metadata.csv`'yi kaydeder |
| 2 | `check_labels.py` | Meyve ve etiket dağılımını gösterir |
| 3 | `select_fruits.py` | Elma, muz, portakal ve çileği ayırır (1600 görsel) |
| 4 | `rename_fruits.py` | `label_0/label_1` klasörlerini `fresh/rotten` yapar |
| 5 | `check_duplicates.py` | SHA-256 ve pHash ile tekrar raporu çıkarır (silmez) |
| 6 | `create_clean_dataset.py` | Birebir aynı dosyaları atlayarak `CBIR_Clean` oluşturur |
| 7 | `filter_raw_dataset.py` | Yığın, düşük çözünürlük ve tekrar kareleri eleyip `CBIR_Raw_Clean` oluşturur |

`download_fruits.py`, 1. adımın ilk ve daha basit sürümüdür.

## 2. Unified veri seti (eğitim seti)

Kaynak: iki parçalı Unified_Dataset zip'i (9 meyve, fresh/rotten)

| Sıra | Script | Ne yapar |
|---|---|---|
| 1 | `filter_unified_raw.py` | Augmente görüntüleri ve tekrar kareleri çıkarır → `Unified_Full_Raw` |
| 2 | `filter_unified_training.py` | Elle belirlenen sorunlu fotoğrafları (`manual_exclude.csv`), düşük çözünürlüklüleri ve neredeyse aynı kareleri çıkarır → `Unified_Full_Clean` |
| 3 | `remove_cbir_overlap.py` | CBIR test setinde de olan fotoğrafları rapor klasörüne taşır (train/test sızıntısını önler) |

Sonuç: 10.201 → 7193 fotoğraf.

`manual_exclude.csv`: görsel kontrolle belirlenen 454 sorunlu fotoğraf ve
nedenleri (filigran, yığın, karışık taze/çürük, yanlış meyve, yanlış etiket vb.).

## Gereksinimler

    pip install pillow numpy imagehash pandas datasets

Klasör yolları (`D:/...`) script'lerin başında sabit olarak tanımlı,
kendi bilgisayarınıza göre değiştirin.
