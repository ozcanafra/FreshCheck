# Meyve Veri Seti Temizleme Çalışmaları

## Projenin amacı

Bu çalışmada farklı kaynaklardan toplanan meyve görüntülerini model
eğitiminden önce incelemek ve düzenlemek amaçlandı. Özellikle:

-   Birebir aynı dosyaları bulmak,
-   Görsel olarak birbirine benzeyen görüntüleri ve olası kopyaları
    incelemek,
-   Dosya adlarından anlaşılabilen artırılmış (augmented) görüntüleri
    belirlemek,
-   Orijinal görüntüleri mümkün olduğunca korumak,
-   İnceleme sonuçlarını raporlamak ve riskli işlemlerde doğrudan silmek
    yerine karantina yaklaşımını kullanmak.

> Not: Aşağıdaki sıra, çalışma sürecinde izlenen sırayı açıklamak
> içindir. GitHub'a eklenen bir dosyanın bulunması, o dosyanın mutlaka
> çalıştırıldığı anlamına gelmez. Her aşamanın çalıştırıldığını
> doğrulamak için ilgili raporlar ve komut geçmişi gerekir.

## İşlem sırası ve kullanılan yöntemler

### 1. `tekrarli_fotolar.py` --- SHA-256 ile birebir kopya kontrolü

Veri klasöründeki görüntü dosyalarının SHA-256 özetleri hesaplandı.
Özeti aynı olan dosyalar aynı grupta toplandı ve `tekrar_raporu.txt`
raporuna yazıldı.

**Amaç:** İçerik olarak birebir aynı dosyaları bulmak.

**Sınırlama:** Dosyanın boyutu, sıkıştırması veya görüntü içeriği
değiştiğinde SHA-256 değeri de değişir. Bu nedenle yeniden
boyutlandırılmış, döndürülmüş ya da kırpılmış benzer görüntüler bu
yöntemle bulunmayabilir.

### 2. `aynilari_sil.py` --- SHA-256 raporundaki kopyaları temizleme denemesi

Önceki aşamada oluşturulan `tekrar_raporu.txt` dosyasını okuyup her
grupta ilk dosyayı korumayı, diğerlerini silinecek adaylar olarak
belirlemeyi amaçlar.

**Dikkat:** Bu kod kalıcı silme riski taşıyan eski bir yaklaşımdır.
Çalıştırmadan önce yedek alınmalı; nihai iş akışında mümkünse
karantina/taşıma yöntemi tercih edilmelidir.

### 3. `name_split_augmented_v4.py` --- Dosya ve klasör adından augmented tespiti

Dosya veya klasör adlarında `rotated`, `flip`, `crop`, `translation`,
`noise`, `augmented` gibi anahtar kelimeler arar. Çalıştırma
seçeneklerine bağlı olarak rapor oluşturabilir, görüntüleri
kopyalayabilir veya taşıyabilir. `--apply` verilmedikçe varsayılan
davranış dry-run'dır.

**Amaç:** Adından artırılmış görüntü olduğu anlaşılabilen dosyaları
belirlemek.

**Sınırlama:** Dosya adı açıklayıcı değilse augmented görüntü gözden
kaçabilir. Bu yöntem görüntünün piksellerini analiz etmez.

### 4. `organize_fruit_dataset_hashonly.py` --- pHash ile görsel benzerlik ve organizasyon

Görüntüler için algısal hash (pHash) üretir ve benzerlikleri sınıf
klasörleri içinde karşılaştırır. Benzer görüntü kümelerinde çözünürlüğü
daha yüksek olan görüntüyü RAW, diğerlerini AUGMENTED olarak
sınıflandırmayı hedefler. Benzeri olmayan görüntüler RAW olarak kalır;
sınıfı belirlenemeyen görüntüler inceleme klasörüne yönlendirilebilir.

Kodun açıklamasına göre varsayılan çalışma biçimi dry-run'dır; `--apply`
kopyalama, `--apply --move` ise taşıma işlemi yapar. Bu çalışma
sırasında 419 görüntünün taşındığı daha önce kaydedilmişti. İşlem
sonrasında `kaggle_cache.npz` dosyası silinirken `PermissionError`
hatası alınmıştı; bu hata önbellek dosyasıyla ilgiliydi.

**Sınırlama:** pHash bazı döndürme ve aynalama türlerini yakalayabilir;
ancak küçük açılı döndürme, kırpma veya yakınlaştırma gibi
değişiklikleri her zaman yakalayamaz. En yüksek çözünürlüklü görüntünün
gerçek orijinal olduğu da kesin değildir; sonuçlar kontrol edilmelidir.

### 5. `benzer_goruntu_tara_yol_sor.py` --- MobileNet ile benzer görüntü taraması

MobileNetV3 Small modelinden görüntü özellikleri çıkarır ve özellik
vektörleri arasındaki kosinüs benzerliğini hesaplar. Kodda benzerlik
eşiği `0.97` olarak tanımlanmıştır. Benzer çiftleri CSV raporuna yazmayı
ve train/valid/test bölümleri arasındaki eşleşmeleri incelemeyi amaçlar.

**Amaç:** Dosya hash'lerinin yakalayamadığı, görsel içeriği benzer
görüntüleri aday olarak bulmak.

**Not:** Benzerlik, iki görüntünün kesin kopya olduğunu kanıtlamaz. Eşik
ve sonuçlar elle değerlendirilmelidir. Bu kodun çalıştırılıp
çalıştırılmadığı mevcut dosyadan tek başına doğrulanamaz.

### 6. `orb_kopya_kontrol.py` --- ORB ile aday çiftleri inceleme

Aday görüntü çiftlerini gri tonlamalı görüntüler ve ORB özellik
eşleşmeleri üzerinden karşılaştırır. Sonuçları CSV raporuna ve insan
incelemesi için HTML raporuna yazabilir; seçimler JSON olarak
kaydedilebilir.

**Amaç:** Benzerlik taramasında bulunan çiftleri daha yakından incelemek
ve yanlış eşleşmeleri azaltmak.

**Sınırlama:** ORB eşleşmeleri de tek başına kesin kopya kanıtı
değildir. Benzer çekim açısı, arka plan veya ortak nesneler sonuçları
etkileyebilir.

### 7. `guvenli_sil.py` --- Onaylanan adaylar için karantina planı

ORB CSV raporunu ve HTML incelemesinden kaydedilen seçimleri kullanarak
bir temizleme planı oluşturur. Kodun açıklamasına göre yalnızca HTML'de
kopya olarak işaretlenen çiftleri işler; önce
`kopya_temizleme_plani.csv` planını üretir ve onay verilirse adayları
`_KARANTINA_KOPYA_ADAYLARI` klasörüne taşır.

**Amaç:** Otomatik ve geri dönüşü zor silme yerine, incelenmiş adayları
kontrollü biçimde ayırmak.

**Önemli:** Planı ve korunacak görüntüleri kontrol etmeden taşıma
işlemini onaylamayın. Train/validation/test bölümlerindeki görüntüler ve
sınıf dağılımı işlem sonrasında tekrar kontrol edilmelidir.

## Klasörde bulunan diğer dosyalar

-   `orb_kopya_kontrol_kayitli_1.py`: `orb_kopya_kontrol.py` dosyasının
    alternatif/kayıtlı sürümüdür. İki sürümün de çalıştırılması gerekli
    değildir; kullanılacak tek sürüm belirlenmelidir.
-   `kopya_gorselleri_guvenli_temizle.py`: Mevcut haliyle bazı bağımlı
    fonksiyon ve değişkenleri (`Path`, `pixels_of`, `is_aug`,
    `split_of`, `SPLIT_PRIORITY`, `graph` gibi) tanımlamadığı için tek
    başına çalıştırılabilir görünmemektedir. Tamamlanmadan
    kullanılmamalıdır.
-   `organize_hashonly_preview/preview_01.png`--`preview_04.png`:
    Önizleme görselleridir; Python betiği değildir.

## Sonuç ve güvenlik notları

Bu çalışma, tek bir yöntemle bütün kopyaları bulmaya çalışmak yerine
farklı seviyelerde kontrol yaklaşımı izler:

1.  SHA-256 ile birebir dosya kopyaları,
2.  Dosya/klasör adlarından augmented adayları,
3.  pHash ile algısal benzerlik,
4.  MobileNet ile özellik tabanlı benzerlik adayları,
5.  ORB ve HTML raporu ile insan kontrolü,
6.  Onaylı adaylar için karantina planı.

Bu yöntemler birbirini tamamlayabilir; ancak otomatik olarak bulunan her
benzer çift kopya değildir. Nihai veri setini kullanmadan önce sınıf
etiketleri, bozuk dosyalar, sınıf başına görüntü sayıları ve veri
bölümleri arasındaki olası kopyalar yeniden kontrol edilmelidir.

### Tekrar üretilebilirlik

Kodları başka bir bilgisayarda çalıştırmadan önce betiklerdeki yerel
klasör yollarını kontrol edin ve gerekli Python paketlerini kurun.
Özellikle `aynilari_sil.py` kalıcı silme riski taşır;
`organize_fruit_dataset_hashonly.py` ve `name_split_augmented_v4.py`
gibi betiklerde de kopyalama/taşıma seçeneklerini kullanmadan önce
yardım çıktısını (`--help`) inceleyin.
