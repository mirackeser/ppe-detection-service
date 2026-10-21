# Aşama 1a — Baret eksikliği veri ve hata incelemesi

Durum: **teşhis tamamlandı; veri yeni eğitime hazır değil.** Model iyileştirme
aşaması henüz tamamlanmadı. Önce etiket/sahne revizyonu gerekiyor.

## Kapsam ve yöntem

1.132 train ve 143 validation görüntüsü/etiketi programatik olarak incelendi.
Held-out test bu incelemede açılmadı. Görüntü ve etiket SHA256 değerleri
[tanı JSON'unda](no-helmet-diagnosis.json) kayıtlı.

Aktif baseline'ın 143 validation görüntüsündeki CPU/640/conf=0,25 tahminleri
yeniden üretildi. Cache; model, görüntü, etiket, çözünürlük, split ve tam görüntü
kapsamasıyla doğrulandı. Sonuç TP=15, FP=10, FN=30: önceki ölçüm aynen tekrarlandı.
Precision=0,60, recall=0,3333. Bunlar nesne kutusu ölçümleri, olay doğruluğu değil.

## Bulgular

- **Boyut tek başına açıklamıyor.** Kaçırılan 30 kutunun 26'sının 640 px model
  girişindeki kısa kenarı en az 32 px. Toplam 45 validation kutusunun 40'ı bu grupta.
  Kullandığımız boyut grupları kısa kenara dayanır; COCO alan metrikleri değildir.
- **Kutu yerleşimi sorunlu.** 10 FP'nin 8'i gerçek etiketle kısmen örtüşüyor fakat
  IoU<0,5; 2'si zaten eşleşmiş bir etikete ek kutu. Bu sonuç yalnızca geometrik
  tanı verir. Gerçek etikette hata varsa model hatasıyla karışır.
- **Somut yanlış etiket örnekleri var.** Train `image1128.jpg` satır 5 ve 6 ile
  validation `image1392.jpg` satır 8'de `no_helmet` kutuları baş yerine boyun/göğüste.
  `image1329.jpg` ve `image1137.jpg` farklı baş alt bölgeleriyle sınırlı kutular içeriyor.
  [Görsel inceleme kaydı](data-review.json) eski satırları ve dosya hash'lerini koruyor.
- **Hedef alan kapsamı zayıf.** `no_helmet` içeren 27 validation görüntüsünün tamamı
  görsel olarak incelendi; görünür şantiye/sanayi PPE çalışma alanı saptanmadı.
  Seed=42 ile seçilmiş 24 train örneği de gündelik, sportif ve resmî etkinlik bağlamlarında.
  Bu gözlem tüm 232 pozitif eğitim görüntüsünün eksiksiz incelemesi değildir.
- **Sınıf anlamı hedefle tam uyuşmuyor.** Train'deki 232 `no_helmet` pozitif
  görüntünün yalnız 2'sinde `helmet` de var; validation'da bu sayı 0.
  Bu iki eğitim görüntüsü ayrıca incelendi: `image1269.jpg` bisiklet kaskı,
  `image1289.jpg` spor kaskı içeriyor. Endüstriyel baret sınıfıyla doğrudan eş anlamlı değiller.
- **Train/validation sahneleri bağımsız değil.** 128-bit dHash/Hamming<=8 taramasında
  12 çift bulundu; tamamında aynı sahne/çekim serisi görsel olarak doğrulandı.
  Bunlar 9 ayrı validation görüntüsüne karşılık geliyor. Tarama tüm sahne
  sızıntılarını bulduğumuzu göstermez; 12 çift bir alt sınırdır.

İnceleme Codex tarafından yapıldı; bağımsız insan etiket doğrulaması değildir.
Kaynak verideki genel `helmet`/`no_helmet` anlamıyla projenin endüstriyel baret
hedefini ayırmak gerekiyor. Gündelik yaşamda baretsiz olmak, iş güvenliği ihlali değildir.

## Karar ve sıradaki alt adım

Sadece epoch, çözünürlük veya kayıp ağırlığı artırarak yeni bir başarı sonucu
üretmeye çalışmıyoruz. Önce [etiketleme sözleşmesine](../docs/annotation-policy.md)
göre yeni veri sürümünü oluşturacağız: hedef saha örnekleri, tutarlı baş/baret
kutuları ve sahne gruplarıyla bağımsız train/validation ayrımı.

Eski veri/etiketler ve aktif ağırlıklar değiştirilmedi. Eski skorlar tarihsel
benchmark olarak kalır; bağımsız saha başarısı olarak sunulmaz. Yeni validation
oluşturulursa eski ve aday model aynı protokolde değerlendirilir. Mevcut modelin
önceden gördüğü sahneleri başka split'e taşımak bağımsızlık sağlamaz.

## Doğrulama

- 45 otomatik test geçti; 13 yeni vaka boyut ölçeklemesi, hata sınıflama,
  cache değişikliği/eksikliği, yalnız train/val kapsamı ve benzerlik sınırlarını sınar.
- Gerçek modelle cache yeniden üretildi ve sabit eşik sonuçları tekrarlandı.
- Yeni tanı çalıştırması 1.275 görüntünün tamamında tamamlandı; 1 px'ten fazla
  görüntü sınırı taşan kutu yok. Bu biçim kontrolü semantik etiket doğruluğu değildir.
- 27 pozitif validation görüntüsü, 24 rastgele train örneği, 2 karma sınıf
  görüntüsü ve 12 yakın sahne çifti görsel olarak incelendi.
- Starlette/httpx tarafından gelen mevcut bir deprecation uyarısı var; test hatası yok.

Tekrar çalıştırma (proje kökünde):

```bash
YOLO_CONFIG_DIR=/tmp/ppe-yolo MPLCONFIGDIR=/tmp/ppe-mpl .venv312/bin/python -m scripts.error_analysis --weights models/best.pt
.venv312/bin/python -m scripts.diagnose_data
.venv312/bin/python -m pytest -q
```

Görsel kanıtlar `runs/data-review/` altında. `data-review.json` görsel inceleme
kararlarını, tanı dosyasının hash'ini ve yeni eğitime geçişteki başarısız kalite
koşullarını saklar. Kaynak veya tanı değişirse görsel kararların geçerliliği de
yeniden kontrol edilmelidir.
