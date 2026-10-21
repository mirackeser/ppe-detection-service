# Veri kapsamını güçlendirme — v2

21 Ekim 2026. **İncelenmiş veri 8 görüntüden 27 görüntüye genişletildi.** Bu adım
veri kapsamını ve dengesini geliştirdi; model doğruluğunda iyileşme iddiası taşımaz.

## Seçim ve inceleme

Önceki 48 adayın dışında 72 yeni aday, model tahminleri kullanılmadan seçildi:
36 baretli, 24 karma ve 12 baretsiz sahne. Seed=20261021, kaynak manifesti ve
önceki incelemenin hash'leri [seçim kaydında](expansion-v2-candidates.json).
`scripts.prepare_expansion` aynı aday listesini tekrar üretiyor; sıra kararlılığı
ve önceki örnekleri dışlama testlerle kontrol edildi.

72 görüntünün temas sayfaları incelendi; 24'ü ham görüntü ve tüm hedef nesne
kutularıyla ayrıntılı değerlendirildi. 19 yeni görüntü kabul edildi. Eksik kişi
etiketi, başın üstünü kapatan pano, yüz/başla karışan baret kutusu ve belirsiz
çalışma bağlamı gibi örnekler dışarıda kaldı. Diğer adaylar tam etiket incelemesi
tamamlanmadığı için kabul edilmedi. Toplam 120 adayın 27'si kabul, 93'ü sürüm dışı.

Kaynak görüntü ve XML'ler değişmedi. İncelenen kırpmalar türetilmiş yeni sürüme
yazıldı; kutular sınır kontrolünden geçti ve 19 yeni çıktı ayrıca görsel olarak
incelendi. İncelemeyi Codex yaptı; bağımsız insan etiket değerlendirmesi yapılmadı.

## Yeni dağılım

- Train: 22 görüntü, 18 sahne grubu; 37 baret, 38 baretsiz baş ve 74 kişi.
- Validation: 5 görüntü, 5 sahne grubu; 9 baret, 9 baretsiz baş ve 18 kişi.
- Önceki pilotun train dağılımı 2 baret / 25 baretsiz baştı.
- Yeni kapsam: döküm alanı, yeraltı maden çalışması, elektrik bakım/yüksekte çalışma,
  karlı dış ortam, ahşap yapı, şantiye asansörü, sondaj ve mobil vinç çevresi.
- Bir görüntüde yerde duran baret de var. `helmet` nesne etiketi, tek başına bir
  kişinin baret taktığı anlamına gelmez; bu örnek inceleme notunda ayrıca kayıtlı.

Maden ve elektrik bakım görüntüleri, kaynak sahne kimliği bilinmediğinden aile
grupları halinde aynı split'te tutuldu. Böylece aynı veya benzer çekim olma
olasılığına karşı muhafazakâr ayrım yapıldı; aynı fiziksel sahne oldukları iddia edilmez.

İlk pilotun 20 grup sınırını aşmak için `reviewed-group-v3` kuralı sabitlendi.
Sahne bütünlüğünü koruyan, sınıf ve görüntü oranlarını yaklaşık %20 validation
hedefine yaklaştıran sınırlı bir arama yapıyor. Girdi sırası değiştiğinde aynı
atama üretiliyor. İnference sonucu veya doğruluk skoru bu kararda kullanılmıyor.
V1 ayrımı ve dosyaları korunuyor; v2 skorları v1/Construction-PPE skorlarıyla
doğrudan karşılaştırılmayacak.

## Adım sonu kontrolleri

- **80 otomatik test geçti**; büyütülmüş veri için kararlı/dengeli grup ayrımı,
  aynı ailenin tek split'te kalması ve bilinmeyen politika reddi kapsanıyor.
- Bağımsız sürüm doğrulaması geçti. Dosya hash'leri, geometri, inceleme kayıtları,
  sınıf sayıları, grup ayrımı ve kırpma sonrası kopya/benzerlik adayları kontrol edildi.
- Gerçek YOLO11n, CPU/320 px/batch=4 ile 1 epoch tamamladı; 22 train / 5 validation
  görüntüsü bozuk örnek olmadan okundu. Bu yalnız veri/eğitim hattı denemesidir.
- Framework cache'leri çalışma kopyasında kaldı; kaynak sürüm eğitimden sonra
  tekrar doğrulandı. Kabul edilen 27 kaynağın görüntü/XML hash'leri korundu.
- Veri ZIP'i başka klasöre açılıp doğrulandı; atıf dosyası mevcut.
- Üç sınıflı gerçek deneme modeli eksik `vest` nedeniyle PPE servisinden reddedildi.
  Aktif PyTorch ve ONNX modellerinin hash'leri değişmedi.
- Mevcut Starlette/httpx deprecation uyarısı testleri etkilemiyor.

Kanıtlar: [inceleme](shel5k-expansion-v2-review.json),
[sürüm manifesti](helmet-pilot-v2-manifest.json),
[kapsam sayımları](data-expansion-v2-coverage.json),
[doğrulama](helmet-pilot-v2-verification.json),
[entegrasyon](data-revision-v2-integration.json),
[test çıktısı](data-expansion-v2-tests.xml).
Eğitim ayarı/ortamı ve ham CSV, `helmet-pilot-v2-smoke-*` dosyalarında saklanıyor.

## Açık kalan koşullar

Veri hâlâ küçük: validation'da sınıf başına yalnız 9 baş/baret kutusu var.
Karanlık ve kar örnekleri yalnız train'de; farklı ışık koşullarında bağımsız
değerlendirme kapsamı tamamlanmış değil. Ziyaretçi ağırlığı azaltılmalı ve daha
fazla bağımsız çalışan/kamera sahnesi incelenmeli. Sert negatif/background görüntüleri
ile yelek gözetimi henüz yok. Video zamanı/kimlik doğrulaması bu veriyle ölçülmez.

Bu nedenle `substantive_training_ready=false`; uzun eğitim engeli korunuyor.
İlk veri kapsamı adımı sürüyor. Tam PPE kapsamı, uzun eğitim, gerçek video, VLM
ve yeni performans deneylerine geçilmedi. İyileşme ölçümü ancak aynı veri/protokol
üzerinde kontrollü model karşılaştırmasıyla yapılacak.

```bash
.venv312/bin/python -m scripts.verify_release datasets/helmet-pilot-v2 --output reports/helmet-pilot-v2-verification.json
YOLO_CONFIG_DIR=/tmp/ppe-yolo MPLCONFIGDIR=/tmp/ppe-mpl .venv312/bin/python -m scripts.check_data_revision --release datasets/helmet-pilot-v2 --package output/helmet-pilot-v2.zip --run runs/train/helmet-pilot-v2-smoke --output reports/data-revision-v2-integration.json
.venv312/bin/python -m pytest -q
```

Colab için veri paketi `output/helmet-pilot-v2.zip`. Kaynak paketini
`python -m scripts.colab_bundle` ile güncelleyin. Colab üzerinde çalıştırma yapılmadı.
