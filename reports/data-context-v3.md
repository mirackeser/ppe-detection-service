# Veri bağlamı ve olumsuz örnek denetimi — v3

21 Ekim 2026. Bu aşama veri hattını doğrular; model başarısının veya saha kullanıma hazır olmanın kanıtı değildir.

## Sonuç

`datasets/helmet-pilot-v3` 32 kaynak fotoğraftan üretilen 43 örnek ve 27 sahne grubu içerir. Bunların 32'si pozitif görüntü, 11'i aynı kaynakların insan ve hedef ekipman içermediği incelenmiş arka plan kırpımlarıdır. Kırpımlar bağımsız yeni sahne sayılmaz. Kaynak fotoğraflar, önceki sürümler ve servisteki model ağırlıkları korunmuştur.

- Train: 35 örnek, 26 kaynak fotoğraf, 22 sahne grubu; 46 baret, 38 baretsiz baş, 83 kişi. Dokuz arka plan kontrolü vardır.
- Validation: 8 örnek, 6 kaynak fotoğraf, 5 sahne grubu; 11 baret, 9 baretsiz baş, 20 kişi. İki arka plan kontrolü vardır.
- Her iki bölümde en az iki normal ışık ve iki düşük ışık sahne grubu vardır. Aynı kaynağın tüm kırpımları aynı bölümde kalır.

Görsel inceleme ajan tarafından yapılmıştır; bağımsız insan anotasyon onayı yerine geçmez. Veri yalnız `helmet`, `no_helmet`, `person` içerir; tam PPE servisinin gerektirdiği `vest` bulunmaz.

## Eklenen korumalar

Boş YOLO etiketi tek başına olumsuz örnek kabul edilmez. Açık inceleme kaydı, hedef sınıfların yokluğunun onayı ve kaynak XML'deki yok sayılan yüz/baş kutuları dahil hiçbir nesne bırakmayan kırpım gerekir. Hedef baş veya baret kutusunu kesen kırpım reddedilir. Boş etiketler sıfır bayt yazılır.

Sürüm doğrulayıcı; kaynak ailesi ayrımını, etiket geometrisini, dosya hash'lerini, görüntü tekrarlarını, rol ve ışık kapsamını bağımsız olarak kontrol eder. Eğitim, değişmez veri sürümünün çalışma kopyasında yürütülür; Ultralytics önbellekleri sürüme yazılmaz.

## Gerçek modelde bulunan hata

Mevcut `models/best.pt`, CPU/640 ve 0,25 güven eşiğinde sabit tutuldu. Train'deki dokuz arka plan kontrolünde hedef tespiti yoktu. Validation'daki **iki kontrolden birinde** fırın/alev alanı 0,631 güvenle kişi olarak tespit edildi (`hard_hat_workers107-bg`). Görsel: `output/data-review/context-v3/false-detection.jpg`.

Bu küçük ve türetilmiş kontrol kümesi genel yanlış pozitif oranı tahminine uygun değildir. Pozitif 32 görüntünün tümünde çıktı alınması yalnız boş çıktı üretmeyen model kontrolüdür; recall veya doğruluk değildir. Tek kare politika kontrolünde eksik baret adayı oluşmaması zamansal yanlış alarm oranını kanıtlamaz. Eski modelin eğitim kaynağı ile tarihsel içerik örtüşmesi dışlanmamıştır.

Sekiz validation örneği `reports/validation-cohort-v3.json` ile sabitlendi. Bu kayıt ilk ölçümden sonra oluşturulmuştur; ön kayıt veya bağımsız son test değildir. Aynı grup üzerinde yapılacak sonraki karşılaştırmalarda kaynak, kırpım, etiket, rol ve sahne değişiklikleri reddedilir. Ek validation verisi açıkça yeni bir grup sürümü gerektirir; bulunan hatayı train'e taşıyarak sonuç iyileştirilmez.

## Doğrulama kanıtı

- 92 otomatik test geçti: `reports/data-context-v3-tests.xml`.
- Gerçek YOLO ile CPU/320, batch 4, bir epoch eğitim hattı denemesi tamamlandı. Boş etiketler ve yalnız arka plan içeren batch işlendi. Bu denemenin metrikleri model seçimi için kullanılmaz.
- Paket taşıma ve yeniden doğrulama, kaynak bütünlüğü, mevcut ağırlıkların korunması ve üç sınıflı pilot modelin PPE servisi tarafından reddi geçti: `reports/data-revision-v3-integration.json`.
- v1, v2 ve v3 sürümleri son kodla bütünlük kontrolünü geçti. Colab GPU çalıştırması yapılmadı.

Tekrarlama komutları, proje kökünde etkin Python ortamıyla:

```bash
python -m scripts.verify_release datasets/helmet-pilot-v3 --output reports/helmet-pilot-v3-verification.json
python -m scripts.evaluate_background datasets/helmet-pilot-v3 --cohort-lock reports/validation-cohort-v3.json --output reports/background-controls-v3-baseline.json
python -m scripts.data_readiness datasets/helmet-pilot-v3 --policy config/data-readiness-v1.json --output reports/data-readiness-v3.json --require-ready
```

Son komut, mevcut kapsam hedeflere ulaşmadığı için çıkış kodu 1 verir. Bu beklenen bir kalite kapısı sonucudur.

## Sıradaki gerekli adım

`config/data-readiness-v1.json`, ilk kontrollü eğitim için proje içi çalışma hedeflerini açıklar; yayınlanmış bir standart veya üretim sertifikası değildir. Dokuz gereksinim henüz karşılanmıyor: pozitif sahne grupları train 22/40, validation 5/10; validation olumsuz sahne grupları 2/10; üç hedef sınıfın örnek sayıları da iki bölümde hedefin altında. Train hedefleri baret/baretsiz baş/kişi için 100/100/200, validation hedefleri 30/30/60'tır.

Öncelik bağımsız çalışma sahnelerini, özellikle baretsiz baş örneklerini ve farklı kaynaklardan gelen validation arka plan kontrollerini artırmaktır. Yeni incelemeler model skorlarıyla etiket değiştirmemeli; mevcut sabit validation grubu korunmalıdır. Uzun eğitim kapısı kapalıdır; mevcut pilot en fazla üç epoch hat denemesine izin verir. Ardından kontrollü eğitim ve aynı değerlendirme protokolü gelir. Tam PPE/yelek kapsamı, gerçek video, VLM ve saha değerlendirmesi ayrıca tamamlanmalıdır. Mevcut held-out Construction-PPE test bölümü bu revizyonda açılmamıştır.
