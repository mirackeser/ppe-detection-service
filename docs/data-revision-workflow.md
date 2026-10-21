# İncelenmiş veri sürümü oluşturma

Bu aşamada baret eksikliği için **SHEL5K v4** kaynak adayı kullanılıyor.
[Resmî yayın](https://data.mendeley.com/datasets/9rcv8mm682/4) CC BY 4.0 olarak
yayımlanmış; kaynak DOI, atıf ve arşiv hash'i `reports/shel5k-source.json` içinde tutulur.
Kaynağın indirilmesi, bütün etiketlerin doğru olduğunun kabulü değildir.

## Sınıf sözleşmesi

- Kaynak `helmet` → hedef `helmet`.
- Kaynak `head` → hedef `no_helmet`; her kabul edilen görüntüde bu anlam görsel olarak doğrulanır.
- Kaynak `person_with_helmet` ve `person_no_helmet` → hedef `person`.
- `head_with_helmet` ve `face`, hedef sınıflar olmadığı için dışarıda bırakılır.
- Bilinmeyen sınıf içeren görüntü karantinaya alınır; sessiz sınıf eşleştirmesi yapılmaz.

Pilot üç sınıflıdır. Yelek etiketi yoktur; mevcut dört kavramlı PPE servisinin
ağırlıklarının yerine doğrudan yüklenemez. Yeleği etiketsiz negatif örnek haline
getirmemek için Construction-PPE ile körlemesine birleştirilmez.

## Kaynak → inceleme → sürüm → doğrulama

```bash
python -m scripts.prepare_shel5k
python -m scripts.shel5k_inventory
```

İlk komut arşivi indirir, yolları ve CRC bütünlüğünü kontrol eder, kaynak klasörünü
yeniden yazmayı reddeder. İkinci komut görüntü/XML eşleşmesini, boyutları, sınıfları,
koordinatları inceler; hash'ler ve 48 görüntülük seed=42 inceleme örneklemi oluşturur.
Bu örneklem, görsel inceleme öncesinde seçilir; skorla örnek seçilmez.

VOC koordinatları dahilî 1 tabanlı biçimden 0 tabanlı xyxy'ye çevrilir. Kaynakta
sıfır minimum koordinatı varsa sınır düzeltmesi ayrıca kaydedilir. XML'deki kaynak
`path` alanı dosya okumada kullanılmaz. Görüntü boyutuyla uyuşmayan XML karantinaya alınır.

`runs/shel5k-review/review-template.json` bir başlangıç şablonudur; `pending`
kararı kabul anlamına gelmez. İnceleme tamamlanınca ayrı bir dosyada saklanır:

- Her kabul edilen görüntü için etiket incelemesi, hedef alan, `scene_id`, gerekçe ve hash'ler gerekir.
- Aynı çekim/sahne tek `scene_id` alır. İsmi sonuçlara göre değiştirilerek validation seçilmez.
- Kutu düzeltmeleri kaynak nesne numarası, eski kutu, yeni kutu ve gerekçe ile kayıt altına alınır.
- Güvenilir etiketlenemeyen veya hedef dışı görüntü bütünüyle dışlanır.
- Kabul edilenler arasındaki benzerlik adayları çözülmeden dışa aktarma yapılmaz.

```bash
python -m scripts.curate_dataset --review reports/shel5k-pilot-review.json --output datasets/helmet-pilot-v1
python -m scripts.verify_release datasets/helmet-pilot-v1 --output reports/helmet-pilot-v1-verification.json
```

Sürümleme aracı kaynak dosyalarını değiştirmez. Var olan sürümün üzerine yazmaz;
geçici klasörde hazırlayıp başarılı olunca taşır. Aynı piksele sahip kopyalar,
incelemesiz örnekler ve çözülmemiş benzer sahneler reddedilir.

Train/validation ataması `pilot-group-v2` kuralıyla yapılır: sahneler sabit SHA256
sırasına dizilir, model skoru kullanılmadan kapsam koşullarını sağlayan ilk grup
kombinasyonu seçilir. Validation'a en az iki, aksi durumda yaklaşık %20 sahne
ayrılır. İlk denenen modulo kuralı 8 sahnenin tamamını train'e atadığı için sürüm
dondurulmadan bu kapsam kontrollü kuralla değiştirildi; görüntüler skorla seçilmedi.
Bu ilk pilot ayırıcısı 4–20 sahne içindir ve v1 sürümünde korunur.
Büyütülmüş v2 sürümü açıkça `reviewed-group-v3` kullanır: 20'den fazla grup için
seed=20261021 ile en fazla 4096 aday grup kombinasyonu incelenir; sınıf ve görüntü
oranlarının %20 validation hedefinden sapması en düşük olan geçerli kombinasyon
seçilir. Eşitlikler sabit grup sırasıyla çözülür. Model skorları girdi değildir.
Bu yöntem bir genelleme garantisi veya istatistiksel örneklem yeterliliği ölçüsü değildir.
Her iki bölümde en az iki
sahne, üç hedef sınıf ve karma baretli/baretsiz örnek gerekir. Bu alt sınırlar
yalnız pilotun çalışabilirliğini sınar; saha kapsamının yeterli olduğunu kanıtlamaz.

Bağımsız doğrulama dosya hash'lerini, YOLO geometrisini, kayıtlı incelemeyi,
sahne ayrımını, sınıf sayılarını ve kayıt dışı dosyaları kontrol eder. Kontrol sonucu
yalnız pilot eğitimi için geçerlidir; veri boyutu ve sahne çeşitliliği artırılmadan
nihai model başarısı olarak sunulmaz. Bu aşamada test split'i oluşturulmaz;
önceki Construction-PPE held-out test kümesi de açılmaz.

## Eğitimden önce

`scripts.train` doğrulayıcıyı yeniden çalıştırır; sonradan değiştirilmiş veriyle
eğitim başlamaz. Doğrulanan sürümün dosyaları `runs/train-inputs/` altında bir
çalışma kopyasına alınır; framework önbellekleri bu kopyaya yazılır. Sürümlü veri
klasörü eğitim sırasında değiştirilmez. Eğitim kaydı kaynak ve çalışma YAML yollarını
ve manifest doğrulamasını saklar. Pilot eğitimi, kaynağın tamamının gözden geçirildiği anlamına gelmez.
Model başarısı, VLM, gerçek video ve performans aşamalarına ilişkin kabul koşulları
ayrı kalır.
Mevcut küçük pilot en fazla 3 epoch eğitim hattı denemesine izin verir; doğrulama
sonucunda `substantive_training_ready=false` kalır. 3 epoch üzeri eğitim komutu
model yüklenmeden reddedilir. Bu sınır veri büyütme ve kapsam değerlendirmesi
tamamlanmadan uzun eğitime geçilmesini önler.

İncelenen kırpma sınırları ve gerekçeleri review dosyasında saklanır. Baş/baret
kutusunu kısmen kesen kırpma reddedilir; kişi kutuları görüntü sınırına kırpılır.
Çerçeve dışında kalan nesneler kaynak nesne numaralarıyla kaydedilir. Kaynak ve
türetilmiş görüntü hash'leri ayrı tutulur. Bağımsız doğrulayıcı kırpma sonrasında da
aynı piksel kopyalarını ve bölmeler arasındaki yakın benzerlikleri denetler.

Colab'a taşınırken yalnız `data.yaml` içindeki yerel `path` yeni klasöre uyarlanır;
görüntü, etiket, inceleme ve manifest içerikleri korunur ve tekrar doğrulanır.

```bash
python -m scripts.package_release datasets/helmet-pilot-v1 --output output/helmet-pilot-v1.zip
python -m scripts.colab_bundle
```

Veri paketi yalnız kayıtlı görüntü/etiketleri, manifest, inceleme ve kaynak atfını
içerir; çalışma önbellekleri veya model ağırlıkları pakete girmez.

## V2 veri genişletme

```bash
python -m scripts.prepare_expansion --output runs/expansion-new-batch
python -m scripts.curate_dataset --review reports/shel5k-expansion-v2-review.json --output datasets/helmet-pilot-v2-reproduced
python -m scripts.verify_release datasets/helmet-pilot-v2 --output reports/helmet-pilot-v2-verification.json
```

İlk komut önceki 48 adayın dışında, en fazla 6 baş/baret kutusu içeren sahnelerden
36 baretli, 24 karma ve 12 baretsiz aday seçer. Seçim sırası ve seed sabittir;
aday seçimi etiket kabulü değildir. Yeni bir klasör gerekir; önceki paketler üzerine yazılmaz.

V2, v1'in kaynak görüntü/etiketlerini değiştirmez. Sahne kimliği bilinmeyen maden
ve elektrik bakım görüntüleri, alt alan aileleri olarak muhafazakâr biçimde
bir arada tutulur. Bu aile grubu aynı fiziksel çekim yeri iddiası değildir.
Genişletme kaydı: [veri raporu](../reports/data-expansion-v2.md).
