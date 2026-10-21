# Veri revizyonu — pilot aşama kaydı

21 Ekim 2026. **Kaynak edinimi, sürümleme hattı ve küçük pilot doğrulandı.**
Asıl model iyileştirmesi için veri kapsamını genişletme aşaması açık kalıyor.

## Oluşan veri

[SHEL5K v4](https://data.mendeley.com/datasets/9rcv8mm682/4) arşivi indirildi;
DOI, CC BY 4.0 atfı, indirme adresi ve SHA256 [kaynak kaydında](shel5k-source.json).
5.000 görüntü ve XML eşleştirildi. 4.995 görüntü biçim/sınıf kontrolünden geçti;
tanımlı altı sınıf dışında `person` içeren 5 görüntü karantinaya alındı. Bu sonuç
4.995 görüntünün semantik olarak doğru etiketlendiği anlamına gelmez.

Seed=42 ile önceden seçilen 48 görüntü görsel olarak incelendi. Yansıtılmış kenarlar,
kolajlar, örtüşmeler, küçük nesneler ve eksik inceleme gereksinimleri nedeniyle
40'ı ilk pilot dışında bırakıldı. 8 görüntünün seçilen sahne bölgeleri ve hedef
nesneleri incelendi. Kararlar, kaynak hash'leri ve kırpma gerekçeleri
[inceleme dosyasında](shel5k-pilot-review.json). İncelemeyi Codex yaptı; bağımsız
insan etiket kalite değerlendirmesi yapılmadı.

`datasets/helmet-pilot-v1` sürümü:

- Train: 6 görüntü / 6 sahne; 2 baret, 25 baretsiz baş, 27 kişi.
- Validation: 2 görüntü / 2 sahne; 5 baret, 2 baretsiz baş, 7 kişi.
- Her iki bölümde karma baretli/baretsiz örnek bulunuyor.
- Yelek etiketi ve test split'i bulunmuyor. Eski Construction-PPE test kümesi açılmadı.
- Kaynak görüntü ve XML'ler değişmedi. Kırpılmış çıktıların ayrı hash'leri var.
- Kırpma baş/baret kutusunu kısmen kesemez; kişi kutusu görüntü sınırına kırpılabilir.

Sahne ayrımı, model skoru kullanılmadan sabit hash sırası ve sınıf kapsamı
koşullarıyla yapıldı. Kırpma öncesi ve sonrası birebir kopya/benzerlik kontrolleri
uygulandı; çözümlenmemiş train/validation benzerliği bulunmadı. Bu yöntem tüm
olası sahne ilişkilerinin bilindiğini garanti etmez.

## Doğrulananlar

- **77 otomatik test geçti.** XML/koordinat hataları, bilinmeyen sınıf, eski hash,
  eksik inceleme, kopya görüntü, belirsiz sahne ilişkisi, yanlış kırpma ve değişmiş
  veriyle eğitime başlama olumsuz senaryoları kapsanıyor.
- Gerçek YOLO11n, CPU/320 px/batch=2 ile 1 epoch tamamladı; 6 train ve 2 validation
  görüntüsünü 0 bozuk örnekle okudu. Bu bir eğitim hattı denemesidir; çıkan metrikler
  doğruluk veya eski modele üstünlük kanıtı değildir.
- Framework ilk denemede veri klasörüne cache yazdı. Eğitim kodu düzeltildi;
  ikinci deneme doğrulanmış çalışma kopyasıyla tamamlandı. İlk cache'ler deneme
  klasörüne taşındı. Sürümlü veri eğitimden sonra tekrar doğrulandı.
- 3 epoch üzeri pilot eğitimi başlamadan reddediliyor. Mekanik pilot doğrulaması
  `passed=true`; asıl eğitim hazırlığı `substantive_training_ready=false`.
- Veri ZIP'i geçici bir klasöre açılıp YAML kökü yeniden konumlandırılarak doğrulandı.
  Bu yerel testtir; Colab GPU çalıştırması değildir.
- Üç sınıflı gerçek deneme ağırlığı PPE adaptörüne verildi ve eksik `vest` sınıfı
  nedeniyle reddedildi. Aktif `models/best.pt` ve `.onnx` hash'leri korunuyor.
- Mevcut Starlette/httpx deprecation uyarısı testleri etkilemiyor.

Kanıtlar: [veri doğrulaması](helmet-pilot-v1-verification.json),
[gerçek entegrasyon kontrolü](data-revision-integration.json),
[test çıktısı](data-revision-tests.xml), [sürüm manifesti](helmet-pilot-v1-manifest.json).
Deneme ortamı/ayarları/ham CSV `helmet-pilot-v1-smoke-*` dosyalarında.

## Sıradaki zorunlu işler

- [ ] Daha fazla farklı şantiye/sanayi sahnesini aynı inceleme akışından geçirmek.
- [ ] Özellikle baretli örnekleri ve karma sahneleri artırmak; mevcut train dağılımı çok dengesiz.
- [ ] Ziyaretçi görüntülerinin ağırlığını azaltıp gerçek çalışan, uzaklık/açı ve örtüşme kapsamını kaydetmek.
- [ ] 20 sahneyi aşan veri için kaynak/sahne gruplarını koruyan yeni bölme planını dondurmak.
- [ ] Yelek gözetimini kaybetmeden baret modelinin tam PPE servisine nasıl katılacağını doğrulamak.
- [ ] Genişletilmiş sürümün kalite koşulları sağlandıktan sonra kontrollü uzun eğitim ve eş protokollü değerlendirme yapmak.

Bu işler kapanmadan model iyileştirme aşaması tamamlanmış sayılmaz; gerçek video,
VLM ve hız karşılaştırmasının sonraki geliştirme aşamalarına geçiş yapılmaz.

## Tekrarlanabilir komutlar

```bash
.venv312/bin/python -m scripts.verify_release datasets/helmet-pilot-v1 --output reports/helmet-pilot-v1-verification.json
YOLO_CONFIG_DIR=/tmp/ppe-yolo MPLCONFIGDIR=/tmp/ppe-mpl .venv312/bin/python -m scripts.check_data_revision
.venv312/bin/python -m pytest -q
```

Colab için `output/helmet-pilot-v1.zip` veri paketi ve
`output/ppe-colab-source.zip` kaynak paketi hazır. Notebook yalnız pilot denemesi
yapar; veri ve kaynak ayrı yüklenir. Güncel kullanım: [iş akışı](../docs/data-revision-workflow.md).
