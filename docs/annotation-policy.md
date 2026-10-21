# Veri revizyonu için etiketleme sözleşmesi — taslak v1

Bu sözleşme yeni veri sürümüne uygulanacak. İndirilen Construction-PPE etiketleri
ve eski benchmark sonuçları tarihsel kayıt olarak korunur. Mevcut veri bu
sözleşmeye göre tamamen yeniden etiketlenmiş değildir.

## Görüntüdeki nesne ile iş güvenliği kararının ayrılması

- `helmet`: görünür endüstriyel koruyucu baretin sıkı kutusu. Normal şapka ve saç
  bu sınıfa dahil edilmez. Baretin kullanımda olması kişi eşleştirmesiyle ayrıca kontrol edilir.
- `no_helmet`: koruyucu baret bulunmadığı açıkça görülen başın görünür tamamının
  sıkı kutusu; alın/saçtan çeneye kadar. Sadece göz, alın, boyun veya göğüs kutusu
  kullanılmaz. Başın üstü kapalı/kadraj dışındaysa veya ekipman türü ayırt edilemiyorsa
  pozitif `no_helmet` etiketi üretilmez; belirsizlik inceleme kaydına alınır.
- `person`: görünür kişinin sıkı kutusu; kadraj dışındaki vücut tahmin edilmez.
- `vest`: görünür güvenlik yeleğinin kutusu. Yeleğin görünmemesi eksiklik etiketi değildir.
- Her görüntüde kapsam içindeki bütün görünür nesneler tutarlı biçimde etiketlenir.
  Kısmen etiketlenmiş sahneler negatif örnek olarak eğitime alınmaz.
- Baretsiz baş bir nesne gözlemidir. İhlal kararı için baret gerektiren alanı operatör
  tanımlar; tören/spor/gündelik yaşam görüntüsünden ihlal zorunluluğu çıkarılmaz.

## İnceleme ve sürümleme

1. Her değişiklikte kaynak görüntü SHA256, eski etiket SHA256, satır numarası,
   eski/yeni kutu, gerekçe ve inceleyen bilgisi kaydedilir.
2. Yeni görüntülerde kaynak, lisans ve kullanım izni bilgisi dosyayla birlikte tutulur.
3. Tam çözünürlükte baş/kutu yerleşimi kontrol edilir. Belirsiz örnekler otomatik
   kutu büyütmeyle düzeltilmez; yeniden inceleme kuyruğuna alınır.
4. Sadece hatalı kutu silip görünür başı etiketsiz bırakmak çözüm değildir. Etiket
   güvenilir biçimde tamamlanamıyorsa görüntü yeni eğitim sürümünden çıkarılır.
5. Yeni sınıf sözleşmesi kullanıldığında eski ve yeni veri aynı etiket anlamına
   sahipmiş gibi körlemesine birleştirilmez.

## Veri ayrımı ve değerlendirme

- Aynı video, çekim serisi veya sahne bir `scene_id` grubunda tutulur ve yalnızca
  bir split'e atanır. Dosya adına göre rastgele bölme yeterli değildir.
- Perceptual hash adayları görsel olarak incelenir. Eşik altında eşleşmemesi,
  sahnelerin bağımsız olduğunu kanıtlamaz. Kaynak/çekim bilgisi de kullanılır.
- Eğitimde ve validation'da hedef sanayi/şantiye bağlamı, farklı görüş açıları,
  uzaklıklar, örtüşmeler, birlikte baretli ve baretsiz kişiler bulunmalıdır.
- Etiket veya split değişirse yeni veri sürümü ve manifest hash'i oluşturulur.
  Eski ve aday model aynı yeni validation protokolünde yeniden ölçülür; önceki
  0,44139 AP50 ile farklı validation kümesinin skoru doğrudan karşılaştırılmaz.
- Mevcut model eski train sahnelerini gördüğünden yalnızca dosyaları yeniden
  bölmek modelin gördüğü sahneleri unutturmaz. Yeni bağımsız sahneler veya yeni
  ayrımla sıfırdan başlatılan kontrollü baseline gerekir.
- Held-out test, model seçimi ve etiket ayarı için kullanılmaz. Yeni veri edinilirse
  kaynak grupları üzerinden ayrılan yeni test kümesi de seçime kadar kapalı tutulur.

## Yeni eğitime geçiş koşulları

- [ ] Yeni sürümde kullanılacak `no_helmet` etiketlerinin tamamı sözleşmeye göre incelendi.
- [ ] Hedef saha ve karma baret durumlarını içeren örneklerin kapsaması kaydedildi.
- [ ] Train/validation sahne grupları ayrıldı; doğrulanmış yakın sahne çakışması kalmadı.
- [ ] Kaynak/lisans, görüntü/etiket hash'leri ve değişiklik manifesti kaydedildi.
- [ ] Biçim/geometri, sınıf eşleşmesi ve split kontrolleri geçti.
- [ ] Eski ve yeni modelleri aynı veri/protokolle karşılaştıracak plan sabitlendi.

Bu koşullar tamamlanmadan yeni eğitim bir veri iyileştirme sonucu olarak sunulmaz.
