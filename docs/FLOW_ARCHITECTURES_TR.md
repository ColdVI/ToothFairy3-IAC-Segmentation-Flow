# TF3 mimarileri — basit anlatım
22 Eylül 2026. İncelenen kod sürümleri ve deney kayıtları esas alınmıştır.
GT gerçek etiket, condition yardımcı bilgi, state ise adımlar boyunca değişen durumdur.

## Hoca: yardımcı SDF ve latent bridge ile eğitilmiş nnU-Net
CBCT nnU-Net'e giriyor. Ağ hem background/kanal segmentasyonu hem yardımcı bir SDF tahmin ediyor. Tahmin SDF'si ve eğitimde GT'den hesaplanan SDF aynı encoder'dan ayrı ayrı geçirilip 16 kanallı özellik hacimlerine çevriliyor. Uzamsal boyut küçülmüyor; bu latent sıkıştırılmış bir AE kodu değil.

Bridge, bu kodlardan oluşturulan ara durumu, zamanı ve nnU-Net decoder'ının görüntü özelliklerini alıp düzeltme üretiyor. Son kod SDF'ye decode edilerek yardımcı kayıplarda kullanılıyor. Paylaşılan kodda t=1 olduğundan eğitim girdisi GT kodu + gürültü oluyor. GT kodu condition değil, eğitimdeki ara durumun parçası; condition görüntü özellikleri.

Normal forward yalnız ana nnU-Net logitlerini döndürüyor. SDF skalerini iki sınıfa eşit ekleyen füzyon argmax'ı değiştirmiyor. Dolayısıyla tamamlanan teacher koşusunda ölçülen maske, flow'dan decode edilen maske değil; yardımcı kayıplarla eğitilmiş nnU-Net'in çıktısı. Bu sürüm binary. Bizim nnU-Net zaten background/sol/sağ üç sınıflı.

Kaynak: paylaşılan create_dataset504_iac_lr.py; NetworkWithSDFAux.forward, forward_with_sdf_and_bridge ve _fuse_logits_with_sdf.

## Bizim başlangıç nnU-Net'imiz
CBCT → background/sol/sağ sınıf skorları → argmax ile L/R maske. Flow yok. Sonraki mimariler bu modelin maskesini, olasılıklarını veya öğrenilmiş görüntü özelliklerini kullanıyor.

## İlk dense-SDF / Prompt3R — flow/
nnU-Net donuk. Sol/sağ tahminleri iki SDF hacmine çevriliyor. Ayrı bir 3D U-Net güncel SDF, zaman ve CBCT/prior bilgilerine bakarak değişim hızı üretiyor. Eğitimde GT SDF hedefi belirliyor; çıkarımda prior SDF adım adım güncelleniyor. Son SDF'nin sıfır sınırından maske üretilip ölçülüyor.

İlk condition CBCT, L/R olasılıkları, kaba SDF ve koordinatları içeriyordu. Prompt3R, koşullama ve t=0 denetimi gibi düzeltmelerin ailesi. Ayrı bir SDF autoencoder'ı yok. İki epoch'luk eski prototipin Dice 0,8156 ve tahmin/GT hacmi 1,452 sonucu, o koşudaki kalınlaşma eğilimini gösterir; bütün SDF mimarilerine genellenmez.

## CanalManifoldFlow: Direct / Linear / Staged / Corrected
nnU-Net maskesinden merkez çizgisi ve kesitler çıkarılıyor. Kanal, merkez kayması, kalınlık, kesit biçimi ve uç konumu gibi sayılarla temsil ediliyor. Ağ görüntü/olasılık profillerine bakarak bu sayıları GT'den çıkarılan hedef şekle yaklaştırıyor. Son parametrelerden tüp maskesi çiziliyor veya değişim özgün SDF'ye uygulanıyor.

Direct tek çağrıda düzeltir. Linear parametreleri birlikte taşır. Staged önce konum/kalınlık/uçları, sonra daha ince biçimi değiştirir. Corrected sürüm koordinat, koşullama ve hedef hatalarını düzeltir. Bu temsil öğrenilmiş bir SDF-AE latent'i değildir.

İlk temsil taraf başına 1443 sayıydı. V2'de daha ayrıntılı harmonik tüp (R1), daha serbest sınır yer değiştirmesi (R2) ve dense-SDF karşılaştırıcısı (R3) kodları da var. Her kod kolu için tamamlanmış eğitim sonucu varsayılmamalı.

Corrected staged aynı 97 vakada Dice 0,905050 → 0,905744 sağladı; clDice ve parça sayısı kötüleşti. Bu küçük Dice artışı genel üstünlük değildir.

## GeoFlow-Newton
Başlangıç donmuş nnU-Net tahmini. State, kesitlerde sınırın kaç mm hareket edeceği. Sınır hareket ettikçe yeni konumunun çevresindeki nnU-Net olasılığı yeniden örnekleniyor. Ağ bu kanıtı ve geometriyi kullanıp yeni sınır hareketini belirliyor. Son hareket özgün SDF'ye uygulanıyor; maskeden metrikler hesaplanıyor.

Ayrı SDF-AE yok. Aynı 97-vaka kohortunda raw nnU-Net Dice 0,905050, GeoFlow'un normal çıktısı 0,903412. Kaynak: experiments/canalmanifold_geoflow.

## Gaussian → SDF — iacflow/
Hazır üç sınıflı nnU-Net ağırlıkları yükleniyor. CBCT encoder'dan görüntü özelliklerine dönüşüyor. İki kanallı Gaussian gürültüden başlanıyor. Güncel state ve zaman adapter'larla görüntü özelliklerine ekleniyor; decoder temiz sol/sağ SDF'yi tahmin ediyor. Bundan hesaplanan hızla state ilerletiliyor. Final entegre SDF maskeye çevrilip ölçülüyor.

Başlangıç hazır maske değil; nnU-Net burada ağırlık başlangıcı ve görüntü koşullaması sağlıyor. Son encoder bloğu ve decoder eğitiliyor. GT yalnız eğitim hedefi/ara durum üretiminde; test girdisi değil. Ayrı SDF-AE yok.

15.000 update sonundaki beş probe vakasında baseline Dice 0,885791; dört çağrılı flow 0,875615. clDice 0,984281 → 0,991324. Bu beş vakalık bir tanılama, tam kohort başarı sonucu değil.

## Yeni IAC-B stokastik köprü — iacb
Donmuş üç sınıflı nnU-Net'ten L/R tahmin alınıp iki SDF'ye çevriliyor. Eğitimde prior SDF ile GT SDF arasında gürültülü ara durumlar kuruluyor. Ayrı StateUNet, CBCT + güncel SDF + zamandan temiz hedef SDF'yi tahmin ediyor. Çıkarım prior SDF'den başlıyor; birkaç bridge adımıyla bir veya çok sayıda aday maske üretiliyor. TC-MBR kolu, adayların örtüşme ve bağlantı bilgisine göre seçim/düzeltme yapıyor.

GT condition değildir. Ayrı sabit prior kanalları tekrar condition'a eklenmez; prior başlangıç state'idir. Model doğrudan endpoint SDF tahmin eder; yalnız velocity-matching ağı diye anlatılmamalı. Varsayılan çalışan kol voxel SDF uzayında. ae_gate.py ayrı bir AE yeniden-üretim deneyidir; önerilen latent bridge uygulanmış varsayılan kol değildir.

Case-balanced v3 aynı mimarinin eğitim/örnekleme düzenidir: her vaka epoch'ta bir kez, iki patch ile görülür. Kayıtlı split 403 train / 32 validation / 97 eval. 45 epoch ve 18.135 update var; en iyi patch-val loss epoch 27. Bu loss Dice değildir. O dizinde full-volume sonuç bulunmadı. Eski fold_0_eval içindeki sıfır-Dice raporunu bu yeni checkpoint'e atamayız.

52 S vakası bu refiner'ın eğitiminde kullanılıyor; bu model için artık untouched external test değildir.

## Segmentasyon + bağlantı flow: tasarım
nnU-Net'in üç sınıf skorlarından ve komşu bölgelerin bağlantı skorlarından başlanması; görüntü özellikleriyle ikisinin birlikte değiştirilmesi önerildi. Bir adımda değişen bağlantı, sonraki adımın sınıf kararını etkileyecekti. Son sınıf skorlarının argmax'ı maske olacaktı. SDF/AE yok. Bu tasarımın uygulaması ve eğitim sonucu bu incelemede doğrulanmadı.

## En kısa ayrım
Hoca yardımcı SDF koluyla ana nnU-Net'i eğitiyor. Dense-SDF ve IAC-B hazır maskeyi voxel düzeyinde düzeltiyor. CMF/GeoFlow hazır maskenin geometrisini düzeltiyor. Gaussian→SDF görüntü bilgisini kullanarak gürültüden maske üretiyor.
