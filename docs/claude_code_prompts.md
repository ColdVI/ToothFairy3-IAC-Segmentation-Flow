# Claude Code görev prompt'ları — TrackB flow pipeline

Kullanım sırası:

1. `CLAUDE.md`'yi repo köküne koy ve commit'le. Claude Code her oturumda otomatik okur.
2. Aşağıdaki prompt'ları **sırayla** ver. Bir aşamanın acceptance criteria'sı geçmeden bir sonrakine geçme.
3. Her prompt kendi başına yeterli; ama `CLAUDE.md` yoksa hiçbiri doğru çalışmaz.

Prompt'lar Türkçe, teknik terimler İngilizce. `CLAUDE.md` İngilizce çünkü repo artefaktı — Hoca veya başka bir ortak da okuyabilmeli.

---

## Prompt 0 — Onboarding / durum tespiti (GPU: 0)

```
Bu repoda TrackB (residual flow) debugging'ine başlıyoruz. Önce CLAUDE.md'yi
tamamen oku, sonra şu dosyaları kaynak koddan oku:

  flow/train.py, flow/losses.py, flow/model.py, flow/datasets.py,
  flow/sampler.py, flow/sliding_window.py, flow/validate.py,
  flow/conditioning.py, configs/flow.yaml, evaluation/metrics.py

Bana kod okumasına dayalı bir rapor yaz, tahmin değil. Şunları istiyorum:

1. CLAUDE.md bölüm 3'teki yedi teşhisin her biri için: kodda gerçekten böyle mi?
   Dosya + fonksiyon + satır göstererek DOĞRULA veya ÇÜRÜT. Çürüttüğün her madde
   için CLAUDE.md'ye düzeltme öner (dosyayı sen değiştirme, öneriyi yaz).

2. train.py'nin training loop'unu satır satır izle ve şu üçünü net söyle:
   - x0'a gürültü ekleniyorsa, x_t gürültülü x0'dan mı yoksa temiz x0'dan mı
     kuruluyor? (kısayolun ne kadar açık olduğunu bu belirliyor)
   - cond tensörü hangi noktada oluşturuluyor ve içindeki coarse SDF kanalları
     gürültüsüz mü?
   - lateral_coord gerçekten total_loss'a hiç geçmiyor mu?

3. Şu anda repoda bulunan HER dosya için tek satırlık "bu ne işe yarıyor +
   şu an çalışıyor mu / ölü kod mu" tablosu.

4. archive/ altındaki v0_flat_flow ile şu anki flow/ arasındaki farkı özetle.
   v0'dan geri taşınması gereken bir şey var mı?

Hiçbir dosyayı değiştirme. Sadece oku ve raporla.
```

---

## Prompt 1 — Aşama 0: teşhis altyapısı (GPU: 0, Mac'te koşar)

```
Aşama 0'a başlıyoruz: hiçbir yeni özellik eklemeden, elimizdeki sayıların
güvenilir olup olmadığını belirleyeceğiz. GPU kullanmayacağız.

Kapsam: flow/train.py, flow/validate.py, evaluation/, tests/, yeni analysis/ ve
scripts/ klasörleri. flow/model.py ve flow/losses.py'nin MATEMATİĞİNİ bu aşamada
DEĞİŞTİRME (sadece logging ekleyebilirsin).

--- Görev 1: loss bileşenlerini ayrı logla ---
losses.total_loss zaten bir `comp` dict döndürüyor ama train.py bunu atıyor.
Epoch boyunca her bileşenin ortalamasını biriktir ve progress.csv'ye ek sütun
olarak yaz: fm, narrowband, cldice, laterality, tv, total.
save_progress()'in kolon listesini history'nin anahtarlarından dinamik türet ki
ileride yeni terim eklediğimizde kırılmasın. progress.png'de bileşenleri ayrı
bir subplot'ta çiz.

--- Görev 2: clDice metriğini doğrula (test-first) ---
tests/test_metrics_sanity.py yaz. evaluation/metrics.py'deki cldice() ve dice()
için, sentetik 3D tüp üzerinde (yarıçap 3 voxel, 40 voxel uzunlukta, hafif
eğri) şu davranışları assert et:

  a) cldice(GT, GT) == 1.0
  b) 1 voxel dilate edilmiş tahmin: dice belirgin düşer, cldice ~1.0 kalır
  c) tüpün ortasından 5 voxel kesilmiş tahmin: cldice belirgin DÜŞER (<0.9)
  d) tüpe kopuk bir küre eklenmiş tahmin: cldice düşer
  e) boş tahmin: cldice == 0.0

(b) beklenen davranıştır, bug değil — clDice kalınlığa duyarsızdır. Testin amacı
metriğin (c) ve (d) durumlarında GERÇEKTEN tepki verdiğini kanıtlamak. Eğer
vermiyorsa metrik bozuktur ve progress.csv'deki sabit 0.993 anlamsızdır; o
durumda skeletonize çağrısını debug et ve rapor et.

Aynı dosyada evaluation/topology_metrics.py'deki betti0_error,
centerline_gap_length, false_branch_length, lr_swap_rate için de bilinen-cevap
testleri yaz.

--- Görev 3: identity baseline (v ≡ 0) ---
scripts/identity_baseline.py yaz. Model yerine sabit sıfır velocity kullanarak
flow/validate.py'nin tam yolunu koştur (sliding window + ODE + SDF decode) ve
TÜM CV üzerinde (max_cases=None, 5 fold'un tüm validation case'leri) metrikleri
çıkar. Sonucu outputs/baselines/identity_prior.json'a yaz.

Bu sayı paper'ın baseline satırı. Beklenti: TrackA'nın 0.9101'ini yeniden
üretmesi. Üretmiyorsa, farkın kaynağını bul ve raporla — şu üç hipotezi ayrı
ayrı test et:
  H1: val_max_cases=20 örnekleme gürültüsü (tüm CV'de fark kapanıyor mu?)
  H2: metrik tanımı — validate() per-side satır biriktirip ortalıyor, nnU-Net
      per-case ortalıyor. İkisini de hesapla ve ikisini de raporla.
  H3: coarse SDF round-trip kaybı. mask -> SDF -> sign -> mask zincirini bir
      case üzerinde birebir karşılaştır; sıfır fark bekliyorum, değilse bug var.

--- Görev 4: "zarar verme" checkpoint gate ---
flow/train.py'de best.pt yazma koşuluna sabit bir taban ekle: Görev 3'ten çıkan
prior skorunun altındaki hiçbir checkpoint best.pt olarak yazılamaz. Tabanı
configs/flow.yaml'a `prior_floor: {dice: ..., cldice: ..., hd95: ..., score: ...}`
olarak koy ve identity_baseline.py'nin bu bloğu otomatik yazabildiği bir
--write-config flag'i ekle. last.pt (resume checkpoint'i) bu kuraldan muaf.

--- Görev 5: run manifest ---
scripts/run_manifest.py: her training run'ı için runs/<config_hash>/ dizini
oluştur ve manifest.json yaz: git commit SHA, dirty flag, resolved config'in
tamamı, config'in sha256'sı, fold, seed, python/torch versiyonları, GPU adı,
başlangıç zamanı. Run bitince/kesilince son metrikleri ve bitiş zamanını
güncelle (atomic write). train.py'yi --out yerine bu dizini kullanacak şekilde
bağla ama eski --out davranışını bozma (geriye uyumlu olsun).

--- Acceptance criteria ---
- pytest tests/ tamamen yeşil
- outputs/baselines/identity_prior.json mevcut ve içinde hem per-side hem
  per-case ortalamalar var
- identity baseline ile TrackA CV arasındaki fark açıklanmış (hangi hipotez)
- flow/train.py --resume ile 3 epoch'luk bir smoke run yapıp kesip devam
  ettirdiğimde progress.csv bozulmuyor
- git log'da her görev ayrı commit

Bittiğinde bana kısa bir özet yaz: hangi sayı değişti, hangi hipotez doğrulandı,
CLAUDE.md'de güncellenmesi gereken ne var.
```

---

## Prompt 2 — Aşama 0.5: kısayol hipotezini kanıtla (GPU: ~15 dk, paper Şekil 1)

```
CLAUDE.md bölüm 3.1-3.2'deki kısayol teşhisini AMPİRİK olarak kanıtlayacağız.
Bu, makalenin Figure 1'i olacak, o yüzden çıktı yayın kalitesinde olmalı.

Elimizde fold 0'ın 125. epoch checkpoint'i var. analysis/shortcut_probe.py yaz.

Not: sigma=0 durumunda (x_t - x0)/t ile (x1 - x0) CEBİRSEL OLARAK AYNI şeydir.
Bu yüzden "modelin çıktısı hangisine yakın" diye sormak anlamsız — ikisi aynı.
Kısayolu ancak modelin NEYE BAKMADIĞINI göstererek kanıtlarız. Üç probe:

Probe A — t'ye göre loss profili (asıl kanıt)
  Validation set'ten ~200 foreground patch al. t ∈ {0.02, 0.05, 0.1, 0.2, ...,
  0.95} grid'inde, her t için FM loss'un ortalamasını hesapla (t'yi rastgele
  değil sabit vererek). Kısayol hipotezi şunu öngörüyor: loss(t) büyük t'de
  çok küçük, t -> 0'a giderken patlıyor. Log-log çiz. Eğer loss ~ 1/t gibi
  davranıyorsa hipotez doğrulanmıştır.

Probe B — görüntü ablasyonu
  Aynı patch'lerde, cond kanal 0'ı (CBCT) sıfırla / gaussian gürültüyle değiştir
  ve v_pred'in ne kadar değiştiğini ölç:
      rel_delta(t) = ||v_full - v_noimg|| / ||v_full||
  t'ye göre çiz. Kısayol hipotezi: büyük t'de rel_delta ≈ 0 (model görüntüye
  bakmıyor), küçük t'de yükseliyor.

Probe C — prior kanal ablasyonu
  cond kanal 3-4'ü (coarse SDF) sıfırla, x_t'ye dokunma. Kısayol hipotezi:
  v_pred çöker, çünkü model x_t'den x0'ı çıkarmayı öğrenmiş.

Çıktılar:
  - outputs/analysis/shortcut_probe.csv (ham sayılar)
  - outputs/analysis/fig1_shortcut.pdf — 3 panel, vektörel, 8pt font,
    matplotlib default stil değil (yayın için okunabilir), colorblind-safe
  - Terminal'e tek paragraflık yorum: hipotez doğrulandı mı, hangi t eşiğinde

Bu script CPU/MPS'te de koşabilmeli (patch sayısını azaltarak). Cihazı
--device ile al.

Hiçbir training kodunu değiştirme. Bu sadece bir okuma/analiz aracı.
```

---

## Prompt 3 — Aşama 1: ablation grid (GPU: ~4 saat, fold 0, 50'şer epoch)

```
Aşama 1: kısayolu kapatan minimal müdahaleleri ablation ile test edeceğiz.
Hiçbirini "kesin çözüm" diye kalıcı yapma — hepsi config flag'i olacak.

--- Görev 1: zero-init head ---
flow/model.py'de ResidualVelocityUNet3D.head'in son Conv3d'sinin weight ve
bias'ını sıfırla (nn.init.zeros_). Böylece t=0'da v ≡ 0 ve endpoint tam olarak
x0 (prior) oluyor; model prior'ın üstüne residual öğreniyor, sıfırdan velocity
uydurmuyor. Bu, normalizing flow (Glow) ve diffusion mimarilerinin standart
tricki. configs/flow.yaml'a `zero_init_head: true` ekle, varsayılan true.

Bir unit test yaz: yeni initialize edilmiş modelde, rastgele girdi için
integrate(model, cond, x0, steps=8) çıktısı x0'a bit-bit eşit olmalı.

--- Görev 2: config'e ablation anahtarları ekle ---
configs/flow.yaml:
  zero_init_head: true
  cond_include_coarse_sdf: true    # false -> cond kanal 3,4 çıkar (COND_CH 8->6)
  noise_frac: 0.5
  train_sigma: 0.1
  w_softdice: 0.0                  # yeni: overlap terimi
  w_cldice: 0.5

cond_include_coarse_sdf=false olduğunda conditioning.py, datasets.py, model.py,
validate.py, sliding_window.py'nin hepsinin tutarlı çalışması lazım — COND_CH'i
tek bir yerden türet, sabit sayı hard-code etme. Kanal sözleşmesini doğrulayan
bir test yaz.

--- Görev 3: soft-Dice loss terimi ---
losses.py'ye soft_dice_loss(occ_pred, occ_true) ekle ve total_loss'a
w_softdice ile bağla. CLAUDE.md bölüm 3.3'teki gerekçe: clDice tek başına
kalınlaşmayı ödüllendiriyor; orijinal clDice makalesi onu soft-Dice ile
birlikte kullanıyor. Bu terim OLMADAN clDice'ı açmak hatalı.

--- Görev 4: ablation runner ---
scripts/run_ablations.py: bir YAML listesi alıp sırayla run'ları başlatan,
her birini kendi runs/<hash>/ dizinine yazan, aralarda ölse kaldığı yerden
devam eden bir sürücü. Şu dört run'ı tanımla (hepsi fold 0, 50 epoch,
val_every 10):

  B0  zero_init_head=true, w_narrowband=0, w_cldice=0, w_laterality=0,
      w_softdice=0, noise_frac=0.5
      -> beklenti: Dice prior'ın ALTINA DÜŞMEMELİ. Düşerse sorun aux loss'larda
         değil, temel formülasyonda demektir.

  B1  B0 + noise_frac=1.0, train_sigma=0.3
      -> kısayolu zayıflatır (hata sigma*eps/t, küçük t'de büyük)

  B2  B1 + cond_include_coarse_sdf=false
      -> kısayolu tamamen kapatır: model artık x_t'den x0'ı çıkaramaz

  B3  B2 + w_narrowband=1.0, w_softdice=1.0
      -> boundary + overlap sinyali geri gelir

--- Acceptance criteria ---
- Her run için 5 epoch'luk smoke test önce geçmiş olacak (patch=32, 2 case)
- outputs/ablation_stage1.md: 4 satırlık tablo (run, dice, cldice, hd95, score,
  prior'a göre delta), her satırda hangi hipotezi test ettiği
- B0 prior'ın altına düşüyorsa DUR ve bana rapor et, B1-B3'ü koşturma

Bana her run bittiğinde tek satırlık sonuç yaz, sonuna kadar sessiz kalma.
```

---

## Prompt 4 — Aşama 2: bridge formülasyonu (GPU: ~10 saat)

```
Aşama 1'de kısayolun sorun olduğu doğrulandıysa, şimdi prensipli çözüme
geçiyoruz: deterministik interpolasyon yerine stochastic interpolant / Brownian
bridge.

Gerekçe (CLAUDE.md 3.1 ile birlikte oku): x0 ve x1 deterministik eşleştiği için
her voxelde p_t bir Dirac — sıfır ölçülü bir doğru parçası. Model eğitimde
sadece bu doğrunun üstünü görüyor, inference'ta kendi trajectory'sini üretip
doğrunun dışına çıkıyor (exposure bias). Brownian bridge gürültüsü hem path'i
pozitif hacme yayıyor hem de kısayolu matematiksel olarak imkansız kılıyor,
çünkü eps modele verilmiyor.

--- Görev 1: bridge path ---
losses.py ve train.py'de path'i şu hale getir:

    x_t = (1-t)*x0 + t*x1 + gamma(t)*eps,    eps ~ N(0, I)
    gamma(t) = sigma_bridge * sqrt(t*(1-t))

configs/flow.yaml: `path: bridge | rectified` ve `sigma_bridge: 0.2`.
rectified mevcut davranış olarak kalsın (ablation için lazım).

--- Görev 2: x1-parametrizasyonu (numerik zorunluluk) ---
gamma'(t) uçlarda patlıyor, o yüzden velocity'yi doğrudan regrese ETME. Bunun
yerine ağ x1'i (endpoint SDF) tahmin etsin, velocity analitik türetilsin:

    x1_hat = f_theta(x_t, t, cond)
    L_main = || x1_hat - x1 ||^2                    (t'ye göre ağırlıklı)
    v(x_t, t) = (x1_hat - x_t) / (1 - t)            sampler'da, t<1 için

Bunu model.py'ye bir `parameterization: velocity | x1` seçeneği olarak ekle.
sampler.integrate() her iki durumu da desteklesin. t -> 1'de (1-t) bölmesi için
son adımı x1_hat ile doğrudan kapat (division guard, epsilon değil).

Bu değişiklik tüm auxiliary loss'ları da sadeleştiriyor: x1_hat artık GT ile
karışmıyor (CLAUDE.md 3.5), yani clDice/narrowband/softDice gerçekten modelin
kendi çıktısını puanlıyor. Bunu kodda açıkça yorumla.

--- Görev 3: train/inference patch dağılımı uyumu ---
CLAUDE.md 3.4: eğitimde fg_prob=0.8, inference'ta tüm hacim taranıyor. İki
düzeltme, ikisi de config flag'i:
  a) `bg_patch_frac: 0.2` — kanal içermeyen patch'leri bilerek eğitime kat.
     Bu patch'lerde hedef v ≡ 0 (x1 == x0 == satüre +1), yani model "dokunma"yı
     öğreniyor.
  b) sliding_window.py'de gaussian blend'i düzelt: `np.maximum(wsum, 1e-6)`
     yerine, ağırlığı belirli bir eşiğin altında kalan voxelleri hiç yazma
     (mask'le) ve o bölgeleri +1 (dış) SDF ile doldur. Şu anki hali köşelerde
     işaret hatası üretebiliyor.

Her ikisi için de önce bir test yaz: tamamen background bir hacimde inference
sıfır foreground üretmeli.

--- Görev 4: laterality'yi gerçekten bağla ---
train.py total_loss'a lateral_coord'u geçmiyor. conditioning.lateral_axis_index()
ile affine'den ekseni belirle (array index varsayma), ilgili koordinat kanalını
datasets.py'den patch'le birlikte döndür, train.py'de total_loss'a geçir.
laterality_coord_weight'i 0.1 yap. Seçilen ekseni ve işaretini run manifest'ine
LOGLA — yanlışsa deney sessizce bozulur.

Ayrıca mevcut overlap terimi 96³ patch'te ölü (iki kanal aynı patch'e girmiyor);
bunu koda yorum olarak yaz ve w_laterality'yi 0'a çek, coord terimi tek aktif
laterality mekanizması olsun.

--- Görev 5: fold 0, 200 epoch, bridge vs rectified ---
İki run: path=bridge ve path=rectified, aksi her şey aynı. Aşama 1'in en iyi
konfigürasyonu üstüne. Karşılaştırma tablosunu outputs/ablation_stage2.md'ye yaz.

--- Acceptance criteria ---
- Tüm testler yeşil, özellikle boş-hacim testi
- bridge run'ı prior floor'u geçiyor (score > prior)
- x1-parameterization ile velocity-parameterization arasındaki fark ölçülmüş
- eksen seçimi manifest'te loglanmış
```

---

## Prompt 5 — Aşama 3: eğri-uzayı flow prototipi (GPU: ~20 saat, ana katkı adayı)

```
Bu, voxel-SDF flow'una alternatif bir kol. Paralel yürütülebilir; Aşama 2'yi
bloklamaz. Yeni klasör: curveflow/ — mevcut flow/ koduna DOKUNMA.

Fikir: IAC yapısal olarak tek bir 1D eğri + yarıçap profili. Voxel uzayında
generatif model kurmak hem pahalı hem topolojiyi garanti etmiyor. Bunun yerine
düşük boyutlu şekil uzayında flow kur:

  taraf başına: K=20 B-spline kontrol noktası (K x 3) + K yarıçap = 80 boyut
  iki taraf: 160 boyut

Bu temsilin sonuçları: (a) spline tüpü tanım gereği tek bağlantılı bileşen ve
deliksiz — Betti-0 hatası yapısal olarak sıfır; (b) 160-D'de flow matching tek
GPU'da dakikalar sürer; (c) sampling gerçekten anlamlı — her örnek "olabilir"
bir kanal yolu, bu da uncertainty iddiasının temeli.

Precedent: PointFlow (Yang et al., ICCV 2019) CNF'i point cloud üstünde kurar;
statistical shape models / point distribution models medikal şekil literatürünün
ana damarı. Bunları koda yorum olarak referansla.

--- Görev 1: temsil ve encode/decode ---
curveflow/representation.py:
  - fit_curve(mask, K) -> (control_points (K,3) mm cinsinden, radii (K,))
    skeletonize -> en uzun yolu al -> yay uzunluğuna göre yeniden örnekle ->
    B-spline fit. Yarıçap: her merkez hattı noktasında GT maskesine olan mesafe
    dönüşümünün değeri.
  - render_tube(control_points, radii, shape, affine) -> binary mask
    Sadece numpy/scipy; differentiable olmasına gerek yok (loss şekil uzayında).
  - Round-trip testi: GT mask -> fit -> render -> Dice. Bu üst sınır. K=10, 20,
    40 için ölç ve raporla. Dice < 0.90 ise temsil yetersiz, K'yı artır veya
    bize haber ver — bu, tüm kolun fizibilitesini belirleyen sayı.

--- Görev 2: koşullu flow ---
curveflow/model.py: koşullu velocity MLP. Conditioning: nnU-Net OOF prior'ından
fit edilmiş coarse eğri (aynı 160-D temsil) + CBCT'den küçük bir 3D CNN
encoder'ın çıkardığı global özellik vektörü (kanal boyunca ROI crop, düşük
çözünürlük — 64³ yeterli).
  x0 = coarse eğri parametreleri
  x1 = GT eğri parametreleri
  Aşama 2'deki bridge path'i ve x1-parametrizasyonunu AYNEN kullan (kodu
  paylaş, kopyalama).

--- Görev 3: değerlendirme ---
Örneklenen eğriyi render_tube ile voxel'e çevir, evaluation/evaluate_cv.py'nin
aynısıyla ölç. Karşılaştırma tablosu üç satır: nnU-Net prior, voxel-SDF flow
(Aşama 2 en iyisi), curve flow.

--- Acceptance criteria ---
- Round-trip Dice tablosu (K=10/20/40) outputs/curveflow_representation.md
- fold 0'da end-to-end bir sonuç, prior ile karşılaştırmalı
- Betti-0 hatası tanım gereği 0 — bunu bir testle doğrula

Görev 1 bitmeden Görev 2'ye geçme. Round-trip Dice tavan; model o tavanı
geçemez.
```

---

## Prompt 6 — Aşama 4: tam CV, uncertainty, paper artefaktları (GPU: ~40 saat)

```
Kazanan konfigürasyonu 5 fold'da koşturup makale çıktılarını üreteceğiz.

--- Görev 1: 5-fold training ---
scripts/run_ablations.py'yi fold döngüsüyle koştur. Her fold ayrı runs/ dizini,
her fold sonunda Drive'a persist. Colab kopmalarına dayanıklı olsun (zaten
--resume var, 5-fold döngüsünde de çalıştığını doğrula).

--- Görev 2: uncertainty ---
Generatif olmanın gerekçesi tek deterministik örnek değil, dağılım. Deterministik
run (sigma=0) headline tahmin; ayrıca K=16 stokastik örnek al (sliding_window.py
zaten global noise field destekliyor — patch başına yeniden örneklenmediğini
doğrula, seam artefaktı olmasın).

Üret:
  - per-voxel foreground olasılığı (K örneğin ortalaması)
  - varyans / entropi haritası
  - "güvenlik zarfı": olasılık > tau seviye kümesi. Klinik anlamı: implant
    planlamasında sinir hasarı riski için güvenlik payı.

Kalibrasyon değerlendirmesi:
  - reliability diagram + ECE
  - zarfın GT'yi kapsama oranı vs zarf hacmi eğrisi; naif dilatasyon baseline'ı
    ile karşılaştır (aynı kapsama için daha küçük hacim = daha iyi)

--- Görev 3: istatistik ---
evaluation/evaluate_cv.py'yi tüm modeller için koştur. compare_bootstrap ile
paired bootstrap %95 CI: her metrik için (flow - prior). Dice'ta non-inferiority,
topoloji metriklerinde superiority iddiası kuruyoruz — CI'ları buna göre raporla.
Çoklu karşılaştırma düzeltmesi uygula ve hangisini kullandığını yaz.

--- Görev 4: makale artefaktları ---
paper/ klasörü altında, hepsi script'ten üretilen (elle çizilmiş hiçbir şey yok):
  - table1_main.tex: prior / voxel-flow / curve-flow x
    {Dice, HD95, NSD, clDice, Betti-0, gap_mm, false-branch_mm, swap_rate},
    mean ± std, en iyi kalın
  - table2_ablation.tex: Aşama 1 ve 2'nin tüm run'ları
  - fig1_shortcut.pdf (Aşama 0.5'ten)
  - fig2_qualitative.pdf: 3 case — prior'ın koptuğu, flow'un bağladığı;
    prior'ın iyi olduğu; ikisinin de battığı (failure case'i GİZLEME)
  - fig3_uncertainty.pdf: bir case'de olasılık haritası + güvenlik zarfı
  - reproduce.sh: sıfırdan tüm tabloları üreten tek komut

--- Acceptance criteria ---
- reproduce.sh temiz bir checkout'ta koşuyor
- Her tablo/figür'ün altında hangi run hash'lerinden üretildiği yazıyor
- Failure case'ler raporlanmış
```

---

## Kısa notlar

**Sıra önemli.** Prompt 3'teki B0 run'ı prior'ın altına düşerse, Prompt 4-5-6 anlamsız — sorun auxiliary loss'larda değil temel formülasyonda demektir ve teşhisi yeniden açman gerekir.

**Prompt 2 en yüksek getirili tek adım.** Sıfıra yakın GPU maliyeti var ve makalenin Figure 1'ini üretiyor. Aşama 1'e geçmeden onu koştur.

**Prompt 5 bağımsız.** Aşama 2 tıkanırsa bile eğri-uzayı kolu tek başına bir workshop paper'ı taşıyabilir, ve GPU maliyeti diğerlerinin çok altında.

**Her prompt'un sonuna ekleyebileceğin ortak kuyruk:**

```
Uzun run başlatmadan önce mutlaka 3-5 epoch'luk smoke test yap (küçük patch,
2 case) ve checkpoint + --resume'un çalıştığını doğrula. Oturumun her an
kopabileceğini varsay. Kapsam dışı bir dosyaya dokunman gerekiyorsa dur ve
bana sor. Her görev ayrı commit.
```
