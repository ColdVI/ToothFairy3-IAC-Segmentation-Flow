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

5. OOF prior artifact'lerini değiştirmeden audit et:
   - hard mask / derived one-hot / true softmax türünü belirle;
   - her kanalın unique değerleri ile min/max/mean'ini çıkar;
   - implicit background dahil p_bg+p_L+p_R toplamını kontrol et;
   - entropy dağılımını raporla;
   - expected validation fold, prediction source fold ve source checkpoint
     provenance'ını doğrula;
   - image/GT/OOF/coarse-SDF shape, affine ve spacing uyumunu kontrol et;
   - conditioning kanal 1-2 ile coarse-SDF kanal 3-4 redundancy/correlation'ını
     ölç.

   `predict_oof.py` mevcut hard segmentation'ı one-hot'a çeviriyorsa bunu true
   probability veya calibrated uncertainty diye adlandırma. Prompt 0 yalnızca
   raporlar; hiçbir artifact'i sessizce dönüştürmez, yeniden adlandırmaz veya
   overwrite etmez.

Hiçbir dosyayı değiştirme. Sadece oku ve raporla.
```

---

## Prompt 1 — Aşama 0: teşhis altyapısı (Colab CPU; local yalnızca edit)

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

Identity sonucunu üç ayrı yol üzerinden vaka bazında karşılaştır:
  A) direct OOF hard segmentation,
  B) direct coarse-SDF sign decode,
  C) zero-velocity sampler + sliding-window full path.
H4 olarak A ile C arasındaki voxel farkını raporla. Full-path farkı açıklanmadan
veya düzeltilmeden baseline güvenilir sayılmaz.

--- Görev 4: checkpoint üçlüsü ve "zarar verme" gate'i ---
Checkpoint'leri `last.pt`, `best_any.pt`, `best_safe.pt` olarak ayır.

- `last.pt`: resume için her checkpoint cadence'inde atomik güncellenir.
- `best_any.pt`: run içindeki leksikografik olarak en iyi validation sonucu;
  prior altında olsa da debugging için korunur.
- `best_safe.pt`: yalnızca önceden tanımlanmış Dice non-inferiority koşulunu
  geçerse yazılır.

Seçimi tek weighted score'a indirgeme. Sıra:
  1) Dice >= prior Dice - predefined_noninferiority_margin,
  2) daha düşük centerline gap / Betti-0 error,
  3) daha düşük HD95,
  4) daha yüksek clDice,
  5) eşitlikte daha erken epoch.

`prior_floor` tam 480-case identity baseline'dan otomatik yazılabilir.
`noninferiority_margin` config alanı olarak eklenir fakat değer UYDURULMAZ;
baseline tamamlandıktan sonra herhangi bir B-run görülmeden önce kullanıcı
tarafından belirlenir. Prior veya margin null/partial ise training başlamaz ve
`best_safe.pt` üretilemez. Legacy `best.pt` davranışını açıkça migrate et.

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

## Prompt 2 — Aşama 0.5: kısayol hipotezi için nedensel diagnostik (GPU: ~15 dk)

```
CLAUDE.md'deki algebraic shortcut'ın checkpoint tarafından gerçekten
kullanıldığı HİPOTEZİNİ test edeceğiz. Çıktı Figure 1 adayıdır; fakat tek bir
loss profili "proof" değildir. analysis/shortcut_probe.py yaz.

Epoch 0, 25 ve 125 checkpoint'lerini aynı vaka/patch/t grid'inde karşılaştır.
Foreground ve pure-background patch'leri ayrı strata olarak raporla.

Probe A — t'ye göre loss profili (supporting diagnostic)
  Validation set'ten foreground ve pure-background patch'ler al. Sabit
  t ∈ {0.02, 0.05, 0.1, 0.2, ..., 0.95} grid'inde FM loss ölç ve log-log çiz.

  Yorum sınırları:
  - Gürültüsüz deterministic batch'lerde küçük-t loss indirgenemez koşullu
    varyans nedeniyle platoya oturabilir.
  - Gürültülü batch'lerde analytic shortcut hatası yaklaşık sigma²/t² ölçeğinde
    büyüyebilir.
  - Büyük t'de düşük loss tek başına shortcut kanıtı değildir; x_t zaten x1
    bilgisi taşır.
  - "loss ~ 1/t, dolayısıyla kanıtlandı" sonucu çıkarma.

Probe B — CBCT nedensel ablasyonu (primary causal probe)
  Aynı state/prior/t korunurken CBCT kanalını:
    1) sıfırla,
    2) eş varyanslı Gaussian noise ile değiştir,
    3) vakalar arasında shuffle et.
  Her müdahalede output sensitivity ölç:
      rel_delta(t) = ||v_full-v_ablated|| / max(||v_full||, eps)
  Case-level bootstrap %95 CI ver. Epoch ve foreground/background strata'larını
  ayrı göster. Düşük CBCT sensitivity shortcut ile uyumlu kanıttır; tek başına
  matematiksel ispat değildir.

Probe C — analytic shortcut ve prior müdahaleleri
  sigma=0 için:
      v_shortcut = (x_t-x0)/t
  pred_v ile v_shortcut arasında cosine similarity ve R² ölç. Ayrıca:
    - conditioning coarse-SDF kanal 3-4'ü sıfırla,
    - başka bir vakanın coarse-SDF kanallarıyla değiştir,
    - x_t'ye dokunmadan doğru/yanlış prior sensitivity'sini karşılaştır.
  Cross-case swap'ın shape uyumunu güvenli crop/pad ile sağla ve provenance'ı
  kaydet.

Probe D — uniform thickening compatibility
  Mevcut prediction'ı fiziksel spacing'i dikkate alarak yaklaşık bir voxel
  erode et. Erosion öncesi/sonrası Dice ve HD95 recovery, prediction/GT volume
  ratio, signed surface-distance dağılımı ve radius-profile farkını raporla.
  Sonuç dili "uniform thickening ile compatible" olabilir; "fully explained"
  deme.

Çıktılar:
  - outputs/analysis/shortcut_probe.csv (case-level ham sayılar)
  - outputs/analysis/shortcut_probe_summary.json (bootstrap CI ve strata)
  - outputs/analysis/fig1_shortcut.pdf (vektörel, 8pt, colorblind-safe)
  - outputs/analysis/thickening_probe.csv
  - Tek paragraf sonuç: yalnızca "The observations provide strong empirical
    evidence consistent with shortcut use" düzeyine kadar iddia kur.

"Shortcut proved" veya "Probe A mathematically demonstrated it" yazma.
Hiçbir training kodunu değiştirme.
```

---

## Prompt 3R — Limited Endpoint Probe Sonrası Flow-v2 Pilotu

```
Bu görev Prompt 0, Prompt 1 ve limited-endpoint Prompt 2 sonrasındaki Aşama
1A'dır. Eski Prompt 3'ü silmez; Prompt 3R acceptance geçene kadar erteler.
Prompt 4, Prompt 5 ve Prompt 6 da bu gate geçmeden başlatılamaz.

KESİN STOP KURALLARI:
- B0-B3 dört adet 50-epoch run'ı başlatma.
- Legacy checkpoint üretme, migrate etme, yeniden kaydetme veya overwrite etme.
- Track A'yı yeniden eğitme.
- Prompt-1 identity artefaktlarını yalnızca salt-okunur kullan.
- Limited diagnostic'i paper proof veya gerçek training trajectory diye sunma.

ÖNCE KAYNAK ARTEFAKT AUDITİ:
Repo ve Drive altında yolları tahmin etmeden şu beş artefaktı bul ve oku:
shortcut_probe_manifest.json, shortcut_probe_summary.json, shortcut_probe.csv,
thickening_probe.csv ve limited_endpoint_diagnostic.pdf. Manifest git HEAD,
dirty flag, resolved config/config hash, checkpoint yolları ve SHA256'ları,
identity-baseline SHA256, GPU, vaka/patch/strata sayıları ve şu flag'leri mevcut
çalışma ağacıyla karşılaştır:

  protocol_deviation=true
  exact_epoch_trajectory_available=false
  historical_per_epoch_checkpoints_were_not_saved=true

Endpoint sözleşmesi:
- best.pt = best_legacy_unknown_epoch; internal epoch yoktur. Epoch 0/1 veya
  başka bir epoch atama. Erken/prior-like olduğu yalnızca hipotezdir.
- last.pt/latest.pt yalnızca internal epoch 129 doğrulanırsa epoch_129'dur.

ÖLÇÜLMÜŞ / ÖLÇÜLMEMİŞ AYRIMI:
Ölçülmüş gözlemler olarak yalnızca kaynak artefaktların desteklediği kapsamı
yaz: legacy best'in örneklenen full-volume vakalarda epoch 129'dan iyi olması;
epoch 129'un daha düşük FM loss'una rağmen daha kötü segmentation üretmesi;
epoch 129'da thickening ile uyumlu volume/surface/radius sinyali; fiziksel
erozyonla birçok tarafta recovery; epoch 129'da güçlü coarse-prior dependence;
ve basit thickening ile düzelmeyen HD95 outlier'ı.

Ölçülmemiş olarak açıkça koru: best.pt exact epoch'u, bozulmanın başladığı epoch,
tüm eğitim boyunca monoton thickening, shortcut'ın matematiksel ispatı ve legacy
best'in başarılı bir flow modeli olduğu iddiası.

KOD ENVANTERİ — DEĞİŞTİRMEDEN ÖNCE:
Kaynak kod ve testlerden şu maddelerin durumunu implemented / partial / missing
olarak raporla: zero-init head; best_any/best_safe; complete-CV prior floor ve
predeclared margin gate; soft-Dice; dynamic conditioning channels;
topology-aware validation; immutable checkpoint history; paired three-path
identity comparison. Kod ile belge çelişirse kodu otomatik olarak belgeye
uydurma; çelişkiyi önce raporla.

DOKÜMANTASYON UZLAŞTIRMASI:
- CLAUDE.md measured state'e historical per-epoch checkpoint'lerin
  saklanmadığını, exact trajectory'nin unavailable olduğunu, legacy best
  epoch'unun unknown olduğunu ve limited diagnostic'in ölçülmüş sonuçlarını ekle.
- Thickening ve prior dependence'i diagnostic observation olarak; yeni
  early-epoch trajectory ihtiyacını motivation olarak; shortcut mekanizmasını
  hypothesis olarak sınıflandır.
- Eski Prompt 3'ü bu uyarının altında aynen koru ve deferred say.

Bu dokümantasyon gate'i tek commit olmalıdır:
  stage1a/docs: reconcile Prompt 3 with limited endpoint evidence

SONRAKİ AŞAMA SINIRI:
Bu metin güvenlik düzeltmeleri veya pilot training için tek başına launch
authorization değildir. Flow-v2 kod değişiklikleri test-first yapılmalı;
prospective pilot her epoch'u immutable ve atomik saklamalı, resume ile aynı
trajectory'yi korumalı ve uzun B0-B3 run'larından önce kısa Fold-0 acceptance
gate'inden geçmelidir. Pilot bütçesi, validation cadence'i ve acceptance
eşikleri ayrıca açıkça tanımlanmadan training başlatma.
```

> **SUPERSEDED TEMPORARILY BY PROMPT 3R**
>
> Do not launch B0–B3 50-epoch runs until Prompt 3R code and pilot acceptance
> criteria pass.

## Prompt 3 — Aşama 1: ablation grid (GPU: ~4 saat, fold 0, 50'şer epoch)

```
Aşama 1: kısayolu kapatan minimal müdahaleleri ablation ile test edeceğiz.
Hiçbirini "kesin çözüm" diye kalıcı yapma — hepsi config flag'i olacak.

--- Görev 1: zero-init head ve gerçek plumbing stop-rule ---
flow/model.py'de ResidualVelocityUNet3D.head'in son Conv3d'sinin weight ve
bias'ını sıfırla (nn.init.zeros_). Böylece t=0'da v ≡ 0 ve endpoint tam olarak
x0 (prior) oluyor; model prior'ın üstüne residual öğreniyor, sıfırdan velocity
uydurmuyor. Bu, normalizing flow (Glow) ve diffusion mimarilerinin standart
tricki. configs/flow.yaml'a `zero_init_head: true` ekle, varsayılan true.

Training update yapılmadan önce epoch-0 validation çalıştır. Zero-init modelin
çıktısı identity prior'ı beklenen tolerans içinde yeniden üretmeli. Bitwise
eşitlik floating-point/sliding-window nedeniyle mümkün değilse:
  - decoded voxel mask equality,
  - maksimum/ortalama SDF absolute error,
  - önceden tanımlanmış explicit numeric tolerance
raporla. Epoch 0 identity'yi üretemezse DUR: bu bir model/sampler/plumbing
bug'ıdır ve hiçbir B-run başlatılmaz. Tek gerçek erken stop koşulu budur.

--- Görev 2: config'e ablation anahtarları ekle ---
configs/flow.yaml:
  zero_init_head: true
  cond_include_coarse_sdf: true    # false -> cond kanal 3,4 çıkar (COND_CH 8->6)
  noise_frac: 0.5
  train_sigma: 0.1
  w_softdice: 0.0                  # yeni: overlap terimi
  w_cldice: 0.5
  noninferiority_margin: null      # baseline sonrası, B sonucu görülmeden karar

cond_include_coarse_sdf=false olduğunda conditioning.py, datasets.py, model.py,
validate.py, sliding_window.py'nin hepsinin tutarlı çalışması lazım — COND_CH'i
tek bir yerden türet, sabit sayı hard-code etme. Kanal sözleşmesini doğrulayan
bir test yaz.

--- Görev 3: soft-Dice loss terimi ---
losses.py'ye soft_dice_loss(occ_pred, occ_true) ekle ve total_loss'a
w_softdice ile bağla. CLAUDE.md bölüm 3.3'teki gerekçe: clDice tek başına
kalınlaşmayı ödüllendiriyor; orijinal clDice makalesi onu soft-Dice ile
birlikte kullanıyor. Bu terim OLMADAN clDice'ı açmak hatalı.

--- Görev 4: checkpoint selection ---
Her run atomik `last.pt`, `best_any.pt`, `best_safe.pt` üretir:
  - last: resume state,
  - best_any: prior altında olsa bile run içindeki en iyi debugging checkpoint,
  - best_safe: yalnızca Dice non-inferiority eligibility sonrası.

Tek weighted score kullanma. Leksikografik sıra:
  1) Dice eligibility,
  2) daha düşük centerline gap ve Betti-0 error,
  3) daha düşük HD95,
  4) daha yüksek clDice,
  5) daha erken epoch.
`prior_floor` veya `noninferiority_margin` null/partial ise training başlama.
Margin'i uydurma; herhangi bir B sonucu görülmeden kullanıcı kararı gerekir.

--- Görev 5: ablation runner ve direct-refiner control ---
scripts/run_ablations.py: bir YAML listesi alıp sırayla run'ları başlatan,
her birini kendi runs/<hash>/ dizinine yazan, aralarda ölse kaldığı yerden
devam eden bir sürücü. Şu beş run'ı tanımla (hepsi fold 0, 50 epoch,
val_every 10):

  B0  zero_init_head=true, w_narrowband=0, w_cldice=0, w_laterality=0,
      w_softdice=0, noise_frac=0.5
      -> temel rectified residual formulation.

  B1  B0 + noise_frac=1.0, train_sigma=0.3
      -> kısayolu zayıflatır (hata sigma*eps/t, küçük t'de büyük)

  B2  B1 + cond_include_coarse_sdf=false
      -> kısayolu tamamen kapatır: model artık x_t'den x0'ı çıkaramaz

  B3  B2 + w_narrowband=1.0, w_softdice=1.0
      -> boundary + overlap sinyali geri gelir

  B4  deterministic direct refiner control
      -> Flow ile aynı CBCT ve prior conditioning; yaklaşık aynı 3-D
         encoder/decoder kapasitesi; NFE=1; ODE ve stochastic sampling yok;
         doğrudan endpoint SDF x1_hat veya residual SDF tahmini; aynı split,
         training budget, validation ve checkpoint selection; soft-Dice ile
         uygun SDF/boundary loss. Amaç flow kazancının yalnızca ikinci bir 3-D
         U-Net/refiner eklemekten gelip gelmediğini test etmek.

B0 training sırasında prior altına düşerse B1 ve B2'yi DURDURMA; bu sonuç
B1/B2'yi çalıştırma gerekçesidir. B3 yalnızca B1/B2 minimal müdahalelerinden
anlamlı sinyal çıkarsa çalıştırılır. B4 final karşılaştırmalarda zorunludur.

--- Acceptance criteria ---
- Her run için 5 epoch'luk smoke test önce geçmiş olacak (patch=32, 2 case)
- Epoch-0 zero-init identity preflight geçecek; geçmezse tüm training duracak
- outputs/ablation_stage1.md: 5 satırlık tablo (run, dice, cldice, HD95,
  gap, Betti-0, safe eligibility,
  prior'a göre delta), her satırda hangi hipotezi test ettiği
- B4 aynı conditioning/capacity/budget sözleşmesini manifest'te gösterecek

Bana her run bittiğinde tek satırlık sonuç yaz, sonuna kadar sessiz kalma.
```

---

## Prompt 4 — Aşama 2: bridge formülasyonu (GPU: ~10 saat)

```
Aşama 1 sonuçları stochastic path'i test etmeyi gerekçelendiriyorsa bridge'i
önce matematik ve 1-D toy distribution üzerinde doğrula. "Bridge fixes it"
önkabulüyle 3-D training başlatma.

--- Görev 1: bridge path ve doğru derivative ---

    x_t = (1-t)x0 + t*x1 + gamma(t)eps,   eps ~ Normal(0,I)
    gamma(t) = sigma_bridge * sin(pi*t)

`sqrt(t(1-t))` kullanma: derivative endpoint'lerde patladığı için bu projede
yasaktır. Path derivative:

    dx_t/dt = x1-x0 + gamma'(t)eps

x0 conditioning'de olduğundan ve

    gamma(t)eps = x_t-(1-t)x0-t*x1

olduğundan tek x1 head ile conditional velocity tahmini:

    x1_hat = f_theta(x_t,t,cond)
    b_hat(x,t) = x1_hat-x0
                 + gamma'(t)/gamma(t)
                   * [x_t-(1-t)x0-t*x1_hat]

gamma(t)=sigma*sin(pi*t) için:

    gamma'(t)/gamma(t) = pi*cot(pi*t)

Bu formülü unit test ile sembolik/numerik finite-difference derivative'a karşı
doğrula. t=0 ve t=1'de doğrudan bölme yapma; endpoint kurallarını açık fonksiyon
olarak tanımla ve test et. Kör epsilon clamp kullanma.

--- Görev 2: x1 parameterization ve auxiliary loss ---
Ağ saf endpoint `x1_hat` tahmin eder; ana kayıp `||x1_hat-x1||²` ve uygun
weighting'dir. Mevcut velocity formundaki

    x1_hat = x_t + (1-t)v

büyük t'de çoğunlukla GT karışımını puanlar ve auxiliary gradient'i `(1-t)` ile
söner. Direct x1 parameterization bu karışımı kaldırmalı. Narrow-band,
soft-Dice ve clDice yalnızca modelin saf `x1_hat` çıktısına uygulanır.

--- Görev 3: sampler'lar ---
Default sampler ancestral bridge refinement'tır. Her adımda:

  1. Model `x1_hat` tahmin eder.
  2. Sonraki state yeni bridge noise ile kurulur:

         x_{t_next} = (1-t_next)x0 + t_next*x1_hat
                      + gamma(t_next)eps_next

  3. t_next=1 olduğunda gamma(1)=0 ve çıktı doğrudan x1_hat olur.

Noise seed/provenance kaydet. ODE sampler yukarıdaki `b_hat` ile yalnızca
ablation olarak kalsın; ancestral ve ODE sonuçları ayrı raporlansın.

--- Görev 4: zorunlu 1-D known-distribution toy gate ---
3-D kodundan bağımsız küçük conditional MLP deneyi yaz:

    x0 ~ Normal(-2, 0.5²)
    x1 ~ Normal( 2, 1.0²)

Train/test seed sabit olsun. t=1 sampled marginal'i hedef Normal(2,1) ile
karşılaştır; ancestral ve ODE için ayrı ayrı:
  - KS distance,
  - mean error,
  - variance error,
  - quantile error
raporla. Acceptance threshold'larını deney sonucunu görmeden test/config içinde
tanımla. Her iki sampler sonucu görünür olsun; toy gate geçmeden 3-D bridge
training BAŞLATMA.

--- Görev 5: train/inference patch dağılımı uyumu ---
Eğitimde fg_prob=0.8, inference'ta tüm hacim taranıyor. İki düzeltme config flag:
  a) `bg_patch_frac: 0.2` — kanal içermeyen patch'leri bilerek eğitime kat.
     Bu patch'lerde hedef v ≡ 0 (x1 == x0 == satüre +1), yani model "dokunma"yı
     öğreniyor.
  b) sliding_window.py'de gaussian blend'i düzelt: `np.maximum(wsum, 1e-6)`
     yerine, ağırlığı belirli bir eşiğin altında kalan voxelleri hiç yazma
     (mask'le) ve o bölgeleri +1 (dış) SDF ile doldur. Şu anki hali köşelerde
     işaret hatası üretebiliyor.

Her ikisi için önce tamamen-background inference testi yaz.

--- Görev 6: laterality'yi gerçekten bağla ---
train.py total_loss'a lateral_coord'u geçmiyor. conditioning.lateral_axis_index()
ile affine'den ekseni belirle (array index varsayma), ilgili koordinat kanalını
datasets.py'den patch'le birlikte döndür, train.py'de total_loss'a geçir.
laterality_coord_weight'i 0.1 yap. Seçilen ekseni ve işaretini run manifest'ine
LOGLA — yanlışsa deney sessizce bozulur.

Ayrıca mevcut overlap terimi 96³ patch'te ölü (iki kanal aynı patch'e girmiyor);
bunu koda yorum olarak yaz; aktif laterality mekanizmasını ayrı ablate et.

--- Görev 7: fold 0 comparison ---
Rectified, ancestral bridge ve bridge-ODE sampler'ı aynı conditioning, capacity,
budget ve checkpoint contract ile karşılaştır. B4 deterministic direct refiner
satırını da tabloda koru. outputs/ablation_stage2.md yaz.

--- Acceptance criteria ---
- 1-D toy gate ve endpoint derivative testleri yeşil olmadan 3-D run yok
- Tüm testler yeşil, özellikle boş-hacim ve endpoint testleri
- Ancestral/ODE/rectified/direct-refiner aynı evaluation contract ile ölçülmüş
- Safe eligibility weighted score yerine predeclared lexicographic gate ile
  raporlanmış
- eksen seçimi manifest'te loglanmış
```

---

## Prompt 5 — Aşama 3: eğri-uzayı flow prototipi (GPU: ~20 saat, ana katkı adayı)

```
Bu, voxel-SDF flow'una alternatif bir kol. Paralel yürütülebilir; Aşama 2'yi
bloklamaz. Yeni klasör: curveflow/ — mevcut flow/ koduna DOKUNMA.

Hipotez: birçok IAC tek bir 1D ana eğri + yarıçap profiliyle yaklaşık temsil
edilebilir. Ancak bifid/accessory anatomy bu varsayımı ihlal edebilir ve
longest-path işlemi gerçek bir dalı silebilir. Önce fizibiliteyi ölç; sonra düşük
boyutlu şekil uzayında flow düşün:

  taraf başına: K=20 B-spline kontrol noktası (K x 3) + K yarıçap = 80 boyut
  iki taraf: 160 boyut

160-D temsil hesap açısından caziptir; fakat rasterisation, self-intersection,
negative/small radius veya failed fit connectivity'yi bozabilir. Betti-0 dahil
tüm topoloji rasterised output üzerinde ölçülür. Sampling'in anlamlı anatomik
varyasyon üretip üretmediği ayrıca test edilir.

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
  - K=10/20/40 için tüm vaka dağılımını raporla: round-trip Dice median,
    quartile ve worst-10; centerline distance; endpoint error; HD95;
    rasterisation sonrası Betti-0; self-intersection; minimum radius; failed-fit
    rate. Tek bir mean sayı fizibiliteyi belirlemek için yeterli değil.

--- Görev 1b: bifid/accessory anatomy audit ---
Branched/bifid olabilecek vakaları component/skeleton graph ölçüleri ve manuel
review listesi için flag et. Bifid ve non-bifid sonuçları ayrı raporla.
Longest-path nedeniyle silinen GT branch uzunluğunu fiziksel mm olarak ölç.
Tek-spline temsil bifid grupta başarısızsa bunu ana katkı olarak ileri sürme;
branched representation veya bu kolu bırakma kararı ver.

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
- K=10/20/40 dağılımları ve worst-10 outputs/curveflow_representation.md'de
- Bifid/non-bifid ayrı sonuç ve silinen branch uzunluğu mevcut
- fold 0'da end-to-end bir sonuç, prior ile karşılaştırmalı
- Rasterised Betti-0, self-intersection, radius ve failed-fit ölçülmüş

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

--- Görev 2: NFE ablation ---
Her stochastic/ODE flow için NFE ∈ {1,2,4,8} ölç:
  - aynı validation vaka ve seed'lerinde kalite metrikleri,
  - case ve volume başına runtime,
  - peak GPU memory.

NFE=1 flow ile B4 direct refiner aynı model değildir; ayrı satırlar olarak
karşılaştır. B4'ün ODE/sampling kullanmadığını manifest'te doğrula.

--- Görev 3: uncertainty ve baselines ---
Generatif olmanın gerekçesi tek deterministik örnek değil, dağılım olabilir;
fakat flow variance'ı tek baseline değildir. Şunları aynı calibration contract
ile karşılaştır:
  - nnU-Net **true-softmax** entropy (hard one-hot değil),
  - test-time augmentation variance,
  - mümkünse MC dropout veya küçük ensemble,
  - naive physical morphological dilation envelope,
  - flow/bridge sample variance.

Deterministik run headline prediction; stokastik sampling için K=16 başlangıç
noktası olabilir. K'yı klinik yeterlilik olarak sunma. K artarken probability,
entropy, coverage-volume curve ve per-voxel variance stabilitesini ölç; Monte
Carlo stability raporla. Global noise field patch'ler arasında bir defa
örneklenmeli; seam artefaktını test et.

Üret:
  - per-voxel foreground olasılığı (K örneğin ortalaması)
  - varyans / entropi haritası
  - "uncertainty envelope" / "candidate safety margin": olasılık > tau seviye
    kümesi. Klinik safety claim'i ayrı doğrulama olmadan kurma.

Kalibrasyon değerlendirmesi:
  - reliability diagram + ECE
  - zarfın GT'yi kapsama oranı vs zarf hacmi eğrisi; naif dilatasyon baseline'ı
    ile karşılaştır (aynı kapsama için daha küçük hacim = daha iyi)

--- Görev 4: istatistik ---
evaluation/evaluate_cv.py'yi tüm modeller için koştur. compare_bootstrap ile
paired bootstrap %95 CI: her metrik için (flow - prior). Dice'ta non-inferiority,
topoloji metriklerinde superiority iddiası kuruyoruz — CI'ları buna göre raporla.
Çoklu karşılaştırma düzeltmesi uygula ve hangisini kullandığını yaz.

--- Görev 5: locked external scanner-shift evaluation ---
52-case S scanner cohort'u yalnızca development configuration, checkpoint
contract, non-inferiority margin ve tüm thresholds kilitlendikten sonra bir kez
aç. Debugging, tuning, model/qualitative case/threshold selection için kullanma.
External sonuç başarısız olsa da raporla; sonucu gördükten sonra development'a
dönüp yeni model seçme.

--- Görev 6: makale artefaktları ---
paper/ klasörü altında, hepsi script'ten üretilen (elle çizilmiş hiçbir şey yok):
  - table1_main.tex: prior / voxel-flow / curve-flow x
    {Dice, HD95, NSD, clDice, Betti-0, gap_mm, false-branch_mm, swap_rate},
    mean ± std, en iyi kalın
  - table2_ablation.tex: Aşama 1 ve 2'nin tüm run'ları
  - fig1_shortcut.pdf (Aşama 0.5'ten)
  - fig2_qualitative.pdf: 3 case — prior'ın koptuğu, flow'un bağladığı;
    prior'ın iyi olduğu; ikisinin de battığı (failure case'i GİZLEME)
  - fig3_uncertainty.pdf: bir case'de probability map + uncertainty envelope
  - reproduce.sh: sıfırdan tüm tabloları üreten tek komut

--- Acceptance criteria ---
- reproduce.sh temiz bir checkout'ta koşuyor
- Her tablo/figür'ün altında hangi run hash'lerinden üretildiği yazıyor
- Failure case'ler raporlanmış
- NFE quality/runtime/memory ablation ve B4 comparison mevcut
- Uncertainty baseline'ları ile Monte Carlo stability raporlanmış
- S-cohort manifest'i final lock zamanını ve tek açılış provenance'ını içeriyor
```

---

## Kısa notlar

**Sıra önemli.** Prompt 3'te tek plumbing stop koşulu, update öncesi zero-init
epoch-0 modelin identity prior'ı tolerance içinde üretememesidir. B0 training
prior altına düşerse B1/B2 devam eder; bu sonuç onların gerekçesidir. B3 yalnızca
minimal müdahaleler sinyal verirse çalışır. B4 final kontrol olarak zorunludur.

**Prompt 2 yüksek getirili diagnostiktir.** Figure 1 adayıdır; nedensel CBCT
ablasyonu ana probe, t-loss profili destekleyici probe'dur. Aşama 1'e geçmeden
koştur.

**Prompt 5 bağımsız fizibilite koludur.** Tek-spline temsili bifid/accessory
anatomiyi kaybediyorsa ana katkı olarak sunulmaz.

**Her prompt'un sonuna ekleyebileceğin ortak kuyruk:**

```
Uzun run başlatmadan önce mutlaka 3-5 epoch'luk smoke test yap (küçük patch,
2 case) ve checkpoint + --resume'un çalıştığını doğrula. Oturumun her an
kopabileceğini varsay. Kapsam dışı bir dosyaya dokunman gerekiyorsa dur ve
bana sor. Her görev ayrı commit.
```
