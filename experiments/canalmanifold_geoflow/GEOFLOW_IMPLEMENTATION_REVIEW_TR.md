# GeoFlow implementasyon ve fikir incelemesi

## Kısa hüküm

Yüklenen ZIP, GeoFlow eklenmeden önceki CanalManifoldFlow v2 ağacıdır.
Yanında verilen `cli.py` ve `evaluate_v2_exact.py`, pakette bulunmayan
`geoflow_model.py` ile `geoflow_train.py` dosyalarını import ettiği için tek
başına çalışabilir bir sürüm oluşturmuyordu. GeoFlow notebook'u da kendi
çağırdığı üç arayüzle uyumsuzdu.

Geometric Boundary Flow fikrinin korunabilir çekirdeği şudur: frozen nnU-Net
sınırından başlamak, sınırsız-rank `h(s,theta)` yüzeyini mm cinsinden taşımak,
koşulu hareket eden yüzeyde yeniden örneklemek ve sonucu exact-identity
`phi0-H` decoder ile geri yazmak. Buna karşılık eski
`crossing-(q0+h)` skalerinin kapalı-çevrim görüntü kanıtı olduğu iddiası doğru
değildi.

Bu pakette yeni primary temsil `geoflow_newton` olarak ayrı namespace'e alındı.
Eski `geoflow` veya R2 checkpoint'i sessizce resume edilemez.

## Doğrulanan implementasyon sorunları

1. **Kayıp çekirdek dosyalar:** ZIP'te `geoflow_model.py`,
   `geoflow_train.py` ve `test_geoflow_contracts.py` yoktu.
2. **CLI yalnız yarım bağlanmıştı:** dışarıdaki `cli.py`, bulunmayan eğitim
   modülünü çağırıyordu.
3. **GeoFlow-only exact evaluation bozuktu:** evaluator,
   `--geoflow-checkpoint` verilmiş olsa bile tube ve surface checkpoint'leri
   yoksa hata veriyordu.
4. **Cross-fold kilidi GeoFlow'u tanımıyordu:** `lock_crossfold_epoch.py`
   yalnız `linear` ve `surface_h` kabul ediyordu.
5. **Cross-fold epoch doğrulaması yoktu:** exact CSV'de
   `geoflow_checkpoint_epoch` yazılmadığı için kilitli epoch kanıtlanamıyordu.
6. **Notebook yanlış dosyayı okuyordu:** evaluator `metrics.csv` yazarken
   notebook `per_case.csv` bekliyordu.
7. **Notebook yanlış report arayüzünü çağırıyordu:** mevcut script
   `--lock`, tekrarlı `--metrics FOLD=PATH` ve `--method` beklerken notebook
   `--run-root/--pattern` veriyordu.
8. **Drive yolları doğrulanmamıştı:** Drive ağacı doğrudan kontrol edildi.
   Proje `ToothFairy/ToothFairy3/CanalManifoldFlow_v2_GeoFlow_Newton`, OOF
   olasılıkları ise `ToothFairy3/iac_runs/canalmanifold_oof_softmax` altında.
   Dataset ve split yolları sırasıyla
   `iac_runs/dataset_cache_colab_v2/Dataset801_IAC_LR` ve
   `iac_runs/configs_cache/splits.json` olarak doğrulandı.

## Matematiksel inceleme

### 1. Eski evidence neden dejenereydi?

`crossing` sabit shell'in bütün radyal profilinden hesaplanıyordu. Yüzey durumu
yalnız çıkarma işleminde giriyordu:

```text
evidence(h) = crossing(shell) - (q0 + h) = constant - h
```

Bu nedenle `evidence+h` her `h` için aynıydı. Hareket sırasında evidence'ın
değişmesi, görüntünün yeni bir şey söylemesi değil, `-h` aritmetiğiydi.
`sampled` profil tensörü ise gerçekten `q0+h+offset` konumlarından okunuyordu;
kapalı çevrimin korunabilir kısmı buydu.

Ek nüans: `q0_ray_radii_mm`, geçerli ışınlarda 64-noktalı `p=0.5` geçişidir.
Geçersiz ışınlarda harmonik fallback ile tamamlanır. Dolayısıyla
`evidence(h=0)≈0` cümlesi yalnız geçerli geçişler için doğrudur.

### 2. Uygulanan düzeltme

Yeni sampler, güncel yüzey çevresindeki `-0.3, 0, +0.3 mm` örneklerinden

```text
slope = (p_plus - p_minus) / 0.6              [probability/mm]
newton = (0.5 - p_at_surface) / slope          [mm]
sharpness = max(-slope, 0)                     [1/mm]
```

hesaplar. Düz/yükselen, radial-grid dışında kalan veya sayısal olarak geçersiz
profiller maskelenir; Newton adımı `±0.60 mm` ile sınırlandırılır. Doğrudan
`slope.clamp(max=-1e-3)` kullanılmadı: pozitif bir eğimi yapay biçimde negatif
eğime çevirmek büyük ve yanlış Newton adımı üretirdi.

`sharpness`, kalibre edilmiş epistemik/aleatorik güven değildir. CBCT
partial-volume, çözünürlük ve nnU-Net olasılık profilinin ortak yerel sınır
kuvvetidir. Ayrıca profil doğrusal değilse `h` ile değişir; “h'den tamamen
bağımsız” diye yazılmamalıdır.

### 3. Hız tabanının identifiability düzeltmesi

Serbest `a(s)` ve serbest `beta(s,theta)` bırakılırsa uniform dilation yalnız
`alpha` içinde yaşamaz: sabit `a` veya sabit beta da aynı modu temsil edebilir.
Yeni hız tabanı bu gauge'i kaldırır:

```text
V = alpha
  + a_zero_mean(s)
  + [beta(s,theta) * r_ref/r_t]_zero_angle_mean
  + gain(s,theta) * newton_local(s,theta)
```

- `a` geçerli istasyonlarda sıfır ortalamalıdır.
- curvature/beta katkısı her istasyonda açısal sıfır ortalamalıdır.
- Bu iki öğrenilmiş residual kolun global ortalaması sıfır olduğundan, görüntü
  kanıtı dışındaki global şişme modu `alpha` ile izlenir.
- Evidence katkısının global ortalaması ayrıca loglanır; görüntüye bağlı bir
  global hareket `alpha` diye yanlış etiketlenmez.

### 4. Velocity drift neden başarı kapısı değildir?

Kullanılan supervised bridge

```text
h_t = h0 + t*Delta + t(1-t)*epsilon
u_t = Delta + (1-2t)*epsilon
```

şeklindedir. Temiz doğrusal yolda (`epsilon=0`) doğru hedef hız sabittir:
`u_t=Delta`. Bu yüzden iyi bir modelde `V(t=0)` ve `V(t=1)` benzer olabilir.
Yüksek `velocity_drift`, kapalı çevrim kullanımı kadar model hatası veya
osilasyon da gösterebilir.

Yeni log ayrımı:

- `val_velocity_drift`: yalnız betimsel hız değişimi;
- `val_profile_remeasurement_rms`: başlangıç ve terminal yüzeylerinde okunan
  `p_at_surface` farkı;
- `val_evidence_gain`, `val_newton_abs_mm`, `val_profile_valid_fraction`:
  feedback kanalının büyüklüğü ve kapsaması;
- exact Dice/HD95/clDice/component farkları: gerçek yöntem sonucu.

Makalede “entegrasyon adımları zorunludur” demek için ayrıca NFE ablation
(`1/2/4/8` Heun adımı) ve hareketli-profil ablation gerekir. Sadece drift bu
iddiayı kanıtlamaz.

### 5. Kalan temsil sınırı

`h(s,theta)` harmonik rank tavanını kaldırır fakat Bishop kesitlerinin
star-shaped olması varsayımını kaldırmaz. `phi0-H` için `h=0` identity exact'tir;
voxel topolojisi garanti değildir. Bu nedenle component, clDice ve kopukluk
metrikleri korunmalıdır.

İki endpoint hâlâ `q1_global[1:3]` içindeki tüp-fit hedefinden gelir; serbest
ray tablosu endpoint'i ayrıca öğrenmez. Bu sınır zaten
`q1_free_surface_ceiling_dice` ölçümünün içindedir fakat “tamamen tavansız
temsil” denmemelidir: harmonik radial rank tavanı kalkmıştır, star-shaped chart
ve endpoint-fit tavanı kalmıştır.

## Yeni/yenilenen dosyalar

- `canalmanifold/geoflow_model.py`: moving local-profile sampler,
  Newton+sharpness, axial-curvature feature, gauge-fixed geometric basis.
- `canalmanifold/geoflow_train.py`: area-weighted flow matching, bridge terminal
  loss, Heun rollout, trust region, resume-safe checkpoint ve diagnostics.
- `scripts/test_geoflow_contracts.py`: dejenerasyon, zero-init, gauge, area
  weight, trust region ve terminal formül sözleşmeleri.
- `canalmanifold/cli.py`: `train-geoflow`.
- `scripts/evaluate_v2_exact.py`: GeoFlow-only çalışma, exact metrics ve vaka
  diagnostics.
- `scripts/lock_crossfold_epoch.py`, `report_crossfold.py`: gerçek locked-epoch
  GeoFlow raporu.
- `configs/geoflow_newton_colab.example.yaml`: corrected config.
- `notebooks/CanalManifoldFlow_GeoFlow_Newton_Colab.ipynb`: daha önce çözülen
  Drive yollarını önceliklendiren tam eğitim akışı.

## Çalıştırma sırası

Colab notebook'u sırayla şunları yapar:

1. Drive yollarını çözer ve üç contract testini çalıştırır.
2. Legacy cache'i CBCT okumadan v2 cache'e yükseltir/var olanı kullanır.
3. Shard'ları Colab local diske kopyalar; checkpoint ve logları Drive'a yazar.
4. Oracle'ı eğitimden bağımsız discussion teşhisi olarak çalıştırır.
5. Fold 0 `geoflow_newton` eğitimini ve exact değerlendirmeyi çalıştırır.
6. Epoch'u fold 0'da kilitler.
7. Kullanıcı açtığında fold 1-4'ü yalnız locked epoch'a kadar eğitir, dört exact
   `metrics.csv` dosyasını tek primary cross-fold raporunda toplar.

## Sonuçların doğru yorumu

İlk gerçek koşuda aşağıdakiler birlikte okunmalıdır:

- `val_soft_dice`, `val_rmse_mm`: seçim-fold proxy'leri;
- `val_alpha_mm`: uniform learned dilation kısayolu;
- `val_profile_remeasurement_rms`: yüzeyin gerçekten başka konumu okuması;
- `val_profile_valid_fraction`: Newton kanalının ne kadar yüzeyde geçerli olduğu;
- exact `raw_pp -> gbf_newton_normal_pp` Dice/HD95/clDice/component farkı;
- fold 1-4 locked-epoch bootstrap CI ve fold yön tutarlılığı.

`profile_remeasurement_rms>0` yöntemin faydalı olduğunu değil, yeniden örnekleme
mekanizmasının çalıştığını gösterir. Akademik başarı iddiası yalnız exact ve
cross-fold sonuçtan çıkarılmalıdır.
