# CanalManifoldFlow v2 — Implementasyon notları

## Bu sürümde gerçekten implemente edilenler

- `scripts/oracle_headroom.py`: O0–O6 headroom ayrıştırması; train-only global
  threshold ve temperature scaling, vaka/side oracle'ları, OOF-probability
  Bayes-Dice kararı, `m≤8` ve free-surface ceiling.
- `canalmanifold/constants.py`, `tube_state.py`: `m=2..8`, `160×17+3=2723`
  R1 state'i; dinamik harmonik fit/loss/decode.
- `canalmanifold/surface_model.py`, `surface_train.py`: rank kısıtsız
  `h(s,θ)∈R^(160×32)` R2 linear flow ve iki endpoint.
- `canalmanifold/model.py`, `surface_model.py`: R1/R2 checkpoint uyumluluğu için
  korunan `crossing-current` kanalı. Bu kanal yeni görüntü kanıtı değil,
  cebirsel bir geri çağırma terimidir.
- `canalmanifold/geoflow_model.py`, `geoflow_train.py`: hareketli yüzeyde yerel
  Newton düzeltmesi + radyal eğim kuvveti, gauge-sabitlenmiş geometrik hız
  tabanı, alan-ağırlıklı mm loss, Heun rollout ve ayrı yeniden-ölçüm teşhisleri.
- `canalmanifold/losses.py`: terminal state üzerinde differentiable polar-shell
  soft-Dice; GT occupancy yalnız supervision tarafında.
- `canalmanifold/paths.py`, `train.py`: uçlarda sıfırlanan exact path-noise,
  q0-jitter, terminal reconstruction ve checkpoint contract kontrolü.
- `canalmanifold/displacement.py`: raw prior'a cebirsel identity sağlayan
  `φ0−H` normal-displacement decode ve fiziksel hacimli küçük-component filtresi.
- `canalmanifold/dense_sdf.py`: aynı prior-coupled linear path, band-weighted
  loss ve terminal Dice kullanan R3 dense-SDF comparator.
- `scripts/evaluate_v2_exact.py`: raw/raw-PP/R1-tube/R1-normal/R2-normal için
  Dice, HD95, clDice, component, volume, patient bootstrap ve paired sign test.
- `scripts/lock_crossfold_epoch.py`, `report_crossfold.py`: epoch'u bir fold'da
  kilitleyip yalnız diğer dört fold'u primary evidence olarak toplama.
- `scripts/safety_envelope.py`: R2 için istasyon/açı bazlı radius quantile zarfı,
  empirical coverage ve zarf genişliği.
- `scripts/upgrade_cache_v2.py`: eski 960 shard'ın image shell'ini koruyup CBCT
  okumadan m≤8/GT-occupancy/free-surface alanlarına yükseltme.
- `notebooks/CanalManifoldFlow_v2_Colab.ipynb`: tarihsel R1/R2 oracle-gated akış;
  corrected GeoFlow için kullanılmamalı.
- `notebooks/CanalManifoldFlow_GeoFlow_Newton_Colab.ipynb`: oracle'ı kapı
  yapmayan, Drive yollarını çözen ve locked fold 1-4 raporunu doğru çağıran
  canonical eğitim akışı.

## Yapılan doğrulamalar

Veri-bağımsız testler:

- state boyutu ve m≤8 harmonik sözleşmesi;
- R1/R2/R3 zero-head identity initialization;
- corrected GeoFlow zero-head identity, yerel profil dejenerasyon testi,
  `alpha/a/beta` gauge sözleşmesi, alan ağırlığı ve trust-region sözleşmesi;
- noisy linear path'in analitik terminal eşitliği;
- decoded soft-Dice'ın finite gradient üretmesi;
- `h=0` için `φ0−H` decoder'ın raw maskeyi bit-for-bit döndürmesi;
- fiziksel küçük-component filtresi;
- notebook JSON ve YAML konfigürasyon geçerliliği.

Sentetik 3B iki-kanal vaka üzerinde geçen uçtan uca adımlar:

1. `precompute` → iki L/R v2 shard;
2. `TubeCacheDataset` + right-chart dönüşümü;
3. R1 linear ve R2 surface flow için birer CPU epoch + archived checkpoint;
4. exact raw/R1/R2 evaluation;
5. R3 dense cache, bir epoch ve identity-anchored exact evaluation;
6. O0–O6 oracle;
7. legacy→v2 cache upgrade;
8. safety envelope;
9. epoch lock ve cross-fold report araçları.

Bu sentetik sayılar akademik sonuç değildir; yalnız yürütme ve sözleşme
doğrulamasıdır. Gerçek 480-vaka TF3 ölçümleri mevcut Drive cache/OOF/label
dosyaları üzerinde Colab notebook'u ile üretilmelidir.

## Bilinçli sınırlamalar

- Eski `m≤4` checkpointleri v2 state boyutuyla uyumlu değildir; yeniden eğitim
  gerekir. Eski cache ise upgrade edilebilir.
- `φ0−H` continuous normal-coordinate argümanının voxel yaklaşımıdır. Identity
  exact'tir; fakat voxel topolojisi garanti diye ilan edilmez, component metriği
  ile ölçülür.
- O4 tüm CBCT bilgisinin teorik Bayes tavanı değildir; kalibre frozen OOF
  probability map'in desteklediği karar sınırıdır.
- Safety-envelope başlangıç jitter'ı açık bir deneysel sampling law'dur. Ölçeği
  yalnız selection fold'da ayarlanmalı, coverage kilitli fold'larda ölçülmelidir.
- Set C, fold-0 seçilmiş checkpoint ile açılmamalıdır. Önce epoch lock + fold
  1–4 tutarlılığı gerekir.

## İlk gerçek koşu

Colab'da sırayla:

1. `scripts/smoke_test.py`, `scripts/test_v2_contracts.py` ve
   `scripts/test_geoflow_contracts.py`;
2. `scripts/upgrade_cache_v2.py`;
3. `scripts/oracle_headroom.py`;
4. oracle'dan bağımsız `train-geoflow` (oracle yalnız discussion teşhisi);
5. corrected GeoFlow exact fold-0 teşhisi;
6. epoch lock;
7. fold 1–4 locked-epoch evaluation;
8. pozitif ve fold-tutarlı sonuç varsa R3/OOD/Set C.
