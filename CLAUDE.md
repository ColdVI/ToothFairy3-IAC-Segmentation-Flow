# CLAUDE.md — proje kapsamı (kısa ve güncel tut)

Claude Code'un projeyi hızlıca anlaması için. Uzun araştırma bağlamı
`docs/project_notes.md`'de; ham browser hafızası `archive/MEMORY.md`'de.

## Amaç (v1.0)
CBCT'de **sol/sağ IAC** segmentasyonu, iki track:
- **Track A:** nnU-Net v2 baseline (3 sınıf: 0=bg, 1=Left, 2=Right).
- **Track B:** **Conditional Residual Flow Matching** — flow SAF GÜRÜLTÜDEN değil,
  **nnU-Net çıktısından** başlar. `x0 = SDF(nnU-Net)`, `x1 = SDF(GT)`, model residual
  hızı öğrenir. **Leakage-free**: flow yalnızca out-of-fold nnU-Net tahminleriyle eğitilir.

## Dataset — KRİTİK
- **ToothFairy3** (TF4 DEĞİL; TF4 rapor üretme, voxel etiketi yok).
- Etiketler **isimle** çözülür; bulunamazsa RAISE (sessiz 3/4 fallback YOK). Bkz.
  `data/prepare_iac_dataset.py`. Çıktı `Dataset801_IAC_LR`, `1=Left, 2=Right`.
- **Split:** development = P+F (aynı cihaz) → 5-fold CV; **S = held-out external OOD test**
  (sadece en sonda açılır). Bkz. `data/create_folds.py` → `configs/splits.json`.

## Domain kuralı — UNUTMA
Sagittal (sol/sağ) mirroring KAPALI. A1=`nnUNetTrainerIAC_NoMirror` (hepsi kapalı),
A2=`nnUNetTrainerIAC_NoLRMirror` (yalnız fiziksel L/R ekseni; eksen loglanır, `IAC_LR_AXIS`
ile override). A1 vs A2 bir deney. TTA mirroring inference'ta kapalı.

## Boru hattı sırası
1. `data/audit_dataset.py` + `data/prepare_iac_dataset.py`
2. `data/create_folds.py` → splits.json
3. `nnunet/run_nnunet_iac.sh` (A0/A1/A2)
4. `nnunet/predict_oof.py` → outputs/oof_probs (leakage-free)
5. `data/compute_gt_sdf.py` + `data/compute_coarse_sdf.py` (mm SDF cache)
6. `flow/train.py --fold f` (residual flow)
7. `evaluation/evaluate_cv.py` → 8. `evaluation/evaluate_external.py` (S, bir kez)

## Flow çekirdeği (koda yansımış kararlar)
- SDF **milimetre** cinsinden (anizotropik spacing), `data/io_utils.py` tek kaynak.
- Conditioning 8 kanal: CBCT, prob_L, prob_R, coarse_SDF_L, coarse_SDF_R, fiziksel x/y/z
  (affine'den, index'ten değil). Flow-state 2 kanal (L/R SDF). Model girişi 10, çıkışı 2.
- Topoloji/sınır kayıpları **endpoint** `x1_hat = x_t+(1-t)v` üzerinde: FM + narrow-band
  + soft-clDice(3D) + laterality (L/R çakışma cezası). TV yalnızca "smoothness", topoloji değil.
- Checkpoint `best.pt` = **0.5·Dice + 0.5·clDice** (validate.py), train loss'a göre DEĞİL.
- Inference deterministik (sigma=0, Heun, 8 adım, Gaussian blend); belirsizlik için global noise.

## Doğrulama
`python flow/selftest.py` (residual makine) + `tests/` (5 suite). GPU/veri gerektiren
adımlar (nnU-Net, SDF cache, tam flow eğitimi) import + unit test ile doğrulandı, uçtan uca değil.

## Konvansiyonlar
- Türkçe iletişim. Yeni deneyi önce selftest/kısa-epoch ile doğrula.
- Kaynakları doğrula (dosya/veri/web); dataset gerçeği ile aktarılan arasında fark çıkabiliyor.
- (yarın gelecek) ofis kodu → ilgili track'e entegre, bu dosyayı güncelle.
