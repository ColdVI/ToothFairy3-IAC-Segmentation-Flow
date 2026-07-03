# Proje: ToothFairy — Sol/Sağ IAC Segmentasyonu → Flow Matching

Bu dosya Claude Code'un proje kapsamını hızlıca anlaması içindir. Kısa ve güncel tut.

## Amaç
CBCT hacimlerinde **sol ve sağ Inferior Alveolar Canal (IAC)** segmentasyonu.
Kısa vadeli hedef: nnU-Net baseline. Nihai hedef: **flow matching** tabanlı
generatif segmentasyon (SEAL-Flow / MedFlowSeg / FlowSDF tarzı), **SDF çıktı**
formatı ile. Yayınlanmış tüm ToothFairy yaklaşımları diskriminatif; flow-based
alan boş — araştırma boşluğu burada.

## Dataset — KRİTİK
- **ToothFairy3** kullanılıyor (ToothFairy4 DEĞİL). TF4 = ODIN 2026 CBCT→rapor
  üretme görevi, segmentasyon değil. IAC voxel etiketleri TF2/TF3'te.
- Format: nnU-Net v2, NIfTI, RPI oryantasyon (eğitim için reorient gerekmez).
- Kaynak etiket id'leri: `3 = Left IAC`, `4 = Right IAC` (dataset.json ile teyit et).
- Çıktı dataset: `0=bg, 1=left_IAC, 2=right_IAC` (bkz. prepare_toothfairy3_iac.py).

## Domain kuralı — UNUTMA
**Sagittal (sol/sağ) mirroring augmentasyonu KAPALI olmalı.** Sol/sağ ayrımını
bozuyor; CVPR2025'te DSC 74->85'e mirroring kapatınca çıktı. Bkz.
nnUNetTrainerIAC.py (nnUNetTrainerIAC_NoMirror).

## Kod haritası
- prepare_toothfairy3_iac.py — TF3 çok-sınıflı → 3-sınıflı IAC dataset + dataset.json
- nnUNetTrainerIAC.py — mirroring kapalı custom trainer'lar (+ kısa-epoch varyantlar)
- (yarın gelecek) mevcut kod ofisten eklenecek — geldiğinde buraya entegre et

## Eğitim (nnU-Net v2)
    export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
    nnUNetv2_plan_and_preprocess -d 111 --verify_dataset_integrity
    nnUNetv2_train 111 3d_fullres 0 -tr nnUNetTrainerIAC_NoMirror

## Değerlendirme metrikleri
DSC, HD95, IoU, NSD; ince tübüler yapı olduğu için clDice / topolojik tutarlılık
(SEAL-Flow motivasyonu) önemli.

## Flow-matching referansları (ileride)
MedFlowSeg, FlowSDF, LatentFM, CurvSegFlow, PolypFlow, FMS2. Doğrusal
interpolasyon binary maskeye uymadığı için **SDF gösterimi** tercih edilir;
3D voxel-uzayında doğrudan flow bellek-yoğun → latent/koşullu yaklaşım.

## Konvansiyonlar
- Türkçe yorum/iletişim tercih edilir.
- Yeni deneyleri kısa smoke-test (50ep) ile doğrula, sonra tam eğitim.
