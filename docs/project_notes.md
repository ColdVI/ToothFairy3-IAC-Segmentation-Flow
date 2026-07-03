# Project notes — research context + v1.0 decisions

Ham/uzun browser hafızası `archive/MEMORY.md`'de. Bu dosya güncel, damıtılmış özet.

## Problem
CBCT'de sol/sağ Inferior Alveolar Canal (IAC) 3D segmentasyonu (danışman: Şaban Hoca).
Çekirdek iddia: **topolojik tutarlılık** (yalnız voxel overlap değil). Literatürdeki
tüm ToothFairy yaklaşımları diskriminatif; flow-based generatif yön açık.

## ToothFairy serisi (doğrulanmış, 2026)
- TF1 (TMI/MICCAI 2023): tek birleşik IAC, 443 CBCT.
- TF2 (CVPR 2025 + MIA 2026): 42 sınıf, 530 hacim; L/R IAC ayrı (3, 4).
- TF3 (MICCAI 2025 ODIN): 77 sınıf, 532 hacim (P 417 / F 63 / S 52); runtime bir metrik.
- TF4 (MICCAI 2026 ODIN): **rapor üretme** (CBCT→metin), voxel etiketi YOK; kardeş Bite2Text.
  → Segmentasyon için TF3 kullanılıyor. (Eski "TF4 yok" notu geçersiz.)

TF3 atıfları (segmentasyon): Bolelli ve ark. *ToothFairy2 challenge* (MIA 2026);
*Segmenting Maxillofacial Structures* (CVPR 2025); Lumetti ve ark. *Patch-Based
Mandibular Canal* (IEEE Access 2024). TF4 atıfları rapor-üretme/vision-language soyu →
bu projeye teknik olarak uzak.

## v1.0 mimari kararları (koda yansıdı)
- **Residual flow:** `x0 = SDF(nnU-Net)`, `x1 = SDF(GT)`, `x_t=(1-t)x0+tx1`, hedef hız `x1-x0`.
  Refinement; saf gürültüden generatif segmentasyon değil. Bu, ekibin önceki başarısız
  saf-flow denemesinin bellek/temsil sorunlarını (3D voxel-uzayı, binary hedef) aşar.
- **Leakage-free:** flow yalnız out-of-fold nnU-Net tahminleriyle eğitilir; aksi halde
  aşırı-temiz (in-sample) maskelerden öğrenir.
- **Fiziksel SDF (mm):** anizotropik `distance_transform_edt(sampling=spacing)`; io_utils tek kaynak.
- **8-kanal conditioning:** CBCT + OOF prob(L,R) + coarse SDF(L,R) + fiziksel x/y/z (affine'den).
  xyz laterality/global-bağlam için (patch tabanlı segmentasyonda konum bilgisi faydası — ICPR 2024).
- **Kayıplar (endpoint üzerinde):** FM + narrow-band (sıfır-seviyesi ağırlıklı) + soft-clDice(3D,
  bağlantısallık) + laterality (L/R çakışma). TV yalnız smoothness, topoloji değil. Endpoint loss
  ilk sürümde zorunlu değil, ayrı ablation.
- **Checkpoint:** S_val = 0.5·Dice + 0.5·clDice (eşitlikte düşük HD95); train loss'a göre değil.
- **Inference:** deterministik (sigma=0, Heun 8 adım, Gaussian blend). Belirsizlik: tek global
  noise field'dan patch crop → seed'ler arası mean/variance/entropy.

## nnU-Net (Track A)
- A0 stock / A1 tüm mirroring kapalı / A2 yalnız fiziksel L/R kapalı. A2'de eksen affine +
  plans transpose ile çözülür ve **loglanır** (yanlışsa deney sessizce bozulur; `IAC_LR_AXIS` override).
- Postprocess connected-component "en büyüğü tut" diye sabit DEĞİL; min hacim / ana-bileşene
  uzaklık CV'de tune edilir; ham + post-processed ayrı raporlanır.

## SEAL-Flow (referans, kaynaktan doğrulandı)
İki aşamalı, **2D** (MoNuSeg/CVC/GlaS). Stage-1 UNet prior + EFD multi-frekans SDF;
Stage-2 SegDiff-tarzı flow (interval-average/MeanFlow, Voronoi bariyer, spektral reg.).
Peer-review'da; bazı bölümler withheld → mimari referans. Bizimki bağımsız **3D residual** yaklaşım.

## Yakın-akraba referanslar
- **FlowSDF** (Bogensperger, IJCV 2025, arXiv 2405.18087) — SDF üzerinde koşullu flow + belirsizlik.
- **clDice** (Shit ve ark., arXiv 2003.07311) — differentiable topoloji-koruyan kayıp (kullandığımız).
- MedFlowSeg (2026), CurvSegFlow, LatentFM; temeller: Lipman flow matching, Liu rectified flow.

## Değerlendirme
Her metrik L/R ayrı sonra bilateral ortalama: Dice, HD95(mm), clDice, NSD + topoloji
(bileşen sayısı, Betti-0 hatası, centerline gap / false-branch mm, L/R swap, empty rate).
Model karşılaştırması: vaka-bazlı paired bootstrap CI (+ Wilcoxon). CV (P+F) → sonra S (OOD, bir kez).

## Başarı kriteri
CV'de clDice/bağlantısallık tutarlı ↑, HD95 ↓, Dice belirgin düşmesin, L/R swap artmasın,
S(OOD)'de baseline'a göre daha iyi/dayanıklı. Dice sabit kalıp topolojik kopukluk azalırsa
yine başarı — asıl iddia topolojik tutarlılık.
