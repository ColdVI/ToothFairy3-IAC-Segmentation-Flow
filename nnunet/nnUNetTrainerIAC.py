# -*- coding: utf-8 -*-
"""
nnUNetTrainerIAC.py
===================
Sol/sağ IAC görevine özel nnU-Net v2 trainer'ları.

NEDEN mirroring kapatılıyor?
  ToothFairy2 çalışmaları (CVPR 2025) sagittal (sol/sağ) mirroring
  augmentasyonunun sol/sağ ayrımını BOZDUĞUNU gösterdi:
      Default nnU-Net ResEnc ............... DSC 74.16 / HD95 14.48
      w/o l/r mirroring .................... DSC 80.79 / HD95 12.37
      w/o l/r mirroring + derinlik ......... DSC 82.11 / HD95 11.86
      + post-processing .................... DSC 84.99 / HD95  8.57
  MICCAI 2024 (ToothFairy2) ilk 3 çözümden 2'si sagittal mirroring'i kapattı.
  Bizim görevimiz TAM OLARAK sol/sağ ayrımı olduğu için mirroring kapatmak şart.

KURULUM:
  Bu dosyayı nnU-Net kurulumundaki şu klasöre kopyalayın:
    nnunetv2/training/nnUNetTrainer/variants/data_augmentation/
  (pip ile kurduysanız: site-packages/nnunetv2/.../variants/data_augmentation/)

  Trainer'ın nnU-Net tarafından görülüp görülmediğini kontrol:
    nnUNetv2_train -h   # hata vermeden trainer adını kabul etmeli

KULLANIM:
    nnUNetv2_train 111 3d_fullres 0 -tr nnUNetTrainerIAC_NoMirror
    # Colab smoke-test (kısa):
    nnUNetv2_train 111 3d_fullres 0 -tr nnUNetTrainerIAC_NoMirror_50ep
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerIAC_NoMirror(nnUNetTrainer):
    """Tüm eksenlerde mirroring augmentasyonunu kapatır."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Colab kopmalarına dayanıklılık: checkpoint_latest.pth'i daha SIK yaz
        # (nnU-Net default 50). --c bundan devam eder; kopmada en fazla ~save_every epoch kaybı.
        self.save_every = 25

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy, initial_patch, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # mirror_axes = None -> hiç mirroring uygulanmaz (train + TTA)
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy, initial_patch, mirror_axes


class nnUNetTrainerIAC_NoMirror_50ep(nnUNetTrainerIAC_NoMirror):
    """Colab'da hızlı deneme için 50 epoch (gerçek eğitim için kullanmayın)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 50
        self.save_every = 5      # kısa koşuda daha da sık checkpoint


class nnUNetTrainerIAC_NoMirror_250ep(nnUNetTrainerIAC_NoMirror):
    """Orta uzunlukta eğitim (kaynak kısıtlıysa iyi bir denge)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 250
        self.save_every = 10
