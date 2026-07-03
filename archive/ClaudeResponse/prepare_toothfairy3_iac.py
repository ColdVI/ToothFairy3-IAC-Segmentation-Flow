#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_toothfairy3_iac.py
===========================
ToothFairy3 (77 sınıflı) datasetinden SADECE sol/sağ Inferior Alveolar Canal
(IAC) sınıflarını çıkarıp, 3 sınıflı (background + left IAC + right IAC) bir
nnU-Net v2 raw datasetine dönüştürür.

Neden ToothFairy3?
  ToothFairy4 (ODIN 2026 / MICCAI 2026) bir SEGMENTASYON değil, CBCT->rapor
  üretme (report generation) görevidir. IAC voxel-level etiketleri
  ToothFairy2/ToothFairy3'te bulunur. Bu yüzden pipeline TF3 üzerine kurulu.

ToothFairy2/3 etiket kuralı (dataset.json içinden TEYİT EDİN):
    0  = background
    1  = Lower Jawbone
    2  = Upper Jawbone
    3  = Left Inferior Alveolar Canal    <-- bize lazım
    4  = Right Inferior Alveolar Canal   <-- bize lazım
    5  = Left Maxillary Sinus
    6  = Right Maxillary Sinus
    ...
    103/104 = Left/Right Incisive Nerve (TF3'te yeni)
    105     = Lingual Nerve (TF3'te yeni)

Çıktı dataset etiketleri:
    0 = background, 1 = Left IAC, 2 = Right IAC

Kullanım:
    python prepare_toothfairy3_iac.py \
        --tf3_images /path/ToothFairy3/imagesTr \
        --tf3_labels /path/ToothFairy3/labelsTr \
        --out_root   /path/nnUNet_raw \
        --dataset_id 111 \
        --dataset_name IAC_LR

    # Önce sadece etiketleri doğrulamak için (dönüştürme yapmadan):
    python prepare_toothfairy3_iac.py --tf3_labels /path/labelsTr --verify_only
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# KONFIG — indirdiğiniz dataset.json ile MUTLAKA karşılaştırın.
# --------------------------------------------------------------------------- #
LEFT_IAC_SRC = 3   # kaynak (TF3) etiketinde Left IAC id'si
RIGHT_IAC_SRC = 4  # kaynak (TF3) etiketinde Right IAC id'si

LEFT_IAC_DST = 1   # çıktı datasetinde Left IAC id'si
RIGHT_IAC_DST = 2  # çıktı datasetinde Right IAC id'si


def find_label_files(labels_dir: Path):
    files = sorted(labels_dir.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"{labels_dir} içinde .nii.gz bulunamadı.")
    return files


def verify_labels(labels_dir: Path, sample: int = 15):
    """
    Rastgele birkaç etiket dosyasında hangi label id'lerinin geçtiğini yazar.
    Sol/sağ IAC id'lerinin gerçekten 3/4 olduğunu bu çıktıyla teyit edin.
    """
    files = find_label_files(labels_dir)
    print(f"[verify] {len(files)} etiket dosyası bulundu. İlk {sample} tanesi kontrol ediliyor...\n")
    global_unique = set()
    for f in files[:sample]:
        arr = np.asarray(nib.load(str(f)).dataobj)
        uniq = np.unique(arr)
        global_unique.update(uniq.tolist())
        has_l = LEFT_IAC_SRC in uniq
        has_r = RIGHT_IAC_SRC in uniq
        print(f"  {f.name:40s} labels={uniq.tolist()[:12]}{'...' if len(uniq) > 12 else ''} "
              f"| L({LEFT_IAC_SRC})={'✓' if has_l else '✗'} R({RIGHT_IAC_SRC})={'✓' if has_r else '✗'}")
    print(f"\n[verify] Örneklerdeki tüm label id'leri: {sorted(global_unique)}")
    print(f"[verify] Beklenen: {LEFT_IAC_SRC}=Left IAC, {RIGHT_IAC_SRC}=Right IAC. "
          f"Uyuşmuyorsa scriptin başındaki KONFIG'i düzeltin.\n")


def remap_label(arr: np.ndarray) -> np.ndarray:
    """Çok-sınıflı etiketi 3-sınıflı (bg / L-IAC / R-IAC) etikete indirger."""
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr == LEFT_IAC_SRC] = LEFT_IAC_DST
    out[arr == RIGHT_IAC_SRC] = RIGHT_IAC_DST
    return out


def match_image_for_label(label_file: Path, images_dir: Path) -> Path:
    """
    Etiket dosyası P001.nii.gz -> görüntü P001_0000.nii.gz eşlemesi.
    TF3 nnU-Net formatında görüntüler _0000 kanal ekiyle gelir.
    """
    case_id = label_file.name.replace(".nii.gz", "")
    candidate = images_dir / f"{case_id}_0000.nii.gz"
    if candidate.exists():
        return candidate
    # bazı sürümlerde ekstra normalizasyon gerekebilir
    alts = list(images_dir.glob(f"{case_id}*.nii.gz"))
    if alts:
        return alts[0]
    raise FileNotFoundError(f"{case_id} için görüntü bulunamadı ({candidate}).")


def build_dataset_json(out_ds: Path, n_training: int):
    dataset_json = {
        "channel_names": {"0": "CBCT"},   # tek kanal; nnU-Net CT normalizasyonu kullanır
        "labels": {
            "background": 0,
            "left_IAC": LEFT_IAC_DST,
            "right_IAC": RIGHT_IAC_DST,
        },
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "description": "ToothFairy3'ten türetilmiş sol/sağ IAC segmentasyonu "
                       "(SEAL-Flow tarzı flow-matching çalışmasına köprü).",
    }
    with open(out_ds / "dataset.json", "w") as fh:
        json.dump(dataset_json, fh, indent=2)
    print(f"[json] dataset.json yazıldı ({n_training} eğitim örneği).")


def convert(images_dir: Path, labels_dir: Path, out_root: Path,
            dataset_id: int, dataset_name: str, copy_images: bool = True):
    ds_folder = out_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    imagesTr = ds_folder / "imagesTr"
    labelsTr = ds_folder / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    label_files = find_label_files(labels_dir)
    kept = 0
    skipped = []

    for lf in tqdm(label_files, desc="Dönüştürülüyor"):
        lbl_nii = nib.load(str(lf))
        arr = np.asarray(lbl_nii.dataobj)

        # bu case'de hiç IAC yoksa atla (nadir; ama güvenli olsun)
        if not (LEFT_IAC_SRC in arr or RIGHT_IAC_SRC in arr):
            skipped.append(lf.name)
            continue

        new_arr = remap_label(arr)
        case_id = lf.name.replace(".nii.gz", "")

        # etiketi kaydet (aynı affine/header ile — hizalama korunur)
        out_lbl = nib.Nifti1Image(new_arr, lbl_nii.affine, lbl_nii.header)
        out_lbl.set_data_dtype(np.uint8)
        nib.save(out_lbl, str(labelsTr / f"{case_id}.nii.gz"))

        # görüntüyü eşle
        img = match_image_for_label(lf, images_dir)
        dst_img = imagesTr / f"{case_id}_0000.nii.gz"
        if copy_images:
            shutil.copy2(img, dst_img)
        else:
            # yer kazanmak için sembolik link (aynı diskteyseniz)
            if not dst_img.exists():
                dst_img.symlink_to(img.resolve())
        kept += 1

    build_dataset_json(ds_folder, kept)
    print(f"\n[done] {kept} örnek yazıldı, {len(skipped)} örnek IAC içermediği için atlandı.")
    if skipped:
        print(f"[done] Atlanan ilk birkaç: {skipped[:5]}")
    print(f"[done] Çıktı klasörü: {ds_folder}")
    print(f"\nSıradaki adım:\n"
          f"  export nnUNet_raw={out_root}\n"
          f"  nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity\n")


def main():
    ap = argparse.ArgumentParser(description="ToothFairy3 -> sol/sağ IAC nnU-Net dataset")
    ap.add_argument("--tf3_images", type=Path, help="TF3 imagesTr klasörü")
    ap.add_argument("--tf3_labels", type=Path, required=True, help="TF3 labelsTr klasörü")
    ap.add_argument("--out_root", type=Path, help="nnUNet_raw kök klasörü")
    ap.add_argument("--dataset_id", type=int, default=111)
    ap.add_argument("--dataset_name", type=str, default="IAC_LR")
    ap.add_argument("--symlink", action="store_true",
                    help="Görüntüleri kopyalamak yerine symlink kullan (disk tasarrufu)")
    ap.add_argument("--verify_only", action="store_true",
                    help="Sadece etiket id'lerini kontrol et, dönüştürme yapma")
    args = ap.parse_args()

    verify_labels(args.tf3_labels)
    if args.verify_only:
        return
    if not args.tf3_images or not args.out_root:
        ap.error("--tf3_images ve --out_root, dönüştürme için zorunludur.")
    convert(args.tf3_images, args.tf3_labels, args.out_root,
            args.dataset_id, args.dataset_name, copy_images=not args.symlink)


if __name__ == "__main__":
    main()
