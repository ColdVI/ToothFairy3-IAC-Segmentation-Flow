# Left / Right IAC Segmentation — ToothFairy3 → nnU-Net baseline + 3D Flow-Matching

Pipeline to segment the **left** and **right inferior alveolar canal (IAC)** from
dental CBCT, with a discriminative baseline (nnU-Net v2) and a generative research
extension (3D continuous-time flow matching, inspired by SEAL-Flow but built for 3D).

## Files

| file | role | where it runs |
|------|------|---------------|
| `convert_tf3_to_iac_lr.py` | reduce ToothFairy3's 77-label masks to `{0:bg, 1:Left_IAC, 2:Right_IAC}`, write nnU-Net `Dataset801_IAC_LR`. IAC ids resolved **by name** from `dataset.json` (avoids mandibular-incisive / lingual canals). | CPU (local or Colab) |
| `run_nnunet_iac.sh` | end-to-end nnU-Net: env vars → convert → plan/preprocess → train 3d_fullres → predict | GPU |
| `flow_matching_iac.py` | the research model: 3D U-Net velocity field, rectified-flow training on signed-distance-field targets, ODE sampling, topology term. Has a CPU `selftest`. | GPU (train) / CPU (selftest) |
| `fm_train.py` | real-data training + sliding-window prediction for the flow model | GPU |
| `IAC_Colab_runner.ipynb` | one notebook running **both** tracks on a Colab GPU | Colab GPU |

## Dataset (validated locally, 532 cases)

- **Geometry:** all 532 cases are **0.3 mm isotropic**, **LPS** orientation, zero image/label shape mismatches.
- **Labels present:** **both** L and R IAC in **every** case (0 missing, 0 one-sided).
- **IAC volume:** L median ≈ 14.5k voxels, R median ≈ 14.7k (range ≈ 5.6k–32.6k).
- Prefixes handled: `ToothFairy3F_` (63), `ToothFairy3P_` (417), `ToothFairy3S_` (52).

## Quick start (Colab)

1. Upload the `ToothFairy3` folder (with `imagesTr/`, `labelsTr/`, `dataset.json`) to Drive.
2. Upload the three `.py` files next to `IAC_Colab_runner.ipynb`.
3. Runtime → GPU. Set `TF3_SRC` in the notebook. Run top to bottom.

## The research contribution (Track B)

Per-voxel classifiers (U-Net/nnU-Net) optimise overlap and **break the thin tubular
canal** where image evidence is weak — the failures that matter for nerve-injury risk.
Track B reframes segmentation as **generative sampling of masks conditioned on the image**:

- **Target:** signed distance field (SDF) per side — smooth, encodes shape/topology in its zero-level-set.
- **Path:** rectified (straight-line) conditional flow, `x_t=(1-t)·noise + t·SDF`, velocity `= SDF − noise`.
- **Model:** 3D U-Net `v_θ(x_t, t, image)` with sinusoidal-time FiLM conditioning.
- **Loss:** flow-matching MSE + boundary-consistency (topology) regulariser.
- **Inference:** integrate `dx/dt = v_θ` from noise (Heun), decode by sign.

Self-test result (synthetic canals, 60 CPU steps): loss 2.14 → 0.14, sampled Dice L 0.33 / R 0.38 and rising — the machinery is verified; real Dice comes from GPU training.

### Evaluate topology, not just Dice
Report Dice **and**: connected components per side (target = 1), Betti-0 error, centerline continuity. That is where a flow-based method should beat the baseline.

## SEAL-Flow relationship
Same flow-matching family. SEAL-Flow's public repo is **2D** (PNG cell/nucleus data) with
trainer + shape-regularisation bodies withheld. This is an independent **3D CBCT** formulation.
