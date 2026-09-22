
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from iacb.train import Case, sample_patch

from iacb.bridge import (
    CorrelatedNoise,
    StateUNet,
    bridge_loss,
    interpolant,
    sdf_cut_ball,
)


# ================================================================================================
# DATASET
# ================================================================================================

class CaseBalancedDataset(Dataset):

    """
    One dataset item == one CASE.

    Every case produces exactly 2 patches:

      patch A -> disagreement-focused
      patch B -> foreground-focused

    Therefore every training case is seen exactly once
    per epoch.

    Randomness remains only inside patch position/jitter
    and augmentation.
    """

    def __init__(
        self,
        cache,
        cases,
        patch,
        epoch,
        seed,
        training=True,
    ):

        self.cache = Path(cache)

        self.cases = list(cases)

        self.patch = tuple(patch)

        self.epoch = int(epoch)

        self.seed = int(seed)

        self.training = bool(training)


    def __len__(self):

        return len(self.cases)


    def __getitem__(self, idx):

        case_name = self.cases[idx]

        rng = np.random.default_rng(
            self.seed
            + self.epoch * 1_000_003
            + idx * 10_007
        )

        c = Case(
            self.cache,
            case_name,
        )


        # ----------------------------------------------------------------------------------------
        # PATCH A — disagreement
        # ----------------------------------------------------------------------------------------

        img_a, x0_a, x1_a = sample_patch(
            c,
            self.patch,

            p_dis=1.0,
            p_fg=0.0,

            rng=rng,
        )


        # ----------------------------------------------------------------------------------------
        # PATCH B — foreground
        # ----------------------------------------------------------------------------------------

        img_b, x0_b, x1_b = sample_patch(
            c,
            self.patch,

            p_dis=0.0,
            p_fg=1.0,

            rng=rng,
        )


        # ----------------------------------------------------------------------------------------
        # TRAIN augmentation
        # ----------------------------------------------------------------------------------------

        if self.training:

            img_a = (
                img_a
                * rng.uniform(
                    0.9,
                    1.1,
                )
                + rng.uniform(
                    -0.1,
                    0.1,
                )
            )

            img_b = (
                img_b
                * rng.uniform(
                    0.9,
                    1.1,
                )
                + rng.uniform(
                    -0.1,
                    0.1,
                )
            )


        img = np.stack(
            [
                img_a,
                img_b,
            ],
            axis=0,
        ).astype(
            np.float32
        )


        x0 = np.stack(
            [
                x0_a,
                x0_b,
            ],
            axis=0,
        ).astype(
            np.float32
        )


        x1 = np.stack(
            [
                x1_a,
                x1_b,
            ],
            axis=0,
        ).astype(
            np.float32
        )


        spacing = float(
            c.meta["spacing"][0]
        )


        return (
            torch.from_numpy(img),

            torch.from_numpy(x0),

            torch.from_numpy(x1),

            torch.tensor(
                spacing,
                dtype=torch.float32,
            ),

            case_name,
        )


# ================================================================================================
# SPLIT
# ================================================================================================

def make_split(
    cache,
    eval_fold,
    val_n,
    seed,
):

    cache = Path(cache)


    metas = [
        json.loads(
            p.read_text()
        )
        for p in sorted(
            cache.glob(
                "*_meta.json"
            )
        )
    ]


    eval_cases = sorted(
        m["case"]
        for m in metas
        if m["fold"] == eval_fold
    )


    # 52 ToothFairy3S cases
    external_s = sorted(
        m["case"]
        for m in metas
        if m["fold"] == -1
    )


    pf_pool = sorted(
        m["case"]
        for m in metas
        if (
            m["fold"] is not None
            and m["fold"] >= 0
            and m["fold"] != eval_fold
        )
    )


    print(
        "\nRaw split counts:"
    )

    print(
        "eval fold       :",
        len(eval_cases),
    )

    print(
        "S external pool :",
        len(external_s),
    )

    print(
        "non-eval P/F    :",
        len(pf_pool),
    )


    assert len(eval_cases) == 97, (
        f"Expected 97 eval cases, got "
        f"{len(eval_cases)}"
    )

    assert len(external_s) == 52, (
        f"Expected 52 S cases, got "
        f"{len(external_s)}"
    )

    assert len(pf_pool) == 383, (
        f"Expected 383 P/F pool cases, got "
        f"{len(pf_pool)}"
    )


    # --------------------------------------------------------------------------------------------
    # Validation ONLY from P/F.
    # ALL 52 S remain in training.
    # --------------------------------------------------------------------------------------------

    rng = random.Random(
        seed
    )

    shuffled = list(
        pf_pool
    )

    rng.shuffle(
        shuffled
    )


    val_cases = sorted(
        shuffled[:val_n]
    )


    train_pf = sorted(
        shuffled[val_n:]
    )


    train_cases = sorted(
        train_pf
        + external_s
    )


    # --------------------------------------------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------------------------------------------

    assert not (
        set(train_cases)
        & set(val_cases)
    )

    assert not (
        set(train_cases)
        & set(eval_cases)
    )

    assert not (
        set(val_cases)
        & set(eval_cases)
    )


    n_s_train = sum(
        "ToothFairy3S_" in x
        for x in train_cases
    )


    assert n_s_train == 52


    assert (
        len(train_cases)
        + len(val_cases)
        + len(eval_cases)
        == 532
    )


    return (
        train_cases,
        val_cases,
        eval_cases,
    )


# ================================================================================================
# VALIDATION
# ================================================================================================

@torch.no_grad()
def run_validation(
    model,
    cache,
    val_cases,
    patch,
    clip_mm,
    device,
    seed,
):

    model.eval()


    # fixed validation patches
    ds = CaseBalancedDataset(
        cache=cache,

        cases=val_cases,

        patch=patch,

        epoch=0,

        seed=seed + 5_000_000,

        training=False,
    )


    # validation dataset item already contains 2 patches
    loader = DataLoader(
        ds,

        batch_size=1,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )


    losses = []

    sdf_losses = []

    bce_losses = []

    dice_losses = []


    for (
        img,
        x0,
        x1,
        spacing,
        names,
    ) in loader:


        img = (
            img.squeeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )


        x0 = (
            x0.squeeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )


        x1 = (
            x1.squeeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )


        B = len(img)


        # ----------------------------------------------------------------------------------------
        # validation at t = 0
        #
        # Question:
        # given the prior directly, does network improve/reconstruct GT?
        # ----------------------------------------------------------------------------------------

        t = torch.zeros(
            B,
            device=device,
        )


        with torch.autocast(
            device_type=device.type,

            enabled=(
                device.type
                == "cuda"
            ),
        ):

            pred = model(
                img,
                x0,
                t,
            )


            loss, parts = bridge_loss(
                pred,
                x1,
                clip_mm,
            )


        losses.append(
            float(
                loss.detach()
                .cpu()
            )
        )


        sdf_losses.append(
            float(
                parts["l_sdf"]
            )
        )


        bce_losses.append(
            float(
                parts["l_bce"]
            )
        )


        dice_losses.append(
            float(
                parts["l_dice"]
            )
        )


    return {

        "loss": float(
            np.mean(losses)
        ),

        "l_sdf": float(
            np.mean(sdf_losses)
        ),

        "l_bce": float(
            np.mean(bce_losses)
        ),

        "l_dice": float(
            np.mean(dice_losses)
        ),
    }


# ================================================================================================
# CHECKPOINT
# ================================================================================================

def save_ckpt(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val,
    best_epoch,
    no_improve,
    widths,
    clip_mm,
    sigma_mm,
    noise_corr_vox,
    patch,
    train_cases,
    val_cases,
    eval_cases,
):

    payload = {

        "model":
            model.state_dict(),

        "opt":
            optimizer.state_dict(),

        "sched":
            scheduler.state_dict(),

        "epoch":
            int(epoch),

        "global_step":
            int(global_step),

        "best_val":
            float(best_val),

        "best_epoch":
            int(best_epoch),

        "no_improve":
            int(no_improve),

        # ----------------------------------------------------------------------------------------
        # infer.py compatibility
        # ----------------------------------------------------------------------------------------

        "widths":
            tuple(widths),

        "clip_mm":
            float(clip_mm),

        "sigma_mm":
            float(sigma_mm),

        "noise_corr_vox":
            float(noise_corr_vox),

        "patch":
            tuple(patch),


        # ----------------------------------------------------------------------------------------
        # reproducibility
        # ----------------------------------------------------------------------------------------

        "train_cases":
            list(train_cases),

        "val_cases":
            list(val_cases),

        "eval_cases":
            list(eval_cases),

        "training_mode":
            "epoch_casebalanced_v3",
    }


    tmp = Path(
        str(path)
        + ".tmp"
    )


    torch.save(
        payload,
        tmp,
    )


    tmp.replace(
        path
    )


# ================================================================================================
# MAIN
# ================================================================================================

def main():

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--cache",
        required=True,
    )


    ap.add_argument(
        "--out",
        required=True,
    )


    ap.add_argument(
        "--epochs",
        type=int,
        required=True,
    )


    ap.add_argument(
        "--eval_fold",
        type=int,
        default=0,
    )


    ap.add_argument(
        "--val_n",
        type=int,
        default=32,
    )


    ap.add_argument(
        "--patch",
        default="96,128,128",
    )


    ap.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )


    ap.add_argument(
        "--widths",
        default="24,48,96,160",
    )


    ap.add_argument(
        "--sigma_mm",
        type=float,
        default=1.5,
    )


    ap.add_argument(
        "--noise_corr_vox",
        type=float,
        default=3.0,
    )


    ap.add_argument(
        "--t0_prob",
        type=float,
        default=0.15,
    )


    ap.add_argument(
        "--src_aug_p",
        type=float,
        default=0.25,
    )


    ap.add_argument(
        "--workers",
        type=int,
        default=6,
    )


    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    ap.add_argument(
        "--lr_patience",
        type=int,
        default=7,
    )


    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=18,
    )


    ap.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
    )


    ap.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )


    a = ap.parse_args()


    # ============================================================================================
    # Seeds
    # ============================================================================================

    torch.manual_seed(
        a.seed
    )

    np.random.seed(
        a.seed
    )

    random.seed(
        a.seed
    )


    device = torch.device(
        a.device
    )


    cache = Path(
        a.cache
    )


    out = Path(
        a.out
    )


    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    patch = tuple(
        int(x)
        for x in a.patch.split(",")
    )


    widths = tuple(
        int(x)
        for x in a.widths.split(",")
    )


    down_factor = (
        2
        ** (
            len(widths)
            - 1
        )
    )


    assert all(
        x % down_factor == 0
        for x in patch
    ), (
        f"Patch {patch} must be divisible by "
        f"{down_factor}"
    )


    # ============================================================================================
    # Split
    # ============================================================================================

    (
        train_cases,
        val_cases,
        eval_cases,
    ) = make_split(
        cache,
        a.eval_fold,
        a.val_n,
        a.seed,
    )


    n_s = sum(
        "ToothFairy3S_" in x
        for x in train_cases
    )


    print(
        "\n"
        + "=" * 90
    )


    print(
        "CASE-BALANCED EPOCH TRAINING"
    )


    print(
        "=" * 90
    )


    print(
        f"train          : "
        f"{len(train_cases)}"
    )


    print(
        f"  P/F train    : "
        f"{len(train_cases) - n_s}"
    )


    print(
        f"  S train      : "
        f"{n_s}"
    )


    print(
        f"validation     : "
        f"{len(val_cases)}"
    )


    print(
        f"untouched test : "
        f"{len(eval_cases)}"
    )


    print(
        f"steps / epoch  : "
        f"{len(train_cases)}"
    )


    print(
        f"patches / epoch: "
        f"{len(train_cases) * 2}"
    )


    print(
        f"max epochs     : "
        f"{a.epochs}"
    )


    print(
        f"device         : "
        f"{device}"
    )


    print(
        "=" * 90,
        flush=True,
    )


    # ============================================================================================
    # Save split
    # ============================================================================================

    split_file = (
        out
        / "epoch_split.json"
    )


    if not split_file.exists():

        split_file.write_text(

            json.dumps(
                {
                    "train":
                        train_cases,

                    "validation":
                        val_cases,

                    "eval_untouched":
                        eval_cases,
                },

                indent=2,
            )
        )


    # ============================================================================================
    # Model config
    # ============================================================================================

    meta = json.loads(

        (
            cache
            / f"{train_cases[0]}_meta.json"
        ).read_text()
    )


    clip_mm = float(
        meta["clip_mm"]
    )


    model = StateUNet(
        widths
    ).to(
        device
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=a.lr,

        weight_decay=1e-4,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=a.lr_patience,

            min_lr=1e-6,
        )
    )


    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=(
            device.type
            == "cuda"
        ),
    )


    sigma = (
        a.sigma_mm
        / clip_mm
    )


    noise = CorrelatedNoise(

        a.noise_corr_vox,

        device,
    )


    # ============================================================================================
    # Resume
    # ============================================================================================

    last_pt = (
        out
        / "last.pt"
    )


    best_pt = (
        out
        / "best.pt"
    )


    start_epoch = 0

    global_step = 0

    best_val = math.inf

    best_epoch = -1

    no_improve = 0


    if last_pt.exists():

        ck = torch.load(
            last_pt,
            map_location=device,
        )


        mode = ck.get(
            "training_mode",
            "",
        )


        if mode != "epoch_casebalanced_v3":

            raise RuntimeError(
                "Existing last.pt belongs to another "
                f"training mode: {mode}"
            )


        model.load_state_dict(
            ck["model"]
        )


        optimizer.load_state_dict(
            ck["opt"]
        )


        scheduler.load_state_dict(
            ck["sched"]
        )


        start_epoch = int(
            ck["epoch"]
        )


        global_step = int(
            ck["global_step"]
        )


        best_val = float(
            ck["best_val"]
        )


        best_epoch = int(
            ck["best_epoch"]
        )


        no_improve = int(
            ck.get(
                "no_improve",
                0,
            )
        )


        print(
            "\n[RESUME]"
            f" epoch={start_epoch}"
            f" global_step={global_step}"
            f" best={best_val:.6f}"
            f" @ epoch={best_epoch}",
            flush=True,
        )


    # ============================================================================================
    # Identity contract before any update
    # ============================================================================================

    if start_epoch == 0:

        c = Case(
            cache,
            train_cases[0],
        )


        rng = np.random.default_rng(
            a.seed
        )


        img_np, x0_np, x1_np = sample_patch(

            c,

            patch,

            p_dis=1.0,

            p_fg=0.0,

            rng=rng,
        )


        img0 = (
            torch.from_numpy(
                img_np[None]
            )
            .to(
                device
            )
        )


        x00 = (
            torch.from_numpy(
                x0_np[None]
            )
            .to(
                device
            )
        )


        with torch.no_grad():

            pred0 = model(

                img0,

                x00,

                torch.zeros(
                    1,
                    device=device,
                ),
            )


        identity_error = float(

            (
                pred0
                - x00
            )
            .abs()
            .max()
            .detach()
            .cpu()
        )


        print(
            f"\n[identity contract] "
            f"max error = "
            f"{identity_error:.3e}",
            flush=True,
        )


        assert (
            identity_error
            < 1e-7
        ), (
            "Zero-init identity contract failed"
        )


    # ============================================================================================
    # Logs
    # ============================================================================================

    log_path = (
        out
        / "epoch_log.jsonl"
    )


    log_f = open(
        log_path,
        "a",
    )


    # ============================================================================================
    # Epoch loop
    # ============================================================================================

    for epoch_idx in range(
        start_epoch,
        a.epochs,
    ):


        epoch_number = (
            epoch_idx
            + 1
        )


        epoch_t0 = time.time()


        train_ds = CaseBalancedDataset(

            cache=cache,

            cases=train_cases,

            patch=patch,

            epoch=epoch_idx,

            seed=a.seed,

            training=True,
        )


        g = torch.Generator()


        g.manual_seed(
            a.seed
            + epoch_idx
        )


        train_loader = DataLoader(

            train_ds,

            batch_size=1,

            shuffle=True,

            generator=g,

            num_workers=a.workers,

            pin_memory=(
                device.type
                == "cuda"
            ),

            persistent_workers=False,

            prefetch_factor=2
            if a.workers > 0
            else None,
        )


        model.train()


        losses = []

        sdf_losses = []

        bce_losses = []

        dice_losses = []


        for case_step, batch in enumerate(
            train_loader,
            start=1,
        ):


            (
                img,
                x0,
                x1,
                spacing,
                names,
            ) = batch


            # [1, 2, C, Z, Y, X]
            # ->
            # [2, C, Z, Y, X]

            img = (
                img
                .squeeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )


            x0 = (
                x0
                .squeeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )


            x1 = (
                x1
                .squeeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )


            spacing_mm = float(
                spacing
                .squeeze()
                .item()
            )


            # source augmentation
            if a.src_aug_p > 0:

                x0 = sdf_cut_ball(

                    x0,

                    clip_mm,

                    spacing_mm,

                    a.src_aug_p,
                )


            B = len(
                img
            )


            # bridge time
            t = torch.rand(

                B,

                device=device,
            )


            # explicit t=0 supervision
            force_t0 = (

                torch.rand(
                    B,
                    device=device,
                )
                < a.t0_prob
            )


            t = torch.where(

                force_t0,

                torch.zeros_like(
                    t
                ),

                t,
            )


            xt = interpolant(

                x0,

                x1,

                t,

                sigma,

                noise,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.autocast(

                device_type=device.type,

                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                pred = model(
                    img,
                    xt,
                    t,
                )


                loss, parts = bridge_loss(

                    pred,

                    x1,

                    clip_mm,
                )


            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(

                    model.parameters(),

                    1.0,
                )
            )


            scaler.step(
                optimizer
            )


            scaler.update()


            global_step += 1


            losses.append(
                float(
                    loss
                    .detach()
                    .cpu()
                )
            )


            sdf_losses.append(
                float(
                    parts["l_sdf"]
                )
            )


            bce_losses.append(
                float(
                    parts["l_bce"]
                )
            )


            dice_losses.append(
                float(
                    parts["l_dice"]
                )
            )


            if (
                case_step % 50 == 0
                or case_step
                == len(train_cases)
            ):

                recent = losses[
                    -50:
                ]


                print(

                    f"[epoch "
                    f"{epoch_number:03d}"
                    f"/{a.epochs}] "

                    f"{case_step:03d}"
                    f"/{len(train_cases)} "

                    f"| global="
                    f"{global_step} "

                    f"| loss="
                    f"{np.mean(recent):.5f} "

                    f"| lr="
                    f"{optimizer.param_groups[0]['lr']:.2e} "

                    f"| gn="
                    f"{float(grad_norm.detach().cpu()):.3f}",

                    flush=True,
                )


        # ========================================================================================
        # Validation
        # ========================================================================================

        train_loss = float(
            np.mean(
                losses
            )
        )


        val_stats = run_validation(

            model=model,

            cache=cache,

            val_cases=val_cases,

            patch=patch,

            clip_mm=clip_mm,

            device=device,

            seed=a.seed,
        )


        val_loss = float(
            val_stats["loss"]
        )


        scheduler.step(
            val_loss
        )


        improved = (
            val_loss
            < (
                best_val
                - a.min_delta
            )
        )


        if improved:

            best_val = val_loss

            best_epoch = epoch_number

            no_improve = 0


        else:

            no_improve += 1


        # ========================================================================================
        # Always save last
        # ========================================================================================

        save_ckpt(

            last_pt,

            model,

            optimizer,

            scheduler,

            epoch_number,

            global_step,

            best_val,

            best_epoch,

            no_improve,

            widths,

            clip_mm,

            a.sigma_mm,

            a.noise_corr_vox,

            patch,

            train_cases,

            val_cases,

            eval_cases,
        )


        # ========================================================================================
        # Save best separately
        # ========================================================================================

        if improved:

            save_ckpt(

                best_pt,

                model,

                optimizer,

                scheduler,

                epoch_number,

                global_step,

                best_val,

                best_epoch,

                no_improve,

                widths,

                clip_mm,

                a.sigma_mm,

                a.noise_corr_vox,

                patch,

                train_cases,

                val_cases,

                eval_cases,
            )


        epoch_sec = (
            time.time()
            - epoch_t0
        )


        record = {

            "epoch":
                epoch_number,

            "global_step":
                global_step,

            "train_loss":
                train_loss,

            "train_sdf":
                float(
                    np.mean(
                        sdf_losses
                    )
                ),

            "train_bce":
                float(
                    np.mean(
                        bce_losses
                    )
                ),

            "train_dice_loss":
                float(
                    np.mean(
                        dice_losses
                    )
                ),

            "val":
                val_stats,

            "lr":
                float(
                    optimizer
                    .param_groups[0]["lr"]
                ),

            "best_val":
                best_val,

            "best_epoch":
                best_epoch,

            "no_improve":
                no_improve,

            "epoch_minutes":
                epoch_sec
                / 60.0,
        }


        log_f.write(
            json.dumps(
                record
            )
            + "\n"
        )


        log_f.flush()


        print(
            "\n"
            + "-" * 90
        )


        print(
            f"EPOCH "
            f"{epoch_number:03d}"
            f"/{a.epochs}"
        )


        print(
            f"train loss     : "
            f"{train_loss:.6f}"
        )


        print(
            f"val loss       : "
            f"{val_loss:.6f}"
        )


        print(
            f"val sdf        : "
            f"{val_stats['l_sdf']:.6f}"
        )


        print(
            f"val BCE        : "
            f"{val_stats['l_bce']:.6f}"
        )


        print(
            f"val Dice loss  : "
            f"{val_stats['l_dice']:.6f}"
        )


        print(
            f"LR             : "
            f"{optimizer.param_groups[0]['lr']:.3e}"
        )


        print(
            f"BEST           : "
            f"{best_val:.6f}"
            f" @ epoch "
            f"{best_epoch}"
        )


        print(
            f"patience       : "
            f"{no_improve}"
            f"/{a.early_stop_patience}"
        )


        print(
            f"epoch time     : "
            f"{epoch_sec / 60:.2f} min"
        )


        print(
            "-" * 90
            + "\n",
            flush=True,
        )


        # ========================================================================================
        # Early stop
        # ========================================================================================

        if (
            no_improve
            >= a.early_stop_patience
        ):

            print(
                "\n[EARLY STOP] "
                f"No validation improvement "
                f"for {no_improve} epochs.",
                flush=True,
            )

            break


    log_f.close()


    print(
        "\n"
        + "=" * 90
    )


    print(
        "TRAINING FINISHED"
    )


    print(
        "=" * 90
    )


    print(
        "best epoch :",
        best_epoch,
    )


    print(
        "best val   :",
        best_val,
    )


    print(
        "best.pt    :",
        best_pt,
    )


    print(
        "last.pt    :",
        last_pt,
    )


    print(
        "=" * 90,
        flush=True,
    )


if __name__ == "__main__":
    main()
