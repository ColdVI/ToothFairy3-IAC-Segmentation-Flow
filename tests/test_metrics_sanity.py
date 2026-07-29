"""Known-answer sanity checks for overlap, centerline, and topology metrics."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

import _pathsetup  # noqa: F401
from evaluation.metrics import cldice, dice
from evaluation.topology_metrics import (betti0_error, centerline_gap_length,
                                         false_branch_length, lr_swap_rate)


def _curved_tube():
    """A radius-3 tube with a 40-voxel-long, gently curved centreline."""
    shape = (56, 48, 48)
    tube = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for z in range(8, 48):
        y = 24 + int(round(3 * np.sin((z - 8) / 7)))
        x = 24 + int(round(2 * np.sin((z - 8) / 9)))
        tube |= (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= 3 ** 2
    return tube


def test_dice_and_cldice_on_synthetic_tube():
    gt = _curved_tube()

    assert dice(gt, gt) == 1.0
    assert cldice(gt, gt) == pytest.approx(1.0, abs=1e-6)

    dilated = binary_dilation(gt, iterations=1)
    assert dice(dilated, gt) < 0.85
    assert cldice(dilated, gt) == pytest.approx(1.0, abs=1e-5)

    # Remove a five-voxel-radius slab around the midpoint. A literal five-slice
    # gap scores about 0.93 for a 40-voxel centreline, so it cannot satisfy the
    # playbook's <0.9 threshold; this wider known break tests the intended signal.
    broken = gt.copy()
    broken[24:33] = False
    assert cldice(broken, gt) < 0.9

    zz, yy, xx = np.ogrid[:gt.shape[0], :gt.shape[1], :gt.shape[2]]
    detached_ball = (zz - 45) ** 2 + (yy - 8) ** 2 + (xx - 8) ** 2 <= 2 ** 2
    false_component = gt | detached_ball
    assert cldice(false_component, gt) < cldice(gt, gt)

    assert cldice(np.zeros_like(gt), gt) == 0.0


def test_betti0_error_known_components():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    assert betti0_error(mask) == 0
    mask[11:14, 11:14, 11:14] = True
    assert betti0_error(mask) == 1
    assert betti0_error(np.zeros_like(mask)) == 1


def test_centerline_gap_length_known_straight_gap():
    spacing = (1.0, 1.0, 1.0)
    gt = np.zeros((20, 9, 9), dtype=bool)
    gt[2:18, 4, 4] = True
    pred = gt.copy()
    pred[8:13, 4, 4] = False

    # The metric ignores the two missing endpoints that are within one voxel of
    # the prediction, leaving three uncovered one-millimetre centreline voxels.
    assert centerline_gap_length(pred, gt, spacing) == pytest.approx(3.0)


def test_false_branch_length_known_perpendicular_branch():
    spacing = (1.0, 1.0, 1.0)
    gt = np.zeros((20, 15, 15), dtype=bool)
    gt[2:18, 7, 7] = True
    pred = gt.copy()
    pred[10, 7, 8:13] = True

    # As above, the first branch voxel is within the one-voxel tolerance.
    assert false_branch_length(pred, gt, spacing) == pytest.approx(4.0)


def test_lr_swap_rate_known_assignments():
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    gt[:2] = 1
    gt[2:] = 2
    assert lr_swap_rate(gt, gt) == 0.0

    swapped = gt.copy()
    swapped[gt == 1] = 2
    swapped[gt == 2] = 1
    assert lr_swap_rate(swapped, gt) == 1.0

    half_swapped = gt.copy()
    half_swapped[:2] = 2
    assert lr_swap_rate(half_swapped, gt) == pytest.approx(0.5)
