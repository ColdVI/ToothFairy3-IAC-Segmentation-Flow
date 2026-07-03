import _pathsetup  # noqa: F401
import numpy as np

from io_utils import mask_to_sdf_mm, normalize_sdf, sdf_stack_to_mask


def test_sdf_sign_and_units():
    mask = np.zeros((20, 20, 20), bool)
    mask[8:12, 8:12, 8:12] = True
    sdf = mask_to_sdf_mm(mask, spacing=(1.0, 1.0, 1.0), clip_mm=10)
    assert sdf[10, 10, 10] < 0                 # inside is negative
    assert sdf[0, 0, 0] > 0                     # far outside is positive
    # anisotropic spacing scales distances physically
    sdf_aniso = mask_to_sdf_mm(mask, spacing=(2.0, 1.0, 1.0), clip_mm=20)
    assert sdf_aniso[0, 10, 10] > sdf[0, 10, 10] - 1e-6


def test_empty_mask_is_positive():
    empty = np.zeros((8, 8, 8), bool)
    sdf = mask_to_sdf_mm(empty, spacing=(1, 1, 1), clip_mm=5)
    assert np.all(sdf == 5)


def test_decode_roundtrip():
    mask = np.zeros((16, 16, 16), np.uint8)
    mask[4:8, 4:8, 4:8] = 1
    mask[4:8, 4:8, 10:14] = 2
    sp = (1.0, 1.0, 1.0)
    left = normalize_sdf(mask_to_sdf_mm(mask == 1, sp))
    right = normalize_sdf(mask_to_sdf_mm(mask == 2, sp))
    dec = sdf_stack_to_mask(np.stack([left, right]))
    inter1 = ((dec == 1) & (mask == 1)).sum() / (mask == 1).sum()
    inter2 = ((dec == 2) & (mask == 2)).sum() / (mask == 2).sum()
    assert inter1 > 0.9 and inter2 > 0.9


if __name__ == "__main__":
    import sys
    _pathsetup.run_module(sys.modules[__name__])
