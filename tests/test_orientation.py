import _pathsetup  # noqa: F401
import numpy as np

from io_utils import orientation_code, physical_coord_grid, normalize_coords
from conditioning import lateral_axis_index


def _rpi_affine(spacing=(0.3, 0.3, 0.3)):
    # R, P, I axes: +x->Right decreasing? Build a diagonal affine with codes RPI.
    a = np.diag([-spacing[0], -spacing[1], -spacing[2], 1.0])  # -> L? check below
    a[0, 0] = spacing[0]   # +i -> +x (R)
    a[1, 1] = -spacing[1]  # +j -> -y (P from A)
    a[2, 2] = -spacing[2]  # +k -> -z (I from S)
    return a


def test_orientation_code_and_axis():
    a = _rpi_affine()
    code = orientation_code(a)
    assert code == "RPI", code
    idx, sign = lateral_axis_index(a)
    assert idx == 0, idx           # first axis is R<->L in RPI


def test_physical_coords_shape_and_norm():
    a = _rpi_affine()
    shape = (8, 10, 12)
    coords = physical_coord_grid(shape, a)
    assert coords.shape == (3, *shape)
    nc = normalize_coords(coords)
    assert nc.shape == (3, *shape)
    assert nc.min() >= -1.0001 and nc.max() <= 1.0001
    # each axis spans roughly the full [-1,1] range
    for c in range(3):
        assert nc[c].max() - nc[c].min() > 1.5


if __name__ == "__main__":
    import sys
    _pathsetup.run_module(sys.modules[__name__])
