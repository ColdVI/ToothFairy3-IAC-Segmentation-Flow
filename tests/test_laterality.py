import _pathsetup  # noqa: F401
import numpy as np
import torch

from losses import laterality_loss, sdf_to_occupancy, soft_cldice_loss


def _sdf_ball(center, D=16, r=3):
    zz, yy, xx = np.meshgrid(*[np.arange(D)] * 3, indexing="ij")
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return (dist - r).astype(np.float32) / 6.0     # normalised-ish SDF


def test_overlap_penalised():
    D = 16
    # disjoint L/R balls
    disj = np.stack([_sdf_ball((8, 8, 4)), _sdf_ball((8, 8, 12))])[None]
    # overlapping: both channels the same ball
    same = _sdf_ball((8, 8, 8))
    over = np.stack([same, same])[None]
    ld = laterality_loss(torch.tensor(disj))
    lo = laterality_loss(torch.tensor(over))
    assert lo > ld, (float(lo), float(ld))


def test_cldice_identity_lower_than_mismatch():
    D = 16
    occ_a = sdf_to_occupancy(torch.tensor(_sdf_ball((8, 8, 8))[None, None]))
    occ_b = sdf_to_occupancy(torch.tensor(_sdf_ball((8, 8, 2))[None, None]))
    same = soft_cldice_loss(occ_a, occ_a, iters=3)
    diff = soft_cldice_loss(occ_a, occ_b, iters=3)
    assert same < diff


if __name__ == "__main__":
    import sys
    _pathsetup.run_module(sys.modules[__name__])
