"""CanalManifold Flow.

The R1 state is a 160-station tube with seventeen local coefficients per
station and three global variables.  The package keeps all geometry in
physical millimetres and treats the frozen nnU-Net output as an OOF prior.
"""

from .constants import GLOBAL_DIM, LOCAL_DIM, N_GLOBAL, N_LOCAL, STATE_DIM

__all__ = ["GLOBAL_DIM", "LOCAL_DIM", "N_GLOBAL", "N_LOCAL", "STATE_DIM"]
__version__ = "0.4.0"
