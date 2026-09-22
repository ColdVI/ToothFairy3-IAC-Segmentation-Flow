"""State layout shared by preprocessing, training and inference.

Version 2 deliberately raises the angular representation ceiling from m<=4
to m<=8.  Positivity still comes from exponentiating the log-radius; harmonic
truncation is therefore a bandwidth choice, not the source of connectivity.
"""

N_LOCAL = 160
MAX_HARMONIC = 8
HARMONICS = tuple(range(2, MAX_HARMONIC + 1))
HARMONIC_START = 3
LOCAL_DIM = HARMONIC_START + 2 * len(HARMONICS)
GLOBAL_DIM = 3
N_GLOBAL = GLOBAL_DIM
STATE_DIM = N_LOCAL * LOCAL_DIM + GLOBAL_DIM

# Per-station state: centre displacement, local log-radius and m=2..8 shape.
D1 = 0
D2 = 1
ELL = 2
HARMONIC_INDEX = {
    harmonic: (HARMONIC_START + 2 * offset, HARMONIC_START + 2 * offset + 1)
    for offset, harmonic in enumerate(HARMONICS)
}
A2, B2 = HARMONIC_INDEX[2]
A3, B3 = HARMONIC_INDEX[3]
A4, B4 = HARMONIC_INDEX[4]
A5, B5 = HARMONIC_INDEX[5]
A6, B6 = HARMONIC_INDEX[6]
A7, B7 = HARMONIC_INDEX[7]
A8, B8 = HARMONIC_INDEX[8]

# Global state: log-calibre and physical axial endpoints.
G = 0
ENDPOINT_START = 1
ENDPOINT_END = 2

LOW_GROUP = (ELL, A2, B2)
HIGH_GROUP = tuple(
    index
    for harmonic in HARMONICS
    if harmonic >= 3
    for index in HARMONIC_INDEX[harmonic]
)
GLOBAL_LOCAL_GROUP = (D1, D2)
