"""等価回路の定義と実験設定。

C1〜C4 の4種類の等価回路を分類対象とする。CIRCUIT_PARAMS には各回路が
実際に使用するパラメータを、PARAM_RANGES には全パラメータの取り得る範囲を
定義している。回路を追加する場合はここに1エントリ追加すればよい
（C5 の枠だけ用意してあるが、デフォルトではサンプル数0のため未使用）。
"""

import matplotlib
import numpy as np

NUM_POINTS = 80
FREQ_RANGE = np.logspace(6, -2, NUM_POINTS)

SAMPLES_CONFIG = {
    "C1": 5000,
    "C2": 5000,
    "C3": 5000,
    "C4": 5000,
    "C5": 0,
}

PARAM_RANGES = {
    "Rs": ("lin", 0.1, 10000),
    "R1": ("lin", 0.1, 10000),
    "R2": ("lin", 0.1, 10000),
    "Q1": ("lin", 1e-8, 1e-4),
    "Q2": ("lin", 1e-8, 1e-4),
    "sigma": ("lin", 4e-4, 4e2),
    "a1": ("lin", 0.5, 1.0),
    "a2": ("lin", 0.5, 1.0),
}

DEFAULT_PARAMS = {
    "Rs": 100.0, "R1": 500.0, "R2": 500.0,
    "Q1": 1e-5, "Q2": 1e-6, "a1": 1.0, "a2": 1.0, "sigma": 50.0,
}

CIRCUIT_PARAMS = {
    "C1": ["Rs", "R1", "Q1", "a1"],
    "C2": ["Rs", "R1", "R2", "Q1", "Q2", "a1", "a2"],
    "C3": ["Rs", "R1", "Q1", "a1", "sigma"],
    "C4": ["Rs", "R1", "R2", "Q1", "Q2", "a1", "a2", "sigma"],
    "C5": ["Rs", "R1", "R2", "Q1", "Q2", "a1", "a2", "sigma"],
}

VAE_SETTINGS = {
    "LATENT_DIM": 2,
    "NUM_EPOCHS": 300,
    "BATCH_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "TEST_SIZE": 0.2,
    "BETA_START": 0.0, "BETA_END": 1.0, "BETA_WARMUP": 50,
    "GAMMA_START": 0.0, "GAMMA_END": 1.0, "GAMMA_WARMUP": 50,
    "GRAD_CLIP": 5.0,
    "LR_DECAY_STEP": 75,
    "LR_DECAY_GAMMA": 0.5,
}

CIRCUIT_TYPES = [c for c, n in SAMPLES_CONFIG.items() if n > 0]
LABEL_MAP = {c: i for i, c in enumerate(CIRCUIT_TYPES)}
NUM_CLASSES = len(CIRCUIT_TYPES)

_BASE_COLORS = ["#440154", "#31688e", "#21908c", "#35b779", "#fde725"]
COLOR_LIST = _BASE_COLORS[:NUM_CLASSES]
CMAP = matplotlib.colors.ListedColormap(COLOR_LIST)
_BOUNDS = [i - 0.5 for i in range(NUM_CLASSES + 1)]
NORM = matplotlib.colors.BoundaryNorm(_BOUNDS, CMAP.N)
TICK_LOCS = list(range(NUM_CLASSES))
