"""等価回路の定義と実験設定。

CIRCUIT_PARAMS には各回路が持つパラメータと、そのサンプリング範囲・スケール
（'lin' or 'log'）を定義している。回路を追加する場合はここに1エントリ追加すればよい。
"""

import numpy as np

CIRCUIT_PARAMS = {
    "C1": {
        "Rs": ("lin", 0.1, 10000),
        "R1": ("lin", 0.1, 10000),
        "Q1": ("lin", 1e-8, 1e-4),
    },
    "C2": {
        "Rs": ("lin", 0.1, 10000),
        "R1": ("lin", 0.1, 10000),
        "Q1": ("lin", 1e-8, 1e-4),
        "R2": ("lin", 0.1, 10000),
        "Q2": ("lin", 1e-8, 1e-4),
    },
    "C3": {
        "Rs": ("lin", 0.1, 10000),
        "R1": ("lin", 0.1, 10000),
        "Q1": ("lin", 1e-8, 1e-4),
        "sigma": ("lin", 4e-4, 4e2),
    },
}

DEFAULT_CONFIG = {
    "INPUT_MODE": "nyquist",       # 'nyquist' (Z', Z'') or 'bode' (|Z|, phase)
    "INCLUDE_FREQ": False,         # 入力に周波数チャンネルを含めるか
    "NUM_POINTS": 80,              # 周波数点数
    "N_SAMPLES": 30000,            # 回路あたりのサンプル数
    "FIXED_A1": 1.0,               # CPE指数（今回は固定）
    "FIXED_A2": 1.0,
    "LATENT_DIM": 4,
    "NUM_EPOCHS": 10000,
    "BATCH_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "TEST_SIZE": 0.2,
    # KL項・回帰損失の重みは学習初期を抑えたのちアニーリングする
    "BETA_START": 0.0, "BETA_END": 0.1, "BETA_WARMUP": 20,
    "DELTA_START": 0.0, "DELTA_END": 500.0, "DELTA_WARMUP": 20,
}


class CircuitEnv:
    """指定した回路種別に応じて、パラメータ数・入力次元などを解決するコンテナ。

    回路ごとに推定対象のパラメータ数が異なるため、モデルや損失関数を組み立てる
    前にこのクラスで諸元を確定させる。
    """

    def __init__(self, circuit_type: str, config: dict | None = None):
        if circuit_type not in CIRCUIT_PARAMS:
            raise ValueError(f"未定義の回路種別です: {circuit_type}")

        self.circuit_type = circuit_type
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.param_ranges = CIRCUIT_PARAMS[circuit_type]
        self.param_keys = list(self.param_ranges.keys())
        self.num_params = len(self.param_keys)
        self.input_channels = 3 if self.config["INCLUDE_FREQ"] else 2
        self.freq_range = np.logspace(6, -2, self.config["NUM_POINTS"])
