"""等価回路インピーダンスのシミュレーションによるデータセット生成。

C1〜C4（および未使用のC5）のインピーダンスを解析式から計算し、
ナイキスト線図（またはBode線図）表現のデータセットを作る。
"""

import numpy as np
import pandas as pd

from config import (
    CIRCUIT_PARAMS,
    CIRCUIT_TYPES,
    DEFAULT_PARAMS,
    FREQ_RANGE,
    LABEL_MAP,
    NUM_POINTS,
    PARAM_RANGES,
    SAMPLES_CONFIG,
)


def get_impedance(params: dict, circuit_type: str, freq: np.ndarray) -> np.ndarray:
    """回路パラメータから複素インピーダンスを計算する。"""
    omega = 2 * np.pi * freq
    j_omega = 1j * omega

    def z_r(r):
        return r

    def z_cpe(q, a):
        return 1 / (q * (j_omega ** a))

    def z_warburg(sigma):
        return sigma * (1 - 1j) / np.sqrt(omega)

    if circuit_type == "C1":
        return z_r(params["Rs"]) + 1 / (1 / z_r(params["R1"]) + 1 / z_cpe(params["Q1"], params["a1"]))

    if circuit_type == "C2":
        z1 = 1 / (1 / z_r(params["R1"]) + 1 / z_cpe(params["Q1"], params["a1"]))
        z2 = 1 / (1 / z_r(params["R2"]) + 1 / z_cpe(params["Q2"], params["a2"]))
        return z_r(params["Rs"]) + z1 + z2

    if circuit_type == "C3":
        branch = z_r(params["R1"]) + z_warburg(params["sigma"])
        return z_r(params["Rs"]) + 1 / (1 / branch + 1 / z_cpe(params["Q1"], params["a1"]))

    if circuit_type == "C4":
        # 表面被覆(C2)と拡散(C3)を組み合わせた回路：
        # 通常のRC並列素子と、Warburgインピーダンスを含む並列素子を直列に接続
        z1 = 1 / (1 / z_r(params["R1"]) + 1 / z_cpe(params["Q1"], params["a1"]))
        branch = z_r(params["R2"]) + z_warburg(params["sigma"])
        z2 = 1 / (1 / branch + 1 / z_cpe(params["Q2"], params["a2"]))
        return z_r(params["Rs"]) + z1 + z2

    if circuit_type == "C5":
        inner = 1 / (1 / z_cpe(params["Q2"], params["a2"]) + 1 / (z_r(params["R2"]) + z_warburg(params["sigma"])))
        branch = z_r(params["R1"]) + inner
        return z_r(params["Rs"]) + 1 / (1 / z_cpe(params["Q1"], params["a1"]) + 1 / branch)

    raise ValueError(f"未対応の回路種別です: {circuit_type}")


def _sample_random_params() -> dict:
    params = DEFAULT_PARAMS.copy()
    for name, (scale, mn, mx) in PARAM_RANGES.items():
        params[name] = 10 ** np.random.uniform(mn, mx) if scale == "log" else np.random.uniform(mn, mx)
    return params


def _to_row(Z: np.ndarray, mode: str, include_freq: bool, log_freq: np.ndarray) -> np.ndarray:
    if mode == "bode":
        mag = np.clip(np.abs(Z), 1e-12, None)
        channels = [np.log10(mag), np.angle(Z)]
    else:
        channels = [Z.real, Z.imag]

    if include_freq:
        channels = [log_freq] + channels
    return np.concatenate(channels)


def generate_dataset(mode: str = "nyquist", include_freq: bool = False):
    """分類用データセットを生成する。

    戻り値:
        X: (N, features) の入力データ
        y: (N,) のクラスラベル
        params_df: 各サンプルの生成パラメータ（回路で未使用のものはNaN）
    """
    freq = FREQ_RANGE
    log_freq = np.log10(freq)

    data_list, labels, params_log = [], [], []

    for circuit_type in CIRCUIT_TYPES:
        n_samples = SAMPLES_CONFIG[circuit_type]
        for _ in range(n_samples):
            params = _sample_random_params()
            Z = get_impedance(params, circuit_type, freq)

            data_list.append(_to_row(Z, mode, include_freq, log_freq))
            labels.append(LABEL_MAP[circuit_type])

            record = {"circuit": circuit_type}
            used_params = CIRCUIT_PARAMS.get(circuit_type, [])
            for name in PARAM_RANGES:
                record[name] = params[name] if name in used_params else np.nan
            params_log.append(record)

    X = np.array(data_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    params_df = pd.DataFrame(params_log)
    return X, y, params_df


def generate_eval_dataset_with_params(mode: str, include_freq: bool, samples_per_class: int = 1000):
    """潜在空間とパラメータ値の対応を見るための、パラメータ付きの評価用データセット。"""
    freq = FREQ_RANGE
    log_freq = np.log10(freq)

    data_list, param_list = [], []
    for circuit_type in CIRCUIT_TYPES:
        for _ in range(samples_per_class):
            params = _sample_random_params()
            Z = get_impedance(params, circuit_type, freq)
            data_list.append(_to_row(Z, mode, include_freq, log_freq))
            param_list.append(params)

    return np.array(data_list, dtype=np.float32), param_list
