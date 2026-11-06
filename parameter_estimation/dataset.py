"""等価回路インピーダンスのシミュレーションによるデータセット生成。

回路パラメータをラテンハイパーキューブサンプリング（LHS）で決定し、対応する
ナイキスト線図（または Bode 線図）を計算する。C2 回路は2つの並列素子が
入れ替え可能（順列不変性）なため、時定数でソートしてラベルスイッチングを防いでいる。
"""

import numpy as np
from scipy.stats import qmc

from config import CircuitEnv


def normalize(val, param_range):
    scale, mn, mx = param_range
    if scale == "log":
        return (np.log10(val) - np.log10(mn)) / (np.log10(mx) - np.log10(mn))
    return (val - mn) / (mx - mn)


def denormalize(val_norm, param_range):
    scale, mn, mx = param_range
    if scale == "log":
        return 10 ** (val_norm * (np.log10(mx) - np.log10(mn)) + np.log10(mn))
    return val_norm * (mx - mn) + mn


def impedance(params, freq, circuit_type, a1=1.0, a2=1.0):
    """回路パラメータから複素インピーダンスを計算する。"""
    omega = 2 * np.pi * freq
    j_omega = 1j * omega

    def z_cpe(q, a):
        return 1 / (q * (j_omega ** a))

    def z_warburg(sigma):
        return sigma * (1 - 1j) / np.sqrt(omega)

    Rs, R1 = params["Rs"], params["R1"]
    Z_Q1 = z_cpe(params["Q1"], a1)

    if circuit_type == "C1":
        return Rs + 1 / (1 / R1 + 1 / Z_Q1)

    if circuit_type == "C2":
        R2 = params["R2"]
        Z_Q2 = z_cpe(params["Q2"], a2)
        return Rs + 1 / (1 / R1 + 1 / Z_Q1) + 1 / (1 / R2 + 1 / Z_Q2)

    if circuit_type == "C3":
        Z_W = z_warburg(params["sigma"])
        return Rs + 1 / (1 / (R1 + Z_W) + 1 / Z_Q1)

    raise ValueError(f"未対応の回路種別です: {circuit_type}")


def _sort_c2_by_time_constant(params, param_norm, param_keys):
    """C2回路の2つの並列素子(R1,Q1)と(R2,Q2)を時定数の大小で並べ替える。

    どちらの並列素子をノード1/2に割り当てても同じインピーダンスが得られてしまう
    ため、時定数 tau = R*Q の大小で一意に順序を固定し、学習の収束を安定させる。
    """
    tau1 = params["R1"] * params["Q1"]
    tau2 = params["R2"] * params["Q2"]
    if tau1 >= tau2:
        return params, param_norm

    params["R1"], params["R2"] = params["R2"], params["R1"]
    params["Q1"], params["Q2"] = params["Q2"], params["Q1"]

    ranges = {"R1": param_keys["R1"], "R2": param_keys["R2"],
              "Q1": param_keys["Q1"], "Q2": param_keys["Q2"]}
    for name, prange in ranges.items():
        idx = list(param_keys).index(name)
        param_norm[idx] = normalize(params[name], prange)

    return params, param_norm


def generate_dataset(env: CircuitEnv, seed: int = 42):
    """LHSサンプリングによりデータセット (X, y) を生成する。

    X: (N, channels, num_points) の入力（Nyquist / Bode 表現）
    y: (N, num_params) の正規化済みパラメータ（0〜1）
    """
    cfg = env.config
    freq = env.freq_range
    log_freq = np.log10(freq)

    sampler = qmc.LatinHypercube(d=env.num_params, seed=seed)
    samples = sampler.random(n=cfg["N_SAMPLES"])

    X_list, y_list = [], []
    for row in samples:
        params = {}
        y_norm = np.zeros(env.num_params)
        for i, name in enumerate(env.param_keys):
            y_norm[i] = row[i]
            params[name] = denormalize(row[i], env.param_ranges[name])

        if env.circuit_type == "C2":
            params, y_norm = _sort_c2_by_time_constant(params, y_norm, env.param_ranges)

        Z = impedance(params, freq, env.circuit_type, cfg["FIXED_A1"], cfg["FIXED_A2"])

        if cfg["INPUT_MODE"] == "bode":
            mag = np.clip(np.abs(Z), 1e-12, None)
            channels = [np.log10(mag), np.angle(Z)]
        else:
            channels = [Z.real, Z.imag]

        if cfg["INCLUDE_FREQ"]:
            channels = [log_freq] + channels

        X_list.append(np.vstack(channels))
        y_list.append(y_norm)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
