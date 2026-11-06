"""物理情報付き損失（PIML: Physics-Informed Machine Learning）。

推定したパラメータ値からインピーダンスを再計算し、真のインピーダンスとの
差（振幅・位相）を損失に加える。パラメータ誤差だけでは物理的に整合しない
組み合わせも許容してしまうため、この項で補正する。
"""

import numpy as np
import torch
import torch.nn as nn

from config import CircuitEnv


class PhysicsInformedLoss(nn.Module):
    def __init__(self, env: CircuitEnv, device: torch.device):
        super().__init__()
        self.env = env
        self.device = device

        self.omega = (2 * torch.pi * torch.tensor(env.freq_range, dtype=torch.float32, device=device)).unsqueeze(0)
        self.param_idx = {name: i for i, name in enumerate(env.param_keys)}

        is_log = [1.0 if env.param_ranges[k][0] == "log" else 0.0 for k in env.param_keys]
        mins = [env.param_ranges[k][1] for k in env.param_keys]
        maxs = [env.param_ranges[k][2] for k in env.param_keys]

        self.is_log = torch.tensor(is_log, device=device)
        self.mins = torch.tensor(mins, dtype=torch.float32, device=device)
        self.maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
        self.log_mins = torch.log10(self.mins)
        self.log_maxs = torch.log10(self.maxs)

    def denormalize(self, p_norm):
        lin_val = p_norm * (self.maxs - self.mins) + self.mins
        log_val = 10 ** (p_norm * (self.log_maxs - self.log_mins) + self.log_mins)
        return self.is_log * log_val + (1.0 - self.is_log) * lin_val

    def _column(self, p_phys, name):
        i = self.param_idx[name]
        return p_phys[:, i:i + 1]

    def compute_impedance(self, p_phys):
        omega = self.omega
        a1 = self.env.config["FIXED_A1"]

        Rs = self._column(p_phys, "Rs")
        R1 = self._column(p_phys, "R1")
        Q1 = self._column(p_phys, "Q1")

        def cpe(q, a):
            mag = torch.pow(omega, a)
            phase = torch.tensor(np.pi / 2.0 * a, dtype=torch.float32, device=self.device)
            j_omega_a = mag * (torch.cos(phase) + 1j * torch.sin(phase))
            return 1.0 / (q * j_omega_a)

        Z_Q1 = cpe(Q1, a1)
        circuit_type = self.env.circuit_type

        if circuit_type == "C1":
            return Rs + 1.0 / (1.0 / R1 + 1.0 / Z_Q1)

        if circuit_type == "C2":
            a2 = self.env.config["FIXED_A2"]
            R2 = self._column(p_phys, "R2")
            Q2 = self._column(p_phys, "Q2")
            Z_Q2 = cpe(Q2, a2)
            return Rs + 1.0 / (1.0 / R1 + 1.0 / Z_Q1) + 1.0 / (1.0 / R2 + 1.0 / Z_Q2)

        if circuit_type == "C3":
            sigma = self._column(p_phys, "sigma")
            Z_W = sigma * (1.0 - 1j) / torch.sqrt(omega)
            return Rs + 1.0 / (1.0 / (R1 + Z_W) + 1.0 / Z_Q1)

        raise ValueError(f"未対応の回路種別です: {circuit_type}")

    def forward(self, param_preds, target_params):
        Z_pred = self.compute_impedance(self.denormalize(param_preds))
        Z_true = self.compute_impedance(self.denormalize(target_params))

        mag_pred = torch.abs(Z_pred) + 1e-8
        mag_true = torch.abs(Z_true) + 1e-8
        loss_mag = torch.mean((torch.log10(mag_pred) - torch.log10(mag_true)) ** 2)
        loss_phase = torch.mean((torch.angle(Z_pred) - torch.angle(Z_true)) ** 2)

        return loss_mag + loss_phase
