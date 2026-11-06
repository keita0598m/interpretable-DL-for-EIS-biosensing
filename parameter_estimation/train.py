"""学習ループ。

損失は 再構築(MSE) + β・KL + δ・(パラメータMAE + α・PIML) + γ・独立性正則化 の
重み付き和。β, δ は学習初期を抑えてから徐々に本来の重みまで引き上げる
（アニーリング）ことで、再構築とパラメータ推定のバランスが崩れるのを防いでいる。
"""

import torch
import torch.nn as nn
from torch import optim

from losses import PhysicsInformedLoss


def _anneal(epoch, start, end, warmup_epochs):
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return end
    return start + (end - start) / warmup_epochs * epoch


def _independence_loss(mu):
    """潜在次元を前半・後半に分け、相関を抑制する正則化項。

    パラメータ数に対して潜在空間の情報が偏らないよう、次元間の共分散を
    ペナルティとして加える。
    """
    if mu.size(0) <= 1 or mu.size(1) < 2:
        return torch.tensor(0.0, device=mu.device)

    centered = mu - mu.mean(dim=0)
    cov = (centered.T @ centered) / (mu.size(0) - 1)
    half = mu.size(1) // 2
    cross_cov = cov[:half, half:]
    return torch.mean(cross_cov ** 2)


def train(model, env, train_loader, device, piml_alpha: float = 0.1, indep_weight: float = 10.0):
    cfg = env.config
    optimizer = optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"])
    mae_loss = nn.L1Loss()
    piml_loss = PhysicsInformedLoss(env, device)

    history = {"total": [], "recon": [], "kl": [], "param": [], "piml": [], "indep": []}
    log_interval = max(1, cfg["NUM_EPOCHS"] // 10)

    for epoch in range(cfg["NUM_EPOCHS"]):
        model.train()
        beta = _anneal(epoch, cfg["BETA_START"], cfg["BETA_END"], cfg["BETA_WARMUP"])
        delta = _anneal(epoch, cfg["DELTA_START"], cfg["DELTA_END"], cfg["DELTA_WARMUP"])
        piml_weight = piml_alpha * min(1.0, epoch / 20.0)

        running = {k: 0.0 for k in history}
        n_batches = len(train_loader)

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            x_recon, param_preds, mu, logvar = model(X)

            recon = nn.functional.mse_loss(x_recon, X)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            param_err = mae_loss(param_preds, y)
            phys_err = piml_loss(param_preds, y)
            indep = _independence_loss(mu)

            regression_term = param_err + piml_weight * phys_err
            loss = recon + beta * kl + delta * regression_term + indep_weight * indep

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running["total"] += loss.item() / n_batches
            running["recon"] += recon.item() / n_batches
            running["kl"] += kl.item() / n_batches
            running["param"] += param_err.item() / n_batches
            running["piml"] += phys_err.item() / n_batches
            running["indep"] += indep.item() / n_batches

        for k, v in running.items():
            history[k].append(v)

        if (epoch + 1) % log_interval == 0:
            print(
                f"[{env.circuit_type}] epoch {epoch + 1:5d}  "
                f"total={running['total']:.4f}  recon={running['recon']:.4f}  "
                f"kl={running['kl']:.4f}  param={running['param']:.4f}  "
                f"piml={running['piml']:.4f}  indep={running['indep']:.4f}"
            )

    return model, history
