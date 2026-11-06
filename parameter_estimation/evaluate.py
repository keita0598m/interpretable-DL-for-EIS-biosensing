"""評価指標の算出とグラフ描画。"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from config import CircuitEnv
from dataset import denormalize


def evaluate(model, env: CircuitEnv, test_loader, device):
    model.eval()
    mu_list, y_true_list, y_pred_list = [], [], []

    import torch
    with torch.no_grad():
        for X, y in test_loader:
            _, preds, mu, _ = model(X.to(device))
            mu_list.append(mu.cpu().numpy())
            y_true_list.append(y.numpy())
            y_pred_list.append(preds.cpu().numpy())

    z_mu = np.concatenate(mu_list)
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    r2_scores = []
    for idx, name in enumerate(env.param_keys):
        yt = denormalize(y_true[:, idx], env.param_ranges[name])
        yp = denormalize(y_pred[:, idx], env.param_ranges[name])
        r2_scores.append(max(0.0, r2_score(yt, yp)))

    return z_mu, y_true, y_pred, r2_scores


def plot_loss_history(history, circuit_type, beta_warmup, delta_warmup):
    epochs = np.arange(1, len(history["total"]) + 1)
    labels = {"total": "Total", "recon": "Recon (MSE)", "kl": "KL Div",
              "param": "Param MAE", "piml": "Physics (PIML)", "indep": "Independence"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Training loss history ({circuit_type})")

    for ax, yscale in zip(axes, ["linear", "log"]):
        for key, label in labels.items():
            vals = np.clip(history[key], 1e-9, None) if yscale == "log" else history[key]
            lw = 2.0 if key == "total" else 1.2
            ls = "-" if key == "total" else "--"
            ax.plot(epochs, vals, lw=lw, ls=ls, label=label)

        if beta_warmup > 0:
            ax.axvline(beta_warmup, ls=":", alpha=0.7, label=f"beta end ({beta_warmup})")
        if delta_warmup > 0:
            ax.axvline(delta_warmup, ls=":", alpha=0.7, label=f"delta end ({delta_warmup})")

        ax.set_yscale(yscale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_r2_scores(r2_scores, env: CircuitEnv):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(env.num_params)
    ax.bar(x, r2_scores, width=0.5, edgecolor="black")
    ax.set_title(f"R2 scores ({env.circuit_type})")
    ax.set_ylabel("R2")
    ax.set_xticks(x)
    ax.set_xticklabels(env.param_keys)
    ax.grid(axis="y", ls="--", alpha=0.7)
    plt.show()


def plot_true_vs_predicted(y_true, y_pred, r2_scores, env: CircuitEnv):
    n_rows = int(np.ceil(env.num_params / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    fig.suptitle(f"True vs predicted ({env.circuit_type})")
    axes = axes.flatten()

    for idx, name in enumerate(env.param_keys):
        ax = axes[idx]
        yt = denormalize(y_true[:, idx], env.param_ranges[name])
        yp = denormalize(y_pred[:, idx], env.param_ranges[name])

        if env.param_ranges[name][0] == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.scatter(yt, yp, alpha=0.3, s=8)
        ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], "r--")
        ax.set_title(f"{name} (R2={r2_scores[idx]:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    for idx in range(env.num_params, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_latent_space(z_mu, y_true, env: CircuitEnv):
    """潜在空間の全2軸ペアを、各パラメータ値でカラーマップして可視化する。"""
    dim_pairs = list(itertools.combinations(range(env.config["LATENT_DIM"]), 2))

    fig, axes = plt.subplots(
        env.num_params, len(dim_pairs),
        figsize=(4 * len(dim_pairs), 3.5 * env.num_params),
        squeeze=False,
    )
    fig.suptitle(f"Latent space projections ({env.circuit_type})")

    for row, name in enumerate(env.param_keys):
        color_vals = y_true[:, row]
        for col, (dx, dy) in enumerate(dim_pairs):
            ax = axes[row][col]
            sc = ax.scatter(z_mu[:, dx], z_mu[:, dy], c=color_vals, cmap="turbo", s=8, alpha=0.8)
            ax.set_title(f"{name}: Z{dx} vs Z{dy}", fontsize=10)
            ax.set_xlabel(f"z_mu_{dx}")
            ax.set_ylabel(f"z_mu_{dy}")
            ax.grid(alpha=0.3)
            if col == len(dim_pairs) - 1:
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
