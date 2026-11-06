"""評価指標の算出とグラフ描画。

分類そのものの評価（潜在空間・混同行列・損失推移）に加えて、VAEの潜在空間が
どのように解釈できるかを見るための可視化（分類マップ・信頼度マップ・
回路パラメータとの対応関係）をまとめている。
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from config import CIRCUIT_PARAMS, CIRCUIT_TYPES, CMAP, COLOR_LIST, NORM, NUM_CLASSES, NUM_POINTS, TICK_LOCS


def plot_latent_space(latent_z, y_true, title="Latent Space"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(latent_z[:, 0], latent_z[:, 1], c=y_true, cmap=CMAP, norm=NORM, s=8, alpha=0.4)
    cbar = plt.colorbar(sc, ax=ax, ticks=TICK_LOCS)
    cbar.ax.set_yticklabels(CIRCUIT_TYPES)
    ax.set_title(title)
    ax.set_xlabel("L1 (Latent Dim 1)")
    ax.set_ylabel("L2 (Latent Dim 2)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, accuracy):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, matrix, title, fmt in zip(
        axes, [cm, cm_norm], ["Confusion Matrix (Count)", "Confusion Matrix (Normalized)"], ["d", ".2f"]
    ):
        im = ax.imshow(matrix, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CIRCUIT_TYPES)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CIRCUIT_TYPES)
        thresh = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if matrix[i, j] > thresh else "black"
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color=color)

    plt.suptitle(f"Overall accuracy: {accuracy * 100:.2f}%", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_loss_history(history, beta_warmup, gamma_warmup):
    epochs = range(1, len(history["total"]) + 1)
    items = [
        ("total", "Total Loss"),
        ("recon", "Reconstruction Loss"),
        ("kl", "KL Divergence"),
        ("cls", "Classification Loss"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, label) in zip(axes.flatten(), items):
        ax.plot(epochs, history[key])
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(ls="--", alpha=0.5)
        if key == "kl":
            ax.axvline(beta_warmup, color="gray", ls=":", label="beta warmup end")
            ax.legend()
        if key == "cls":
            ax.axvline(gamma_warmup, color="gray", ls=":", label="gamma warmup end")
            ax.legend()

    plt.suptitle("Training Loss History")
    plt.tight_layout()
    plt.show()


def _latent_mesh(latent_z, resolution=300, margin=0.5):
    l1_min, l1_max = latent_z[:, 0].min() - margin, latent_z[:, 0].max() + margin
    l2_min, l2_max = latent_z[:, 1].min() - margin, latent_z[:, 1].max() + margin
    l1_mesh, l2_mesh = np.meshgrid(
        np.linspace(l1_min, l1_max, resolution), np.linspace(l2_min, l2_max, resolution)
    )
    extent = [l1_min, l1_max, l2_min, l2_max]
    return l1_mesh, l2_mesh, extent


def plot_classification_and_confidence_maps(model, latent_z, y_true, device, resolution=300, batch=4096):
    """潜在空間全体に対する分類マップと、その確信度（最大softmax確率）マップを描画する。"""
    l1_mesh, l2_mesh, extent = _latent_mesh(latent_z, resolution)
    mesh_points = torch.tensor(
        np.stack([l1_mesh.ravel(), l2_mesh.ravel()], axis=1), dtype=torch.float32
    ).to(device)

    preds, confidences = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(mesh_points), batch):
            logits = model.classifier(mesh_points[i:i + batch])
            probs = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            confidences.append(probs.max(dim=1).values.cpu().numpy())

    pred_mesh = np.concatenate(preds).reshape(resolution, resolution)
    conf_mesh = np.concatenate(confidences).reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(pred_mesh, origin="lower", extent=extent, cmap=CMAP, norm=NORM, aspect="auto", alpha=0.5)
    ax.scatter(latent_z[:, 0], latent_z[:, 1], c=y_true, cmap=CMAP, norm=NORM, s=5, alpha=0.3)
    handles = [matplotlib.patches.Patch(color=COLOR_LIST[i], label=CIRCUIT_TYPES[i]) for i in range(NUM_CLASSES)]
    ax.legend(handles=handles, loc="upper right", title="Circuit Class")
    ax.set_title("Latent Space Classification Map")
    ax.set_xlabel("L1"); ax.set_ylabel("L2")
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_mesh, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Max softmax probability")
    ax.scatter(latent_z[:, 0], latent_z[:, 1], c="white", s=3, alpha=0.2)
    ax.set_title("Classification Confidence Map")
    ax.set_xlabel("L1"); ax.set_ylabel("L2")
    plt.tight_layout(); plt.show()


def plot_decoded_grid(model, latent_z, scaler, device, grid_n=5, span=1.0):
    """潜在空間の一部を格子状にサンプリングし、デコードしてナイキスト線図として並べる。

    潜在空間の各点が「実際にどんなインピーダンス応答に対応するか」を
    直感的に確認できる図。
    """
    z_mean = latent_z.mean(axis=0)
    l1_centers = np.linspace(z_mean[0] - span, z_mean[0] + span, grid_n)
    l2_centers = np.linspace(z_mean[1] - span, z_mean[1] + span, grid_n)

    decoded, predicted_class = {}, {}
    model.eval()
    with torch.no_grad():
        for row in range(grid_n):
            for col in range(grid_n):
                z_pt = torch.tensor([[l1_centers[col], l2_centers[row]]], dtype=torch.float32).to(device)
                x_dec = model.decode(z_pt).cpu().numpy()[0]
                x_dec = scaler.inverse_transform(x_dec.reshape(1, -1))[0]
                decoded[(row, col)] = (x_dec[:NUM_POINTS], x_dec[NUM_POINTS:])
                predicted_class[(row, col)] = torch.argmax(model.classifier(z_pt), dim=1).item()

    fig, axes = plt.subplots(grid_n, grid_n, figsize=(15, 12))
    fig.suptitle("Decoded Nyquist plots across the latent grid (row: L2, col: L1)")
    for row in range(grid_n):
        for col in range(grid_n):
            ax = axes[grid_n - 1 - row][col]
            real, imag = decoded[(row, col)]
            cls = predicted_class[(row, col)]
            color = COLOR_LIST[cls]
            ax.plot(real, -imag, color=color, linewidth=1.0)
            ax.text(0.97, 0.97, CIRCUIT_TYPES[cls], transform=ax.transAxes, fontsize=6,
                    ha="right", va="top", color=color, fontweight="bold")
            ax.tick_params(labelsize=4, length=2)
            if row != 0:
                ax.set_xticklabels([])
            if col != 0:
                ax.set_yticklabels([])

    handles = [matplotlib.patches.Patch(color=COLOR_LIST[i], label=CIRCUIT_TYPES[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.01), title="Predicted class")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    plt.show()


def plot_parameter_maps(latent_z, params_df):
    """潜在空間上に、各回路パラメータの真値をカラーマップとして重ねる。

    その回路で使われていないパラメータはグレーの背景点として表示する。
    """
    plot_params = list(CIRCUIT_PARAMS["C4"])  # 全パラメータを含む回路を基準に列挙
    ncols = 3
    nrows = (len(plot_params) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(plot_params):
        ax = axes[i]
        values = params_df[name].values
        valid = ~np.isnan(values)

        ax.scatter(latent_z[:, 0], latent_z[:, 1], c="lightgray", s=3, alpha=0.3)
        if valid.sum() == 0:
            ax.set_title(f"{name} (該当なし)")
            continue

        vmin, vmax = np.percentile(values[valid], [1, 99])
        sc = ax.scatter(latent_z[valid, 0], latent_z[valid, 1], c=values[valid],
                         cmap="plasma", s=8, alpha=0.6, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="Value")

        used_by = [c for c, ps in CIRCUIT_PARAMS.items() if name in ps and c in CIRCUIT_TYPES]
        ax.set_title(f"{name}\n(used by: {', '.join(used_by)})")
        ax.set_xlabel("L1"); ax.set_ylabel("L2")

    for j in range(len(plot_params), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Circuit parameters mapped onto latent space\n(gray = parameter not used by that circuit)")
    plt.tight_layout()
    plt.show()
