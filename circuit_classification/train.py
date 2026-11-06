"""学習ループ。

損失は 再構築(MSE) + β・KL + γ・分類誤差(CrossEntropy) の重み付き和。
β, γ は学習初期を0近くに抑えてから徐々に引き上げる（アニーリング）ことで、
分類器が未成熟な潜在空間に引っ張られて学習が不安定になるのを防いでいる。
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from model import VAEClassifier


def _anneal(epoch, start, end, warmup_epochs):
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return end
    return start + (end - start) / warmup_epochs * epoch


def train_and_evaluate(X_train, y_train, X_test, y_test, settings: dict, num_classes: int, device):
    input_dim = X_train.shape[1]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=settings["BATCH_SIZE"], shuffle=True,
    )

    model = VAEClassifier(input_dim, settings["LATENT_DIM"], num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=settings["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=settings["LR_DECAY_STEP"], gamma=settings["LR_DECAY_GAMMA"]
    )

    history = {"total": [], "recon": [], "kl": [], "cls": []}

    model.train()
    for epoch in range(settings["NUM_EPOCHS"]):
        beta = _anneal(epoch, settings["BETA_START"], settings["BETA_END"], settings["BETA_WARMUP"])
        gamma = _anneal(epoch, settings["GAMMA_START"], settings["GAMMA_END"], settings["GAMMA_WARMUP"])

        running = {"total": 0.0, "recon": 0.0, "kl": 0.0, "cls": 0.0}

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            x_recon, logits, mu, logvar = model(X_b)
            recon_loss = nn.functional.mse_loss(x_recon, X_b)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            cls_loss = nn.functional.cross_entropy(logits, y_b)

            loss = recon_loss + beta * kl_loss + gamma * cls_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["GRAD_CLIP"])
            optimizer.step()

            n = X_b.size(0)
            running["total"] += loss.item() * n
            running["recon"] += recon_loss.item() * n
            running["kl"] += kl_loss.item() * n
            running["cls"] += cls_loss.item() * n

        scheduler.step()
        n_train = len(train_loader.dataset)
        for k in history:
            history[k].append(running[k] / n_train)

        if (epoch + 1) % 20 == 0:
            print(
                f"epoch {epoch + 1:3d}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"total={history['total'][-1]:.4f}  recon={history['recon'][-1]:.4f}  "
                f"kl={history['kl'][-1]:.4f}  cls={history['cls'][-1]:.4f}"
            )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        x_recon_test, logits, mu, _ = model(X_test_t)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        latent_z = mu.cpu().numpy()
        x_recon_test = x_recon_test.cpu().numpy()

    accuracy = float(np.mean(y_pred == y_test))
    return model, accuracy, y_test, y_pred, latent_z, history, x_recon_test
