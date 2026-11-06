"""

使い方:
    python main.py --circuits C1 C2 C3
    python main.py --circuits C3 --epochs 2000 --latent-dim 4
"""

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import CircuitEnv
from dataset import generate_dataset
from evaluate import evaluate, plot_latent_space, plot_loss_history, plot_r2_scores, plot_true_vs_predicted
from model import CircuitParameterVAE
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="EIS回路パラメータ推定モデルの学習")
    parser.add_argument("--circuits", nargs="+", default=["C1", "C2", "C3"], help="学習する回路（例: C1 C2 C3）")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=30000, help="回路あたりのサンプル数")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def scale_inputs(X):
    """(N, C, L) を チャンネルごとに標準化する。"""
    n, c, length = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, c)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat)
    return scaled.reshape(n, length, c).transpose(0, 2, 1), scaler


def run_circuit(circuit_type, args, device):
    print(f"\n--- {circuit_type} の学習を開始 ---")

    env = CircuitEnv(circuit_type, config={
        "N_SAMPLES": args.samples,
        "NUM_EPOCHS": args.epochs,
        "LATENT_DIM": args.latent_dim,
    })

    X, y = generate_dataset(env, seed=args.seed)
    X, _ = scale_inputs(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=env.config["TEST_SIZE"], random_state=args.seed
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=env.config["BATCH_SIZE"], shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=env.config["BATCH_SIZE"], shuffle=False,
    )

    model = CircuitParameterVAE(
        in_channels=env.input_channels,
        param_keys=env.param_keys,
        latent_dim=env.config["LATENT_DIM"],
    ).to(device)

    model, history = train(model, env, train_loader, device)

    z_mu, y_true, y_pred, r2_scores = evaluate(model, env, test_loader, device)
    print(f"{circuit_type} R2: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(env.param_keys, r2_scores)))

    plot_loss_history(history, circuit_type, env.config["BETA_WARMUP"], env.config["DELTA_WARMUP"])
    plot_r2_scores(r2_scores, env)
    plot_true_vs_predicted(y_true, y_pred, r2_scores, env)
    plot_latent_space(z_mu, y_true, env)

    return model, r2_scores


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results = {}
    for circuit_type in args.circuits:
        _, r2_scores = run_circuit(circuit_type, args, device)
        results[circuit_type] = r2_scores

    print("\n全回路の学習が完了しました。")
    for circuit_type, r2_scores in results.items():
        print(f"  {circuit_type}: {np.round(r2_scores, 3).tolist()}")


if __name__ == "__main__":
    main()
