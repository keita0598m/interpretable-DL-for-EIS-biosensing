"""等価回路分類モデルの学習・評価エントリーポイント。

使い方:
    python main.py
    python main.py --epochs 300 --latent-dim 2
"""

import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import torch

from config import CIRCUIT_TYPES, NUM_CLASSES, VAE_SETTINGS
from dataset import generate_dataset, generate_eval_dataset_with_params
from evaluate import (
    plot_classification_and_confidence_maps,
    plot_confusion_matrix,
    plot_decoded_grid,
    plot_latent_space,
    plot_loss_history,
    plot_parameter_maps,
)
from train import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="EIS等価回路分類モデルの学習")
    parser.add_argument("--epochs", type=int, default=VAE_SETTINGS["NUM_EPOCHS"])
    parser.add_argument("--latent-dim", type=int, default=VAE_SETTINGS["LATENT_DIM"])
    parser.add_argument("--input-mode", choices=["nyquist", "bode"], default="nyquist")
    parser.add_argument("--include-freq", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"circuits: {CIRCUIT_TYPES}")

    settings = {**VAE_SETTINGS, "NUM_EPOCHS": args.epochs, "LATENT_DIM": args.latent_dim}

    X_raw, y_raw, params_df = generate_dataset(mode=args.input_mode, include_freq=args.include_freq)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y_raw, range(len(y_raw)),
        test_size=settings["TEST_SIZE"], random_state=args.seed, stratify=y_raw,
    )
    params_test = params_df.iloc[list(idx_test)].reset_index(drop=True)

    model, accuracy, y_true, y_pred, latent_z, history, _ = train_and_evaluate(
        X_train, y_train, X_test, y_test, settings, NUM_CLASSES, device
    )

    print(f"\ntest accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CIRCUIT_TYPES))

    plot_latent_space(latent_z, y_true)
    plot_confusion_matrix(y_true, y_pred, accuracy)
    plot_loss_history(history, settings["BETA_WARMUP"], settings["GAMMA_WARMUP"])

    if settings["LATENT_DIM"] == 2:
        plot_decoded_grid(model, latent_z, scaler, device)
        plot_classification_and_confidence_maps(model, latent_z, y_true, device)
        plot_parameter_maps(latent_z, params_test)
    else:
        print("潜在空間が2次元でないため、格子デコード・マップ系の可視化はスキップしました。")

    return model, accuracy


if __name__ == "__main__":
    main()
