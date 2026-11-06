"""EISデータ用の1D CNN VAE。

全結合層のみの VAE では周波数方向の局所的な特徴を捉えにくかったため、
Encoder を VGG 風の1次元畳み込みブロックに置き換えている。潜在変数からは
Decoder（再構築用）と、パラメータごとに独立した推定ヘッドの両方に分岐する。
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
        nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )


class CircuitParameterVAE(nn.Module):
    def __init__(self, in_channels: int, param_keys: list[str], latent_dim: int = 4):
        super().__init__()
        self.param_keys = param_keys

        self.encoder_cnn = nn.Sequential(
            _conv_block(in_channels, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        # 80点入力を5回ダウンサンプリングすると系列長は2になる
        self.flatten_dim = 256 * 2

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, self.flatten_dim), nn.ReLU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(size=5, mode="nearest"),
            nn.Conv1d(256, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, in_channels, kernel_size=3, padding=1),
        )

        # パラメータごとに独立した推定ヘッド（0〜1に正規化した値をSigmoidで出力）
        self.estimators = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid(),
            )
            for name in param_keys
        })

    def encode(self, x):
        h = self.encoder_cnn(x)
        h = h.flatten(start_dim=1)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, -1)
        return self.decoder_cnn(h)

    def estimate_params(self, mu):
        preds = [self.estimators[name](mu) for name in self.param_keys]
        return torch.cat(preds, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        param_preds = self.estimate_params(mu)  # 推定はサンプリング前のmuを使用
        return x_recon, param_preds, mu, logvar
