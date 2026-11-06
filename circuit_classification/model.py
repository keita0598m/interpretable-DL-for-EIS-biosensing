"""等価回路分類用の VAE。

潜在変数（既定では2次元）に分類器を接続し、再構築誤差・KLダイバージェンス・
分類誤差を同時に最小化する。潜在空間を2次元に絞ることで、学習後に
散布図としてそのまま可視化・解釈できるようにしている。
"""

import torch
import torch.nn as nn


class VAEClassifier(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        class_logits = self.classifier(z)
        return x_recon, class_logits, mu, logvar
