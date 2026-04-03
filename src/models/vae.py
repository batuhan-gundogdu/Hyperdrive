import torch
import torch.nn as nn

import config


class ResBlock1d(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    """1D-CNN encoder with residual blocks.

    Input: (B, 250) -> reshape to (B, 1, 250)
    Conv layers: 250 -> 125 -> 63 -> 32
    Residual blocks after each downsampling for better feature extraction.
    Then flatten + FC to get mu and logvar.
    """
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),   # 250 -> 125
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            ResBlock1d(32, dropout=0.1),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 125 -> 63
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            ResBlock1d(64, dropout=0.1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # 63 -> 32
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResBlock1d(128, dropout=0.1),
        )
        # 128 * 32 = 4096
        self.fc = nn.Sequential(
            nn.Linear(128 * 32, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (B, 250) -> (B, 1, 250)
        h = x.unsqueeze(1)
        h = self.conv(h)
        h = h.flatten(1)  # (B, 4096)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class Decoder(nn.Module):
    """1D-CNN decoder with residual blocks.

    Mirrors the encoder: FC -> reshape -> ConvTranspose1d layers.
    32 -> 63 -> 125 -> 250
    """
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 128 * 32),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            ResBlock1d(128, dropout=0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 32 -> 63
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            ResBlock1d(64, dropout=0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0),   # 63 -> 125
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            ResBlock1d(32, dropout=0.1),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),    # 125 -> 250
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 32)
        h = self.deconv(h)
        return h.squeeze(1)  # (B, 250)


class VAE(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x):
        """Encode without reparameterization (for inference)."""
        return self.encoder(x)

    def reconstruct(self, x):
        """Full forward pass using the mean (no sampling noise)."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)
