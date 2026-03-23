import torch
import torch.nn as nn

import config


class Encoder(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM,
                 latent_dim=config.LATENT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.tanh(self.fc1(x))
        h = self.tanh(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM, hidden_dim=config.HIDDEN_DIM,
                 output_dim=config.INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        h = self.tanh(self.fc1(z))
        h = self.tanh(self.fc2(h))
        return self.fc_out(h)


class VAE(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, hidden_dim=config.HIDDEN_DIM,
                 latent_dim=config.LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

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
