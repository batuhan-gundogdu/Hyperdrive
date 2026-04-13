import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FactoredTDNNLayer(nn.Module):
    """Time Delay Neural Network layer using factored linear projections.

    Equivalent to a 1D convolution with kernel_size=3, but implemented as
    three separate Linear(dim, dim) projections (left/center/right context)
    whose outputs are summed. This keeps every matmul at dim x dim (square),
    fitting the hardware constraint of max 256 I/O.

    Uses pre-norm residual connection for stable deep stacking.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_left = nn.Linear(dim, dim)
        self.proj_center = nn.Linear(dim, dim)
        self.proj_right = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)

        # Shift for left context (frame t-1) — pad a zero frame at the start
        left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        # Shift for right context (frame t+1) — pad a zero frame at the end
        right = F.pad(x[:, 1:, :], (0, 0, 0, 1))

        out = self.proj_left(left) + self.proj_center(x) + self.proj_right(right)
        out = F.leaky_relu(out, 0.2)
        out = self.dropout(out)
        return out + residual


class PointwiseFFN(nn.Module):
    """Pointwise feedforward: Linear(dim, dim) with pre-norm residual.

    Square matmul that adds per-frame capacity between TDNN layers.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        return x + residual


class TDNNBlock(nn.Module):
    """One TDNN layer followed by a pointwise FFN, both with residuals."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.tdnn = FactoredTDNNLayer(dim, dropout)
        self.ffn = PointwiseFFN(dim, dropout)

    def forward(self, x):
        x = self.tdnn(x)
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    """TDNN encoder for VAE.

    Splits input into frames, projects to feature dim, processes with
    stacked TDNN blocks, then pools and projects to latent space.

    All Linear layers are either square (dim x dim) or smaller than 256 I/O.
    """
    def __init__(self, input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
                 frame_size=config.FRAME_SIZE, feature_dim=config.FEATURE_DIM,
                 n_blocks=config.ENCODER_BLOCKS, dropout=0.1):
        super().__init__()
        self.frame_size = frame_size
        self.n_frames = input_dim // frame_size

        # Frame embedding: Linear(frame_size, feature_dim)
        self.frame_embed = nn.Linear(frame_size, feature_dim)
        self.embed_norm = nn.LayerNorm(feature_dim)

        # Stacked TDNN blocks
        self.blocks = nn.ModuleList([
            TDNNBlock(feature_dim, dropout) for _ in range(n_blocks)
        ])

        # Final projection to latent space
        self.final_norm = nn.LayerNorm(feature_dim)
        self.fc_pre = nn.Linear(feature_dim, feature_dim)  # square
        self.fc_pre_norm = nn.LayerNorm(feature_dim)
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

    def forward(self, x):
        # x: (B, 250)
        B = x.shape[0]

        # Split into frames: (B, 250) -> (B, 10, 25)
        x = x.view(B, self.n_frames, self.frame_size)

        # Frame embedding: (B, 10, 25) -> (B, 10, feature_dim)
        x = self.frame_embed(x)
        x = self.embed_norm(x)
        x = F.leaky_relu(x, 0.2)

        # TDNN blocks
        for block in self.blocks:
            x = block(x)

        # Global average pool over frames: (B, 10, D) -> (B, D)
        x = self.final_norm(x)
        x = x.mean(dim=1)

        # Project to latent
        x = self.fc_pre(x)
        x = self.fc_pre_norm(x)
        x = F.leaky_relu(x, 0.2)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class Decoder(nn.Module):
    """TDNN decoder for VAE.

    Projects latent code to feature dim, broadcasts to all frame positions
    with learned positional embedding, refines with TDNN blocks, then
    decodes each frame back to samples.
    """
    def __init__(self, input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
                 frame_size=config.FRAME_SIZE, feature_dim=config.FEATURE_DIM,
                 n_blocks=config.DECODER_BLOCKS, dropout=0.1):
        super().__init__()
        self.frame_size = frame_size
        self.n_frames = input_dim // frame_size

        # Latent to feature dim
        self.fc_in = nn.Linear(latent_dim, feature_dim)
        self.fc_in_norm = nn.LayerNorm(feature_dim)
        self.fc_expand = nn.Linear(feature_dim, feature_dim)  # square
        self.fc_expand_norm = nn.LayerNorm(feature_dim)

        # Learned positional embedding for frame positions
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_frames, feature_dim) * 0.02)

        # Stacked TDNN blocks
        self.blocks = nn.ModuleList([
            TDNNBlock(feature_dim, dropout) for _ in range(n_blocks)
        ])

        # Frame decode: feature_dim -> frame_size per frame
        self.final_norm = nn.LayerNorm(feature_dim)
        self.frame_decode = nn.Linear(feature_dim, frame_size)

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc_in(z)
        x = self.fc_in_norm(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc_expand(x)
        x = self.fc_expand_norm(x)
        x = F.leaky_relu(x, 0.2)

        # Broadcast to all frame positions: (B, D) -> (B, n_frames, D)
        x = x.unsqueeze(1).expand(-1, self.n_frames, -1)
        x = x + self.pos_embed

        # TDNN blocks
        for block in self.blocks:
            x = block(x)

        # Decode frames: (B, n_frames, D) -> (B, n_frames, frame_size)
        x = self.final_norm(x)
        x = self.frame_decode(x)

        # Reshape: (B, n_frames, frame_size) -> (B, input_dim)
        return x.reshape(x.shape[0], -1)


class VAE(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

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
