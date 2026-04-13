import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from src.models.vae import FactoredTDNNLayer, PointwiseFFN, TDNNBlock


class VectorQuantizer(nn.Module):
    """Vector quantization layer with class-conditional codebook routing.

    During training with labels: normal inputs are routed to normal codes,
    abnormal inputs to abnormal codes (masked nearest-neighbor).
    During inference (no labels): unmasked nearest-neighbor assignment.
    """

    def __init__(self, num_codes=config.NUM_CODES, embed_dim=config.EMBED_DIM,
                 normal_code_ids=None, commitment_cost=config.COMMITMENT_COST,
                 codebook_cost=config.CODEBOOK_COST):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.codebook_cost = codebook_cost
        self.normal_code_ids = set(normal_code_ids or config.NORMAL_CODE_IDS)

        self.codebook = nn.Embedding(num_codes, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e, labels=None):
        """
        Args:
            z_e: (B, embed_dim) encoder output
            labels: (B,) binary labels, 0=normal 1=abnormal. None for inference.

        Returns:
            z_q: (B, embed_dim) quantized vectors (straight-through)
            indices: (B,) selected codebook indices
            codebook_loss: scalar
            commitment_loss: scalar
            cls_loss: classification loss (0 if no labels)
        """
        # Distances: (B, num_codes)
        distances = torch.cdist(z_e.unsqueeze(1), self.codebook.weight.unsqueeze(0)).squeeze(1)

        # Classification loss: push encoder to separate classes in codebook space
        cls_loss = torch.tensor(0.0, device=z_e.device)
        if labels is not None:
            # Softmax over negative distances → soft assignment probabilities
            logits = -distances  # (B, K) — closer = higher logit
            probs = F.softmax(logits, dim=1)  # (B, K)
            # Sum probabilities assigned to normal codes
            normal_mask_codes = torch.tensor(
                [i in self.normal_code_ids for i in range(self.num_codes)],
                device=z_e.device,
            )
            p_normal = probs[:, normal_mask_codes].sum(dim=1)  # (B,)
            # Binary cross-entropy: label 0 = normal (target p_normal=1)
            target_normal = (labels == 0).float()
            cls_loss = F.binary_cross_entropy(p_normal, target_normal)

            # Class-conditional routing: mask invalid codes per sample
            mask = torch.full_like(distances, float("inf"))
            for i in range(self.num_codes):
                if i in self.normal_code_ids:
                    mask[:, i] = torch.where(labels == 0, 0.0, float("inf"))
                else:
                    mask[:, i] = torch.where(labels == 1, 0.0, float("inf"))
            distances_masked = distances + mask
        else:
            distances_masked = distances

        indices = distances_masked.argmin(dim=1)  # (B,)
        z_q = self.codebook(indices)               # (B, embed_dim)

        # Losses
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, codebook_loss, commitment_loss, cls_loss

    def get_code_type(self, indices):
        """Return 0 for normal codes, 1 for abnormal codes."""
        is_abnormal = torch.ones_like(indices)
        for nid in self.normal_code_ids:
            is_abnormal = is_abnormal & (indices != nid)
        return is_abnormal.long()


class VQEncoder(nn.Module):
    """TDNN encoder for VQ-VAE. Outputs a single continuous vector (no mu/logvar)."""

    def __init__(self, input_dim=config.INPUT_DIM, embed_dim=config.EMBED_DIM,
                 frame_size=config.FRAME_SIZE, feature_dim=config.FEATURE_DIM,
                 n_blocks=config.ENCODER_BLOCKS, dropout=0.1):
        super().__init__()
        self.frame_size = frame_size
        self.n_frames = input_dim // frame_size

        self.frame_embed = nn.Linear(frame_size, feature_dim)
        self.embed_norm = nn.LayerNorm(feature_dim)

        self.blocks = nn.ModuleList([
            TDNNBlock(feature_dim, dropout) for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(feature_dim)
        self.fc_pre = nn.Linear(feature_dim, feature_dim)
        self.fc_pre_norm = nn.LayerNorm(feature_dim)
        self.fc_out = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.n_frames, self.frame_size)
        x = self.frame_embed(x)
        x = self.embed_norm(x)
        x = F.leaky_relu(x, 0.2)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)

        x = self.fc_pre(x)
        x = self.fc_pre_norm(x)
        x = F.leaky_relu(x, 0.2)
        z_e = self.fc_out(x)
        return z_e


class VQDecoder(nn.Module):
    """TDNN decoder for VQ-VAE. Takes quantized codebook vector as input."""

    def __init__(self, input_dim=config.INPUT_DIM, embed_dim=config.EMBED_DIM,
                 frame_size=config.FRAME_SIZE, feature_dim=config.FEATURE_DIM,
                 n_blocks=config.DECODER_BLOCKS, dropout=0.1):
        super().__init__()
        self.frame_size = frame_size
        self.n_frames = input_dim // frame_size

        self.fc_in = nn.Linear(embed_dim, feature_dim)
        self.fc_in_norm = nn.LayerNorm(feature_dim)
        self.fc_expand = nn.Linear(feature_dim, feature_dim)
        self.fc_expand_norm = nn.LayerNorm(feature_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_frames, feature_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TDNNBlock(feature_dim, dropout) for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(feature_dim)
        self.frame_decode = nn.Linear(feature_dim, frame_size)

    def forward(self, z_q):
        x = self.fc_in(z_q)
        x = self.fc_in_norm(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc_expand(x)
        x = self.fc_expand_norm(x)
        x = F.leaky_relu(x, 0.2)

        x = x.unsqueeze(1).expand(-1, self.n_frames, -1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = self.frame_decode(x)
        return x.reshape(x.shape[0], -1)


class VQVAE(nn.Module):
    def __init__(self, embed_dim=config.EMBED_DIM, num_codes=config.NUM_CODES):
        super().__init__()
        self.encoder = VQEncoder(embed_dim=embed_dim)
        self.vq = VectorQuantizer(num_codes=num_codes, embed_dim=embed_dim)
        self.decoder = VQDecoder(embed_dim=embed_dim)

    def forward(self, x, labels=None):
        """
        Args:
            x: (B, 250) input signal
            labels: (B,) 0=normal, 1=abnormal. None for inference.
        Returns:
            x_recon, z_e, z_q, indices, codebook_loss, commitment_loss, cls_loss
        """
        z_e = self.encoder(x)
        z_q, indices, cb_loss, commit_loss, cls_loss = self.vq(z_e, labels=labels)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, indices, cb_loss, commit_loss, cls_loss

    def encode(self, x):
        """Encode and quantize (no labels — unmasked assignment)."""
        z_e = self.encoder(x)
        z_q, indices, _, _, _ = self.vq(z_e, labels=None)
        return z_e, z_q, indices

    def reconstruct(self, x):
        """Full forward pass for inference (no labels)."""
        z_e = self.encoder(x)
        z_q, indices, _, _, _ = self.vq(z_e, labels=None)
        x_recon = self.decoder(z_q)
        return x_recon, indices
