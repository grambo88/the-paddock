"""
model.py
========
Transformer-based race encoder for Win / Place / Show probability prediction.

Architecture
------------
Each race is treated as a SET of horses. The Transformer encoder lets every
horse attend to every other horse in the same race, learning inter-horse
relationships (e.g. "this horse is a heavy favourite in a weak field").

Input  : (batch, MAX_HORSES, N_FEATURES)
Output : (batch, MAX_HORSES, 3)  — softmax probabilities for Win, Place, Show

The output probabilities for each target (Win, Place, Show) sum to 1.0
across all horses in a race, i.e. "which horse is most likely to win".

Framework: PyTorch (framework-agnostic design — easy to port to TF/JAX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from features import N_FEATURES, MAX_HORSES


class HorseEmbedding(nn.Module):
    """
    Projects raw per-horse features into a higher-dimensional embedding space.
    Acts as the input layer before the Transformer.
    """
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (batch, horses, features)
        return self.net(x)


class RaceTransformer(nn.Module):
    """
    Transformer encoder that lets horses attend to each other within a race.
    Padding slots (empty horses) are masked out via key_padding_mask.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,   # (batch, seq, features)
            norm_first     = True,   # Pre-norm = more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, padding_mask):
        # padding_mask: True = IGNORE (padding), False = attend (real horse)
        # PyTorch TransformerEncoder uses True = ignore convention
        return self.encoder(x, src_key_padding_mask=~padding_mask)


class RacePredictor(nn.Module):
    """
    Full model: embedding → transformer → per-horse probability heads.

    Output is three softmax distributions (Win, Place, Show),
    each summing to 1.0 across all real horses in the race.
    """
    def __init__(
        self,
        n_features:      int   = N_FEATURES,
        d_model:         int   = 64,
        n_heads:         int   = 4,
        n_layers:        int   = 3,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.embedding   = HorseEmbedding(n_features, d_model, dropout)
        self.transformer = RaceTransformer(d_model, n_heads, n_layers,
                                           dim_feedforward, dropout)
        # Three separate output heads — one per target
        self.win_head   = nn.Linear(d_model, 1)
        self.place_head = nn.Linear(d_model, 1)
        self.show_head  = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        """
        Args:
            x    : (batch, MAX_HORSES, N_FEATURES)  float tensor
            mask : (batch, MAX_HORSES)               bool tensor — True = real horse

        Returns:
            probs: (batch, MAX_HORSES, 3)  softmax probabilities
                   [:, :, 0] = win prob
                   [:, :, 1] = place prob
                   [:, :, 2] = show prob
                   Padding positions have probability 0.
        """
        # Embed features → (batch, horses, d_model)
        emb = self.embedding(x)

        # Transformer — horses attend to each other
        enc = self.transformer(emb, mask)

        # Per-horse logits for each target
        win_logits   = self.win_head(enc).squeeze(-1)    # (batch, horses)
        place_logits = self.place_head(enc).squeeze(-1)
        show_logits  = self.show_head(enc).squeeze(-1)

        # Mask padding before softmax — set padding logits to -inf
        INF = float('-inf')
        win_logits   = win_logits.masked_fill(~mask,   INF)
        place_logits = place_logits.masked_fill(~mask, INF)
        show_logits  = show_logits.masked_fill(~mask,  INF)

        # Softmax across horses — probabilities sum to 1 per race per target
        win_probs   = F.softmax(win_logits,   dim=-1)
        place_probs = F.softmax(place_logits, dim=-1)
        show_probs  = F.softmax(show_logits,  dim=-1)

        # Stack → (batch, horses, 3)
        probs = torch.stack([win_probs, place_probs, show_probs], dim=-1)

        # Zero out padding positions explicitly
        probs = probs * mask.unsqueeze(-1).float()

        return probs


def masked_cross_entropy(probs, targets, mask):
    """
    Cross-entropy loss for softmax race predictions.

    For each target (win/place/show), the "correct" horse should get
    probability 1.0 and all others 0.0. We use KL divergence against
    the true label distribution.

    Args:
        probs   : (batch, horses, 3)  model output
        targets : (batch, horses, 3)  ground truth (0/1 labels)
        mask    : (batch, horses)     True = real horse

    Returns:
        scalar loss
    """
    eps   = 1e-8
    loss  = 0.0
    field = mask.float().sum(dim=1, keepdim=True).unsqueeze(-1)  # (batch, 1, 1)

    for i in range(3):  # win, place, show
        p    = probs[:, :, i].clamp(eps, 1.0)      # predicted probs
        t    = targets[:, :, i]                     # true labels
        # Normalize targets so they sum to 1 (handles multi-horse place/show)
        t_sum = t.sum(dim=1, keepdim=True).clamp(eps)
        t_norm = t / t_sum
        # Cross-entropy: -sum(t * log(p)) averaged over races
        ce   = -(t_norm * p.log()).sum(dim=1).mean()
        loss += ce

    return loss / 3.0


def build_model(device: str = "cpu") -> RacePredictor:
    model = RacePredictor(
        n_features      = N_FEATURES,
        d_model         = 32,    # smaller — fewer params, less overfitting
        n_heads         = 4,
        n_layers        = 2,    # shallower
        dim_feedforward = 128,  # smaller feedforward
        dropout         = 0.3,  # stronger regularization
    )
    return model.to(device)


if __name__ == "__main__":
    # Smoke test
    model = build_model()
    batch = 4
    x    = torch.randn(batch, MAX_HORSES, N_FEATURES)
    mask = torch.ones(batch, MAX_HORSES, dtype=torch.bool)
    mask[:, 10:] = False  # simulate 10-horse fields

    probs = model(x, mask)
    print(f"Output shape: {probs.shape}")
    print(f"Win probs sum (should be ~1.0): {probs[:, :, 0].sum(dim=1)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
