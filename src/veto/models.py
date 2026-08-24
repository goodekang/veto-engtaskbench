from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedObsEncoder(nn.Module):
    """Encodes a variable-length observation bag (IFC entities or CAD ops)."""

    def __init__(self, d_obs: int = 32, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_obs = d_obs
        self.d_model = d_model
        self.proj = nn.Linear(d_obs, d_model)
        self.enc = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, D], lengths: [B]
        x = self.proj(obs)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, h = self.enc(packed)
        pooled = self.norm(h[-1])
        return pooled


class VetoPolicy(nn.Module):
    """Offline policy used for replay: broker scores, verdict, CSR, per-item widths."""

    def __init__(
        self,
        d_obs: int = 32,
        d_model: int = 128,
        n_layers: int = 2,
        n_tools: int = 42,
    ):
        super().__init__()
        self.encoder = CachedObsEncoder(d_obs, d_model, n_layers)
        self.verdict = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(d_model, 1),
        )
        self.csr_head = nn.Linear(d_model, 1)
        self.broker = nn.Linear(d_model, n_tools)
        self.item_width = nn.Sequential(
            nn.Linear(d_obs, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.item_pass = nn.Sequential(
            nn.Linear(d_obs, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def encode(self, obs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs, lengths)

    def forward(
        self, obs: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.encode(obs, lengths)
        return {
            "logit": self.verdict(h).squeeze(-1),
            "csr": torch.sigmoid(self.csr_head(h)).squeeze(-1),
            "broker": self.broker(h),
            "hidden": h,
        }

    def score_items(self, items: torch.Tensor) -> dict[str, torch.Tensor]:
        # Channel 0 holds the cached measurement (metres for doors, unitless for CAD).
        residual = self.item_width(items).squeeze(-1)
        width = items[:, 0] * 1000.0 + residual
        return {
            "width": width,
            "pass_logit": self.item_pass(items).squeeze(-1),
        }

    def encode_tokens(self, obs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.encoder.proj(obs)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        token_out, _ = self.encoder.enc(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(token_out, batch_first=True)
        return self.encoder.norm(padded)

    def broker_probs(self, obs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encode(obs, lengths)
        return torch.softmax(self.broker(h), dim=-1)


def count_params(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": int(trainable),
        "frozen": int(frozen),
        "total": int(trainable + frozen),
    }


def load_checkpoint(path, map_location: str = "cpu") -> tuple[VetoPolicy, dict[str, Any]]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = blob.get("cfg", {})
    model = VetoPolicy(
        d_obs=cfg.get("d_obs", 32),
        d_model=cfg.get("d_model", 128),
        n_layers=cfg.get("n_layers", 2),
        n_tools=cfg.get("n_tools", 42),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
