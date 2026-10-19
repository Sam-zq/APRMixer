"""CSRampFormer compact model (archival variant: NoCross), standalone module.

API: Model(configs)(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B,H,1].
Only torch and Python standard library imports; no local model/layer imports.
Historical power is the last input channel in capacity-factor units.
Marks end in the SIX frozen empirical_harmonic fields [s,d,e,sin(t),cos(t),e].
Those fields must be prepared by the original training-only fitted PV loader.
They are NOT future observed weather and NOT an astronomical clear-sky model.

Compatibility note: the original Transformer/Conv helper constructors below
are used ONLY to replay initialization RNG before replacing those modules.
They do NOT remain in the final model state_dict or forward computation.
The exported Model additionally replaces both attention readers with MeanContext.
This preserves NoCross initial weights and checkpoint keys exactly.
Architecture: Patch projection -> Mixer memory; pointwise fluctuation memory;
adaptive persistence -> future conditioning + dual temporal mean -> main + short
correction -> asymmetric bounds -> clip[0,1] -> physical daylight mask.
"""
from __future__ import annotations
from typing import Dict, Tuple
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


def _time_difference(x: torch.Tensor) -> torch.Tensor:
    """First difference, left padded to preserve sequence length."""
    if x.size(1) < 2:
        return torch.zeros_like(x)
    dx = x[:, 1:, :] - x[:, :-1, :]
    return F.pad(dx, (0, 0, 1, 0))

class PatchEncoder(nn.Module):
    """Patch embedding followed by temporal Transformer self-attention."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch_len: int,
        stride: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Patch embedding: flattened time patch -> d_model token.
        self.patch_projection = nn.Linear(input_dim * patch_len, d_model)

        # Self-attention is inside every TransformerEncoderLayer below.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

    def _tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) < self.patch_len:
            raise ValueError(
                f"seq_len={x.size(1)} must be >= patch_len={self.patch_len}"
            )
        # unfold: [B, L, C] -> [B, N_patch, C, patch_len]
        patches = x.unfold(1, self.patch_len, self.stride)
        flat_patches = patches.contiguous().flatten(start_dim=2)
        tokens = self.patch_projection(flat_patches)
        return flat_patches, tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, tokens = self._tokenize(x)
        return self.norm(self.encoder(tokens))

    def forward_with_trace(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat_patches, tokens = self._tokenize(x)
        encoded = self.encoder(tokens)
        memory = self.norm(encoded)
        return memory, {
            "B01_flat_patches": flat_patches,
            "B02_patch_embeddings": tokens,
            "B03_transformer_output": encoded,
            "B04_M_smooth": memory,
        }

class CausalConvBlock(nn.Module):
    """Causal dilated convolution + GroupNorm + GELU + dropout + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        y = F.pad(x, (self.left_padding, 0))
        y = self.conv(y)
        y = self.dropout(F.gelu(self.norm(y)))
        return y + residual

class DaylightFluctuationEncoder(nn.Module):
    """Serial dilation-1/2/4 encoder for daylight PV fluctuations."""

    def __init__(self, enc_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        # k, delta-k, delta2-k, delta of all encoder variables, daylight,
        # and clear-sky power: enc_in + 5 total channels.
        input_dim = enc_in + 5
        self.blocks = nn.Sequential(
            CausalConvBlock(input_dim, d_model, 3, 1, dropout),
            CausalConvBlock(d_model, d_model, 3, 2, dropout),
            CausalConvBlock(d_model, d_model, 3, 4, dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def build_features(
        x_enc: torch.Tensor, past_physics: torch.Tensor
    ) -> torch.Tensor:
        power = x_enc[..., -1:].clamp(0.0, 1.0)
        clear_power = past_physics[..., 0:1].clamp(0.0, 1.0)
        daylight = past_physics[..., 1:2].clamp(0.0, 1.0)

        clear_index = (power / clear_power.clamp_min(0.03)).clamp(0.0, 1.5)
        clear_index = clear_index * daylight
        delta_index = _time_difference(clear_index)
        curvature_index = _time_difference(delta_index)
        delta_inputs = _time_difference(x_enc)

        return torch.cat(
            [
                clear_index,
                delta_index,
                curvature_index,
                delta_inputs,
                daylight,
                clear_power,
            ],
            dim=-1,
        )

    def forward(
        self, x_enc: torch.Tensor, past_physics: torch.Tensor
    ) -> torch.Tensor:
        features = self.build_features(x_enc, past_physics)
        encoded = self.blocks(features.transpose(1, 2)).transpose(1, 2)
        return self.norm(encoded)

    def forward_with_trace(
        self, x_enc: torch.Tensor, past_physics: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.build_features(x_enc, past_physics)
        y = features.transpose(1, 2)
        block_outputs = []
        for block in self.blocks:
            y = block(y)
            block_outputs.append(y.transpose(1, 2))
        memory = self.norm(y.transpose(1, 2))
        return memory, {
            "C01_F_fluct": features,
            "C02_dilated_conv_d1": block_outputs[0],
            "C03_dilated_conv_d2": block_outputs[1],
            "C04_dilated_conv_d4": block_outputs[2],
            "C05_M_fluct": memory,
        }

class PointFluct(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c.enc_in + 5, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model))
    def forward(self, x, physics):
        return self.net(DaylightFluctuationEncoder.build_features(x, physics))

class MixerBlock(nn.Module):
    def __init__(self,n,d,dff,dropout):
        super().__init__()
        self.time_norm=nn.LayerNorm(d)
        self.time_mlp=nn.Sequential(nn.Linear(n,2*n),nn.GELU(),nn.Dropout(dropout),
                                    nn.Linear(2*n,n),nn.Dropout(dropout))
        self.feature_norm=nn.LayerNorm(d)
        self.feature_mlp=nn.Sequential(nn.Linear(d,dff),nn.GELU(),nn.Dropout(dropout),
                                       nn.Linear(dff,d),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.time_mlp(self.time_norm(x).transpose(1,2)).transpose(1,2)
        return x+self.feature_mlp(self.feature_norm(x))

class _InitializationCompatibleBase(nn.Module):
    """Frozen final architecture, compatible with existing checkpoints."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.phys_dim = getattr(configs, "phys_dim", 6)

        # Validation-selected T009 settings; identical across stations and horizons.
        self.persistence_period = getattr(configs, "persistence_period", 96)
        self.recent_tau = getattr(configs, "recent_tau", 4.0)
        self.recent_decay = getattr(configs, "recent_decay", 48.0)
        self.recent_mix = getattr(configs, "recent_mix", 0.85)
        self.up_residual_scale = getattr(configs, "up_residual_scale", 0.30)
        self.down_residual_scale = getattr(configs, "down_residual_scale", 0.60)
        self.short_pool_window = int(getattr(configs, "short_pool_window", 96))
        self.short_pool_tau = float(getattr(configs, "short_pool_tau", 16.0))
        self.short_decay_steps = float(getattr(configs, "short_decay_steps", 16.0))

        self._last_correction = None
        self._last_daylight_mask = None
        self._last_short_gate = None
        self._last_short_decay = None

        d_model = configs.d_model
        history_dim = configs.enc_in + self.phys_dim
        stride = getattr(configs, "stride", 8)
        hidden = max(32, d_model // 2)

        # B. Smooth temporal memory.
        self.smooth_encoder = PatchEncoder(
            input_dim=history_dim,
            d_model=d_model,
            patch_len=configs.patch_len,
            stride=stride,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
        )

        # C. Cloud-induced fluctuation memory.
        self.fluctuation_encoder = DaylightFluctuationEncoder(
            enc_in=configs.enc_in,
            d_model=d_model,
            dropout=configs.dropout,
        )

        # D. Future-query embedding: six physical features + three baselines.
        self.future_projection = nn.Sequential(
            nn.Linear(self.phys_dim + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.horizon_embedding = nn.Parameter(
            torch.zeros(1, self.pred_len, d_model)
        )
        nn.init.trunc_normal_(self.horizon_embedding, std=0.02)

        # A. Learned delta applied to the analytical persistence-gate prior.
        self.regime_gate_delta = nn.Sequential(
            nn.Linear(self.phys_dim + 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.regime_gate_delta[-1].weight)
        nn.init.zeros_(self.regime_gate_delta[-1].bias)

        # D. Q=future query; K,V=the two historical memories.
        self.smooth_cross_attention = nn.MultiheadAttention(
            d_model,
            configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.fluctuation_cross_attention = nn.MultiheadAttention(
            d_model,
            configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.residual_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head.bias)

        # E. Horizon-damped recent-cloud-state correction.
        self.short_state_norm = nn.LayerNorm(d_model)
        self.short_correction = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.short_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.short_correction[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.short_correction[-1].bias)
        nn.init.zeros_(self.short_gate[-1].weight)
        nn.init.zeros_(self.short_gate[-1].bias)
        # Replay original C1 -> PatchMixer replacement order exactly.
        self.fluctuation_encoder = PointFluct(configs)
        n = (configs.seq_len - configs.patch_len) // stride + 1
        if n < 1:
            raise ValueError('Invalid patch count')
        self.smooth_encoder.encoder = nn.Sequential(*[
            MixerBlock(n, d_model, configs.d_ff, configs.dropout)
            for _ in range(configs.e_layers)])


    def _adaptive_persistence(
        self,
        x_enc: torch.Tensor,
        past_physics: torch.Tensor,
        future_physics: torch.Tensor,
        return_trace: bool = False,
    ):
        """A. Build daily/recent bases and learn their adaptive mixture."""
        if x_enc.size(1) < self.persistence_period:
            raise ValueError(
                f"seq_len={x_enc.size(1)} must be >= "
                f"persistence_period={self.persistence_period}"
            )

        power = x_enc[:, -self.persistence_period :, -1:].clamp(0.0, 1.0)
        past_clear = past_physics[
            :, -self.persistence_period :, 0:1
        ].clamp(0.0, 1.0)
        past_day = past_physics[
            :, -self.persistence_period :, 1:2
        ].clamp(0.0, 1.0)
        future_clear = future_physics[..., 0:1].clamp(0.0, 1.0)

        daily_ratio_profile = (
            power / past_clear.clamp_min(0.03)
        ).clamp(0.0, 1.5) * past_day
        horizon_index = (
            torch.arange(self.pred_len, device=x_enc.device)
            % self.persistence_period
        )
        daily_ratio = daily_ratio_profile.index_select(1, horizon_index)
        daily_base = (future_clear * daily_ratio).clamp(0.0, 1.0)

        ages = torch.arange(
            self.persistence_period - 1,
            -1,
            -1,
            device=x_enc.device,
            dtype=x_enc.dtype,
        ).view(1, -1, 1)
        recent_weights = torch.exp(-ages / self.recent_tau) * past_day
        weight_sum = recent_weights.sum(dim=1, keepdim=True)
        recent_ratio = (
            (recent_weights * daily_ratio_profile).sum(dim=1, keepdim=True)
            / weight_sum.clamp_min(1e-6)
        ).clamp(0.0, 1.5)
        recent_base = (future_clear * recent_ratio).clamp(0.0, 1.0)
        recent_base = recent_base.expand(-1, self.pred_len, -1)

        centered = daily_ratio_profile - recent_ratio
        recent_std = torch.sqrt(
            (recent_weights * centered.square()).sum(dim=1, keepdim=True)
            / weight_sum.clamp_min(1e-6)
            + 1e-6
        )
        ratio_delta = _time_difference(daily_ratio_profile)
        recent_slope = (
            (recent_weights * ratio_delta).sum(dim=1, keepdim=True)
            / weight_sum.clamp_min(1e-6)
        )
        ramp_energy = (
            (recent_weights * ratio_delta.abs()).sum(dim=1, keepdim=True)
            / weight_sum.clamp_min(1e-6)
        )
        confidence = (weight_sum / 4.0).clamp(0.0, 1.0)

        regime_stats = torch.cat(
            [
                recent_ratio,
                recent_std,
                recent_slope,
                ramp_energy,
                confidence,
            ],
            dim=-1,
        ).expand(-1, self.pred_len, -1)
        horizon = torch.arange(
            self.pred_len, device=x_enc.device, dtype=x_enc.dtype
        ).view(1, -1, 1)
        horizon_norm = horizon / max(self.pred_len - 1, 1)

        gate_features = torch.cat(
            [
                future_physics,
                daily_base,
                recent_base,
                regime_stats,
                horizon_norm.expand(x_enc.size(0), -1, -1),
            ],
            dim=-1,
        )
        gate_delta = self.regime_gate_delta(gate_features)

        decay_prior = self.recent_mix * torch.exp(
            -horizon / self.recent_decay
        )
        decay_prior = decay_prior.clamp(1e-4, 1.0 - 1e-4)
        prior_logit = torch.log(decay_prior) - torch.log1p(-decay_prior)
        alpha = confidence * torch.sigmoid(prior_logit + gate_delta)

        adaptive_base = alpha * recent_base + (1.0 - alpha) * daily_base

        if not return_trace:
            return adaptive_base, daily_base, recent_base
        return adaptive_base, daily_base, recent_base, {
            "A01_daily_ratio_profile": daily_ratio_profile,
            "A02_B_daily": daily_base,
            "A03_B_recent": recent_base,
            "A04_regime_statistics": regime_stats,
            "A05_gate_features": gate_features,
            "A06_gate_delta": gate_delta,
            "A07_decay_prior": decay_prior,
            "A08_alpha": alpha,
            "A09_B_adapt": adaptive_base,
        }

    def _recent_cloud_state(
        self,
        fluctuation_memory: torch.Tensor,
        past_physics: torch.Tensor,
        return_trace: bool = False,
    ):
        """E. Pool the latest valid daylight fluctuation states."""
        window = min(self.short_pool_window, fluctuation_memory.size(1))
        memory = fluctuation_memory[:, -window:, :]
        daylight = past_physics[:, -window:, 1:2].clamp(0.0, 1.0)
        ages = torch.arange(
            window - 1,
            -1,
            -1,
            device=memory.device,
            dtype=memory.dtype,
        ).view(1, window, 1)
        recency = torch.exp(-ages / self.short_pool_tau)
        daylight_weights = recency * daylight
        daylight_sum = daylight_weights.sum(dim=1, keepdim=True)

        # If the history has no daylight mark, use recency weights alone.
        fallback = recency.expand(memory.size(0), -1, -1)
        weights = torch.where(
            (daylight_sum > 1e-6).expand_as(daylight_weights),
            daylight_weights,
            fallback,
        )
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        state = (weights * memory).sum(dim=1, keepdim=True) / weight_sum
        state = self.short_state_norm(state)
        confidence = (1.0 - torch.exp(-daylight_sum / 4.0)).clamp(0.0, 1.0)

        if not return_trace:
            return state, confidence
        return state, confidence, {
            "E01_recency_weights": weights,
            "E02_recent_state": state,
            "E03_recent_confidence": confidence,
        }

    def correction_regularization(self):
        if self._last_correction is None:
            return None
        return self._last_correction.square().mean()

    def daylight_mask(self):
        return self._last_daylight_mask

    def short_branch_diagnostics(self):
        return self._last_short_gate, self._last_short_decay

    def _contexts(self, q, smooth, fluct):
        cs, _ = self.smooth_cross_attention(q, smooth, smooth, need_weights=False)
        cf, _ = self.fluctuation_cross_attention(q, fluct, fluct, need_weights=False)
        return cs, cf

    def _short_raw(self, q, fluct, past):
        state, confidence = self._recent_cloud_state(fluct, past)
        state = state.expand(-1, self.pred_len, -1)
        confidence = confidence.expand(-1, self.pred_len, -1)
        features = torch.cat([q, state], dim=-1)
        gate = torch.sigmoid(self.short_gate(torch.cat([features, confidence], dim=-1))) * confidence
        h = torch.arange(1, self.pred_len+1, device=q.device, dtype=q.dtype).view(1,-1,1)
        decay = torch.exp(-h / self.short_decay_steps)
        self._last_short_gate, self._last_short_decay = gate, decay
        return decay * gate * self.short_correction(features)

    def _bound(self, raw, base, future):
        clear = future[..., :1].clamp(0,1)
        up = self.up_residual_scale * (0.05 + clear) * (1-base)
        down = self.down_residual_scale * base
        return torch.where(raw >= 0, up, down) * torch.tanh(raw)

    def _compute(self, x, xm, ym, trace=False):
        past = xm[..., -self.phys_dim:]
        future = ym[:, -self.pred_len:, -self.phys_dim:]
        base, daily, recent = self._adaptive_persistence(x, past, future)
        smooth = self.smooth_encoder(torch.cat([x, past], dim=-1))
        fluct = self.fluctuation_encoder(x, past)
        q = self.future_projection(torch.cat([future, base, daily, recent], dim=-1)) + self.horizon_embedding
        cs, cf = self._contexts(q, smooth, fluct)
        main = self.residual_head(self.fusion(torch.cat([q, cs, cf], dim=-1)))
        short = self._short_raw(q, fluct, past)
        raw = main + short
        delta = self._bound(raw, base, future)
        self._last_correction = delta
        self._last_daylight_mask = future[..., 1:2].clamp(0,1)
        prediction = (base + delta).clamp(0,1) * self._last_daylight_mask
        if trace:
            return prediction, OrderedDict(past_physics=past, future_physics=future,
                B_daily=daily, B_recent=recent, B_adapt=base,
                M_smooth=smooth, M_fluct=fluct, Q=q, C_smooth=cs, C_fluct=cf,
                r_main=main, r_short=short, r_raw=raw, bounded_delta=delta,
                daylight_mask=self._last_daylight_mask, Y_hat=prediction)
        return prediction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_dec and mask are accepted for TSLib compatibility, intentionally unused.
        return self._compute(x_enc, x_mark_enc, x_mark_dec)

    def forward_with_trace(self, x_enc, x_mark_enc, x_mark_dec):
        return self._compute(x_enc, x_mark_enc, x_mark_dec, trace=True)


class MeanContext(nn.Module):
    """Parameter-free temporal mean; q supplies forecast length, not weights."""
    def forward(self, q, k, v, **kwargs):
        return v.mean(dim=1, keepdim=True).expand(-1, q.size(1), -1), None


class Model(_InitializationCompatibleBase):
    """Frozen NoCross implementation; no attention modules in the final graph."""
    def __init__(self, configs):
        super().__init__(configs)
        self.smooth_cross_attention = MeanContext()
        self.fluctuation_cross_attention = MeanContext()
