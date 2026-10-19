"""APRMixer final active model, for architecture drawing and inference.

Frozen NoCross + PointFluct + PatchMixer architecture. No self-attention,
cross-attention or dilated convolution. Contexts use temporal mean pooling.
Historical x: [B,L,C], last channel = capacity-factor power.
Marks end with six empirical features: [s,d,e,sin(time),cos(time),e].
Future weather observations are not inputs. Features are built by the PV loader.
Training objective/checkpoint criterion: all-day MSE, outside this model file.

Checkpoint keys are preserved; load_state_dict(strict=True) is supported.
Removing discarded constructors changes RNG consumption during initialization:
this clean file is NOT a seed-identical replacement for rerunning old training.
Use the frozen training package for exact experimental reproduction.

Forward: adaptive reference -> patch/pointwise memories -> future conditioning
and mean contexts -> main + recent correction -> asymmetric residual bounds
-> add adaptive reference -> clip -> empirical daylight mask.
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

def build_fluctuation_features(x_enc: torch.Tensor, past_physics: torch.Tensor) -> torch.Tensor:
    power = x_enc[..., -1:].clamp(0.0, 1.0)
    clear_power = past_physics[..., 0:1].clamp(0.0, 1.0)
    daylight = past_physics[..., 1:2].clamp(0.0, 1.0)
    clear_index = (power / clear_power.clamp_min(0.03)).clamp(0.0, 1.5)
    clear_index = clear_index * daylight
    delta_index = _time_difference(clear_index)
    curvature_index = _time_difference(delta_index)
    delta_inputs = _time_difference(x_enc)
    return torch.cat([clear_index, delta_index, curvature_index, delta_inputs, daylight, clear_power], dim=-1)

class MixerBlock(nn.Module):

    def __init__(self, n, d, dff, dropout):
        super().__init__()
        self.time_norm = nn.LayerNorm(d)
        self.time_mlp = nn.Sequential(nn.Linear(n, 2 * n), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * n, n), nn.Dropout(dropout))
        self.feature_norm = nn.LayerNorm(d)
        self.feature_mlp = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dff, d), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.time_mlp(self.time_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.feature_mlp(self.feature_norm(x))

class PatchEncoder(nn.Module):
    """Flattened multivariate patches -> linear embedding -> PatchMixer blocks -> LayerNorm. No attention."""

    def __init__(self, input_dim: int, d_model: int, patch_len: int, stride: int, n_heads: int, e_layers: int, d_ff: int, dropout: float, seq_len=96) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_projection = nn.Linear(input_dim * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.encoder = nn.Sequential(*[MixerBlock((seq_len - patch_len) // stride + 1, d_model, d_ff, dropout) for _ in range(e_layers)])

    def _tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) < self.patch_len:
            raise ValueError(f'seq_len={x.size(1)} must be >= patch_len={self.patch_len}')
        patches = x.unfold(1, self.patch_len, self.stride)
        flat_patches = patches.contiguous().flatten(start_dim=2)
        tokens = self.patch_projection(flat_patches)
        return (flat_patches, tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (_, tokens) = self._tokenize(x)
        return self.norm(self.encoder(tokens))

class PointFluct(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c.enc_in + 5, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model))

    def forward(self, x, physics):
        return self.net(build_fluctuation_features(x, physics))

class Model(nn.Module):
    """Frozen final architecture, compatible with existing checkpoints."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.phys_dim = getattr(configs, 'phys_dim', 6)
        self.persistence_period = getattr(configs, 'persistence_period', 96)
        self.recent_tau = getattr(configs, 'recent_tau', 4.0)
        self.recent_decay = getattr(configs, 'recent_decay', 48.0)
        self.recent_mix = getattr(configs, 'recent_mix', 0.85)
        self.up_residual_scale = getattr(configs, 'up_residual_scale', 0.3)
        self.down_residual_scale = getattr(configs, 'down_residual_scale', 0.6)
        self.short_pool_window = int(getattr(configs, 'short_pool_window', 96))
        self.short_pool_tau = float(getattr(configs, 'short_pool_tau', 16.0))
        self.short_decay_steps = float(getattr(configs, 'short_decay_steps', 16.0))
        self._last_correction = None
        self._last_daylight_mask = None
        self._last_short_gate = None
        self._last_short_decay = None
        d_model = configs.d_model
        history_dim = configs.enc_in + self.phys_dim
        stride = getattr(configs, 'stride', 8)
        hidden = max(32, d_model // 2)
        self.smooth_encoder = PatchEncoder(input_dim=history_dim, d_model=d_model, patch_len=configs.patch_len, stride=stride, n_heads=configs.n_heads, e_layers=configs.e_layers, d_ff=configs.d_ff, dropout=configs.dropout, seq_len=configs.seq_len)
        self.future_projection = nn.Sequential(nn.Linear(self.phys_dim + 3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.pred_len, d_model))
        nn.init.trunc_normal_(self.horizon_embedding, std=0.02)
        self.regime_gate_delta = nn.Sequential(nn.Linear(self.phys_dim + 8, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.regime_gate_delta[-1].weight)
        nn.init.zeros_(self.regime_gate_delta[-1].bias)
        self.fusion = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Dropout(configs.dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.residual_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.residual_head.bias)
        self.short_state_norm = nn.LayerNorm(d_model)
        self.short_correction = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.short_gate = nn.Sequential(nn.Linear(2 * d_model + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.short_correction[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.short_correction[-1].bias)
        nn.init.zeros_(self.short_gate[-1].weight)
        nn.init.zeros_(self.short_gate[-1].bias)
        self.fluctuation_encoder = PointFluct(configs)

    def _adaptive_persistence(self, x_enc: torch.Tensor, past_physics: torch.Tensor, future_physics: torch.Tensor, return_trace: bool=False):
        """A. Build daily/recent bases and learn their adaptive mixture."""
        if x_enc.size(1) < self.persistence_period:
            raise ValueError(f'seq_len={x_enc.size(1)} must be >= persistence_period={self.persistence_period}')
        power = x_enc[:, -self.persistence_period:, -1:].clamp(0.0, 1.0)
        past_clear = past_physics[:, -self.persistence_period:, 0:1].clamp(0.0, 1.0)
        past_day = past_physics[:, -self.persistence_period:, 1:2].clamp(0.0, 1.0)
        future_clear = future_physics[..., 0:1].clamp(0.0, 1.0)
        daily_ratio_profile = (power / past_clear.clamp_min(0.03)).clamp(0.0, 1.5) * past_day
        horizon_index = torch.arange(self.pred_len, device=x_enc.device) % self.persistence_period
        daily_ratio = daily_ratio_profile.index_select(1, horizon_index)
        daily_base = (future_clear * daily_ratio).clamp(0.0, 1.0)
        ages = torch.arange(self.persistence_period - 1, -1, -1, device=x_enc.device, dtype=x_enc.dtype).view(1, -1, 1)
        recent_weights = torch.exp(-ages / self.recent_tau) * past_day
        weight_sum = recent_weights.sum(dim=1, keepdim=True)
        recent_ratio = ((recent_weights * daily_ratio_profile).sum(dim=1, keepdim=True) / weight_sum.clamp_min(1e-06)).clamp(0.0, 1.5)
        recent_base = (future_clear * recent_ratio).clamp(0.0, 1.0)
        recent_base = recent_base.expand(-1, self.pred_len, -1)
        centered = daily_ratio_profile - recent_ratio
        recent_std = torch.sqrt((recent_weights * centered.square()).sum(dim=1, keepdim=True) / weight_sum.clamp_min(1e-06) + 1e-06)
        ratio_delta = _time_difference(daily_ratio_profile)
        recent_slope = (recent_weights * ratio_delta).sum(dim=1, keepdim=True) / weight_sum.clamp_min(1e-06)
        ramp_energy = (recent_weights * ratio_delta.abs()).sum(dim=1, keepdim=True) / weight_sum.clamp_min(1e-06)
        confidence = (weight_sum / 4.0).clamp(0.0, 1.0)
        regime_stats = torch.cat([recent_ratio, recent_std, recent_slope, ramp_energy, confidence], dim=-1).expand(-1, self.pred_len, -1)
        horizon = torch.arange(self.pred_len, device=x_enc.device, dtype=x_enc.dtype).view(1, -1, 1)
        horizon_norm = horizon / max(self.pred_len - 1, 1)
        gate_features = torch.cat([future_physics, daily_base, recent_base, regime_stats, horizon_norm.expand(x_enc.size(0), -1, -1)], dim=-1)
        gate_delta = self.regime_gate_delta(gate_features)
        decay_prior = self.recent_mix * torch.exp(-horizon / self.recent_decay)
        decay_prior = decay_prior.clamp(0.0001, 1.0 - 0.0001)
        prior_logit = torch.log(decay_prior) - torch.log1p(-decay_prior)
        alpha = confidence * torch.sigmoid(prior_logit + gate_delta)
        adaptive_base = alpha * recent_base + (1.0 - alpha) * daily_base
        if not return_trace:
            return (adaptive_base, daily_base, recent_base)
        return (adaptive_base, daily_base, recent_base, {'A01_daily_ratio_profile': daily_ratio_profile, 'A02_B_daily': daily_base, 'A03_B_recent': recent_base, 'A04_regime_statistics': regime_stats, 'A05_gate_features': gate_features, 'A06_gate_delta': gate_delta, 'A07_decay_prior': decay_prior, 'A08_alpha': alpha, 'A09_B_adapt': adaptive_base})

    def _recent_cloud_state(self, fluctuation_memory: torch.Tensor, past_physics: torch.Tensor, return_trace: bool=False):
        """E. Pool the latest valid daylight fluctuation states."""
        window = min(self.short_pool_window, fluctuation_memory.size(1))
        memory = fluctuation_memory[:, -window:, :]
        daylight = past_physics[:, -window:, 1:2].clamp(0.0, 1.0)
        ages = torch.arange(window - 1, -1, -1, device=memory.device, dtype=memory.dtype).view(1, window, 1)
        recency = torch.exp(-ages / self.short_pool_tau)
        daylight_weights = recency * daylight
        daylight_sum = daylight_weights.sum(dim=1, keepdim=True)
        fallback = recency.expand(memory.size(0), -1, -1)
        weights = torch.where((daylight_sum > 1e-06).expand_as(daylight_weights), daylight_weights, fallback)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
        state = (weights * memory).sum(dim=1, keepdim=True) / weight_sum
        state = self.short_state_norm(state)
        confidence = (1.0 - torch.exp(-daylight_sum / 4.0)).clamp(0.0, 1.0)
        if not return_trace:
            return (state, confidence)
        return (state, confidence, {'E01_recency_weights': weights, 'E02_recent_state': state, 'E03_recent_confidence': confidence})

    def correction_regularization(self):
        if self._last_correction is None:
            return None
        return self._last_correction.square().mean()

    def daylight_mask(self):
        return self._last_daylight_mask

    def short_branch_diagnostics(self):
        return (self._last_short_gate, self._last_short_decay)

    def _contexts(self, q, smooth, fluct):
        return (smooth.mean(dim=1, keepdim=True).expand(-1, q.size(1), -1), fluct.mean(dim=1, keepdim=True).expand(-1, q.size(1), -1))

    def _short_raw(self, q, fluct, past):
        (state, confidence) = self._recent_cloud_state(fluct, past)
        state = state.expand(-1, self.pred_len, -1)
        confidence = confidence.expand(-1, self.pred_len, -1)
        features = torch.cat([q, state], dim=-1)
        gate = torch.sigmoid(self.short_gate(torch.cat([features, confidence], dim=-1))) * confidence
        h = torch.arange(1, self.pred_len + 1, device=q.device, dtype=q.dtype).view(1, -1, 1)
        decay = torch.exp(-h / self.short_decay_steps)
        (self._last_short_gate, self._last_short_decay) = (gate, decay)
        return decay * gate * self.short_correction(features)

    def _bound(self, raw, base, future):
        clear = future[..., :1].clamp(0, 1)
        up = self.up_residual_scale * (0.05 + clear) * (1 - base)
        down = self.down_residual_scale * base
        return torch.where(raw >= 0, up, down) * torch.tanh(raw)

    def _compute(self, x, xm, ym, trace=False):
        past = xm[..., -self.phys_dim:]
        future = ym[:, -self.pred_len:, -self.phys_dim:]
        (base, daily, recent) = self._adaptive_persistence(x, past, future)
        smooth = self.smooth_encoder(torch.cat([x, past], dim=-1))
        fluct = self.fluctuation_encoder(x, past)
        q = self.future_projection(torch.cat([future, base, daily, recent], dim=-1)) + self.horizon_embedding
        (cs, cf) = self._contexts(q, smooth, fluct)
        main = self.residual_head(self.fusion(torch.cat([q, cs, cf], dim=-1)))
        short = self._short_raw(q, fluct, past)
        raw = main + short
        delta = self._bound(raw, base, future)
        self._last_correction = delta
        self._last_daylight_mask = future[..., 1:2].clamp(0, 1)
        prediction = (base + delta).clamp(0, 1) * self._last_daylight_mask
        if trace:
            return (prediction, OrderedDict(past_physics=past, future_physics=future, B_daily=daily, B_recent=recent, B_adapt=base, M_smooth=smooth, M_fluct=fluct, Q=q, C_smooth=cs, C_fluct=cf, r_main=main, r_short=short, r_raw=raw, bounded_delta=delta, daylight_mask=self._last_daylight_mask, Y_hat=prediction))
        return prediction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self._compute(x_enc, x_mark_enc, x_mark_dec)

    def forward_with_trace(self, x_enc, x_mark_enc, x_mark_dec):
        return self._compute(x_enc, x_mark_enc, x_mark_dec, trace=True)
