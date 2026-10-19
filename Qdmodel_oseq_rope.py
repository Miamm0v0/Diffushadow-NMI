import inspect
import math

import torch
import torch.nn as nn

try:
    from transformers.models.llama import modeling_llama as _llama_modeling
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    _LlamaLinearScalingRotaryEmbedding = getattr(
        _llama_modeling, "LlamaLinearScalingRotaryEmbedding", None
    )
    _LlamaDynamicNTKScalingRotaryEmbedding = getattr(
        _llama_modeling, "LlamaDynamicNTKScalingRotaryEmbedding", None
    )
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    _LlamaLinearScalingRotaryEmbedding = None
    _LlamaDynamicNTKScalingRotaryEmbedding = None

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


def _apply_rotary_pos_emb_requires_position_ids():
    """4.37.x: position_ids 为必选且用 cos[position_ids]；4.40+ 多为默认 None 且 cos 已为 [B,S,D]。"""
    try:
        p = inspect.signature(apply_rotary_pos_emb).parameters.get("position_ids")
        if p is None:
            return False
        return p.default is inspect.Parameter.empty
    except (ValueError, TypeError):
        return False


def _normalize_rope_scaling_type(rope_scaling_type):
    if rope_scaling_type is None:
        return None
    s = str(rope_scaling_type).strip().lower()
    if s in ("", "none", "off", "false"):
        return None
    if s in ("ntk", "dynamic_ntk"):
        return "dynamic"
    return s


def _build_llama_rotary_embedding(
    head_dim,
    hidden_dim,
    head_count,
    max_seq_len,
    rope_theta=10000.0,
    rope_scaling_type=None,
    rope_scaling_factor=1.0,
):
    """
    构建 HF Llama RoPE。支持：
    - 默认：LlamaRotaryEmbedding
    - linear：LlamaLinearScalingRotaryEmbedding（位置缩放）
    - dynamic：LlamaDynamicNTKScalingRotaryEmbedding（NTK-aware，序列可长于 max_position_embeddings 时重算 inv_freq）

    兼容：旧版 (dim, max_position_embeddings, base, scaling_factor)；新版仅 LlamaConfig 时在 config 上设 rope_scaling。
    """
    rst = _normalize_rope_scaling_type(rope_scaling_type)
    factor = float(rope_scaling_factor)

    rope_init = list(inspect.signature(LlamaRotaryEmbedding.__init__).parameters.keys())
    second = rope_init[1] if len(rope_init) > 1 else None

    if second == "config":
        from transformers.models.llama.configuration_llama import LlamaConfig

        cfg_kw = dict(
            vocab_size=32000,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_hidden_layers=1,
            num_attention_heads=head_count,
            max_position_embeddings=max_seq_len,
            rope_theta=rope_theta,
        )
        if rst == "linear":
            cfg_kw["rope_scaling"] = {"rope_type": "linear", "factor": factor}
        elif rst == "dynamic":
            cfg_kw["rope_scaling"] = {"rope_type": "dynamic", "factor": factor}
        elif rst is not None:
            raise ValueError(f"Unknown rope_scaling_type for config API: {rope_scaling_type!r}")

        try:
            rope_cfg = LlamaConfig(**cfg_kw)
        except TypeError:
            rope_cfg = LlamaConfig(
                vocab_size=32000,
                hidden_size=hidden_dim,
                intermediate_size=hidden_dim * 4,
                num_hidden_layers=1,
                num_attention_heads=head_count,
                max_position_embeddings=max_seq_len,
                rope_theta=rope_theta,
            )
            if rst == "linear":
                rope_cfg.rope_scaling = {"type": "linear", "factor": factor}
            elif rst == "dynamic":
                rope_cfg.rope_scaling = {"type": "dynamic", "factor": factor}

        return LlamaRotaryEmbedding(rope_cfg)

    # 经典 (dim, max_position_embeddings, base, ...) API，如 transformers 4.37.x
    if rst is None:
        return LlamaRotaryEmbedding(
            head_dim, max_position_embeddings=max_seq_len, base=rope_theta
        )
    if rst == "linear":
        if _LlamaLinearScalingRotaryEmbedding is None:
            raise RuntimeError(
                "当前 transformers 未提供 LlamaLinearScalingRotaryEmbedding，无法使用 rope_scaling_type=linear"
            )
        return _LlamaLinearScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
            scaling_factor=factor,
        )
    if rst == "dynamic":
        if _LlamaDynamicNTKScalingRotaryEmbedding is None:
            raise RuntimeError(
                "当前 transformers 未提供 LlamaDynamicNTKScalingRotaryEmbedding，无法使用 rope_scaling_type=dynamic"
            )
        return _LlamaDynamicNTKScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
            scaling_factor=factor,
        )
    raise ValueError(f"Unknown rope_scaling_type: {rope_scaling_type!r}")


class QuantumDiffusionModel_oseq_rope(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=3,
        head_count=4,
        max_seq_len=4096,
        max_N=25,
        rope_scaling_type=None,
        rope_scaling_factor=1.0,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(3, hidden_dim)  # g, P, b
        # self.size_embedding = nn.Embedding(max_N + 1, hidden_dim)
        self.g_proj = nn.Linear(1, hidden_dim)
        self.P_embedding = nn.Embedding(3, hidden_dim)
        self.b_embedding = nn.Embedding(3, hidden_dim)
        head_dim = hidden_dim // head_count
        self.rope = _build_llama_rotary_embedding(
            head_dim,
            hidden_dim,
            head_count,
            max_seq_len,
            rope_theta=rope_theta,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, head_count, self.rope) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.MASK_TOKEN_ID = -1.0

    def forward(self, x, mask_indices=None, return_attn_weights=False):
        """
        与 Qdmodel_oseq 一致：序列布局为 [g, P1, b1, P2, b2, ..., PN, bN]；
        注意力与 RoPE（TransformerBlock）与 Qdmodel_nseq_rope 相同。
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        length = seq_len
        qubit = (length - 1) // 2
        g = x[:, :1]
        r_raw = x[:, 1:length:2].squeeze(-1)
        b_raw = x[:, 2:length:2].squeeze(-1)

        r_tokens = (r_raw - 2).long()

        b_tokens = b_raw.long().clone()
        b_tokens[mask_indices] = 2

        x_g = self.g_proj(g)
        x_r = self.P_embedding(r_tokens)
        x_b = self.b_embedding(b_tokens)

        x = torch.zeros(batch_size, length, x_g.shape[-1], device=x.device, dtype=x_g.dtype)
        x[:, :1] = x_g
        x[:, 1:length:2] = x_r
        x[:, 2:length:2] = x_b

        type_ids = torch.zeros(length, dtype=torch.long, device=x.device)
        type_ids[1:length:2] = 1
        type_ids[2:length:2] = 2
        x = x + self.type_embedding(type_ids).unsqueeze(0)

        if return_attn_weights:
            attn_maps = []
            for layer in self.layers:
                x = layer(x)
                attn_maps.append(layer.attention_weights)
        else:
            for layer in self.layers:
                x = layer(x)

        if return_attn_weights:
            return self.output_layer(x[:, 2:length:2]), attn_maps
        return self.output_layer(x[:, 2:length:2])


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, head_count, rope):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        self.rope = rope
        rope_fwd = inspect.signature(rope.forward).parameters
        self._rope_uses_position_ids = "position_ids" in rope_fwd
        self._rope_uses_seq_len = "seq_len" in rope_fwd
        self._apply_rotary_pass_position_ids = _apply_rotary_pos_emb_requires_position_ids()

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attention_weights = None

    def forward(self, x, key_padding_mask=None):
        batch, seq_len, _ = x.shape
        position_ids = torch.arange(
            seq_len, device=x.device, dtype=torch.long
        ).unsqueeze(0).expand(batch, -1)


        q = self.q_proj(x).view(batch, seq_len, self.head_count, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.head_count, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.head_count, self.head_dim)

        # [B, H, S, D]；4.37.x: rope(x, seq_len=int)+apply 需 position_ids；4.40+: rope(x, position_ids)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        s = int(q.shape[2])
        if self._rope_uses_position_ids:
            cos, sin = self.rope(q, position_ids)
        elif self._rope_uses_seq_len:
            cos, sin = self.rope(q, seq_len=s)
        else:
            cos, sin = self.rope(q)

        if self._apply_rotary_pass_position_ids:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = torch.softmax(attn_weights, dim=-1)

        self.attention_weights = attn_weights.detach().cpu()

        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
