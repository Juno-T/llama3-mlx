import os
from typing import Any, Dict
import mlx.core as mx
from functools import partial


# vocab_size = 4096
# emb_dim = 2222

# context_size = 100
# base_theta = 10000.0

# num_heads = 2
# attn_dim = 333

# ffn_dim = 444

# embedding_table = mx.random.normal([vocab_size, emb_dim])

# W_Q = mx.random.normal([num_heads * attn_dim, emb_dim])
# W_K = mx.random.normal([num_heads * attn_dim, emb_dim])
# W_V = mx.random.normal([num_heads * attn_dim, emb_dim]) # V_down

# W_O = mx.random.normal([emb_dim, num_heads * attn_dim]) # V_up


def embed_token(embedding_table, tokens):
    return embedding_table[tokens]


# slower for 1d tokens. Fast for 2d tokens (batched)
@partial(mx.vmap, in_axes=(None, 0), out_axes=0)
def embed_token_vmapped(embedding_table, tokens):
    return embed_token(embedding_table, tokens)


def precompute_rope_cos_sin(max_positions, emb_dim, base_theta=10000.0, use_scaled_rope=True, stream=mx.cpu):
    d = emb_dim
    theta_i = mx.power(
        base_theta, (-2 * mx.divide(mx.arange(0, d // 2), d, stream=stream)), stream=stream)

    if use_scaled_rope:
        from .ref_impl import apply_rope_scaling
        theta_i = apply_rope_scaling(theta_i)
    indices = mx.stack([mx.arange(0, d // 2)]*2).T.flatten()  # (d,)
    positions = mx.arange(0, max_positions)  # (max_positions,)
    # (max_positions, d)
    pos_theta_i = mx.outer(positions, theta_i[indices], stream=stream)
    cos = mx.cos(pos_theta_i, stream=stream)  # (max_positions, d)
    sin = mx.sin(pos_theta_i, stream=stream)  # (max_positions, d)
    # (1, max_positions, d)
    return mx.expand_dims(cos, axis=0), mx.expand_dims(sin, axis=0)


def apply_rope(q, k, cos, sin, expand_axis=1):
    """ Applies rope to q and k matrices.

    Why q and k, and not just x?
    The essense of RoPE method is to have __relative__ positional embeddings. The relative position information
    will only occur when we take dot product between the two embedded vectors, which is meant to be the step of
    Q @ K_T in the self-attention mechanism. We can just use single argument, but making it explicit here will
    prevent any confusion.

    Args:
        q: query matrix of shape (..., seq_len, dim)
        k: key matrix of shape (..., seq_len, dim)

    """
    def rearrange(x):
        """Rearrange the hidden dims of the input, according to the RoPE paper.

        From x1, x2, x3, x4, x5 ...
        To  -x2, x1,-x4, x3,-x6, x5, ...
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return mx.stack([-x2, x1], axis=-1).flatten(-2)

    seq_len = q.shape[-2]
    cos = cos[:seq_len, :]
    sin = sin[:seq_len, :]
    cos = mx.expand_dims(cos, axis=expand_axis)
    sin = mx.expand_dims(sin, axis=expand_axis)
    print(cos.shape, q.shape)
    q_pos_embed = (q * cos) + (rearrange(q) * sin)
    k_pos_embed = (k * cos) + (rearrange(k) * sin)
    return q_pos_embed, k_pos_embed


def apply_multihead_self_attn(
        x: mx.array,
        W_Q: mx.array,
        W_K: mx.array,
        W_V: mx.array,
        W_O: mx.array,
        cos: mx.array,
        sin: mx.array,
        num_heads: int,
        num_kv_heads: int,
        attn_dim: int):
    seq_len = x.shape[0]
    num_reps = num_heads // num_kv_heads
    # (seq_len, num_heads * attn_dim) = [[q1], [q2], [q3]]
    Q = mx.matmul(x, W_Q.T)
    # (seq_len, num_kv_heads * attn_dim) = [[k1], [k2], [k3]]
    K = mx.matmul(x, W_K.T)
    # (seq_len, num_kv_heads * attn_dim) = [[v1], [v2], [v3]]
    V = mx.matmul(x, W_V.T)

    Q = Q.reshape(seq_len, num_heads, attn_dim).transpose(
        1, 0, 2)  # (num_heads, seq_len, attn_dim)
    K = K.reshape(seq_len, num_kv_heads, attn_dim).transpose(
        1, 0, 2)  # (num_kv_heads, seq_len, attn_dim)
    V = V.reshape(seq_len, num_kv_heads, attn_dim).transpose(
        1, 0, 2)  # (num_kv_heads, seq_len, attn_dim)

    Q, K = apply_rope(Q, K, cos, sin, expand_axis=1)
    # (num_kv_heads * num_reps = num_heads, seq_len, attn_dim)
    K = mx.concatenate([K] * num_reps)
    # (num_kv_heads * num_reps = num_heads, seq_len, attn_dim)
    V = mx.concatenate([V] * num_reps)

    attn_score = mx.matmul(Q, K.transpose(0, -1, -2))
    # (num_heads, seq_len, seq_len)
    attn_score = mx.softmax(attn_score / mx.sqrt(attn_dim), axis=-1)

    # (num_heads, seq_len, attn_dim)
    delta_x_down = mx.matmul(attn_score, V)
    delta_x_down = delta_x_down.transpose(1, 0, 2).reshape(
        seq_len, num_heads * attn_dim)

    delta_x = mx.matmul(delta_x_down, W_O.T)

    # Q.reshape(-1, num_heads, attn_dim).transpose(-2, 0, -1) # (num_head, seq_len, attn_dim)
    # mx.stack(mx.split(Q, num_heads, axis=-1)).shape
    # attn_score = mx.matmul(Q, K.T) # (seq_len, seq_len) = [[q1k1, q1k2, q1k3], [q2k1, q2k2, q2k3], [q3k1, q3k2, q3k3]]
    # delta_x_down = mx.matmul(attn_score, V) # (seq_len, attn_dim) = [[q1k1v1 + q1k2v2 + q1k3v3], [q2k1v1 + q2k2v2 + q2k3v3], [q3k1v1 + q3k2v2 + q3k3v3]]

    # # attn_dim -> emb_dim
    # delta_x = mx.matmul(delta_x_down, W_O) # (seq_len, emb_dim)
    # delta_x.shape

    # residual
    # return x + delta_x
    return delta_x


def silu(x):
    return x * mx.sigmoid(x)


def gated_mlp(x, W_gate, W_up, W_down):
    """ SwiGLU/gated_mlp

    Not clear why it is used. The paper said that it's a simpler attention module.
    But no explanation on why it works better, or why llama use it.

    # x: (context_size, emb_dim)
    # W_gate: (emb_dim, ffn_hidden_dim)
    # W_up: (ffn_hidden_dim, emb_dim)
    # W_down: (emb_dim, ffn_hidden_dim)
    """
    x1 = mx.matmul(x, W_gate)  # (context_size, ffn_hidden_dim)
    x1 = silu(x1)
    x3 = mx.matmul(x, W_up)  # (context_size, ffn_hidden_dim)
    return mx.matmul(x1 * x3, W_down)


def rms_norm(hidden_states: mx.array, weight: mx.array, eps: float = 1e-6):
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(mx.float32)
    variance = mx.power(hidden_states, 2).mean(-1, keepdim=True)
    hidden_states = hidden_states * mx.rsqrt(variance + eps)
    return (weight * hidden_states).astype(original_dtype)


def transformer_block(
    x: mx.array,
    W_Q: mx.array,
    W_K: mx.array,
    W_V: mx.array,
    W_O: mx.array,
    cos: mx.array,
    sin: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
    W_down: mx.array,
    W_rms_norm: mx.array,
    rms_eps: float,
    num_heads: int,
    num_kv_heads: int,
    attn_dim: int
) -> mx.array:
    h1 = rms_norm(x, W_rms_norm, rms_eps)
    h1 = x + apply_multihead_self_attn(
        h1, W_Q, W_K, W_V, W_O, cos, sin, num_heads, num_kv_heads, attn_dim)
    h2 = rms_norm(h1, W_rms_norm, rms_eps)
    return h1 + gated_mlp(h2, W_gate, W_up, W_down)


def llama(
    tokens: mx.array,
    weights: Dict[str, mx.array],
    hparams: Dict[str, Any],
):
    rms_eps: float = hparams["rms_eps"]
    context_size: int = hparams["context_size"]
    num_heads: int = hparams["num_heads"]
    num_kv_heads: int = hparams["num_kv_heads"]
    attn_dim: int = hparams["attn_dim"]
    num_layers: int = hparams["num_layers"]
    vocab_size: int = hparams["vocab_size"]
    emb_dim: int = hparams["emb_dim"]
    ffn_hidden_dim: int = hparams["ffn_hidden_dim"]
    rope_theta: float = hparams["rope_theta"]
    use_scaled_rope: bool = hparams["use_scaled_rope"]

    _bsz, seqlen = tokens.shape

    embedding_table = weights["model.embedding_table"]
    h = embed_token_vmapped(embedding_table, tokens)
    cos, sin = precompute_rope_cos_sin(
        context_size * 2, attn_dim, rope_theta, use_scaled_rope)

    # mask = None
    # if seqlen > 1:
    #     mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

    #     mask = torch.triu(mask, diagonal=1)

    #     # When performing key-value caching, we compute the attention scores
    #     # only for the new sequence. Thus, the matrix of scores is of size
    #     # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
    #     # j > cache_len + i, since row i corresponds to token cache_len + i.
    #     mask = torch.hstack(
    #         [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
    #     ).type_as(h)

    for l in range(num_layers):
        W_Q = weights[f"model.layers.{l}.attn.W_Q"]
        W_K = weights[f"model.layers.{l}.attn.W_K"]
        W_V = weights[f"model.layers.{l}.attn.W_V"]
        W_O = weights[f"model.layers.{l}.attn.W_O"]
        W_gate = weights[f"model.layers.{l}.attn.W_gate"]
        W_up = weights[f"model.layers.{l}.attn.W_up"]
        W_down = weights[f"model.layers.{l}.attn.W_down"]
        W_rms_norm = weights[f"model.layers.{l}.attn.W_rms_norm"]
        h = transformer_block(
            h,
            W_Q,
            W_K,
            W_V,
            W_O,
            cos,
            sin,
            W_gate,
            W_up,
            W_down,
            W_rms_norm,
            rms_eps,
            num_heads,
            num_kv_heads,
            attn_dim,
        )
    W_rms_norm = weights["model.norm_out.W_rms_norm"]
    W_out = weights["lm_head.weight"]
    h = rms_norm(h, W_rms_norm, rms_eps)
    output = mx.matmul(h, W_out)
    return output


def generate():
    pass


def load_weights(weight_path: str):
    import torch
    weights = torch.load(weight_path)
    return weights


def load_hparams(hparams_path: str):
    import json
    with open(hparams_path) as f:
        hparams_llama = json.load(f)
    # hparams_llama = {
    #   "dim": 2048,
    #   "ffn_dim_multiplier": 1.5,
    #   "multiple_of": 256,
    #   "n_heads": 32,
    #   "n_kv_heads": 8,
    #   "n_layers": 16,
    #   "norm_eps": 1e-05,
    #   "rope_theta": 500000.0,
    #   "use_scaled_rope": true,
    #   "vocab_size": 128256
    # }
    hparams = {
        "emb_dim": hparams_llama["dim"],
        "attn_dim": None,                           # ? To be calculated
        "num_heads": hparams_llama["n_heads"],
        "num_kv_heads": hparams_llama["n_kv_heads"],
        "num_layers": hparams_llama["n_layers"],
        "vocab_size": hparams_llama["vocab_size"],
        "context_size": None,                       # ! Not presented
        "ffn_hidden_dim": None,                         # ? To be calculated
        "rms_eps": hparams_llama["norm_eps"],
        "rope_theta": hparams_llama["rope_theta"],
        "use_scaled_rope": hparams_llama["use_scaled_rope"],
    }
    hparams["attn_dim"] = hparams["emb_dim"] // hparams["num_heads"]

    def llama_ffn_hidden_dim_calculator(dim, multiple_of, ffn_dim_multiplier):
        hidden_dim = dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)
        return hidden_dim

    hparams["ffn_hidden_dim"] = llama_ffn_hidden_dim_calculator(
        hparams["attn_dim"], hparams_llama["multiple_of"], hparams_llama["ffn_dim_multiplier"]
    )
    hparams["context_size"] = 8192

    return hparams


def load_tokenizer(tokenizer_path: str):
    from llama_models.llama3.api import Tokenizer
    return Tokenizer(tokenizer_path)


if __name__ == "__main__":
    model_dir = "../weights/checkpoints/Llama3.2-1B"
    weights = load_weights(os.path.join(model_dir, "consolidated.00.pth"))
    hparams = load_hparams(os.path.join(model_dir, "hparams.json"))
    tokenizer = load_tokenizer(os.path.join(model_dir, "tokenizer.model"))
    hparams["model_path"] = model_dir
