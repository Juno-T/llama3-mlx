import os
from typing import Any, Dict
import mlx.core as mx
from functools import partial


from llama_models.llama3.api import Tokenizer

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


def precompute_rope_cos_sin(max_positions, attn_dim, base_theta=10000.0, use_scaled_rope=True, stream=mx.cpu):
    d = attn_dim
    theta_i = mx.power(
        base_theta, (-2 * mx.divide(mx.arange(0, d // 2), d, stream=stream)), stream=stream)

    if use_scaled_rope:
        from ref_impl import apply_rope_scaling
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
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2:]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return mx.stack([-x2, x1], axis=-1).flatten(-2)

    seq_len = q.shape[-2]
    cos = cos[:, :seq_len, :]  # (bsz, seq_len, attn_dim)
    sin = sin[:, :seq_len, :]  # (bsz, seq_len, attn_dim)
    cos = mx.expand_dims(cos, axis=expand_axis)
    sin = mx.expand_dims(sin, axis=expand_axis)
    # print(cos.shape, q.shape)
    q_pos_embed = (q * cos) + (rearrange(q) * sin)
    k_pos_embed = (k * cos) + (rearrange(k) * sin)
    return q_pos_embed, k_pos_embed


def apply_multihead_self_attn(
        x: mx.array,
        mask: mx.array,
        W_Q: mx.array,
        W_K: mx.array,
        W_V: mx.array,
        W_O: mx.array,
        cos: mx.array,
        sin: mx.array,
        num_heads: int,
        num_kv_heads: int,
        attn_dim: int):
    _bsz, seq_len, emb_dim = x.shape
    num_reps = num_heads // num_kv_heads
    mask = mx.expand_dims(mask, axis=[0, 1])  # (bsz, 1, seq_len, seq_len)
    # (seq_len, num_heads * attn_dim) = [[q1], [q2], [q3]]
    Q = mx.matmul(x, W_Q.T)
    # (seq_len, num_kv_heads * attn_dim) = [[k1], [k2], [k3]]
    K = mx.matmul(x, W_K.T)
    # (seq_len, num_kv_heads * attn_dim) = [[v1], [v2], [v3]]
    V = mx.matmul(x, W_V.T)

    Q = Q.reshape(-1, seq_len, num_heads, attn_dim).transpose(
        0, 2, 1, 3)  # (bsz, num_heads, seq_len, attn_dim)
    K = K.reshape(-1, seq_len, num_kv_heads, attn_dim).transpose(
        0, 2, 1, 3)  # (bsz, num_kv_heads, seq_len, attn_dim)
    V = V.reshape(-1, seq_len, num_kv_heads, attn_dim).transpose(
        0, 2, 1, 3)  # (bsz, num_kv_heads, seq_len, attn_dim)

    Q, K = apply_rope(Q, K, cos, sin, expand_axis=1)
    # (num_kv_heads * num_reps = num_heads, seq_len, attn_dim)
    K = mx.concatenate([mx.expand_dims(K, 2)] * num_reps, axis=2).reshape(
        _bsz, num_heads, seq_len, attn_dim
    )
    # (num_kv_heads * num_reps = num_heads, seq_len, attn_dim)
    V = mx.concatenate([mx.expand_dims(V, 2)] * num_reps, axis=2).reshape(
        _bsz, num_heads, seq_len, attn_dim
    )

    attn_score = mx.matmul(Q, K.transpose(0, 1, 3, 2)
                           ) / (attn_dim ** 0.5)
    # (bsz, num_heads, seq_len, seq_len)
    # attn_score = mx.softmax(attn_score / mx.sqrt(attn_dim), axis=-1)
    # print(attn_score[0, 15, -1], (attn_dim ** -0.5), attn_score.shape)
    attn_score = mx.softmax(attn_score + mask, axis=-1)
    # (bsz, num_heads, seq_len, attn_dim)
    delta_x_down = mx.matmul(attn_score, V)
    delta_x_down = delta_x_down.transpose(0, 2, 1, 3).reshape(
        _bsz, seq_len, num_heads * attn_dim)

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


def gated_mlp(x: mx.array, W_gate: mx.array, W_up: mx.array, W_down: mx.array):
    """ SwiGLU/gated_mlp

    Not clear why it is used. The paper said that it's a simpler attention module.
    But no explanation on why it works better, or why llama use it.

    # x: (context_size, emb_dim)
    # W_gate: (emb_dim, ffn_hidden_dim)
    # W_up: (ffn_hidden_dim, emb_dim)
    # W_down: (emb_dim, ffn_hidden_dim)
    """
    x1 = mx.matmul(x, W_gate.T)  # (context_size, ffn_hidden_dim)
    x1 = silu(x1)
    x3 = mx.matmul(x, W_up.T)  # (context_size, ffn_hidden_dim)
    return mx.matmul(x1 * x3, W_down.T)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5):
    original_dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.power(x, 2).mean(-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    return (weight * x).astype(original_dtype)


def transformer_block(
    x: mx.array,
    mask: mx.array,
    W_Q: mx.array,
    W_K: mx.array,
    W_V: mx.array,
    W_O: mx.array,
    cos: mx.array,
    sin: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
    W_down: mx.array,
    W_attn_rms_norm: mx.array,
    W_ffn_rms_norm: mx.array,
    rms_eps: float,
    num_heads: int,
    num_kv_heads: int,
    attn_dim: int
) -> mx.array:
    h = x + apply_multihead_self_attn(
        rms_norm(x, W_attn_rms_norm, rms_eps),
        mask, W_Q, W_K, W_V, W_O, cos, sin, num_heads, num_kv_heads, attn_dim)
    return h + gated_mlp(
        rms_norm(h, W_ffn_rms_norm, rms_eps), W_gate, W_up, W_down)


def create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32):
    mask = mx.triu(mx.ones((seq_len, seq_len,),
                   dtype=dtype) * -mx.inf, k=1)
    return mask


def llama(
    weights: Dict[str, mx.array],
    hparams: Dict[str, Any],
    tokens: mx.array,
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
    # use_scaled_rope: bool = False

    _bsz, seq_len = tokens.shape

    embedding_table = weights["tok_embeddings.weight"]
    h = embed_token_vmapped(embedding_table, tokens)
    cos, sin = precompute_rope_cos_sin(
        context_size * 2, attn_dim, rope_theta, use_scaled_rope)

    mask = create_causal_mask(seq_len)

    # q = mx.array(mx.arange(emb_dim * 10, dtype=mx.float32)
    #              ).reshape(1, 10, emb_dim)
    # mask = create_causal_mask(10)

    for l in range(num_layers):
        W_Q = weights[f"layers.{l}.attention.wq.weight"]
        W_K = weights[f"layers.{l}.attention.wk.weight"]
        W_V = weights[f"layers.{l}.attention.wv.weight"]
        W_O = weights[f"layers.{l}.attention.wo.weight"]
        W_gate = weights[f"layers.{l}.feed_forward.w1.weight"]
        W_up = weights[f"layers.{l}.feed_forward.w3.weight"]
        W_down = weights[f"layers.{l}.feed_forward.w2.weight"]
        W_attn_rms_norm = weights[f"layers.{l}.attention_norm.weight"]
        W_ffn_rms_norm = weights[f"layers.{l}.ffn_norm.weight"]
        # q = apply_multihead_self_attn(
        #     q, mask, W_Q, W_K, W_V, W_O, cos, sin, num_heads, num_kv_heads, attn_dim)
        # print(q[0, 0], q.shape)
        # raise Exception
        h = transformer_block(
            h,
            mask,
            W_Q,
            W_K,
            W_V,
            W_O,
            cos,
            sin,
            W_gate,
            W_up,
            W_down,
            W_attn_rms_norm,
            W_ffn_rms_norm,
            rms_eps,
            num_heads,
            num_kv_heads,
            attn_dim,
        )
    W_rms_norm = weights["norm.weight"]
    W_out = weights["output.weight"]
    h = rms_norm(h, W_rms_norm, rms_eps)
    output = mx.matmul(h, W_out.T)
    return output


def load_weights(weight_path: str):
    import torch
    weights_pt = torch.load(weight_path, map_location="cpu")
    keys = list(weights_pt.keys())
    weights = {}
    for k in keys:
        weights[k] = mx.array(weights_pt[k].to(
            torch.float32).numpy(), dtype=mx.float32)
        del weights_pt[k]
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
    return Tokenizer(tokenizer_path)


def generate(
        text,
        model,
        tokenizer: Tokenizer,
        top_k=100,
        temperature=1,
        max_generation=50,
        context_size=2048,
        seed=42):
    import numpy as np
    mx.random.seed(seed)
    np.random.seed(seed+1)
    pad_id = tokenizer.pad_id
    tokens = tokenizer.encode(text, bos=True, eos=False)
    # tokens = [pad_id] * (context_size - len(tokens)) + tokens
    print(text, end="")
    heatmap = []
    for i in range(max_generation):
        h = model(mx.expand_dims(mx.array(tokens[-context_size:]), axis=0))
        logits = h[0, -1, :]
        temperature = 0.6
        probs = np.array(mx.softmax(logits / temperature, axis=-1))
        top_k_indices = probs.argsort()[-top_k:][::-1]
        p = probs[top_k_indices]
        p = p / np.sum(p)
        next_token = np.random.choice(
            top_k_indices, p=p).item()
        print(tokenizer.decode([next_token]), end="", flush=True)
        tokens.append(next_token)
        heatmap.append(np.array(probs)[:500])


if __name__ == "__main__":
    model_dir = "./weights/checkpoints/Llama3.2-1B-Instruct"
    weights = load_weights(os.path.join(model_dir, "consolidated.00.pth"))
    hparams = load_hparams(os.path.join(model_dir, "params.json"))
    tokenizer = load_tokenizer(os.path.join(model_dir, "tokenizer.model"))
    hparams["model_path"] = model_dir
    # print(hparams)

    model = partial(llama, weights, hparams)

    prompt = """
System: You are an expert in Japan tourist guide. You must answer with an elaborated response that is a complete sentence.
User: Hello
Assistant: Hello, how can I help you today? I can introduce you to some of the most popular tourist attractions in Japan.
User: {sentence}
Assistant:"""
    sentence = "What is your favorite place in Japan?"
    generate(
        prompt.format(sentence=sentence),
        model,
        tokenizer,
        top_k=100,
        temperature=1,
        max_generation=50,)

    # Example response:
    # "I like to visit the Japanese garden. It is a place where you can relax and enjoy the beautiful"
