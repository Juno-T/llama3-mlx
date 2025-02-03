import mlx.core as mx
import numpy as np
import torch
import math


def apply_rope_scaling(freqs: mx.array) -> mx.array:
    freqs = torch.tensor(np.array(freqs), dtype=torch.float32)
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq /
                             scale_factor + smooth * freq)
    freqs = torch.tensor(new_freqs, dtype=freqs.dtype).cpu().numpy()
    return mx.array(freqs, dtype=mx.float32)
