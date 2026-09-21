import numpy as np


def _stop_ids(eos_token_id):
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset({int(eos_token_id)})
    return frozenset(int(t) for t in eos_token_id)


def _make_sampler(do_sample, temperature, top_k, seed):
    """Softmax sampling over decrypted logits (client-feedback path), or None for argmax."""
    if not do_sample:
        return None
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 for sampling, got {temperature}")
    if top_k is not None and int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    rng = np.random.default_rng(seed)

    def sample(logits):
        z = np.asarray(logits, dtype=np.float64) / float(temperature)
        if top_k is not None and int(top_k) < z.shape[0]:
            keep = np.argpartition(z, -int(top_k))[-int(top_k):]
            masked = np.full_like(z, -np.inf)
            masked[keep] = z[keep]
            z = masked
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(z.shape[0], p=p))

    return sample
