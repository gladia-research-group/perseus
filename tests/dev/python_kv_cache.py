import numpy as np
from linear import *

def pre_multi_head_comp(X, W_Q, W_K, W_V):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V

    Q = Q.reshape(Q.shape[0], H, -1)
    K = K.reshape(K.shape[0], H, -1)
    V = V.reshape(V.shape[0], H, -1)

    return Q, K, V
    

def multi_head_attention(Q, K, V, H, max_diff):
    """Ground-truth QK^T and QKV outputs for testing.
    Args:
        Q, K, V: shape (n_q, H, d_head)
        H: number of heads
    Returns:
        gt_qkt: shape (n_q, n_k) raw QK^T scores
        gt_attn: shape (n_q, d) final attention output after softmax and V matmul
    """

    gt_qkt = np.einsum('qhd,khd->qkh', Q, K) * V.shape[-1] ** -0.5
    attn_weights = np.exp(gt_qkt - max_diff) # subtract max for numerical stability in softmax
    attn_weights = attn_weights / np.sum(attn_weights, axis=1, keepdims=True)
    gt_attn = np.einsum('qkh,khd->qhd', attn_weights, V).reshape(Q.shape[0], -1) # reshape back to (n_q, d)

    return gt_qkt, attn_weights, gt_attn

def gt_qkt_repack(gt_qkt, N, d):
    """Repack ground-truth QK^T into head-interleaved N-slot format."""
    n_q, n_k, H = gt_qkt.shape
    d_head = d // H
    t = N // d
    assert n_k <= t * H, f"Too many keys {n_k} for N={N}, d={d}, H={H}"
    qkt_repacked = np.zeros((n_q, N))
    for gt, rep in zip(gt_qkt, qkt_repacked):
        for k, k_row in enumerate(gt):
            for h, h_val in enumerate(k_row):
                rep[k//t * t * H + h * t + k%t] = h_val

    return qkt_repacked

def gt_attn_repack(gt_attn, H, N, d):
    """Repack ground-truth attention output into head-interleaved N-slot format."""
    n_q, d_total = gt_attn.shape
    real_head_d = d_total // H
    t = N // d_total
    attn_repacked = np.zeros((n_q, N))
    for gt, rep in zip(gt_attn, attn_repacked):
        for h in range(H):
            gt_head = gt[h * real_head_d:(h + 1) * real_head_d]
            for i, v in enumerate(gt_head):
                rep[_pos(i, h, 0, t, H)] = v
    return attn_repacked

def rearrange_q_k_v(W, H):
    # perm[r] = (r % H) · d_head + ⌊r / H⌋  →  old_col = h · d_head + ld

    new_W = np.zeros_like(W)
    d_in, d_out = W.shape
    d_head = d_out // H
    for r in range(d_out):
        h = r % H
        ld = r // H
        new_W[:, r] = W[:, h * d_head + ld]
    return new_W

def rearrange_w_o(W, H):
    new_W = np.zeros_like(W)
    d_in, d_out = W.shape
    d_head = d_in // H
    for r in range(d_in):
        h = r % H
        ld = r // H
        new_W[r, :] = W[h * d_head + ld, :]
    return new_W

def rearrange_input_ct(ct, N, d, H):
    new_ct = np.zeros_like(ct)
    t = N // d
    d_head = d // H
    for r in range(d):
        h = r % H
        ld = r // H
        old_block = h * d_head + ld
        new_ct[r*t:(r+1)*t] = ct[old_block*t:(old_block+1)*t]
    return new_ct
    

def _pos(local_dim, head, tok, t, H):
    """Ciphertext slot index for a (local_dim, head, token) triple."""
    return local_dim * t * H + head * t + tok


def encode_q_interleaved(q, N, d, H=1):
    """
    Encode one query vector into head-interleaved N-slot format.
    Zeros in all token slots except tok=0.
    """
    t      = N // d
    d_head = d // H
    ct     = np.zeros(N)
    for h in range(H):
        for ld in range(d_head):
            ct[_pos(ld, h, 0, t, H)] = q[h * d_head + ld]
    return ct

def neg_inf_mask(nk, N, d, H, given_min=-1e30):
    t = N // d
    m = np.full(N, given_min)
    for h in range(H):
        for tok in range(nk):
            m[_pos(tok // t, h, tok % t, t, H)] = 0.0
    return m


def encode_kcache_group(keys, N, d, H=1):
    """
    Pack up to t key vectors into one head-interleaved K cache ciphertext.
    For H=1 reduces to: ct[dim * t + tok] = keys[tok][dim]
    """
    t      = N // d
    d_head = d // H
    assert len(keys) <= t, f"group has {len(keys)} keys but t={t}"
    ct = np.zeros(N)
    for tok_idx, k_vec in enumerate(keys):
        for h in range(H):
            for ld in range(d_head):
                ct[_pos(ld, h, tok_idx, t, H)] = k_vec[h * d_head + ld]
    return ct

def qkt_single_group(cq_rep, kcache_ct, N, d, H, group_idx, num_tok):
    def _sum_by_rot(result, N, d, H):
        s = (N // d) * H
        while s < N:
            result = result + rot(result, s)
            s *= 2
        return result


    def _qkt_mask(group_idx, num_tok, N, d, H):
        t, tH, mask = N // d, (N // d) * H, np.zeros(N)
        for i in range(N):
            if i // tH == group_idx and i % t < num_tok:
                mask[i] = 1.0
        return mask
    
    result = cq_rep * kcache_ct # elementwise product in N-slot
    result = _sum_by_rot(result, N, d, H)
    mask = _qkt_mask(group_idx, num_tok, N, d, H)
    return result * mask

def head_reduce_sum(ct, N, d, H):
    t  = N // d
    tH = t * H
    out = ct.copy()

    step = 1
    while step < t:
        mask_nw = np.array([1.0 if (i % t) + step < t else 0.0 for i in range(N)])
        mask_w  = 1.0 - mask_nw
        # shift left inner head by step, shift right inner head by step - t to align values, mask to be sure we are not pulliting wrap around blocks
        shifted = rot(out, step) * mask_nw + rot(out, step - t) * mask_w
        # aggregate
        out = out + shifted
        step *= 2

    # inter block aggregations, standard heap
    step = tH
    while step < N:
        out = out + rot(out, step)
        step *= 2

    return out

def softmax(scores, nk, N, d, H, gt_max):
    out = np.zeros_like(scores)
    ninf = neg_inf_mask(nk, N, d, H, -gt_max)
    ct = scores + ninf # to get zeros in empty positions
    e = np.exp(ct - gt_max) # exp for every row, interleaved by nature
    s = head_reduce_sum(e, N, d, H) # 
    out = e / s
    return out

class KCache:
    """Helper for building K cache ciphertexts from a key matrix."""
    def __init__(self, N, d, H):
        self.curr_keys = 0
        self.N = N
        self.d = d
        self.H = H
        self.ciphertexts = []

    def push_back(self, key):
        if self.curr_keys % (self.N // self.d) == 0: # New group needed
            # Start new group
            self.ciphertexts.append(np.zeros(self.N))
        # Add to current group
        mask = np.zeros(self.N) # one-hot for position in group
        for i in range(self.N):
            mask[i] = i % (self.N // self.d) == 0
        key = key * mask # zero out head dims that are not part of this key
        self.ciphertexts[-1] += rot(key, -(self.curr_keys % (self.N // self.d))) # rotate to correct token position and add to group
        self.curr_keys += 1

    def qkt(self, query):
        """
        Compute QK^T for a single query vector against this cache.

        Args:
            query: encoded query vector in head-interleaved N-slot format

        Returns:
            ct_windows: list of N-slot ciphertext windows
        """

        keys_per_ct = self.N // self.d

        # Mask junk
        mask = np.zeros(self.N) # one-hot for position in group
        for i in range(self.N):
            mask[i] = i % (self.N // self.d) == 0
        query = query * mask

        # Fill empty space
        step = 1
        while step < self.N // self.d:
            query = query + rot(query, -step)
            step *= 2

        attn_ct = np.zeros(self.N)
        for global_ct_indx in range(len(self.ciphertexts)):
            first_local_key = global_ct_indx * keys_per_ct
            num_tok = min(keys_per_ct, self.curr_keys - first_local_key)
            partial = qkt_single_group(query, self.ciphertexts[global_ct_indx], self.N, self.d, self.H, global_ct_indx, num_tok)
            
            attn_ct += partial

        return attn_ct * ((self.d // self.H) ** -0.5) # scale by sqrt(d_head)


class VCache:
    """Helper for building V cache ciphertexts from a value matrix."""
    def __init__(self, N, d, H=1):
        self.curr_values = 0
        self.N = N
        self.d = d
        self.H = H
        self.metaciphertext = None

    def push_back(self, value):
        """Append one value vector using the same layout as _build."""

        if self.metaciphertext is None:
            self.metaciphertext = [np.zeros(self.N) for _ in range(self.d // self.H)]

        right_rot = self.curr_values % (self.N // self.d)
        value = rot(value, -right_rot)
        for i in range(self.d // self.H):
            mask = np.zeros(self.N)
            for h in range(self.H):
                mask[i * (self.N // self.d) * self.H + h * (self.N // self.d) + right_rot] = 1.0
            self.metaciphertext[((self.curr_values // (self.N // self.d)) - i) % (self.d // self.H)] += value * mask

        self.curr_values += 1

    def softmaxV(self, softmax_scores):
        """Compute weighted sum of V cache values given attention weights."""
        res = np.zeros(self.N)
        for ct in self.metaciphertext:
            res += ct * softmax_scores
            softmax_scores = rot(softmax_scores, (self.N // self.d) * self.H) # rotate to align with next value

        step = 1
        while step < self.N // self.d:
            res = res + rot(res, step)
            step *= 2

        mask = np.zeros(self.N)
        for i in range(self.N):
            mask[i] = i % (self.N // self.d) == 0
        res = res * mask # zero out non-head dims
        return res


def _print_group_interleaving(ct, N, d, H, num_tokens, label):
    """Print one packed KCache group to inspect raw slot interleaving."""
    print(f"  {label}")
    print(f"    raw slots: {ct}")

def _run(label, Q, K, N, d, H):
    nq  = 1 if Q.ndim == 1 else Q.shape[0]
    nk  = K.shape[0]
    t   = N // d
    print(f"  {label:12s} N={N:4d} d={d:2d} H={H} t={t:2d} "
          f"nq={nq:2d} nk={nk:3d}:", end=' ')
    gt = _gt(Q, K, H, N)
    kcache = KCache(N, d, K, H)
    scores = kcache.qkt(Q)
    print(scores)
    err = np.max(np.abs(gt - scores))
    tag = "OK" if err < 1e-10 else "FAIL"
    print(f"{err:.2e} {tag}")
    return err < 1e-10

def _run_softmax(label, Q, K, N, d, H):
    nk = K.shape[0]
    nq = 1 if Q.ndim == 1 else Q.shape[0]
    if Q.ndim == 1: Q = Q[None]
    t = N // d
    print(f"  {label:12s} N={N:4d} d={d:2d} H={H} t={t:2d} nq={nq:2d} nk={nk:3d}:", end=' ')
    scores = KCache(N, d, K, H).qkt(Q)
    gt_sm, gt_sum, gt_max = _soft_gt(Q, K, H, N)
    fhe_sm = softmax(scores, nk, N, d, H, gt_max)
    err = np.max(np.abs(gt_sm - fhe_sm))
    tag = "OK" if err < 1e-10 else "FAIL"
    print(f"{err:.2e} {tag}")
    return err < 1e-10


def _same_kcache_layout(a, b):
    if a.curr_keys != b.curr_keys or len(a.ciphertexts) != len(b.ciphertexts):
        return False
    return all(np.allclose(ga, gb) for ga, gb in zip(a.ciphertexts, b.ciphertexts))


def _run_kcache_path_test(label, Q, K_base, K_new, N, d, H):
    K_full = np.vstack([K_base, K_new]) if K_new.shape[0] > 0 else K_base.copy()

    cache_full = KCache(N, d, K_full, H)
    cache_inc  = KCache(N, d, K_base, H)
    for i in range(K_new.shape[0]):
        cache_inc.push_back(K_new[i])

    scores_full = cache_full.qkt(Q)
    scores_inc  = cache_inc.qkt(Q)
    gt = _gt(Q, K_full, H, N)

    err_full = np.max(np.abs(gt - scores_full))
    err_inc  = np.max(np.abs(gt - scores_inc))
    err_diff = np.max(np.abs(scores_full - scores_inc))
    same_layout = _same_kcache_layout(cache_full, cache_inc)

    ok = (err_full < 1e-10) and (err_inc < 1e-10) and (err_diff < 1e-10) and same_layout
    tag = "OK" if ok else "FAIL"
    nq  = 1 if Q.ndim == 1 else Q.shape[0]
    nk  = K_full.shape[0]
    print(
        f"  {label:12s} N={N:4d} d={d:2d} H={H} nq={nq:2d} nk={nk:3d} "
        f"(+{K_new.shape[0]}): full={err_full:.2e} inc={err_inc:.2e} "
        f"delta={err_diff:.2e} layout={'OK' if same_layout else 'FAIL'} {tag}"
    )
    return ok


def _print_vcache_layout(vcache, label):
    print(f"  {label}: metagroups={len(vcache.metaciphertexts)}")
    for mg_idx, metacipher in enumerate(vcache.metaciphertexts):
        print(f"    metacipher {mg_idx}")
        for lane_idx, lane_ct in enumerate(metacipher):
            print(f"      lane {lane_idx}: {lane_ct}")
            

def main():
    ok = True

    # ── Multi-head QK^T ──
    print("\n=== QK^T multi-head decoding ===")
    for lab, N, d, H, nk in [("2h",8,4,2,3), ("4h",64,8,4,12),("8h",128,16,8,8)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(d), np.random.randn(nk, d), N, d, H)

    print("\n=== QK^T multi-head classification ===")
    for lab, N, d, H, n in [("2h",8,4,2,3),("4h",64,8,4,12),("8h",128,16,8,8)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(n, d), np.random.randn(n, d), N, d, H)

    # ── KCache construction paths ──
    print("\n=== KCache build paths (full vs incremental add) ===")
    np.random.seed(42)
    N, d, H = 128, 16, 8

    Q = np.random.randn(5, d)
    K_full = np.random.randn(11, d)
    cache_full = KCache(N, d, K_full, H)
    scores_cache = cache_full.qkt(Q)
    gt_full = _gt(Q, K_full, H, N)
    err_full_build = np.max(np.abs(gt_full - scores_cache))
    tag = "OK" if err_full_build < 1e-10 else "FAIL"
    print(f"  {'whole-k':12s} N={N:4d} d={d:2d} H={H} nq={Q.shape[0]:2d} nk={K_full.shape[0]:3d}: {err_full_build:.2e} {tag}")
    ok &= (err_full_build < 1e-10)

    K_base = np.random.randn(8, d)
    K_more = np.random.randn(4, d)
    ok &= _run_kcache_path_test("add-1", Q, K_base, K_more[:1], N, d, H)
    ok &= _run_kcache_path_test("add-many", Q, K_base, K_more, N, d, H)
    
    print("\n=== Softmax (oracle max, FHE sum) ===")
    for lab, N, d, H, nk in [
        ("small",      16, 4, 2, 5),
        ("gpt2", 32768, 1024, 16, 5)
    ]:
        np.random.seed(42)
        Q = np.random.randn(12, d)
        K = np.random.randn(nk, d)
        ok &= _run_softmax(lab, Q, K, N, d, H)
    
    print("\n=== VCache storage demo (constant vectors) ===")
    N, d, H = 16, 4, 1
    n_vals = 7
    V_demo = np.vstack([np.full((d,), i + 1.0) for i in range(n_vals)])
    print(f"  params: N={N}, d={d}, H={H}, n_values={n_vals}")
    print("  input values (row i is filled with i+1):")
    print(f"  {V_demo}")
    vcache_demo = VCache(N, d, V_demo, H)
    _print_vcache_layout(vcache_demo, "stored V-cache")

    print("  incremental push_back view:")
    vcache_inc = VCache(N, d, V_demo[:-1], H)
    _print_vcache_layout(vcache_inc, "before push_back")
    vcache_inc.push_back(V_demo[-1])
    _print_vcache_layout(vcache_inc, "after push_back")


    print("\n=== Visual interleaving demo (incremental add) ===")
    N, d, H = 8, 4, 1
    t = N // d
    n_demo = t + 2  # force spill into a second KCache group

    K_demo = np.vstack([np.full((d,), i + 1.0) for i in range(n_demo)])
    print(f"  params: N={N}, d={d}, H={H}, t={t}, n_keys={n_demo}")
    print("  input keys (row i is filled with i+1):")
    print(f"  {K_demo}")
    kcache_inc = KCache(N, d, K_demo[:-1], H)
    print(f"  before add: n_keys={kcache_inc.curr_keys}")
    for g, ct in enumerate(kcache_inc.ciphertexts):
        start = g * t
        num_tok = min(t, kcache_inc.curr_keys - start)
        _print_group_interleaving(ct, N, d, H, num_tok, f"group {g} (keys {start}..{start + num_tok - 1})")

    kcache_inc.push_back(K_demo[-1])
    print(f"  after add : n_keys={kcache_inc.curr_keys} (added key value {n_demo})")
    for g, ct in enumerate(kcache_inc.ciphertexts):
        start = g * t
        num_tok = min(t, kcache_inc.curr_keys - start)
        _print_group_interleaving(ct, N, d, H, num_tok, f"group {g} (keys {start}..{start + num_tok - 1})")

    print(f"\n{'ALL PASSED' if ok else 'SOME FAILURES'}")
    
def compute_avg_interleaved(ct, N, d):
    t  = N // d
    out = ct.copy()

    step = 1
    while step < t:
        out += rot(out, step)
        step *= 2
    return out / d

# TODO: do this when export weights as it is anightmare in c++
def calculate_per_head_logit_bounds(W_q, W_k, norm_type="layernorm"):
    """
    Calculates the theoretical maximum and minimum values (logits) that can enter 
    the softmax of an attention mechanism, separated by head.
    
    Args:
        W_q: Query weights of shape (num_heads, d_model, d_k)
        W_k: Key weights of shape (num_heads, d_model, d_k)
        norm_type: "layernorm" (mean=0, var=1) or "rmsnorm" (var=1). 
                   Both result in vectors with an L2 norm of sqrt(d_model).
                   
    Returns:
        dict: Containing 1D numpy arrays of shape (num_heads,) with the bounds.
    """
    assert W_q.shape == W_k.shape, "W_q and W_k must have the same shape."
    assert W_q.ndim == 3, "Expected shape (num_heads, d_model, d_k)."
    
    num_heads, d_model, d_k = W_q.shape
    
    # 1. Calculate the base matrix M for all heads
    # Shape: (num_heads, d_model, d_model)
    M = np.matmul(W_q, W_k.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # 2. Project onto the normalized subspace
    if norm_type == "layernorm":
        # LayerNorm constraints: variance=1, mean=0.
        # Project onto a mean-zero subspace using projection matrix P
        P = np.eye(d_model) - (1.0 / d_model) * np.ones((d_model, d_model))
        
        # NumPy matmul broadcasting works beautifully here: 
        # (d_model, d_model) @ (num_heads, d_model, d_model) @ (d_model, d_model)
        M_proj = P @ M @ P
    elif norm_type == "rmsnorm":
        # RMSNorm constraint: root-mean-square=1. Mean is NOT forced to zero.
        M_proj = M
    else:
        raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")

    # Scaling factor because norm equals sqrt(d_model) instead of 1
    # ||x||_2 * ||y||_2 = sqrt(d_model) * sqrt(d_model) = d_model
    scale = float(d_model)
    
    # 3. Bounds for DIFFERENT tokens (i != j) -> Uses Singular Values
    # compute_uv=False saves computation by only returning the singular values
    # Shape of S_vals: (num_heads, d_model)
    S_vals = np.linalg.svd(M_proj, compute_uv=False)
    max_singular_vals = np.max(S_vals, axis=-1)  # Shape: (num_heads,)
    
    max_diff = scale * max_singular_vals

    return max_diff


if __name__ == "__main__":
    # main()

    N = 32
    d = 8
    H = 2
    n = 3
    assert n <= N // H, "Too many samples for demo parameters"

    curr_X = np.random.randn(1, d)
    curr_X = curr_X / np.linalg.norm(curr_X, axis=-1, keepdims=True)# normalize to match layernorm constraints
    W_Q = np.random.randn(d, d)
    old_X = np.random.randn(n, d)
    old_X = old_X / np.linalg.norm(old_X, axis=-1, keepdims=True)# normalize to match layernorm constraints
    W_K = np.random.randn(d, d)
    W_V = np.random.randn(d, d)
    W_O = np.random.randn(d, d)

    max_diff = calculate_per_head_logit_bounds(W_Q.reshape(d, H, d//H).transpose(1,0,2), W_K.reshape(d, H, d//H).transpose(1,0,2), norm_type="layernorm")
    max_diff = np.max(max_diff)

    Q = curr_X @ W_Q
    K = old_X @ W_K
    V = old_X @ W_V
    Q = Q.reshape(Q.shape[0], H, -1)
    K = K.reshape(K.shape[0], H, -1)
    V = V.reshape(V.shape[0], H, -1)

    gt_qkt, soft_out, gt_attn = multi_head_attention(Q, K, V, H=H, max_diff=max_diff)
    out = gt_attn @ W_O
    gt_res_out = out + curr_X


    rearranged_w_q = rearrange_q_k_v(W_Q, H) # d x d == d x (H * d_head)
    rearranged_w_k = rearrange_q_k_v(W_K, H) # d x d == d x (H * d_head)
    rearranged_w_v = rearrange_q_k_v(W_V, H) # d x d == d x (H * d_head)
    rearranged_w_o = rearrange_w_o(W_O, H)
    encoded_W_q = encode_W(rearranged_w_q, N, d, alpha=1, is_up=True)
    encoded_W_k = encode_W(rearranged_w_k, N, d, alpha=1, is_up=True)
    encoded_W_v = encode_W(rearranged_w_v, N, d, alpha=1, is_up=True)
    encoded_W_o = encode_W(rearranged_w_o, N, d, alpha=1, is_up=True)

    computed_params = compute_params(N, d, alpha=1, is_up=True)

    kcache = KCache(N, d, H)
    vcache = VCache(N, d, H)
    for i, x in enumerate(old_X):
        encoded_x = encode_x(x, N, d, alpha=1, is_up=True)
        encoded_K = cachemir_vmm(encoded_x, encoded_W_k, N, d, d, alpha=1, is_up=True, computed_params=computed_params)
        encoded_V = cachemir_vmm(encoded_x, encoded_W_v, N, d, d, alpha=1, is_up=True, computed_params=computed_params)
        kcache.push_back(encoded_K)
        vcache.push_back(encoded_V)

    print("VCache layout:")
    for ct in vcache.metaciphertext:
        print(ct)
    print(V)

    encoded_x = encode_x(curr_X[0], N, d, alpha=1, is_up=True)
    encoded_Q = cachemir_vmm(encoded_x, encoded_W_q, N, d, d, alpha=1, is_up=True, computed_params=computed_params)
    attn_val = kcache.qkt(encoded_Q)
    soft = softmax(attn_val, n, N, d, H, gt_max=max_diff)
    res = vcache.softmaxV(soft)
    print("")
    print(res)
    encoded_O = cachemir_vmm(res, encoded_W_o, N, d, d, alpha=1, is_up=True, computed_params=computed_params)
    print(out)
    print(encoded_O)
    
    res_res = encoded_O + encoded_x
    
    print(res_res)
    print(gt_res_out)