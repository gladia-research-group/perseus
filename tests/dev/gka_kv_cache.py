import numpy as np
from linear import *

def pre_multi_head_comp(X, W_Q, W_K, W_V):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V

    Q = Q.reshape(Q.shape[0], H, -1)
    K = K.reshape(K.shape[0], H, -1)
    V = V.reshape(V.shape[0], H, -1)

    return Q, K, V
    

def multi_head_gka_attention(Q, K, V, H, max_diff):
    Q_norm = np.linalg.norm(Q, axis=-1)[:, None] ** 2
    K_norm = np.linalg.norm(K, axis=-1)[None, :] ** 2
    diff = Q_norm + K_norm - 2 * np.einsum('qhd,khd->qkh', Q, K) # shape (n_q, n_k, H)

    attn_weights = np.exp(-diff) # subtract max for numerical stability in softmax
    gt_attn = np.einsum('qkh,khd->qhd', attn_weights, V).reshape(Q.shape[0], -1) # reshape back to (n_q, d)

    return diff, attn_weights, gt_attn

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
    
    result = cq_rep - kcache_ct # elementwise product in N-slot
    result = result * (-result) # square to get elementwise squared differences
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

    step = tH
    while step < N:
        out = out + rot(out, step)
        step *= 2

    return out

def exp(scores, nk, N, d, H, gt_max):
    ninf = neg_inf_mask(nk, N, d, H, -gt_max)
    ct = scores + ninf # to get zeros in empty positions
    return np.exp(ct)

class GKA_KCache:
    """Helper for building K cache ciphertexts from a key matrix."""
    def __init__(self, N, d, H):
        self.curr_keys = 0
        self.N = N
        self.d = d
        self.H = H
        self.ciphertexts = []

    def push_back(self, key):
        if self.curr_keys % (self.N // self.d) == 0: # New group needed
            self.ciphertexts.append(np.zeros(self.N))

        mask = np.zeros(self.N) # one-hot for position in group
        for i in range(self.N):
            mask[i] = i % (self.N // self.d) == 0
        key = key * mask # zero out head dims that are not part of this key
        self.ciphertexts[-1] += rot(key, -(self.curr_keys % (self.N // self.d))) # rotate to correct token position and add to group
        self.curr_keys += 1

    def qkt(self, query):
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

        return attn_ct # scale by sqrt(d_head)


class GKA_VCache:
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

def calculate_per_head_logit_bounds(W_q, W_k, norm_type="layernorm"):
    assert W_q.shape == W_k.shape, "W_q and W_k must have the same shape."
    assert W_q.ndim == 3, "Expected shape (num_heads, d_model, d_k)."
    
    num_heads, d_model, d_k = W_q.shape
    
    M = np.matmul(W_q, W_k.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    if norm_type == "layernorm":
        P = np.eye(d_model) - (1.0 / d_model) * np.ones((d_model, d_model))
        
        M_proj = P @ M @ P
    elif norm_type == "rmsnorm":
        M_proj = M
    else:
        raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")

    scale = float(d_model)
    
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

    max_diff = calculate_per_head_logit_bounds(W_Q.reshape(d, H, d//H).transpose(1,0,2), W_K.reshape(d, H, d//H).transpose(1,0,2), norm_type="layernorm")
    max_diff = np.max(max_diff)

    Q = curr_X @ W_Q
    K = old_X @ W_K
    V = old_X @ W_V
    Q = Q.reshape(Q.shape[0], H, -1)
    K = K.reshape(K.shape[0], H, -1)
    V = V.reshape(V.shape[0], H, -1)

    gt_qkt, soft_out, gt_attn = multi_head_gka_attention(Q, K, V, H=H, max_diff=max_diff)


    rearranged_w_q = rearrange_q_k_v(W_Q, H) # d x d == d x (H * d_head)
    rearranged_w_k = rearrange_q_k_v(W_K, H) # d x d == d x (H * d_head)
    rearranged_w_v = rearrange_q_k_v(W_V, H) # d x d == d x (H * d_head)
    encoded_W_q = encode_W(rearranged_w_q, N, d, alpha=1, is_up=True)
    encoded_W_k = encode_W(rearranged_w_k, N, d, alpha=1, is_up=True)
    encoded_W_v = encode_W(rearranged_w_v, N, d, alpha=1, is_up=True)

    computed_params = compute_params(N, d, alpha=1, is_up=True)

    kcache = GKA_KCache(N, d, H)
    vcache = GKA_VCache(N, d, H)
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
    soft = exp(attn_val, n, N, d, H, gt_max=max_diff)
    res = vcache.softmaxV(soft)
    print(res)
    print(gt_attn)