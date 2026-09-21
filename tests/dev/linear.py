import numpy as np

def rot(v, k):
    """Cyclic rotation: rot(v, k)[i] = v[(i+k) % len(v)]."""
    return np.roll(v, -k)


def compute_params(N, d, alpha, is_up):
    """Compute all algorithm parameters from N, d, alpha, direction."""
    t  = N // d                        
    tp = N // (alpha * d)              

    tp_in  = t  if is_up else tp                # (t1 and t2 in Cachemir)
    tp_out = tp if is_up else t                 # (t1 and t2 in Cachemir)

    d_in  = d if is_up else alpha * d
    n_pt  = d_in // tp_out                      # from Chachemir's go code this coincides
    r_i   = max(1, d * d // N)          
    r_i   = min(r_i, n_pt)
    r_o   = n_pt // r_i

    return t, tp, tp_in, tp_out, r_i, r_o, n_pt


def interleave_idx(m, d, dim):
    a = dim // d if dim > d else 1
    return (m // a + (m % a) * d) % dim


def encode_x(x, N, d, alpha, is_up):
    """
    Encode activation vector into N-slot ciphertext.

    Up  (x ∈ R^d):   c^x[i] = x[i/t] · 1{t|i}
    Down (x ∈ R^alpha* d):  c^x[i] = x[interleave(i/t')] · 1{t'|i}
    """
    t, tp, _, _, _, _, _ = compute_params(N, d, alpha, is_up)
    d_in = d if is_up else alpha * d
    M = N // tp  # = αd

    ptx = np.zeros(N)
    if is_up:
        for i in range(d):
            ptx[i * t] = x[i]
    else:
        for m in range(M):
            ptx[m * tp] = x[interleave_idx(m, d, d_in)]
    return ptx


def encode_W(W, N, d, alpha, is_up):
    """
    Encode weight matrix into n_pt = r_i·r_o plaintext vectors.

    Row formula (index into d_in):
        ((i//t + j·t + i%tp_in) % d) + ((i%t)//tp_in) · d

    Col formula (index into d_out, with interleaved output mapping):
        interleave_idx((i//tp_out - k·cascade_shift) % M_out, d_out)
    """
    d_in, d_out = W.shape
    t, tp, tp_in, tp_out, r_i, r_o, n_pt = compute_params(N, d, alpha, is_up)

    M_out = N // tp_out
    cascade_shift = (t * tp) // tp_out

    pt = np.zeros((n_pt, N))
    for j in range(r_i):
        for k in range(r_o):
            for i in range(N):
                row = ((i // t + j * t + i % tp_in) % d) \
                    + ((i % t) // tp_in) * d

                m_shifted = (i // tp_out - k * cascade_shift) % M_out
                col = interleave_idx(m_shifted, d, d_out)

                pt[j * r_o + k, i] = W[row, col]
    return pt

def decode_output(cy, N, t, tp, alpha, d, d_out, is_up):
    M = N // tp
    if is_up and alpha > 1:
        y = np.zeros(d_out)
        for m in range(M):
            idx = interleave_idx(m, d, d_out)
            if idx < d_out:
                y[idx] = cy[m * tp]
    else:
        y = cy[::t][:d_out]
    
    return y 


def cachemir_vmm(x, W, N, d, d_out, alpha, is_up, computed_params):
    t, tp, tp_in, tp_out, r_i, r_o, n_pt = computed_params

    ptx_prime = x.copy()
    step = 1
    while step < tp_in:
        ptx_prime = ptx_prime + rot(ptx_prime, step * (t - 1))
        step *= 2

    rot2 = t * t
    cy_prime = np.zeros((r_o, N))

    ptx_rotated = [rot(ptx_prime, j * rot2) for j in range(r_i)]
                 
    for k in range(r_o):
        for j in range(r_i):
            cy_prime[k] += ptx_rotated[j] * W[j * r_o + k]

    cascade_rot = t * tp
    for k in range(r_o - 1, 0, -1):
        cy_prime[k - 1] += rot(cy_prime[k], cascade_rot)

    cy = cy_prime[0]

    step = 1
    while step < tp_out:
        cy = cy + rot(cy, step)
        step *= 2
    
    return cy

def _run_vmm(label, N, d, alpha, is_up):
    """Encode x and W, run the homomorphic VMM, decode, and compare to x @ W.

    Shapes (d is the base dim, alpha the expansion factor):
        up   : x ∈ R^d,        W ∈ R^{d × alpha·d}   -> y ∈ R^{alpha·d}
        down : x ∈ R^{alpha·d}, W ∈ R^{alpha·d × d}   -> y ∈ R^d
        (alpha=1, is_up=True is the square case)
    """
    d_in, d_out = (d, alpha * d) if is_up else (alpha * d, d)

    x  = np.random.randn(d_in)
    W  = np.random.randn(d_in, d_out)
    gt = x @ W

    params  = compute_params(N, d, alpha, is_up)
    t, tp   = params[0], params[1]

    enc_x = encode_x(x, N, d, alpha, is_up)
    enc_W = encode_W(W, N, d, alpha, is_up)
    cy    = cachemir_vmm(enc_x, enc_W, N, d, d_out, alpha, is_up, params)
    y     = decode_output(cy, N, t, tp, alpha, d, d_out, is_up)

    err = np.max(np.abs(gt - y))
    tag = "OK" if err < 1e-10 else "FAIL"
    print(f"  {label:6s} N={N:6d} d={d:5d} alpha={alpha} is_up={int(is_up)} "
          f"d_in={d_in:5d} d_out={d_out:5d}: {err:.2e} {tag}")
    return err < 1e-10


def main():
    np.random.seed(42)
    ok = True

    print("=== Cachemir VMM (encode x, encode W, compare to x @ W) ===")
    ok &= _run_vmm("square", 65536, 1024, 1, True)   # 1024 ->  1024
    ok &= _run_vmm("up",     65536, 1024, 4, True)   # 1024 ->  4096
    ok &= _run_vmm("down",   65536, 1024, 4, False)  # 4096 ->  1024

    print(f"\n{'ALL PASSED' if ok else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()