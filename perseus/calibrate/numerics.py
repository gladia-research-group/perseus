import logging
import math

import numpy as np
import torch
from scipy.optimize import root

log = logging.getLogger(__name__)


def polyval_torch(x, coeffs):
    """Horner evaluation of an ascending-order coefficient list [c0, c1, ...]."""
    val = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        val = val * x + coeffs[i]
    return val


def get_chebyshev_nodes(a, b, n):
    k = np.arange(1, n + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb


def cheb_exp_series(lo, hi, degree, min_nodes=200):
    """Chebyshev coefficients of exp on [lo, hi], normalized so the series
    value at 0 is 1 (the FHE post-exp squaring base stays ~1)."""
    n = max(8 * (degree + 1), min_nodes)
    x = get_chebyshev_nodes(lo, hi, n)
    series = np.polynomial.chebyshev.Chebyshev.fit(x, np.exp(x), degree, domain=[lo, hi])
    return (series.coef / float(series(0.0))).astype(np.float64)


def rational_remez(target_func, a, b, n, m, max_iter=50, tol=1e-12, grid_size=5000):
    num_points = n + m + 2

    x_nodes = np.sort(get_chebyshev_nodes(a, b, num_points))

    cheb_dense = np.cos((2*np.arange(1, grid_size+1) - 1) / (2*grid_size) * np.pi)
    x_dense = np.sort(0.5 * (b - a) * cheb_dense + 0.5 * (b + a))

    p_coeffs = np.zeros(n + 1)
    q_coeffs = np.zeros(m + 1)
    q_coeffs[0] = 1.0

    for iteration in range(max_iter):
        y_nodes = target_func(x_nodes)

        def equations(vars):
            p = vars[: n + 1]
            q_high = vars[n + 1 : n + m + 1]
            E = vars[-1]

            q = np.concatenate(([1.0], q_high))

            P_val = np.polynomial.polynomial.polyval(x_nodes, p)
            Q_val = np.polynomial.polynomial.polyval(x_nodes, q)

            signs = (-1.0) ** np.arange(num_points)

            return P_val - y_nodes * Q_val * (1.0 + signs * E)

        guess = np.zeros(num_points)
        if iteration == 0:
            guess[0] = np.mean(y_nodes)
        else:
            guess[: n + 1] = p_coeffs
            guess[n + 1 : n + m + 1] = q_coeffs[1:]

        sol = root(equations, guess, method="lm")

        if not sol.success and iteration > 0:
            log.info("  Convergence stalled in solver. Returning previous best.")
            break

        p_coeffs = sol.x[: n + 1]
        q_coeffs = np.concatenate(([1.0], sol.x[n + 1 : n + m + 1]))
        E_curr = sol.x[-1]

        y_dense = target_func(x_dense)
        num = np.polynomial.polynomial.polyval(x_dense, p_coeffs)
        den = np.polynomial.polynomial.polyval(x_dense, q_coeffs)

        if np.min(np.abs(den)) < 1e-9:
            log.warning("  Warning: Denominator near zero. Pole detected.")

        approx = num / den
        rel_error = (approx - y_dense) / y_dense
        abs_rel_error = np.abs(rel_error)

        max_err_idx = np.argmax(abs_rel_error)
        max_err_val = abs_rel_error[max_err_idx]
        x_extremum = x_dense[max_err_idx]
        ext_sign = np.sign(rel_error[max_err_idx])

        diff = abs(max_err_val - abs(E_curr))

        if diff < tol:
            break

        n_num = np.polynomial.polynomial.polyval(x_nodes, p_coeffs)
        n_den = np.polynomial.polynomial.polyval(x_nodes, q_coeffs)
        node_errs = (n_num / n_den - y_nodes) / y_nodes
        node_signs = np.sign(node_errs)

        idx = np.searchsorted(x_nodes, x_extremum)

        if idx == 0:
            if np.sign(ext_sign) == np.sign(node_signs[0]):
                x_nodes[0] = x_extremum
            else:
                x_nodes = np.insert(x_nodes, 0, x_extremum)[:-1]
        elif idx == num_points:
            if np.sign(ext_sign) == np.sign(node_signs[-1]):
                x_nodes[-1] = x_extremum
            else:
                x_nodes = np.append(x_nodes, x_extremum)[1:]
        else:
            s_left = node_signs[idx - 1]
            s_right = node_signs[idx]

            if np.sign(ext_sign) == np.sign(s_left):
                x_nodes[idx - 1] = x_extremum
            elif np.sign(ext_sign) == np.sign(s_right):
                x_nodes[idx] = x_extremum

        x_nodes = np.sort(x_nodes)

    x_dense = np.linspace(a, b, 1000)

    final_den_vals = np.polynomial.polynomial.polyval(x_dense, q_coeffs)
    q_min = np.min(final_den_vals)
    q_max = np.max(final_den_vals)

    return p_coeffs, q_coeffs, q_min, q_max



_QUANTILE_MAX = 1 << 23


def _quantile_input(t: torch.Tensor) -> torch.Tensor:
    flat = t.reshape(-1)
    if flat.numel() > _QUANTILE_MAX:
        gen = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randint(0, flat.numel(), (_QUANTILE_MAX,), generator=gen)
        flat = flat[idx.to(flat.device)]
    return flat


def safe_quantile(t: torch.Tensor, q: float) -> float:
    flat = _quantile_input(t)
    return float(torch.quantile(flat, q))


def linear_gs_init(d_min, d_max):
    alpha = 4.0 / (d_min + d_max)
    return alpha, alpha * alpha / 4.0


def chebyshev_gs_init(d_min, d_max):
    s = d_min + d_max
    beta = 8.0 / (s * s + 4.0 * d_min * d_max)
    return beta * s, beta


GS_INITS = {"linear": linear_gs_init, "chebyshev": chebyshev_gs_init}


@torch.no_grad()
def gs_converged_iters(S, alpha, beta, target, max_iters):
    """Iterations of the Goldschmidt reciprocal y <- y·(2 - y·S), seeded
    y0 = alpha - beta·S, until max |1 - y·S| < target over the samples.
    0 = the init alone suffices; None = no convergence within max_iters."""
    y = alpha - beta * S
    for it in range(max_iters + 1):
        if (1.0 - y * S).abs().max().item() < target:
            return it
        y = y * (2.0 - y * S)
    return None


@torch.no_grad()
def goldschmidt_reciprocal(D, alpha, beta, iters):
    """1/D by the RUNTIME Goldschmidt recurrence, where `iters` counts the init
    product as iteration 1 (the convention `gs_iters` is stored in)."""
    F = alpha - beta * D
    N_cur = F
    D_cur = -D * F
    F = 2.0 + D_cur
    for _ in range(iters - 1):
        N_cur = N_cur * F
        D_cur = D_cur * F
        F = 2.0 + D_cur
    return N_cur


def _perturb(x, eps, mode, gen):
    if eps <= 0.0:
        return x
    if mode == "+":
        return x + eps
    if mode == "-":
        return x - eps
    u = torch.rand(x.shape, generator=gen, dtype=x.dtype, device=x.device)
    return x + eps * (2.0 * u - 1.0)


@torch.no_grad()
def goldschmidt_reciprocal_noisy(D, alpha, beta, iters, eps, mode, gen):
    F = alpha - beta * D
    N_cur = _perturb(F, eps, mode, gen)
    D_cur = _perturb(-D * F, eps, mode, gen)
    F = 2.0 + D_cur
    for _ in range(iters - 1):
        N_cur = _perturb(N_cur * F, eps, mode, gen)
        D_cur = _perturb(D_cur * F, eps, mode, gen)
        F = 2.0 + D_cur
    return N_cur


@torch.no_grad()
def gs_error_under_noise(S, alpha, beta, iters, eps, trials, seed, q=0.999):
    gen = torch.Generator(device=S.device).manual_seed(int(seed))
    worst = 0.0
    modes = ["+", "-"] + ["rand"] * max(0, int(trials))
    for mode in modes:
        y = goldschmidt_reciprocal_noisy(S, alpha, beta, iters, eps, mode, gen)
        e = ((y * S) - 1.0).abs()
        e = e[torch.isfinite(e)]
        if e.numel() == 0:
            return float("inf")
        err = float(torch.quantile(e, q)) if e.numel() > 1 else float(e[0])
        if err > worst:
            worst = err
        if not math.isfinite(worst):
            return float("inf")
    return worst


@torch.no_grad()
def fit_gs_under_noise(S, d_min, d_max, max_iters, method, eps,
                       trials=6, seed=20260822, span=0.25, grid=7, slack=1.0,
                       max_samples=4096, iters=None):
    alpha0, beta0 = GS_INITS[method](d_min, d_max)
    in_band = S[(S >= d_min) & (S <= d_max)]
    S_test = in_band if in_band.numel() > 0 else S
    if S_test.numel() > max_samples:                      # deterministic thinning
        step = int(S_test.numel() // max_samples) + 1
        S_test = S_test[::step]
    S_test = S_test.reshape(-1)

    factors = [2.0 ** (span * (2.0 * i / (grid - 1) - 1.0)) for i in range(grid)] \
        if grid > 1 else [1.0]
    counts = [int(iters)] if iters is not None else list(range(1, int(max_iters) + 1))
    scored = []
    for fa in factors:
        for fb in factors:
            a, b = alpha0 * fa, beta0 * fb
            for it in counts:
                scored.append((a, b, it,
                               gs_error_under_noise(S_test, a, b, it, eps, trials, seed)))
    err_best = min(c[3] for c in scored)
    ok = [c for c in scored if c[3] <= slack * err_best]
    return min(ok, key=lambda c: (c[2], c[3]))


@torch.no_grad()
def fit_nr_iters_under_noise(z, seed_y, max_iters, eps, trials, seed, slack=1.05):
    truth = z ** -0.5
    gen = torch.Generator(device=z.device).manual_seed(int(seed))
    modes = ["+", "-"] + ["rand"] * max(0, int(trials))
    errs = []
    for it in range(int(max_iters) + 1):
        worst = 0.0
        for mode in modes:
            y = seed_y.clone()
            for _ in range(it):
                y = _perturb(0.5 * y * (3.0 - z * y * y), eps, mode, gen)
            e = ((y - truth).abs() / truth).max().item()
            worst = max(worst, e if math.isfinite(e) else float("inf"))
        errs.append(worst)
    best_it = min(range(len(errs)), key=lambda i: errs[i])
    for it in range(best_it):                             # cheapest within slack
        if errs[it] <= slack * errs[best_it]:
            return it, errs[it]
    return best_it, errs[best_it]


@torch.no_grad()
def nr_converged_iters(z, y0, target, max_iters):
    truth = z ** -0.5
    y = y0.clone()
    for it in range(max_iters + 1):
        if ((y - truth).abs() / truth).max().item() < target:
            return it
        y = 0.5 * y * (3.0 - z * y * y)
    return None


@torch.no_grad()
def estimate_gs_iters(S, d_min, d_max, target, max_iters, method):
    alpha, beta = GS_INITS[method](d_min, d_max)
    in_band = S[(S >= d_min) & (S <= d_max)]
    S_test = in_band if in_band.numel() > 0 else S
    it = gs_converged_iters(S_test, alpha, beta, target, max_iters)
    if it is None:
        log.warning(f"[gs] WARN: did not converge to {target:g} in {max_iters} iters "
              f"(range=[{float(d_min):.3g}, {float(d_max):.3g}], "
              f"|S|={int(S.numel())}, in_band={int(in_band.numel())}); "
              f"saturating at {max_iters}")
        return max_iters
    return it
