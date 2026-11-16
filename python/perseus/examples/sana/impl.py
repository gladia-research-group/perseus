"""SANA-Sprint's sampler encrypted on the impl driver: `SanaImpl` runs, per step of hf.STEPS, the patch embedding, the
28 blocks and the output head as stages (capture / plan stages on a session; the CPU twin for the capture), the linears
through the session's PCMM engine (StC-first, perseus.impl.pcmm). Stage state: the residual X[g][k] (G x 3
ciphertexts), the carry C[g] (G) and the prompt's caption rows T[k] (3 ciphertexts, one chunk: the text tokens on the
real lanes), from which each block takes its cross-attention keys and values (the public BOS row's are plaintext).
A step's head adds its F, times the coefficient F enters with, to the carry: the next step's model input (renoise:
p x_in + r z2, the client's, plus q F), then the last's latent (a x_in + b F, a = 1), which the client decrypts."""
from __future__ import annotations

import math
import os

import numpy as np

from perseus.impl.driver import ImplModel
from perseus.impl.prefill import name_inputs, split

from . import fold as FD
from . import hf
from . import kernels as KN
from .kernels import room_hint
from . import layout as L
from . import model as M
from .layout import G, LANES, N, NCH


class Pcmms:
    """model.Lin -> PcmmLinear (encoded once, on its layout's N / lin.nch token lanes), applied per chunk pair with the
    bias."""

    def __init__(self, t=LANES):
        self.t = t
        self._enc = {}
        self._meta = {}

    def weights(self, lin: M.Lin):
        """lin's weights as encoded: W in_scale, the rows a pre-divided producer hands it scaled back up
        (lin.row_div: [(row slice, the linear whose in_div that producer applies)], SANA_ATTN_MERGE's qx reading the
        attention output divided for the merged out)."""
        W = lin.W * lin.in_scale
        for rows, other in getattr(lin, "row_div", None) or ():
            W = W.copy()
            W[rows] *= self.in_div(other)
        return W

    def get(self, lin: M.Lin):
        from perseus.impl.pcmm import PcmmLinear, band_shift
        if lin.name not in self._enc:
            p = PcmmLinear.encode(self.weights(lin), N // lin.nch, 1.0, lin.b,
                                  shift=band_shift(lin.name), name=lin.name)
            # the StC-first input's factors are set for a stamp w_scale 2^shift of 2^20, and only a band shift above 1
            # is pre-divided down to it (pc-gpu matmul::pcmm::prediv_log2, planner.pcmm_prediv_caps): a larger stamp at
            # shift <= 1 (24-bit weights on a small |y|: ff_in) came out 2.7% wrong, so it takes shift 2
            # PF_PREDIV_SHIFT_FIX=1 (default off until run E's GPU run has started; levers.RUNTIME sets it for new captures)
            if (os.environ.get("PF_PREDIV_SHIFT_FIX", "0") not in ("", "0") and p.shift <= 1
                    and round(math.log2(p.w_scale * 2.0 ** p.shift)) > 20):
                p.shift = 2
            if lin.name.rsplit(".", 1)[-1] in KN.PREDIV_FOLD:      # its producer hands it x / 2^s (in_div)
                s = max(0, round(math.log2(p.w_scale * 2.0 ** p.shift)) - 20)
                p.w_scale /= 2.0 ** s
                p.in_div = 2.0 ** s
            self._enc[lin.name] = p
        return self._enc[lin.name]

    def in_div(self, lin: M.Lin):
        """What lin's producer divides its input by (SANA_PREDIV_FOLD; 1: nothing)."""
        if FD.FOLD:
            return self.meta(lin)[2]
        return getattr(self.get(lin), "in_div", 1.0)

    def meta(self, lin: M.Lin):
        """(weight scale, shift, in_div) that `get` gives the unsplit linear, without encoding it: the parts of a
        linear split by the hidden fold share them (one stamp, one band, one pre-division)."""
        from perseus.impl.pcmm import band_shift, weight_bits
        if lin.name not in self._meta:
            wbits = weight_bits()
            nb = (wbits + 7) // 8
            top = min(2 ** (wbits - 1) - 1, 127 * (256 ** nb - 1) // 255)
            ws = 2.0 ** math.floor(math.log2(top / np.abs(self.weights(lin)).max()))
            shift = band_shift(lin.name)
            if (os.environ.get("PF_PREDIV_SHIFT_FIX", "0") not in ("", "0") and shift <= 1
                    and round(math.log2(ws * 2.0 ** shift)) > 20):
                shift = 2
            in_div = 1.0
            if lin.name.rsplit(".", 1)[-1] in KN.PREDIV_FOLD:
                in_div = 2.0 ** max(0, round(math.log2(ws * 2.0 ** shift)) - 20)
            self._meta[lin.name] = (ws, shift, in_div)
        return self._meta[lin.name]

    def part(self, lin: M.Lin, name, W, b, m_orig):
        """A part of lin (weights W, bias b already in lin's input scale) under `name`, on lin's scale / shift /
        pre-division; its outputs keep the unsplit linear's indices for the raise tables (m_orig)."""
        from perseus.impl.pcmm import PcmmLinear
        if name not in self._enc:
            ws, shift, in_div = self.meta(lin)
            p = PcmmLinear.encode(W, N // lin.nch, 1.0, b, shift=shift, name=name, ws=ws)
            if in_div != 1.0:
                p.w_scale /= in_div
                p.in_div = in_div
            p.m_orig, p.n_out_orig = list(m_orig), lin.n_out
            self._enc[name] = p
        return self._enc[name]

    def _finish(self, rt, p, ys, bias_of, a, b, nch):
        """The outputs' biases (masked to the filled lanes) and realized rescales."""
        t = N // nch
        fac = (np.arange(N) % t < a).astype(float) + 1j * (np.arange(N) % t < b)
        return [rt.ops.realize(rt.add_mask(y, ("sana.bias", p.name, i, a, b), lambda i=i: p.bias[i] * fac)
                               if np.any(bias_of(i)) else y) for i, y in enumerate(ys)]

    def _folded(self, rt, X, lin: M.Lin, lanes, fold_out):
        """lin with the hidden fold: X an FH (pair g's block 2 = the folded ciphertext rotated down) or blocks; with
        `fold_out` the hidden outputs come back folded: blocks 0, 1 per pair (the main part, lin's name) and each
        group's folded block from one block-diagonal linear over the group's inputs."""
        from perseus.impl.pcmm import prep as pcmm_prep
        from perseus.impl.pcmm import run as pcmm_run
        nch = lin.nch
        t = N // nch
        a, b = lanes or (t, t)
        rows_in = X.pcmm_rows(rt, t) if isinstance(X, FD.FH) else X
        if not fold_out:
            p = self.get(lin)
            return [self._finish(rt, p, pcmm_run(rt, row, p), lambda m: lin.b[m * nch:(m + 1) * nch], a, b, nch)
                    for row in rows_in]
        W = self.weights(lin)
        fcols = FD.fold_out_cols(lin.n_out)
        mcols = [m for m in range(lin.n_out) if m not in fcols]
        Wm = np.concatenate([W[:, m * nch:(m + 1) * nch] for m in mcols], axis=1)
        bm = np.concatenate([lin.b[m * nch:(m + 1) * nch] for m in mcols])
        pm = self.part(lin, lin.name, Wm, bm, mcols)
        # each pair's blocks feed its main call and its group's folded call: rotated and hinted once for both
        rows_in = [pcmm_prep(rt, row, t) for row in rows_in]
        mains = [self._finish(rt, pm, pcmm_run(rt, row, pm, prepped=True), lambda i: bm[i * nch:(i + 1) * nch],
                              a, b, nch) for row in rows_in]
        pre, _, tail = lin.name.rpartition(".")
        fouts = []
        for f in range(FD.NF):
            gs = FD.group(f)
            Wf, bf = FD.fold_weight(W, lin.b, len(rows_in[0]), nch, len(gs), fcols)
            name = f"{pre}.f{len(gs)}.{tail}" if pre else f"{tail}.fold{len(gs)}"
            pf = self.part(lin, name, Wf, bf, fcols)
            ins = [x for g in gs for x in rows_in[g]]
            fouts.append(self._finish(rt, pf, pcmm_run(rt, ins, pf, prepped=True),
                                      lambda i, bf=bf: bf[i * nch:(i + 1) * nch], a, b, nch))
        per = len(mcols) // len(fcols)
        outs = [FD.FH([mains[g][per * k:per * (k + 1)] for g in range(G)], [fouts[f][k] for f in range(FD.NF)])
                for k in range(len(fcols))]
        return outs[0] if len(outs) == 1 else outs

    def __call__(self, rt, X, lin: M.Lin, lanes=None, fold_out=False):
        """lin on X[g] (blocks) for every g; `lanes` (a, b): the filled lanes of each half (default all). With the
        hidden fold (fold.FOLD): X may be an FH and `fold_out` returns the hidden output(s) folded."""
        from perseus.impl.pcmm import run as pcmm_run
        if FD.FOLD and (fold_out or isinstance(X, FD.FH)):
            return self._folded(rt, X, lin, lanes, fold_out)
        p = self.get(lin)
        t, nch = N // lin.nch, lin.nch
        a, b = lanes or (t, t)
        fac = (np.arange(N) % t < a).astype(float) + 1j * (np.arange(N) % t < b)
        out = []
        for row in X:
            ys = pcmm_run(rt, row, p)
            # the finish's pending rescale realized in the graph: the plan and the compile then agree on the degree
            out.append([rt.ops.realize(rt.add_mask(y, ("sana.bias", lin.name, m, a, b), lambda m=m: p.bias[m] * fac)
                                       if np.any(lin.b[m * nch:(m + 1) * nch]) else y) for m, y in enumerate(ys)])
        return out

    def linears(self):
        return dict(self._enc)


def text_lins(W, b):
    """Block b's cross-attention k / v linears on the caption rows (3 blocks in, x-head layout out): k carries
    norm_k.weight on its columns (the RMS statistic reads 1 / w)."""
    t = lambda name: W[f"transformer_blocks.{b}.{name}"].numpy()            # noqa: E731
    xp = KN.xhead_perm()
    wk = t("attn2.norm_k.weight")
    H3 = 3 * NCH
    k = M.Lin(f"b{b}.tk", M._pad(t("attn2.to_k.weight").T * wk, H3, 4 * NCH, None, xp),
              M._padv(t("attn2.to_k.bias") * wk, 4 * NCH, xp))
    v = M.Lin(f"b{b}.tv", M._pad(t("attn2.to_v.weight").T, H3, 4 * NCH, None, xp),
              M._padv(t("attn2.to_v.bias"), 4 * NCH, xp))
    return k, v, wk


def text_kv(rt, T, W, b, nl, lin, n_tok, bos_kb, bos_vb):
    """(Kb, Vb) for block b: the BOS row's (public) then one lane-broadcast row per prompt token (lane j of T)."""
    ops = rt.ops
    lk, lv, wk = text_lins(W, b)
    K = lin(rt, [T], lk, (n_tok, 0))[0]
    V = lin(rt, [T], lv, (n_tok, 0))[0]
    site = f"b{b}.xa.knorm"
    c = nl.stat_scale(site)
    inv_w = np.zeros(4 * NCH)
    inv_w[KN.xhead_perm()] = 1.0 / wk
    z = KN.token_stat(rt, [split(ops, x)[0] for x in K], c * inv_w, site)
    z = rt.add_mask(z, (site, "eps", c), lambda: np.full(N, c * c * 1e-5))
    rk = KN.packed_lane_map(rt, [z], lambda v: nl.inv_sqrt(rt, v, site), site)[0]
    msc = KN.xa_scale(nl, f"b{b}.xa")                # the scores' 1 / M rides the keys
    if KN.XA_KEY_FOLD:                               # ... or the scores' window mask (kernels.cross_attention): the
        bos_kb = [KN.Pt(x.v * np.sqrt(KN.DX), x.tag + ("unit",)) for x in bos_kb]    # rows stay at their RMS scale
    else:
        rk = rt.mult_mask(rk, (site, "dx") if msc == 1.0 else (site, "dx", msc),
                          lambda: np.full(N, 1.0 / (np.sqrt(KN.DX) * msc)))
        if msc != 1.0:
            bos_kb = [KN.Pt(x.v / msc, x.tag + ("M", msc)) for x in bos_kb]
    Kn = [ops.mult(x, rk) for x in K]
    Kb, Vb = [bos_kb], [bos_vb]
    if KN.KV_HINT_EARLY:           # one refresh site per key / value ciphertext, before the rows are broadcast
        Kn = [room_hint(rt, x, 4) for x in Kn]
        V = [room_hint(rt, x, 3) for x in V]
    for j in range(n_tok):
        sel = (np.arange(N) % LANES == j).astype(float)

        def bcast(x, tag):
            y = rt.mult_mask(x, ("sana.tok", j), lambda: sel)
            y = ops.rotate(y, j) if j else y
            y = ops.rotate_and_sum(y, -1, LANES)
            return y
        # each row's refresh site: the score / value products' room
        if KN.KV_HINT_EARLY:
            Kb.append([bcast(x, "k") for x in Kn])
            Vb.append([bcast(x, "v") for x in V])
        else:
            Kb.append([room_hint(rt, bcast(x, "k"), 3) for x in Kn])
            Vb.append([room_hint(rt, bcast(x, "v"), 2) for x in V])
    return Kb, Vb


def bos_rows(W, b, txt_bos):
    """The BOS key / value rows of block b (plaintext): KN.Pt per block."""
    Kb, Vb = M.text_keys(W, b, txt_bos[None], np.array([False]))
    return Kb[0], Vb[0]


class SanaImpl(ImplModel):
    def __init__(self, inf, W, nl, *, core=None, n_layers=None, profile=False):
        super().__init__(inf, core=core, profile=profile, dims=L.geometry()[0])
        assert len(hf.STEPS) <= 2, "the carry: renoise's p x_in is the client's on the first step only"
        self.W, self.nl = W, nl
        self.Ws = [W.step(i) for i in range(len(hf.STEPS))]
        self.n_layers = W.n_layers if n_layers is None else n_layers
        self.lin = Pcmms()
        self._bw = {}

    @property
    def r(self):
        return getattr(self.nl, "resid_scale", np.ones(L.D))

    def bw(self, i, b):
        if (i, b) not in self._bw:
            self._bw[i, b] = M.fold_block(self.Ws[i], b, self.r, self.nl, hf.step_tag(i))
        return self._bw[i, b]

    def embed_lin(self):
        Wp = self.W["patch_embed.proj.weight"][:, :, 0, 0].numpy().T / self.r  # (32, D), the stream's 1 / r
        return M.Lin("embed", M._pad(Wp, NCH, 3 * NCH),
                     M._padv(self.W["patch_embed.proj.bias"].numpy() / self.r, 3 * NCH))

    def head_lin(self, i):
        """Step i's head: F times the coefficient it enters the next model input with (renoise's q), or the latent
        (the last step's b; its a is 1: the carry adds x_in as is)."""
        t = hf.STEPS[i]
        k = hf.scm_constants(t)
        assert abs(k["a"] - 1.0) < 1e-12
        c = k["b"] if i + 1 == len(hf.STEPS) else hf.renoise(hf.STEPS[i + 1], t)["q"]
        return M.out_lin(self.Ws[i], c, hf.step_tag(i) + "out")

    def set_prompt(self, x_in, txt_rows, txt_bos, z2=None):
        """The prompt the next forward / recording runs on: the model input x_in (32, IMG, IMG), the caption rows of the
        prompt's tokens (n_tok, D) and of BOS (D,) (public: the transformer's constant first key), the second step's
        noise z2 (32, IMG, IMG; the client's, as x_in)."""
        self.x_in, self.txt_rows, self.txt_bos = np.asarray(x_in), np.asarray(txt_rows), np.asarray(txt_bos)
        self.z2 = None if z2 is None else np.asarray(z2)

    @staticmethod
    def carry(x_in, z2):
        """The client's carry: what the first head's output adds to its F (the next model input's p x_in + r z2; with
        one step the latent's x_in)."""
        if len(hf.STEPS) == 1:
            return x_in
        rn = hf.renoise(hf.STEPS[1], hf.STEPS[0])
        return rn["p"] * x_in + rn["r"] * z2

    def stages(self):
        """[(label, fn(state) -> state)], state = the model input's G ciphertexts (embed) or the residual's (row-major
        [g][k]), then C's G, then T's 3; a head's out: the next model input's G then T's 3 (the next embed passes the
        input on as C: a pass-through copy), the last's the latent's G. Fewer layers: the first step's embed and blocks
        alone."""
        rt, nl = self.rt, self.nl
        ops = rt.ops
        _d, B, ch = L.geometry()
        n_tok, txt_bos = len(self.txt_rows), self.txt_bos

        def unflat(xs, nb):
            if FD.FOLD and nb == 3:                       # the hidden width: folded
                return FD.FH.from_flat(xs)
            return [xs[g * nb:(g + 1) * nb] for g in range(G)]

        def parts(state):
            return state[:-3 - G], state[-3 - G:-3], state[-3:]

        def out(X, C, T):
            # a stage produces its outputs (C's and T's copies), rescales realized (one degree at every stage boundary)
            xs = X.flat() if isinstance(X, FD.FH) else [x for row in X for x in row]
            return [ops.realize(x) for x in xs] + [ops.copy(c) for c in C] + [ops.copy(t) for t in T]

        def embed(i):
            def fn(state):
                if i == 0:
                    xs, C, T = parts(state)
                else:                                         # the model input is the carry too (a = 1): passed on
                    xs, T = state[:-3], state[-3:]
                    C = xs
                return out(self.lin(rt, unflat(xs, 1), self.embed_lin(), fold_out=FD.FOLD), C, T)
            return fn

        def blk(i, b):
            def fn(state):
                xs, C, T = parts(state)
                bk, bv = bos_rows(self.W, b, txt_bos)
                Kb, Vb = text_kv(rt, T, self.W, b, nl, self.lin, n_tok, bk, bv)
                blockf = FD.block if FD.FOLD else M.block
                return out(blockf(rt, unflat(xs, 3), self.bw(i, b), Kb, Vb, nl, self.lin, b), C, T)
            return fn

        def head(i):
            def fn(state):
                from perseus.impl.blocked import norm_blocked
                xs, C, T = parts(state)
                if FD.FOLD:
                    N2 = FD.norm(rt, FD.unscale(rt, unflat(xs, 3), self.r, "out.ln"), nl.ln("out.ln"), ch, B)
                else:
                    N2 = norm_blocked(rt, M._unscale(rt, unflat(xs, 3), self.r, "out.ln"), nl.ln("out.ln"), ch, B)
                Y = [ops.realize(ops.add(row[0], c)) for row, c in zip(self.lin(rt, N2, self.head_lin(i)), C)]
                if i + 1 == len(hf.STEPS):
                    return Y                                  # the latent
                return Y + [ops.copy(t) for t in T]           # the next model input (its embed passes it on as C)
            return fn
        full = self.n_layers == self.W.n_layers
        st = []
        for i in range(len(hf.STEPS) if full else 1):
            s = hf.step_tag(i)
            st += [(s + "embed", embed(i))] + [(f"{s}blk{b}", blk(i, b)) for b in range(self.n_layers)] + \
                ([(s + "head", head(i))] if full else [])
        return st

    def prefill_stages(self, ch=None):
        def stage(i, lab, fn):
            def compute(state):
                cap = self.enter_stage(i, 0)
                name_inputs(self.inf, state)
                # the pf_in names name_inputs just gave inputs 1.. (an in-place refresh hint inside the stage renames an
                # input ciphertext: the recorder reads these, not the names after the stage ran). Input 0 keeps None:
                # it carries a stale name here and gets its graph name (anon_<n>) at its first read in the stage
                self.__dict__.setdefault("_entry_names", {})[i] = (
                    [None] + [getattr(c, "name", None) for c in state[1:]] if hasattr(self.inf, "name_ct") else None)
                with self.rt.step(lab):
                    state = fn(state)
                self.exit_stage(i, cap)
                return state
            return compute
        return [(lab, None, stage(i, lab, fn)) for i, (lab, fn) in enumerate(self.stages())]

    def encrypt(self, vecs):
        ops = self.rt.ops
        zero = ops.core.encode_token_input(ops.inf, np.zeros(self.rt.dims.dim))
        return [ops.add_pt(zero, v) for v in vecs]

    def input_values(self, x_in=None, txt_rows=None, z2=None):
        """The client's slot vectors: the model input's rows (32 channels, G chunk pairs), the carry's (`carry`), then
        the prompt's caption rows (n_tok tokens on the real lanes of one chunk, 3 blocks)."""
        x_in = self.x_in if x_in is None else x_in
        z2 = self.z2 if z2 is None else z2
        txt_rows = self.txt_rows if txt_rows is None else txt_rows
        T = np.zeros((LANES, 3 * NCH))
        T[:len(txt_rows), :L.D] = txt_rows
        tv = [np.ascontiguousarray(T[:, k * NCH:(k + 1) * NCH].T.reshape(N)).astype(complex) for k in range(3)]
        return [v for img in (x_in, self.carry(x_in, z2)) for row in L.encode(L.to_slots(img), 1) for v in row] + tv

    def prefill(self, rows=None, on_block=None):
        """The forward on the set prompt: (output ciphertexts, None, token record); `rows` unused (PlannedPrefill's
        signature)."""
        self.start()
        self.token_begin(0)
        stages = self.prefill_stages(None)
        if on_block is not None:
            stages = [(lab, pre, (lambda s, fn=fn, i=i: on_block(i, fn(s)))) for i, (lab, pre, fn) in enumerate(stages)]
        state = self.run_stages(self.encrypt(self.input_values()), 0, stages)
        return state, None, self.token_end(0)

    def decrypt_latent(self, xs):
        """The latent (32, IMG, IMG) from the last head's G ciphertexts (decrypted here, or slot vectors)."""
        ops = self.rt.ops
        rows = L.decode([[x if isinstance(x, np.ndarray) else ops.decrypt_slots_complex(x)] for x in xs], 32)
        return L.from_slots(rows)
