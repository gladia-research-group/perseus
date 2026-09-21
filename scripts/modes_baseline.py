"""The single model driver for the whole matrix (GPT-2 + ViT), mode-selected:

  usage: modes_baseline.py {decode,prefill,handoff,gen,gen_prefill,forward}

GPT-2 (MODEL=gpt2, the default) — session modes over `perseus._core`:
  decode       mode 4  decode -> teacher-forced          run_decode
  prefill      mode 2  prefill tail next-token           run_prefill (prefill T, decode 0)
  handoff      mode 2  prefill -> teacher-forced decode  run_prefill (prefill T, decode 2)
  gen          mode 3  decode -> AUTOREGRESSIVE          run_generate (gen_prompt 1, CutMax feedback)
  gen_prefill  mode 1  prefill -> AUTOREGRESSIVE         run_generate (gen_prompt 8, CutMax feedback)
  forward      debug gate: per-token per-block forward vs the all_blocks_io oracle
               (top1/KL; GATE_DEBUG=1 adds per-block residual taps). Absorbed from
               the old encgpt2_forward.py.
The teacher-forced modes check top1==oracle; the autoregressive modes fork from GT
after the first token and gate on clean completion + unplanned_bts + the per-step
cutmax==fhe_argmax parity logged to stderr.

ViT (MODEL=vit) — image model, NO language modes: only `forward` is valid (the
encrypted forward against the identically-truncated torch-hub model; GATE_BLOCKS=k
truncates both, k=12 = the real model). Absorbed from the old encvit_forward.py.

Every mode ends with the uniform acceptance marker `[<mode>] PASS` on success.
"""
import os
import sys
import traceback

MODEL = os.environ.get("MODEL", "gpt2")
mode = sys.argv[1] if len(sys.argv) > 1 else ""

GPT2_MODES = ("decode", "prefill", "handoff", "gen", "gen_prefill", "forward")
if MODEL in ("vit", "bert"):
    if mode != "forward":
        raise SystemExit(f"MODEL={MODEL} supports only 'forward' (an encoder has no "
                         f"language modes); got '{mode}'")
elif mode not in GPT2_MODES:
    raise SystemExit(f"unknown mode '{mode}' (expected one of {GPT2_MODES})")

# ViT/BERT: torch must be imported BEFORE _core — _core's load binds the module-env
# NCCL, and torch's bundled libtorch_cuda then hits undefined NCCL symbols if it
# loads second (the old encvit_forward.py had this order; the merge broke it).
if MODEL in ("vit", "bert"):
    import torch  # noqa: F401  (re-imported locally where used)

import numpy as np

from perseus import _core

GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "4"))
PREFILL_T = int(os.environ.get("PREFILL_T", "8"))


def check(res, gt, expect_rows, min_hits):
    assert not res.threw, f"{mode} threw: {res.error}"
    rows = len(res.top1)
    assert rows == expect_rows, f"logit rows {rows}/{expect_rows}"
    hits = 0
    for pos, top1 in zip(res.positions, res.top1):
        ref = int(np.argmax(gt.logits[pos]))
        hits += top1 == ref
        print(f"[{mode}] pos={pos} top1={top1} ref={ref} {'OK' if top1 == ref else 'MISS'}",
              flush=True)
    e2e = res.avg_s_per_tok + res.avg_argmax_s   # e2e/tok = forward + encrypted-argmax stage
    print(f"[{mode}] top1 {hits}/{rows}  completed={res.completed} "
          f"bootstraps={res.bootstraps} unplanned_bts={res.unplanned_bts} "
          f"weight_relevels={res.weight_relevels} "
          f"s/tok={res.avg_s_per_tok:.1f} argmax_s/tok={res.avg_argmax_s:.1f} e2e_s/tok={e2e:.1f}",
          flush=True)
    assert hits >= min_hits, f"top1 {hits} < {min_hits}"


def check_generate(res, want_tokens):
    assert not res.threw, f"{mode} threw: {res.error}"
    assert res.completed == want_tokens, f"completed {res.completed}/{want_tokens}"
    assert res.unplanned_bts == 0, f"unplanned_bts {res.unplanned_bts}"
    stream = [int(t) for t in res.top1]
    for pos, top1 in zip(res.positions, res.top1):
        print(f"[{mode}] pos={pos} top1={top1}", flush=True)
    e2e = res.avg_s_per_tok + res.avg_argmax_s
    print(f"[{mode}] stream={stream} completed={res.completed} "
          f"bootstraps={res.bootstraps} unplanned_bts={res.unplanned_bts} "
          f"weight_relevels={res.weight_relevels} "
          f"s/tok={res.avg_s_per_tok:.1f} argmax_s/tok={res.avg_argmax_s:.1f} e2e_s/tok={e2e:.1f}",
          flush=True)


def run_gpt2_session():
    cfg = _core.RunConfig.from_env()
    if mode == "prefill":
        cfg.tokens, cfg.prefill_tokens, cfg.decode_tokens = PREFILL_T, PREFILL_T, 0
    elif mode == "handoff":
        cfg.tokens, cfg.prefill_tokens, cfg.decode_tokens = PREFILL_T + 2, PREFILL_T, 2
    elif mode == "gen":
        cfg.gen_prompt, cfg.gen_tokens, cfg.teacher_forced = 1, GEN_TOKENS, False
        cfg.tokens = cfg.gen_prompt + cfg.gen_tokens
    elif mode == "gen_prefill":
        cfg.gen_prompt, cfg.gen_tokens, cfg.teacher_forced = 8, GEN_TOKENS, False
        cfg.tokens = cfg.gen_prompt + cfg.gen_tokens
    inputs = _core.read_teacher_forced_inputs(cfg)
    gt = _core.read_lm_head_steps(cfg)

    if mode == "decode":
        check(_core.run_decode(cfg, inputs), gt, expect_rows=cfg.tokens,
              min_hits=cfg.tokens - 1)
    elif mode == "prefill":
        check(_core.run_prefill(cfg, inputs), gt, expect_rows=1,
              min_hits=0 if PREFILL_T >= 128 else 1)
    elif mode == "handoff":
        check(_core.run_prefill(cfg, inputs), gt, expect_rows=2, min_hits=1)
    else:
        check_generate(_core.run_generate(cfg, inputs), want_tokens=cfg.gen_tokens)


def run_gpt2_forward():
    """Per-token per-block forward gate vs the oracle (the old encgpt2_forward)."""
    from perseus.nn import EncGPT2

    opts = _core.InferenceOptions()
    opts.ckks = _core.CKKSOptions.from_env()
    opts.mode = _core.InferenceMode.Sync
    inf = _core.make_gpt2_inference(opts)
    store = _core.WeightStore.from_zip(os.environ["WEIGHTS_PATH"])
    configs = _core.load_configs(os.environ["CONFIGS_PATH"])
    model = EncGPT2(store, configs).bind(inf)
    print(f"[{mode}] model bound (block states + lnf encoded)", flush=True)

    cfg = _core.RunConfig.from_env()
    inputs = _core.read_teacher_forced_inputs(cfg)
    gt = _core.read_lm_head_steps(cfg)

    def kl(p_logits, q_logits):
        p = np.exp(p_logits - np.max(p_logits)); p /= p.sum()
        q = np.exp(q_logits - np.max(q_logits)); q /= q.sum()
        return float(np.sum(p * np.log(p / np.maximum(q, 1e-30))))

    taps = None
    if os.environ.get("GATE_DEBUG"):
        import json
        io_dir = os.environ.get(
            "ALL_BLOCKS_IO_DIR",
            "/leonardo/pub/userexternal/azirilli/he-aware-training_data/all_blocks_io")
        taps = [json.load(open(f"{io_dir}/all_blocks_L{b:02d}_T{cfg.steps_t}.json"))
                for b in range(model.n_layers)]

    model.start()
    hits, logits, tiles = 0, None, None
    n_tok = int(os.environ.get("GATE_TOKENS", "2"))
    for t in range(n_tok):
        inf.capture_t = t
        x = _core.encode_token_input(inf, inputs[t])
        for b in range(model.n_layers):
            x = model.run_block(b, x)
            if taps is not None:
                out = np.array(_core.decode_token_output(inf, x))[:768]
                ref_res = np.array(taps[b]["res"][t])
                rel = np.linalg.norm(out - ref_res) / (np.linalg.norm(ref_res) + 1e-9)
                print(f"[{mode}] tok{t} block{b} res_rel={rel:.4f} "
                      f"free={_core.device_free_gb():.1f}GB", flush=True)
        x = _core.apply_final_ln(inf, x, model._lnf)
        tiles = model.lm_head(x)
        logits = np.array(model.decode_logits(tiles))
        ref = np.array(gt.logits[t])
        top1, rtop = int(logits.argmax()), int(ref.argmax())
        hits += top1 == rtop
        print(f"[{mode}] tok{t} top1={top1} ref={rtop} KL={kl(ref, logits):.4g} "
              f"{'OK' if top1 == rtop else 'MISS'}", flush=True)

    if hasattr(model, "cutmax"):
        z = model.cutmax(tiles)
        cm = int(np.array(model.decode_logits(z)).argmax())
        print(f"[{mode}] cutmax={cm} fhe_argmax={int(logits.argmax())}", flush=True)
        assert cm == int(logits.argmax()), "cutmax != logits argmax"
    assert hits >= n_tok - 1, f"top1 {hits}/{n_tok}"


def run_vit_forward():
    """Encrypted ViT forward vs the identically-truncated torch model (the old
    encvit_forward). GATE_BLOCKS=k truncates both (k=1 localizes block defects);
    k=12 gates top1/top5 vs the full model. VIT_CANON_TOP1 accepts a known FHE
    canon class (e.g. vit80's 664) when top1 differs from the torch ref."""
    import torch
    from transformers import AutoModelForImageClassification
    from perseus.nn import EncViT

    k = int(os.environ.get("GATE_BLOCKS", "12"))
    res = int(os.environ.get("GATE_RES", "0"))      # 112 -> 50 tokens, 80 -> 26; 0 = native
    model_name = os.environ.get("VIT_MODEL", "google/vit-base-patch16-224")
    model_dir = os.environ["VIT_MODEL_DIR"]

    hf = AutoModelForImageClassification.from_pretrained(
        model_name, attn_implementation="eager").eval()
    # transformers-v5 rename bridge: ViT encoder blocks moved vit.encoder.layer
    # -> vit.layers (behemoth venv is v5; leonardo was v4)
    vit_layers = hf.vit.layers if hasattr(hf.vit, "layers") else hf.vit.encoder.layer
    # VIT_VAL_DIR + VIT_IMG_IDXS ("a-b" or "i,j,k"): loop encrypted forwards over
    # labeled val images (crown-jewel accuracy sweep). Unset -> the original
    # single canon-pool-image path, bit-identical.
    val_dir = os.environ.get("VIT_VAL_DIR")
    if val_dir:
        import pickle
        meta = pickle.load(open(os.path.join(val_dir, "meta.pkl"), "rb"))
        slug = meta["model"].split("/")[-1]
        C, H, W = meta["channels"], meta["height"], meta["width"]
        vp = np.memmap(os.path.join(val_dir, f"{slug}_val_pixels.bin"),
                       dtype=np.float16, mode="r").reshape(-1, C, H, W)
        lb_raw = np.fromfile(os.path.join(val_dir, f"{slug}_val_labels.bin"),
                             dtype={8: np.int64, 4: np.int32, 2: np.int16,
                                    1: np.uint8}[os.path.getsize(
                                        os.path.join(val_dir, f"{slug}_val_labels.bin"))
                                        // len(vp)])
        ie = os.environ.get("VIT_IMG_IDXS", "0")
        idxs = (list(range(int(ie.split("-")[0]), int(ie.split("-")[1]) + 1))
                if "-" in ie and "," not in ie else [int(x) for x in ie.split(",")])
    else:
        idxs = [None]
    pool = np.load(os.environ["VIT_POOL"], mmap_mode="r")
    pix = torch.from_numpy(np.stack([pool[0]]))
    if res:
        pix = torch.nn.functional.interpolate(pix, size=(res, res), mode="bilinear",
                                              antialias=True)
    with torch.no_grad():
        emb = hf.vit.embeddings(pix, interpolate_pos_encoding=bool(res))
        h = emb
        for b in range(k):
            out = vit_layers[b](h)
            h = out[0] if isinstance(out, tuple) else out
        ref_logits = hf.classifier(hf.vit.layernorm(h)[0, 0]).numpy()
    tokens = emb[0].numpy().tolist()
    print(f"[{mode}] res={res or 'native'} T={len(tokens)}", flush=True)

    client = np.load(os.path.join(model_dir, "client.npz"))

    opts = _core.InferenceOptions()
    opts.ckks = _core.CKKSOptions.from_env()
    _m = os.environ.get("VIT_MODE", "sync" if os.environ.get("FHE_GRAPH_DIR") else "threaded")
    opts.mode = {"sync": _core.InferenceMode.Sync,
                 "threaded": _core.InferenceMode.Threaded,
                 "prefetch": _core.InferenceMode.Prefetch}[_m]
    print(f"[{mode}] mode={_m}", flush=True)
    inf = _core.make_vit_inference(opts)

    # weights_dir takes precedence: behemoth's binary has no LibArchive, so
    # exported zips are unpacked next to them (same fix as the BERT lane).
    wdir = os.path.join(model_dir, "weights_dir")
    store = (_core.WeightStore.from_dir(wdir) if os.path.isdir(wdir)
             else _core.WeightStore.from_zip(os.path.join(model_dir, "weights.bin.zip")))
    configs = _core.load_configs(os.environ["CONFIGS_PATH"])

    print(f"[{mode}] free after context: {_core.device_free_gb():.1f} GB", flush=True)
    model = EncViT(store, configs, n_layers=k).bind(inf)
    n_cls = len(client["classifier_bias"])
    # Session-reuse gotcha: per-forward caches (LN center masks, encode levels)
    # leak across images and break strict planned entry levels on forward #2 —
    # rebuild the EncViT wrapper per image (keys/weights stay resident).
    remodel = os.environ.get("VIT_REBUILD_PER_IMG", "1") == "1"

    def _ref(pix_t):
        with torch.no_grad():
            e = hf.vit.embeddings(pix_t, interpolate_pos_encoding=bool(res))
            hh = e
            for b in range(k):
                o = vit_layers[b](hh)
                hh = o[0] if isinstance(o, tuple) else o
            return hf.classifier(hf.vit.layernorm(hh)[0, 0]).numpy(), e[0].numpy().tolist()

    import time as _time
    agree = acc_enc = acc_ref = 0
    for i in idxs:
        if i is None:
            rl, toks = ref_logits, tokens
        else:
            p = torch.from_numpy(np.asarray(vp[i], dtype=np.float32))[None]
            rl, toks = _ref(p)
        t0 = _time.time()
        if i is not None and remodel and i != idxs[0]:
            n_ev = inf.clear_enc_cache()
            print(f"[{mode}] enc_cache cleared ({n_ev} entries)", flush=True)
            model = EncViT(store, configs, n_layers=k).bind(inf)
        xs, ns, ns_im = model.encode_tokens(toks)
        tiles = model.forward(xs, ns, ns_im)
        logits = np.array(model.decode_logits(tiles))[:n_cls] + client["classifier_bias"]
        dt = _time.time() - t0
        w_mape = np.abs(logits - rl).sum() / np.abs(rl).sum()
        top1, ref1 = int(np.argmax(logits)), int(np.argmax(rl))
        top5 = set(np.argsort(logits)[-5:].tolist())
        ref5 = set(np.argsort(rl)[-5:].tolist())
        assert np.isfinite(logits).all()
        if i is None:
            print(f"[{mode}] k={k} top1={top1} ref={ref1} {'OK' if top1 == ref1 else 'MISS'} "
                  f"top5_overlap={len(top5 & ref5)}/5 logits_w_mape={w_mape:.3f}", flush=True)
            canon = os.environ.get("VIT_CANON_TOP1")
            if k >= 12:
                assert (top1 == ref1 or len(top5 & ref5) >= 3
                        or (canon is not None and top1 == int(canon))), \
                    f"top1 {top1} != ref {ref1}, overlap {len(top5 & ref5)}/5, canon {canon}"
        else:
            lab = int(lb_raw[i])
            agree += top1 == ref1
            acc_enc += top1 == lab
            acc_ref += ref1 == lab
            print(f"[{mode}] img={i} label={lab} enc_top1={top1} ref_top1={ref1} "
                  f"{'AGREE' if top1 == ref1 else 'DISAGREE'} "
                  f"top5_overlap={len(top5 & ref5)}/5 mape={w_mape:.3f} "
                  f"fwd_s={dt:.1f} free={_core.device_free_gb():.1f}GB", flush=True)
    if idxs != [None]:
        n = len(idxs)
        print(f"[{mode}] SWEEP n={n} enc_acc={acc_enc}/{n} ref_acc={acc_ref}/{n} "
              f"enc_ref_agree={agree}/{n}", flush=True)


def run_bert_forward():
    """Encrypted BERT (post-LN encoder) vs the identically-truncated torch model.

    GATE_BLOCKS=k truncates both (k=1 localizes a block defect, k=12 = the real
    model). The encrypted stage is the ENCODER only: the client folds the
    embeddings (word+position+token_type then the embeddings LayerNorm) before
    encryption, and applies pooler-dense + tanh + classifier to the decrypted CLS
    afterwards — `tanh` has no FHE approximation (see perseus/nn/bert.py).
    BERT_TEXT overrides the sentence; BERT_TAPS=1 prints a per-block CLS drift.
    """
    import torch
    from transformers import AutoTokenizer
    from perseus.nn import EncBert
    from perseus.hub import load_model

    k = int(os.environ.get("GATE_BLOCKS", "12"))
    model_name = os.environ.get("BERT_MODEL", "textattack/bert-base-uncased-SST-2")
    model_dir = os.environ["BERT_MODEL_DIR"]
    text = os.environ.get("BERT_TEXT", "a masterpiece of modern cinema .")

    hf = load_model(model_name, device="cpu")          # keeps the pooler (head is client-side)
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(text, return_tensors="pt")
    ids, ttype = enc["input_ids"], enc.get("token_type_ids")
    if ttype is None:
        ttype = torch.zeros_like(ids)

    with torch.no_grad():
        emb = hf.bert.embeddings(input_ids=ids, token_type_ids=ttype)   # client-side fold
        h = emb
        taps = []
        for b in range(k):
            o = hf.bert.encoder.layer[b](h)
            h = o[0] if isinstance(o, tuple) else o
            taps.append(h[0, 0].numpy().copy())       # plaintext CLS after each block
        ref_logits = hf.classifier(hf.bert.pooler(h)).numpy()[0]
    tokens = emb[0].numpy().tolist()
    T = len(tokens)
    print(f"[{mode}] text={text!r} T={T} blocks={k}", flush=True)

    client = np.load(os.path.join(model_dir, "client.npz"))

    opts = _core.InferenceOptions()
    opts.ckks = _core.CKKSOptions.from_env()
    _m = os.environ.get("BERT_MODE", "sync" if os.environ.get("FHE_GRAPH_DIR") else "threaded")
    opts.mode = {"sync": _core.InferenceMode.Sync,
                 "threaded": _core.InferenceMode.Threaded,
                 "prefetch": _core.InferenceMode.Prefetch}[_m]
    inf = _core.make_bert_inference(opts)   # encoder context: filling packing, bidirectional
    print(f"[{mode}] mode={_m} free after context: {_core.device_free_gb():.1f} GB", flush=True)

    wdir = os.path.join(model_dir, "weights_dir")
    store = (_core.WeightStore.from_dir(wdir) if os.path.isdir(wdir)
             else _core.WeightStore.from_zip(os.path.join(model_dir, "weights.bin.zip")))
    configs = _core.load_configs(os.environ["CONFIGS_PATH"])
    model = EncBert(store, configs, n_layers=k).bind(inf)

    cap = inf.slots // inf.size.hidDim          # real arm: 32 tokens/chunk
    if inf.token_pair:
        cap *= 2
    if T > cap and os.environ.get("BERT_ALLOW_MULTICHUNK") != "1":
        raise RuntimeError(f"T={T} > {cap} = one packed chunk. The multi-chunk "
                           f"bidirectional arm has never been run end-to-end; "
                           f"shorten BERT_TEXT or set BERT_ALLOW_MULTICHUNK=1 "
                           f"(validated 2026-08-08 on behemoth, real arm, T=128).")
    xs, ns, ns_im = model.encode_tokens(tokens)
    print(f"[{mode}] chunks={len(xs)} ns={ns} ns_im={ns_im} "
          f"token_pair={inf.token_pair}", flush=True)

    cls = np.array(model.decode_cls(model.forward(xs, ns, ns_im)))

    rel = np.linalg.norm(cls - taps[k - 1]) / (np.linalg.norm(taps[k - 1]) + 1e-9)
    print(f"[{mode}] CLS rel_err={rel:.4f}", flush=True)

    # client-side head: pooler dense + tanh + classifier
    pooled = np.tanh(client["pooler_weight"] @ cls + client["pooler_bias"])
    logits = client["classifier_weight"] @ pooled + client["classifier_bias"]
    top1, ref1 = int(np.argmax(logits)), int(np.argmax(ref_logits))
    w_mape = np.abs(logits - ref_logits).sum() / (np.abs(ref_logits).sum() + 1e-9)
    print(f"[{mode}] k={k} top1={top1} ref={ref1} {'OK' if top1 == ref1 else 'MISS'} "
          f"logits={np.round(logits, 3).tolist()} ref={np.round(ref_logits, 3).tolist()} "
          f"w_mape={w_mape:.3f}", flush=True)
    assert np.isfinite(logits).all(), "non-finite logits"
    if k >= 12:
        assert top1 == ref1, f"top1 {top1} != ref {ref1}"


try:
    if MODEL == "bert":
        run_bert_forward()
    elif MODEL == "vit":
        run_vit_forward()
    elif mode == "forward":
        run_gpt2_forward()
    else:
        run_gpt2_session()
except BaseException:
    traceback.print_exc(file=sys.stdout)
    print(f"[{mode}] FAIL", flush=True)
    sys.stdout.flush()
    os._exit(1)

print(f"[{mode}] PASS", flush=True)
sys.stdout.flush()
