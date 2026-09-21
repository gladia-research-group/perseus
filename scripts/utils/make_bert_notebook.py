"""Generate notebooks/bert_torch_forward.ipynb — EncBert forward, planned.

Sentence chosen inline in the first cell, like GEN_PROMPT/IMAGE_INDEX in the other two.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BERT-base as a torch module — encrypted, planned

Sentiment classification under CKKS. `perseus.nn.EncBert` runs the 12 encoder blocks on
the frozen `planned_bert_base` plan, nothing decrypted mid-chain.

Client-side: the embeddings, and the head (pooler + `tanh` + classifier) — `tanh` has no
FHE approximation, so the encrypted stage ends at the encoder.
""")

code('''
import os, time

import numpy as np
import torch                                   # torch before _core (NCCL load order)
from transformers import AutoTokenizer

from perseus import _core
from perseus.hub import load_model
from perseus.nn import EncBert

# ── write your sentence here ──────────────────────────────────────────────────
TEXT = "a masterpiece of modern cinema ."
# ──────────────────────────────────────────────────────────────────────────────
SENTIMENT = {0: "negative", 1: "positive"}

k = int(os.environ.get("GATE_BLOCKS", "12"))
model_name = os.environ.get("BERT_MODEL", "textattack/bert-base-uncased-SST-2")
model_dir = os.environ["BERT_MODEL_DIR"]

hf = load_model(model_name, device="cpu")
tok = AutoTokenizer.from_pretrained(model_name)
ids = tok(TEXT, return_tensors="pt")["input_ids"]

with torch.no_grad():
    emb = hf.bert.embeddings(input_ids=ids, token_type_ids=torch.zeros_like(ids))
    h = emb
    for b in range(k):
        out = hf.bert.encoder.layer[b](h)
        h = out[0] if isinstance(out, tuple) else out   # BertLayer returns Tensor or tuple
    ref_logits = hf.classifier(hf.bert.pooler(h)).numpy()[0]

tokens = emb[0].numpy().tolist()
ref_id = int(np.argmax(ref_logits))
print(f"sentence : {TEXT!r}")
print(f"tokens   : {tok.convert_ids_to_tokens(ids[0])}")
print(f"plaintext: {SENTIMENT[ref_id]}")
''')

md("""
## Build the encrypted model
""")

code('''
opts = _core.InferenceOptions()
opts.ckks = _core.CKKSOptions.from_env()
_mode = os.environ.get("BERT_MODE", "threaded")
opts.mode = {"sync": _core.InferenceMode.Sync,
             "threaded": _core.InferenceMode.Threaded}[_mode]
inf = _core.make_bert_inference(opts)

store = _core.WeightStore.from_zip(os.path.join(model_dir, "weights.bin.zip"))
configs = _core.load_configs(os.environ["CONFIGS_PATH"])
model = EncBert(store, configs, n_layers=k).bind(inf)

plan = os.environ.get("FHE_BOOTSTRAP_PLACEMENTS_DIR", "(eager)")
print(f"EncBert bound: {k} blocks   plan = {os.path.basename(plan)}   mode = {_mode}")
''')

md("""
## Encrypted forward

One packed chunk: the real arm carries `slots / hidDim` = 32 tokens and the multi-chunk
bidirectional arm is unvalidated, so the driver raises above that rather than rely on it.
""")

code('''
cap = inf.slots // inf.size.hidDim * (2 if inf.token_pair else 1)
if len(tokens) > cap:
    raise RuntimeError(f"T={len(tokens)} > {cap} = one packed chunk; shorten TEXT.")

xs, ns, ns_im = model.encode_tokens(tokens)
t0 = time.perf_counter()
cls = np.array(model.decode_cls(model.forward(xs, ns, ns_im)))
print(f"forward: {time.perf_counter() - t0:.1f} s   ->  CLS vector of {cls.shape[0]} features")
''')

md("""
## The prediction

Client head on the decrypted CLS: pooler dense, `tanh`, classifier.
""")

code('''
c = np.load(os.path.join(model_dir, "client.npz"))
logits = c["classifier_weight"] @ np.tanh(
    c["pooler_weight"] @ cls + c["pooler_bias"]) + c["classifier_bias"]
top1 = int(np.argmax(logits))
w_mape = np.abs(logits - ref_logits).sum() / np.abs(ref_logits).sum()

print(f"sentence  : {TEXT!r}")
print(f"encrypted : {SENTIMENT[top1]}   {np.round(logits, 3).tolist()}")
print(f"plaintext : {SENTIMENT[ref_id]}   {np.round(ref_logits, 3).tolist()}   "
      f"{'MATCH' if top1 == ref_id else 'DIFF'}")
print(f"w_mape    : {w_mape:.3f}")
assert np.isfinite(logits).all()
assert top1 == ref_id, f"encrypted {top1} != plaintext {ref_id}"
''')

md("""
## Notes

- Change `TEXT` above; up to 32 tokens.
- Encrypted logits come out compressed toward zero — argmax is reliable, magnitudes are
  not calibrated.
""")

nb.cells = C
nb.metadata = {"language_info": {"name": "python"}}
nbf.write(nb, "notebooks/bert_torch_forward.ipynb")
print("wrote notebooks/bert_torch_forward.ipynb")
