"""Regenerate notebooks/vit_torch_forward.ipynb — inline image, encrypt/decrypt round-trip.

The image is chosen inline in the first cell (an index into the calibration pool) and
displayed, so it is obvious what is actually being classified.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# ViT-B/16 as a torch module — encrypted, planned

An image classified under CKKS on GPU. `perseus.nn.EncViT` is the torch-style module;
the encrypted forward runs the 12 encoder blocks + classifier head under the frozen
`planned_vit_complex_112` plan — nothing decrypted mid-chain.

Only the patch embedding runs client-side. Pick the image inline in the first cell.
""")

code('''
import os, time

import numpy as np
import torch                                   # torch before _core (NCCL load order)
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification

from perseus import _core
from perseus.nn import EncViT

# ── pick the image here ───────────────────────────────────────────────────────
IMAGE_INDEX = 0          # index into the calibration image pool
RES         = int(os.environ.get("GATE_RES", "112"))
# ──────────────────────────────────────────────────────────────────────────────

k = int(os.environ.get("GATE_BLOCKS", "12"))
model_name = os.environ.get("VIT_MODEL", "google/vit-base-patch16-224")
model_dir = os.environ["VIT_MODEL_DIR"]

hf = AutoModelForImageClassification.from_pretrained(
    model_name, attn_implementation="eager").eval()

pool = np.load(os.environ["VIT_POOL"], mmap_mode="r")
pix = torch.from_numpy(np.stack([pool[IMAGE_INDEX]]))
pix = torch.nn.functional.interpolate(pix, size=(RES, RES), mode="bilinear", antialias=True)
print(f"image {IMAGE_INDEX}: {tuple(pix.shape)}  (res={RES})")


def show(t, title):
    """Processor tensors are normalised to [-1, 1]; undo that for display."""
    img = ((t[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    plt.figure(figsize=(2.6, 2.6)); plt.imshow(img); plt.axis("off")
    plt.title(title, fontsize=9); plt.show()


show(pix, f"input image {IMAGE_INDEX}")
''')

md("""
## Client side: patch embedding, then the plaintext reference
""")

code('''
with torch.no_grad():
    emb = hf.vit.embeddings(pix, interpolate_pos_encoding=True)
    h = emb
    for b in range(k):
        out = hf.vit.encoder.layer[b](h)
        h = out[0] if isinstance(out, tuple) else out
    ref_logits = hf.classifier(hf.vit.layernorm(h)[0, 0]).numpy()

tokens = emb[0].numpy().tolist()
ref_id = int(np.argmax(ref_logits))
print(f"patch tokens : {len(tokens)} x {len(tokens[0])}")
print(f"plaintext top-1 = {ref_id} ({hf.config.id2label[ref_id]})")
''')

md("""
## Build the encrypted model, encrypt the image
""")

code('''
opts = _core.InferenceOptions()
opts.ckks = _core.CKKSOptions.from_env()
_mode = os.environ.get("VIT_MODE", "threaded")
opts.mode = {"sync": _core.InferenceMode.Sync,
             "threaded": _core.InferenceMode.Threaded}[_mode]
inf = _core.make_vit_inference(opts)

store = _core.WeightStore.from_zip(os.path.join(model_dir, "weights.bin.zip"))
configs = _core.load_configs(os.environ["CONFIGS_PATH"])
model = EncViT(store, configs, n_layers=k).bind(inf)

plan = os.environ.get("FHE_BOOTSTRAP_PLACEMENTS_DIR", "(eager)")
print(f"EncViT bound: {k} blocks   plan = {os.path.basename(plan)}   mode = {_mode}")

xs, ns, ns_im = model.encode_tokens(tokens)          # <- encrypted here
print(f"encrypted {len(tokens)} tokens -> {len(xs)} chunk(s)  ns={ns} token_pair={inf.token_pair}")
''')

md("""
## Encrypted forward
""")

code('''
t0 = time.perf_counter()
tiles = model.forward(xs, ns, ns_im)          # <- fully encrypted, planned
client = np.load(os.path.join(model_dir, "client.npz"))
n_cls = len(client["classifier_bias"])
logits = np.asarray(model.decode_logits(tiles))[:n_cls] + client["classifier_bias"]
print(f"forward: {time.perf_counter() - t0:.1f} s  ->  {n_cls} class logits")

top1 = int(np.argmax(logits))
w_mape = np.abs(logits - ref_logits).sum() / np.abs(ref_logits).sum()
print(f"encrypted top-1 : {top1}  ({hf.config.id2label[top1]})")
print(f"plaintext top-1 : {ref_id}  ({hf.config.id2label[ref_id]})   "
      f"{'MATCH' if top1 == ref_id else 'DIFF'}")
print(f"logits agreement (w_mape): {w_mape:.3f}")
assert np.isfinite(logits).all()
''')

md("""
## What this showed

- **Encrypted ViT-B/16** classified the image shown above as a `torch.nn.Module` — 12
  encoder blocks + head under CKKS, nothing decrypted mid-chain, on the frozen plan.

Change `IMAGE_INDEX` to classify a different image from the pool.

Note the encrypted top-1 differs from plaintext on this row: ViT-80/112's FHE canon class
is a known open item (accuracy, not speed), tracked separately from the plan work.
""")

nb.cells = C
nb.metadata = {"language_info": {"name": "python"}}
nbf.write(nb, "notebooks/vit_torch_forward.ipynb")
print("wrote notebooks/vit_torch_forward.ipynb")
