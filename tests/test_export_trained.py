"""perseus-export --checkpoint: an HE-aware-trained GPT-2 state dict into the stock model."""
import json
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from perseus.export import export_weights, load_trained_backbone


def _tiny():
    cfg = transformers.GPT2Config(n_layer=1, n_embd=8, n_head=2, n_positions=16, vocab_size=32)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def _trained_state(src):
    """What the trainer saves: attention as nn.Linear (d_out, d_in) plus training-only extras."""
    sd = {k: v.clone() for k, v in src.state_dict().items() if not k.endswith(".attn.bias")}
    for k in ("transformer.h.0.attn.c_attn.weight", "transformer.h.0.attn.c_proj.weight"):
        sd[k] = sd[k].t().contiguous()
    sd["transformer.h.0.ln_1.inv_sqrt_approx.ponder_newton.halt_logits"] = torch.zeros(3)
    sd["transformer.h.0.attn.softmax.init_alpha"] = torch.tensor(1.0)
    return sd


def test_trained_checkpoint_round_trip(tmp_path):
    src = _tiny()
    ck = tmp_path / "model.pt"
    torch.save(_trained_state(src), ck)
    dst = _tiny()
    for p in dst.parameters():
        torch.nn.init.zeros_(p)
    load_trained_backbone(dst, str(ck))
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v), k

    zp = export_weights(dst, str(tmp_path), "heat")
    with zipfile.ZipFile(zp) as zf:
        man = {e["name"]: e for e in json.loads(zf.read("manifest.json"))["tensors"]}
        e = man["transformer.h.0.attn.c_proj.weight"]
        got = np.frombuffer(zf.read(e["path"]), dtype=e["dtype"]).reshape(e["shape"])
    # the store holds nn.Linear (d_out, d_in), whatever layout the checkpoint used
    np.testing.assert_array_equal(got, src.transformer.h[0].attn.c_proj.weight.detach().numpy().T)


def test_trained_checkpoint_shape_mismatch_is_an_error(tmp_path):
    sd = _trained_state(_tiny())
    sd["transformer.h.0.mlp.c_fc.bias"] = torch.zeros(5)
    ck = tmp_path / "model.pt"
    torch.save(sd, ck)
    with pytest.raises(ValueError, match="c_fc.bias"):
        load_trained_backbone(_tiny(), str(ck))


def test_trained_checkpoint_missing_weights_is_an_error(tmp_path):
    sd = _trained_state(_tiny())
    del sd["transformer.ln_f.weight"]
    ck = tmp_path / "model.pt"
    torch.save(sd, ck)
    with pytest.raises(ValueError, match="ln_f.weight"):
        load_trained_backbone(_tiny(), str(ck))
