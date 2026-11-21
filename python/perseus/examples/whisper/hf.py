"""Whisper's encoder as the calibrator (`python -m perseus.calibrate model=whisper_base_en`) sees it: HF's
WhisperEncoder forward with the front-end's two GELUs as modules (HF calls them functionally), so their sites are
collected and fitted like the blocks' (`frontend.conv1.act`, `frontend.conv2.act` in the runtime's names)."""
from __future__ import annotations

import torch
from torch import nn


def fit_window(model):
    """A transformers Whisper model (or its WhisperModel) cut to weights.T encoder positions, so generate's
    input-length check takes the window's N_FRAMES; returns it."""
    from . import weights as W
    inner = getattr(model, "model", model)
    enc = inner.encoder
    if enc.embed_positions.num_embeddings != W.T:
        enc.embed_positions.weight = nn.Parameter(enc.embed_positions.weight.data[:W.T].clone(), requires_grad=False)
        enc.embed_positions.num_embeddings = W.T
        enc.max_source_positions = W.T
        enc.config.max_source_positions = W.T
    return model


def window(input_features):
    """input_features (B, 80, >= N_FRAMES) cut to the window's weights.N_FRAMES."""
    from . import weights as W
    return input_features[..., :W.N_FRAMES]


class WhisperEncoderTaps(nn.Module):
    """HF WhisperEncoder's modules under their own names (conv1, conv2, embed_positions, layers, layer_norm) plus
    `act1` / `act2`, the front-end GELUs; forward(input_features (B, 80, >= N_FRAMES)) = WhisperEncoder's last hidden
    state over the window (`window`)."""

    def __init__(self, encoder, config):
        super().__init__()
        self.config = config
        self.conv1, self.conv2 = encoder.conv1, encoder.conv2
        self.embed_positions, self.layers, self.layer_norm = encoder.embed_positions, encoder.layers, encoder.layer_norm
        self.act1, self.act2 = nn.GELU(), nn.GELU()

    def forward(self, input_features):
        x = self.act2(self.conv2(self.act1(self.conv1(window(input_features))))).permute(0, 2, 1)
        x = x + self.embed_positions.weight[:x.shape[1]]
        for layer in self.layers:
            x = layer(x, None)
            x = x[0] if isinstance(x, tuple) else x
        return self.layer_norm(x)


def encoder(pretrained_model_name_or_path, attn_implementation="eager", **kw):
    """The calibration model (hydra `_target_`): the pretrained encoder, eager attention (the softmax tap)."""
    from transformers import WhisperModel
    m = WhisperModel.from_pretrained(pretrained_model_name_or_path, attn_implementation=attn_implementation, **kw)
    return WhisperEncoderTaps(m.encoder, m.config)


@torch.no_grad()
def check_encoder(path, features):
    """max |WhisperEncoderTaps - HF WhisperEncoder| on one batch (the wrapper's own test)."""
    from transformers import WhisperModel
    m = WhisperModel.from_pretrained(path, attn_implementation="eager").eval()
    taps = WhisperEncoderTaps(m.encoder, m.config).eval()
    x = torch.as_tensor(features, dtype=torch.float32)
    return float((taps(x) - m.encoder(x).last_hidden_state).abs().max())


class WhisperDecoderTaps(nn.Module):
    """HF WhisperDecoder (`decoder`: its layers, layer_norm) teacher-forced on the greedy transcript of each clip, the
    calibrator's view of the decode. forward(input_features (B, 80, >= N_FRAMES), cut to the window) runs the encoder
    and the greedy decoding on
    a second copy outside the module tree (no site sees them), then the decoder over [the forced start + the
    transcript] one clip at a time; `.logits`: the rows the decode's argmax reads (each position from the last forced
    token to the one emitting end-of-text), over the tokens it may emit (weights.Generation.allowed)."""

    def __init__(self, model, gen_model, config, allowed, max_new_tokens=128):
        super().__init__()
        self.config = config
        self.decoder = model.model.decoder
        self._hidden = [gen_model]                       # not a submodule: no hooks on the encoder or the generation
        self.register_buffer("allowed", torch.as_tensor(allowed, dtype=torch.long), persistent=False)
        self.max_new_tokens = max_new_tokens

    def forward(self, input_features):
        import types
        g = self._hidden[0]
        if next(g.parameters()).device != input_features.device:
            g.to(input_features.device)
        x = window(input_features).to(next(g.parameters()).dtype)
        with torch.no_grad():
            enc = g.model.encoder(x).last_hidden_state
            seqs = g.generate(input_features=x, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        eot, n0 = int(g.generation_config.eos_token_id), 2
        dt = next(self.decoder.parameters()).dtype
        rows = []
        for b in range(seqs.shape[0]):
            s = seqs[b].tolist()
            end = s.index(eot, n0) + 1 if eot in s[n0:] else len(s)
            ids = torch.tensor([s[:end]], device=x.device)
            h = self.decoder(input_ids=ids, encoder_hidden_states=enc[b:b + 1].to(dt)).last_hidden_state
            lg = h[0, n0 - 1:end - 1] @ self.decoder.embed_tokens.weight.T
            rows.append(lg[:, self.allowed])
        return types.SimpleNamespace(logits=torch.cat(rows))


def decoder(pretrained_model_name_or_path, attn_implementation="eager", max_new_tokens=128, **kw):
    """The decode's calibration model (hydra `_target_`): WhisperDecoderTaps on two copies of the pretrained model,
    eager attention (the softmax tap)."""
    from transformers import WhisperForConditionalGeneration
    from . import weights as W
    load = lambda: WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                                    attn_implementation=attn_implementation, **kw).eval()
    m, g = fit_window(load()), fit_window(load())
    gen = W.Generation(W.snapshot())
    return WhisperDecoderTaps(m, g, m.config, gen.allowed(m.config.vocab_size).nonzero()[0], max_new_tokens)
