"""The Hugging Face side of the port: the pinned SANA-Sprint checkpoint, the sampler's constants (the pipeline's
default 2 steps) and the prompt layout (the public complex-human-instruction prefix, the encrypted prompt tokens, the
masked padding)."""
from __future__ import annotations

import dataclasses
import os

import numpy as np

NAME = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
REVISION = "aa76e7f4f4928f378716b6716a2130fba3caf5b1"
MAX_SEQ = 300                   # max_sequence_length: the text tokens the transformer attends
RES = int(os.environ.get("SANA_RES", "768"))   # the image's side in pixels (the 1024 px checkpoint at 768: 24 x 24
                                # latent tokens). SANA_RES=704: 22 x 22 = 484 tokens, 4 chunk pairs instead of 5 (a plan is
                                # cut for one resolution: export it for capture, plan and run, like SANA_TIGHT_ROWS)
GUIDANCE = 4.5                  # the pipeline's embedded guidance scale
T_MAX = 1.57080                 # max_timesteps: the first step at t = pi / 2
T_MID = 1.3                     # intermediate_timesteps: the second step's t
STEPS = (T_MAX, T_MID)[:int(os.environ.get("SANA_STEPS", "2"))]   # the pipeline's default 2 steps; SANA_STEPS=1:
                                # (T_MAX,) (ghosts compositional prompts). A plan is cut for one: export it for capture,
                                # plan and run, like SANA_RES

# SanaSprintPipeline.__call__'s default complex_human_instruction
CHI = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",  # noqa: E501
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",  # noqa: E501
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",  # noqa: E501
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",  # noqa: E501
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",  # noqa: E501
    "User Prompt: ",
]


def snapshot() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(NAME, revision=REVISION)


def load_pipeline(dtype="float32"):
    import torch
    from diffusers import SanaSprintPipeline
    pipe = SanaSprintPipeline.from_pretrained(snapshot(), torch_dtype=getattr(torch, dtype))
    pipe.text_encoder.config._attn_implementation = "eager"     # attn_logit_softcapping is eager-only
    return pipe


@dataclasses.dataclass
class PromptLayout:
    """One prompt through _get_gemma_prompt_embeds + encode_prompt's select_index. Gemma runs on the
    n_chi + MAX_SEQ - 2 ids (n_chi: the instruction's own token count); the transformer reads position 0 (BOS)
    and the last MAX_SEQ - 1 positions. Gemma is causal, so the n_pub positions the instruction fixes are public
    (the instruction's trailing space merges into the prompt's first word: n_pub = n_chi - 1); the n_tok prompt
    positions after them are the encrypted rows, then padding (masked)."""
    ids: np.ndarray             # (n_chi + MAX_SEQ - 2,) Gemma input ids
    mask: np.ndarray            # same shape, 1 on BOS..last prompt token
    n_chi: int                  # tokenize(instruction), BOS included
    n_pub: int                  # public positions: the common prefix of tokenize(instruction + prompt) and it
    n_tok: int                  # prompt tokens (encrypted)

    @property
    def select(self) -> np.ndarray:
        L = len(self.ids)
        return np.array([0] + list(range(L - MAX_SEQ + 1, L)))

    @property
    def prompt_pos(self) -> slice:
        return slice(self.n_pub, self.n_pub + self.n_tok)

    @property
    def prompt_sel(self) -> slice:
        """The prompt rows' indices in the transformer's MAX_SEQ text tokens."""
        i0 = int(np.searchsorted(self.select[1:], self.n_pub)) + 1
        return slice(i0, i0 + self.n_tok)


def prompt_layout(tok, prompt: str) -> PromptLayout:
    tok.padding_side = "right"              # as _get_gemma_prompt_embeds (Gemma's tokenizer pads left)
    chi = "\n".join(CHI)
    chi_ids = tok.encode(chi)
    n_chi = len(chi_ids)
    L = n_chi + MAX_SEQ - 2
    t = tok([chi + prompt], padding="max_length", max_length=L, truncation=True, add_special_tokens=True,
            return_tensors="np")
    ids, mask = t.input_ids[0], t.attention_mask[0]
    n_pub = next(i for i in range(n_chi + 1) if i == n_chi or ids[i] != chi_ids[i])
    if n_pub < len(ids) - MAX_SEQ + 1:
        raise ValueError(f"prompt_layout: {n_chi - n_pub} instruction tokens change with the prompt")
    return PromptLayout(ids, mask, n_chi, n_pub, int(mask.sum()) - n_pub)


def step_tag(i: int) -> str:
    """Step i's prefix of names (stage labels, its time-dependent linears, ref_float's captures): none on the first."""
    return f"s{i + 1}." if i else ""


def scm_constants(t: float = T_MAX):
    """The pipeline's arithmetic around the transformer F at the trigflow time t (one step): the model input is
    x_in = (x_t / sigma_data) c_in, the step's denoised latent (over sigma_data) a x_in + b F(x_in)."""
    s = np.sin(t) / (np.cos(t) + np.sin(t))
    c_in = np.sqrt(s * s + (1 - s) ** 2)
    norm = np.sqrt(s * s + (1 - s) ** 2)
    # noise_pred = ((1 - 2s) x_in + (1 - 2s + 2s^2) F) / norm * sd ; denoised = cos(t) x_t - sin(t) noise_pred
    # with x_t = noise * sd: denoised / sd = cos(t) x_in / c_in - sin(t) noise_pred / sd
    a = np.cos(t) / c_in - np.sin(t) * (1 - 2 * s) / norm
    b = -np.sin(t) * (1 - 2 * s + 2 * s * s) / norm
    return dict(s=float(s), c_in=float(c_in), a=float(a), b=float(b))


def renoise(t: float, prev: float):
    """The next step's model input from the previous step's (at time `prev`) model input x and output F and fresh
    noise z (all over sigma_data): x' = p x + q F + r z (SCMScheduler.step: x_t = cos(t) x0 + sin(t) z sigma_data, x0
    = a x + b F)."""
    k0, k = scm_constants(prev), scm_constants(t)
    c = k["c_in"]
    return dict(p=c * np.cos(t) * k0["a"], q=c * np.cos(t) * k0["b"], r=c * np.sin(t))


# the shipped approximation configs (fit.py's) and the PCMM linears' output bands (a --pcmm-calib run's)
CONFIGS = "configs/model/approximation/sana_sprint_t2i/configs.json"
PCMM_CALIB = "configs/model/approximation/sana_sprint_t2i/pcmm_calib.json"


def scratch(*parts) -> str:
    """A path under the scratch root (SANA_SCRATCH, default $HF_HOME/perseus/sana): the float dumps, statistics,
    fitted configs and outputs."""
    root = os.environ.get("SANA_SCRATCH") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "perseus", "sana")
    p = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def data(name) -> str:
    """A configs / calibration file: its repo or package spelling when one exists (perseus._paths.resolve), else a
    name under the scratch root (fit.py's or a --pcmm-calib run's output)."""
    from perseus._paths import resolve
    p = resolve(name)
    return p if os.path.exists(p) else scratch(name)
