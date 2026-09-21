"""Typed session profiles: the env contract as an object instead of scattered exports.

The runtime is steered by env vars whose semantics-bearing subset must agree with the
plan being loaded (see perseus.plan.contract). A SessionProfile makes that subset
explicit, applies it with env-as-override semantics (an already-exported var always
wins, so run scripts keep working unchanged), and hands back InferenceOptions.

    from perseus.profile import SessionProfile
    prof = SessionProfile.gpt2_decode_n32()
    prof.apply()                      # setdefault into os.environ
    inf = _core.make_gpt2_inference(prof.options())

`extra` is a free-form env mapping for everything without a field: sparse routing, the
softmax fold, arena sizes, the rotation-key band, planner units. The subset that shapes
the executed graph is also the plan contract (perseus.plan.contract); the rest only moves
WHEN work happens.
"""
import dataclasses
import os


def _device_free_gb():
    """Free memory of the visible GPU, or None when no GPU is visible or the extension is
    not built (the clamp is then skipped, never guessed)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0") in ("", "-1"):
        return None
    try:
        from . import _core
        gb = float(_core.device_free_gb())
    except Exception:
        return None
    return gb if gb > 0 else None


@dataclasses.dataclass
class SessionProfile:
    chain: str = "n64"
    logN: int = 16
    auto_bts_level: int = 24          # PRIME-granular on n32 (46/49), CKKS-level on n64
    bts_iterations: int = 1
    ckks_complex: bool = True
    packing: str = "cachemir"
    gpt2_cache: bool = True
    fold_ln1: bool = True
    fold_ln2: bool = True
    fold_lnf: bool = False
    cache_read_level_k: int = 17
    cache_read_level_v: int = 17
    lmhead_cap: int = 22
    mode: str = "threaded"            # sync | prefetch | threaded (scheduling-only)
    extra: dict = dataclasses.field(default_factory=dict)

    def _env(self):
        e = {
            "CHAIN": self.chain,
            "LOGN": str(self.logN),
            "AUTO_BTS_LEVEL": str(self.auto_bts_level),
            "BTS_ITERATIONS": str(self.bts_iterations),
            "CKKS_COMPLEX": "1" if self.ckks_complex else "0",
            "GPT2_PACKING": self.packing,
            "GPT2_CACHE": "1" if self.gpt2_cache else "0",
            "GPT2_FOLD_LN1": "1" if self.fold_ln1 else "0",
            "GPT2_FOLD_LN2": "1" if self.fold_ln2 else "0",
            "GPT2_FOLD_LNF": "1" if self.fold_lnf else "0",
            "CACHE_READ_LEVEL_K": str(self.cache_read_level_k),
            "CACHE_READ_LEVEL_V": str(self.cache_read_level_v),
            "FHE_LMHEAD_CAP": str(self.lmhead_cap),
            "GPT2_INFERENCE_MODE": self.mode,
        }
        e.update({k: str(v) for k, v in self.extra.items()})
        return e

    #: device-memory knobs: fraction of the free device memory a preset may claim
    DEVICE_CLAMPS = {"KV_ARENA_GB": 0.25}

    def env(self, host=True):
        e = self._env()
        if host:
            free_gb = _device_free_gb()
            if free_gb is not None:
                for k, frac in self.DEVICE_CLAMPS.items():
                    if k in e and float(e[k]) > frac * free_gb:
                        e[k] = str(max(1, int(frac * free_gb)))
        return e

    def apply(self):
        """setdefault the profile into os.environ (exported env always wins). Arena sizes
        are clamped to this device's free memory first."""
        for k, v in self.env().items():
            os.environ.setdefault(k, v)
        return self

    def options(self):
        """InferenceOptions of the loaded extension (perseus._core, else perseus._client)
        read from the environment, with the profile's mode."""
        from . import _backend
        ext = _backend.default()
        loaded = getattr(ext, "chain", None)
        if loaded and loaded != self.chain:
            import warnings
            warnings.warn(f"SessionProfile(chain={self.chain!r}) but the loaded {ext.__name__} "
                          f"is the {loaded} build; the two chains are numerically different "
                          f"libraries — relink (scripts/local_build_core.sh / "
                          f"local_build_client.sh) or pick the matching preset", stacklevel=2)
        opts = ext.InferenceOptions()
        opts.ckks = ext.CKKSOptions.from_env()
        opts.mode = _backend.mode(ext, self.mode) or ext.InferenceMode.Threaded
        return opts

    # ── presets (the validated recipes, not guesses) ─────────────────────────

    @classmethod
    def gpt2_decode_n32(cls):
        """The paper's 32-bit decode row, the same configuration scripts/run_task.sh sets."""
        return cls(chain="n32", auto_bts_level=46, lmhead_cap=44,
                   cache_read_level_k=34, cache_read_level_v=34,
                   extra={"KV_ARENA_GB": 24, "MALLOC_ARENA_MAX": 2, "STEPS_T": 128,
                          "FIDESLIB_ROT_KEY_BAND": 22, "FHE_STAGE_ARENA_GB": 24,
                          # these shape the executed graph: a plan is bound to them
                          "SPARSE_BTS_SLOTS": "512,1", "SPARSE_AUTO": 2,
                          "FUSED_SM_DEN": 1, "FUSED_LN_VAR": 0,
                          "FHE_PT_COEFF_ENCODE": 0,
                          # planner units for this chain (make_plans.sh n32 arm)
                          "PLAN_MAX_LEVEL": 46, "PLAN_BTS_LEVEL": 34,
                          "PLAN_SRC_LEVEL": 34, "PLAN_CACHE_READ_LEVEL": 34,
                          "PLAN_LEVEL_UNIT": 2})

    @classmethod
    def gpt2_decode_n64(cls):
        """The 64-bit reference decode row (scripts/run_task.sh CHAIN=n64)."""
        return cls(chain="n64", auto_bts_level=24,
                   extra={"MALLOC_ARENA_MAX": 2, "STEPS_T": 128,
                          "FIDESLIB_ROT_KEY_BAND": 11, "SPARSE_BTS_SLOTS": "512,1",
                          "SPARSE_AUTO": 2, "CORRECTION_FACTOR": 7,
                          "FHE_PT_COEFF_ENCODE": 0})

    @classmethod
    def custom_n64(cls):
        """run_notebooks.sh `custom`: the self-planning custom-model toy env."""
        return cls(chain="n64", auto_bts_level=24, mode="sync")

    @classmethod
    def custom_n32(cls):
        """The n32 custom-model probe env (prime-granular levels + planner units)."""
        return cls(chain="n32", auto_bts_level=49, mode="sync",
                   extra={"FIDESLIB_ROT_KEY_BAND": 22,
                          "PLAN_MAX_LEVEL": 49, "PLAN_BTS_LEVEL": 34,
                          "PLAN_SRC_LEVEL": 34, "PLAN_CACHE_READ_LEVEL": 34,
                          "PLAN_LEVEL_UNIT": 2})
