import contextlib
import datetime
import json
import logging
import os
import warnings

import numpy as np

from .. import __version__, _backend
from .._env import current_chain
from ..errors import BundleError, FHEError, LayoutError, MaskError, PlanError, SecurityError

log = logging.getLogger(__name__)

_ext = None   # the extension the module runs on; None = resolve on first use (tests may set it)


def _default_ext():
    """The loaded extension (perseus._core, else perseus._client), resolved on first use so
    the package imports without either being built."""
    return _ext if _ext is not None else _backend.default()


@contextlib.contextmanager
def _translate(ext):
    """Re-raise `ext`'s error classes as perseus.errors' (the _core classes when both
    extensions are built) so `except perseus.errors.FHEError` holds for the client role
    whichever extension backs it."""
    try:
        yield
    except RuntimeError as e:
        base = getattr(ext, "FHEError", None)
        if base is None or isinstance(e, FHEError) or not isinstance(e, base):
            raise
        for name, cls in (("PlanError", PlanError), ("MaskError", MaskError),
                          ("LayoutError", LayoutError)):
            typed = getattr(ext, name, None)
            if typed is not None and isinstance(e, typed):
                raise cls(str(e)) from e
        raise FHEError(str(e)) from e


MANIFEST_NAME = "bundle.json"
MANIFEST_FORMAT = 1

_FAMILIES = {
    "gpt2": "make_gpt2_inference",
    "vit": "make_vit_inference",
    "bert": "make_bert_inference",
    "generic": "make_inference",
}

_CKKS_FIELDS = (
    "logN", "depth", "scale_bits", "enable_bootstrap", "btp_depth_overhead", "level_budget",
    "bootstrap_slots", "sparse_bts_slots_list", "sparse_level_budget", "btp_scale_bits",
    "correction_factor", "first_mod_bits", "num_large_digits", "auto_bts_level_override",
    "batch_size", "ckks_complex_payload", "h_weight",
    "extra_rot_steps", "deferred_rot_steps", "bts_iterations", "bts_precision",
    "composite_degree", "defer_heavy_setup",
)
_INF_FIELDS = ("dim", "expanded", "hidDim", "expDim", "numHeads", "numHeadsReal", "seqLen")


def _plain(v):
    return list(v) if isinstance(v, (list, tuple)) else v


def options_to_manifest(options, family):
    """The JSON view of an InferenceOptions: what a server needs to rebuild the same
    context (parameters, model widths, packing) — nothing process-local."""
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}, got {family!r}")
    ck = options.ckks
    return {
        "format": MANIFEST_FORMAT,
        "family": family,
        "ckks": {k: _plain(getattr(ck, k)) for k in _CKKS_FIELDS},
        "inference": {
            **{k: int(getattr(options, k)) for k in _INF_FIELDS},
            "packing_kind": options.packing_kind.name,
            "aux_packing_kinds": [k.name for k in options.aux_packing_kinds],
        },
    }


def options_from_manifest(manifest, ext=None):
    """Inverse of options_to_manifest (keys_dir / skip_gpu_load / mode left default), as
    `ext`'s option structs (default: the loaded extension)."""
    ext = ext or _default_ext()
    if manifest.get("format") != MANIFEST_FORMAT:
        raise BundleError(f"bundle manifest format {manifest.get('format')!r} is not "
                          f"the supported {MANIFEST_FORMAT}")
    opts = ext.InferenceOptions()
    ck = ext.CKKSOptions()
    for k in _CKKS_FIELDS:
        if k in manifest["ckks"]:
            setattr(ck, k, manifest["ckks"][k])
    opts.ckks = ck
    inf = manifest["inference"]
    for k in _INF_FIELDS:
        if k in inf:
            setattr(opts, k, int(inf[k]))
    opts.packing_kind = getattr(ext.PackingKind, inf["packing_kind"])
    opts.aux_packing_kinds = [getattr(ext.PackingKind, n) for n in inf["aux_packing_kinds"]]
    return opts


def manifest_diff(expected, given):
    """Human-readable list of the fields on which two manifests disagree."""
    out = []
    if expected.get("family") != given.get("family"):
        out.append(f"family: bundle={expected.get('family')!r} caller={given.get('family')!r}")
    for section in ("ckks", "inference"):
        a, b = expected.get(section, {}), given.get(section, {})
        for k in sorted(set(a) | set(b)):
            if _plain(a.get(k)) != _plain(b.get(k)):
                out.append(f"{section}.{k}: bundle={a.get(k)!r} caller={b.get(k)!r}")
    return out


def _resolve_options(options, profile, ext=None):
    if options is not None and profile is not None:
        raise ValueError("pass options= or profile=, not both")
    if options is not None:
        return options
    if profile is not None:
        profile.apply()
        return profile.options()
    ext = ext or _default_ext()
    opts = ext.InferenceOptions()
    opts.ckks = ext.CKKSOptions.from_env()
    return opts


def _copy_options(options, family, ext=None):
    """A deep copy that cannot alias the caller's object, as `ext`'s structs: the C++ copy
    constructor when `options` already belongs to `ext` (exact, hidden fields included),
    else the manifest round trip (which is how the other extension's structs — e.g.
    _core's from SessionProfile.options() on a GPU box — become _client's)."""
    ext = ext or _default_ext()
    try:
        opts = ext.InferenceOptions(options)
        opts.ckks = ext.CKKSOptions(options.ckks)
        return opts
    except TypeError:
        opts = options_from_manifest(options_to_manifest(options, family), ext=ext)
        opts.mode = getattr(ext.InferenceMode, options.mode.name)
        return opts


class EncClient:
    _encode_level: int | None = None
    _formula_logged: bool = False
    _ext_module = None

    def __init__(self, options=None, profile=None, family: str = "gpt2",
                 backend: str | None = None):
        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}, got {family!r}")
        self.family = family
        ext = _backend.client_extension(backend)
        self._ext_module = ext
        opts = _copy_options(_resolve_options(options, profile, ext), family, ext)
        opts.ckks.skip_gpu_load = True     # a client machine needs no GPU
        opts.ckks.keys_dir = ""            # generate, never load
        opts.mode = ext.InferenceMode.Sync
        self.options = opts
        with _translate(ext):
            self.inf = getattr(ext, _FAMILIES[family])(opts)
        self.d = self.inf.size.dim         # real feature width of one token

    @property
    def backend(self):
        """The extension module backing this client (perseus._core or perseus._client)."""
        return self._ext_module if self._ext_module is not None else _default_ext()

    def __repr__(self):
        return (f"EncClient(family={self.family!r}, d={self.d}, logN={self.options.ckks.logN}, "
                f"backend={self.backend.__name__})")

    def _local_level(self) -> int:
        """This session's own `bootstrap_output_level()`: the probe on a GPU session, the
        parameter formula on a GPU-less one (the probe needs a bootstrap)."""
        return int(self.inf.fhe.bootstrap_output_level())

    @property
    def encode_level(self) -> int:
        return self._encode_level if self._encode_level is not None else self._local_level()

    def accept(self, manifest: dict | None) -> int:
        level = (manifest or {}).get("bootstrap_output_level")
        if level is None:
            if not self._formula_logged:
                log.info("EncClient: the server's manifest carries no bootstrap_output_level; "
                         "encoding fresh inputs at this session's level %d (a GPU session "
                         "may probe a different one; strict plans expect the probed level)",
                         self._local_level())
                self._formula_logged = True
            return self._local_level()
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or level <= 0:
            raise ValueError(f"accept: bootstrap_output_level must be a positive integer, "
                             f"got {level!r}")
        grid = int(self.options.ckks.composite_degree)
        if grid > 1 and level % grid != 0:
            raise ValueError(f"accept: bootstrap_output_level {level} is off the "
                             f"composite-degree grid (must be a multiple of {grid})")
        self._encode_level = int(level)
        return self._encode_level

    def manifest(self):
        """The bundle manifest this client's keys were generated under."""
        m = options_to_manifest(self.options, self.family)
        m.update({
            "perseus_version": __version__,
            "chain": current_chain(),
            "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "bootstrap_output_level": int(self.encode_level),
        })
        return m

    def save_bundle(self, path: str) -> str:
        """Write the server's bundle: context + public + eval keys + bundle.json.
        No secret key."""
        os.makedirs(path, exist_ok=True)   # save_keys writes INTO the dir
        with _translate(self.backend):
            self.backend.save_keys(self.inf, path)
        with open(os.path.join(path, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(self.manifest(), f, indent=1)
        return path

    def save_secret_key(self, path: str) -> str:
        """Write the secret key (owner-only permissions). This file never leaves the
        client."""
        with _translate(self.backend):
            self.backend.save_secret_key(self.inf, path)
        os.chmod(path, 0o600)
        return path

    def encrypt(self, values) -> bytes:
        """Plaintext vector (<= d real features; shorter is zero-padded) -> ciphertext
        bytes (the request payload), encoded at `encode_level`."""
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"encrypt: expected a 1-D vector of up to {self.d} features, "
                             f"got shape {x.shape}")
        if x.shape[0] > self.d:
            raise ValueError(f"encrypt: {x.shape[0]} features do not fit the session's "
                             f"{self.d}-wide token (values past {self.d} would be dropped)")
        if not np.isfinite(x).all():
            raise ValueError("encrypt: input contains NaN/inf")
        ext = self.backend
        with _translate(ext):
            ct = ext.pack_tokens(self.inf, x[None, :], int(self.encode_level))
            return ext.serialize_ct(self.inf, ct)

    def decrypt(self, blob: bytes, d: int | None = None) -> np.ndarray:
        """Ciphertext bytes (the response payload) -> plaintext vector. d trims to the
        first d lanes (default: every lane, padding included)."""
        ext = self.backend
        with _translate(ext):
            out = np.array(ext.decode_token_output(self.inf, ext.deserialize_ct(self.inf, blob)))
        return out[:d] if d is not None else out


class EncServer:

    def __init__(self, bundle: str, options=None, profile=None, family: str | None = None,
                 strict: bool = True):
        from .. import _core  # the server role is _core-only (ImportError with the recipe)
        self.bundle = bundle
        self.manifest = None
        mpath = os.path.join(bundle, MANIFEST_NAME)
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                self.manifest = json.load(f)
            fam = self.manifest.get("family", "gpt2")
            if family is not None and family != fam:
                raise BundleError(f"bundle {bundle!r} was keyed for family {fam!r}, not "
                                  f"{family!r}: the rotation-key band would not match")
            if options is not None or profile is not None:
                given = options_to_manifest(_resolve_options(options, profile, _core), fam)
                diff = manifest_diff(self.manifest, given)
                if diff and strict:
                    raise BundleError(f"bundle {bundle!r} disagrees with the caller's options "
                                      f"on {len(diff)} field(s):\n  " + "\n  ".join(diff))
                if diff:
                    warnings.warn(f"EncServer: using the bundle's parameters over the "
                                  f"caller's on: {', '.join(d.split(':')[0] for d in diff)}",
                                  stacklevel=2)
            opts = options_from_manifest(self.manifest, ext=_core)
        else:
            fam = family or "gpt2"
            if options is None and profile is None:
                warnings.warn(f"EncServer: {bundle!r} has no {MANIFEST_NAME}; taking the "
                              f"CKKS parameters from the environment (save the bundle with "
                              f"EncClient.save_bundle to fix this)", stacklevel=2)
            opts = _copy_options(_resolve_options(options, profile, _core), fam, _core)
        self.family = fam
        opts.ckks.keys_dir = bundle        # deserialize keys; no keygen
        opts.ckks.skip_gpu_load = False
        opts.mode = _core.InferenceMode.Sync
        self.options = opts
        self._core = _core
        self.inf = getattr(_core, _FAMILIES[fam])(opts)
        if self.inf.fhe.has_secret_key:
            raise SecurityError(f"server session built from {bundle!r} holds a secret key; "
                                f"a server must never be able to decrypt")
        want = (self.manifest or {}).get("bootstrap_output_level")
        have = int(self.inf.fhe.bootstrap_output_level())
        self.client_encode_level = want
        self.fresh_encode_level = have
        if want is not None and int(want) != have:
            log.info("EncServer: the bundle says the client encodes at %s but this session "
                     "probes %s; hand EncClient.accept(server.session_manifest()) to the "
                     "client so strict plans see the probed level (eager runs are "
                     "unaffected)", want, have)

    def __repr__(self):
        return f"EncServer(bundle={self.bundle!r}, family={self.family!r})"

    def session_manifest(self) -> dict:
        m = dict(self.manifest or {})
        m["bootstrap_output_level"] = int(self.fresh_encode_level)
        return m

    def run(self, model, blob: bytes) -> bytes:
        """Request bytes -> model (bound to this session) -> response bytes."""
        y = model(self._core.deserialize_ct(self.inf, blob))
        return self._core.serialize_ct(self.inf, y)
