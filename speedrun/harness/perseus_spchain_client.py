"""The alternating-chain entry's client crypto (Perseus on its Rust runtime): pc-client through perseus._pc, no GPU.

Used by harness/client.py --backend perseus-spchain, which harness/with_perseus.sh runs with that checkout's toolchain.
It is harness/perseus_keyset_client.py with one more chain, n32sp (pc-core `ChainSpec::n32sp`: n32a4 with its 9 user
levels replaced by 6 single primes of ~27.85 bits alternating with 6 pairs of ~2 x 26.15 bits, every prime below 2^28;
log QP 1729), whose keys are the same roles and shapes over its primes; the policy below is the keyset client's. Its
roles: harness/perseus_lean_client.py's (chains n32 and n32a4) and three more switching-key roles, every one an RLWE
sample under the dense uniform ternary main secret (sigma 3.19 errors) at a modulus inside Q x P:
- `sparse_rot` (s_old = the h = 32 sparse secret, s_new = sigma_g^-1(s)): the raise's sparse -> dense switch folded into
  the next hoisted rotations; full shape, modulus Q x P.
- `rot` over a prefix of P (P'' = the first 6 of n32's special primes) at q_top 4: StC-first's input stages; modulus
  q0..q4 x P''.
- `relin_level` (s_old = s^2, s_new = s): EvalMod's level relinearization keys, their special modulus P x the Q primes
  idle above the level; modulus q0..q_(q_top) x P.
So `refuse` sorts every role by the secret its samples are under (`SECRET`) and bounds each key's modulus, the product of
the primes its shape stores: under the dense secret at most log QP 1734 (n32a4's main-key modulus, <= 1747 on the HE
standard's ternary table at N = 2^16; n32sp's is 1729); under a sparse secret (h = 32 encapsulation, SPRU block key)
only at the bottom shape, q0 q1 x at most n32's 9 special primes (308 bits, record 1's Lattice Estimator verdict; n32sp
keeps q0 and those 9 primes). A role it does not know is refused. The client generates the keys the request lists,
after checking that it is for the chain it names (params_hash).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

PERSEUS = "ea62bcc3bd7f052d8005e9039c74cc476f19f364"     # the Perseus commit this client runs (runner/run.py)
CHAINS = ("n32", "n32a4", "n32sp")
VOCAB = 50257
SPRU_H = 64                                            # the SPRU block key: a single 1 in each of 64 blocks
SPARSE = ("enc_dense_to_sparse", "spru_atob")          # the switching keys to a sparse secret (Hamming weight 32, SPRU)
BOTTOM = 1                                             # their Q limbs: q0 and q1 (the bottom level)
P_SPARSE = 9                                           # and their special primes: n32's 9 (record 1's modulus q0 x P)
# the secret every role's samples are under (s_new of b = -a s_new + e + P g s_old): the dense uniform ternary main
# secret, or a sparse one (pc-client client.rs `keygen_in_pool`)
SECRET = {"relin": "dense", "relin_level": "dense", "rot": "dense", "conj": "dense", "auto": "dense",
          "enc_sparse_to_dense": "dense", "sparse_rot": "dense", "enc_dense_to_sparse": "sparse", "spru_atob": "sparse"}
LOG_QP_DENSE = 1734                                    # samples under the dense secret: at most n32a4's log QP


def parse(keyspec: bytes) -> dict:
    """The server's key request as pc-client reads it: a repeated JSON key, which two parsers may resolve differently,
    is refused."""
    def unique(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError(f"a repeated key in the key request: {[k for k, _ in pairs]}")
        return dict(pairs)
    return json.loads(keyspec, object_pairs_hook=unique)


def modulus_bits(shape: dict, prime_log2: list, counts: tuple) -> float | None:
    """log2 of the modulus a key of `shape` lives at: the product of the Q primes 0..q_top and the first kp special
    primes it stores (None for a shape outside the chain)."""
    nq, np_, _ = counts
    q_top, kp = shape.get("q_top"), shape.get("kp")
    if not isinstance(q_top, int) or not isinstance(kp, int) or not 0 <= q_top < nq or not 0 < kp <= np_:
        return None
    return sum(prime_log2[:q_top + 1]) + sum(prime_log2[nq:nq + kp])


def refuse(spec: dict, prime_log2: list | None = None, counts: tuple | None = None) -> str | None:
    """Why the client will not generate the keys `spec` asks for (None if it will): only the declared keys, each at
    128 bits where it is used (README Rule 2). Every key's samples are under the secret `SECRET` names for its role: a
    dense one at a modulus of at most `LOG_QP_DENSE` bits, a sparse one only at q0 x P' with P' n32's 9 special primes
    (`prime_log2` / `counts`: the chain's prime sizes and (Q, P, P') counts, from the client). The SPRU block key has
    64 blocks, and CCMM keys (ring 2^13) are not declared. (pc-client also refuses a key whose shape is not one the
    chain has.)"""
    if spec.get("chain") not in CHAINS:
        return f"chain {spec.get('chain')!r}, not one of {CHAINS}"
    spru = spec.get("spru")
    if spru is not None and (spru.get("h") != SPRU_H or spru.get("n") != 2):
        return f"a SPRU block key of {spru.get('h')} blocks, not {SPRU_H}"
    if spec.get("ccmm"):
        return "CCMM keys, which this entry does not declare"
    for k in spec.get("keys", []):
        role, shape = k.get("role"), k.get("shape", {})
        if role not in SECRET:
            return f"a key of role {role!r}, which this client does not declare"
        if SECRET[role] == "sparse":
            if not shape.get("q_top", BOTTOM + 1) <= BOTTOM:
                return f"a {role} key up to limb {shape.get('q_top')}, above the bottom level (limb {BOTTOM})"
            if not shape.get("kp", P_SPARSE + 1) <= P_SPARSE:
                return f"a {role} key over {shape.get('kp')} special primes, more than n32's {P_SPARSE}"
        if prime_log2 is not None:
            bits = modulus_bits(shape, prime_log2, counts)
            if bits is None:
                return f"a {role} key of shape {shape}, outside the chain"
            if SECRET[role] == "dense" and bits > LOG_QP_DENSE:
                return f"a {role} key at log modulus {bits:.1f}, above {LOG_QP_DENSE}"
    return None


class PerseusSpChain:
    def __init__(self, track: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the client never touches a GPU
        from perseus.impl.layout import Dims
        from speedrun import record1
        from perseus.client import pack_token_rows
        self.r1, self.pack_rows, self.d, self.track = record1, pack_token_rows, Dims.gpt2(), track
        self.client = self.level = self.metas = None

    def keys(self, path: str, keyspec: bytes) -> None:
        from perseus.client import Client, ClientSecret
        spec = parse(keyspec)
        if why := refuse(spec):
            raise RuntimeError(f"the server's key request asks for {why}")
        self.client = Client(ClientSecret.generate(spec["chain"]), chain=spec["chain"], complex=True)
        if spec.get("params_hash") != self.client.params_hash:
            raise RuntimeError(f"the server's key request is for another chain than {spec['chain']}")
        # every key's modulus, from the chain this client runs (the request's params_hash is that chain's)
        if why := refuse(spec, list(self.client.prime_log2), tuple(self.client.prime_counts)):
            raise RuntimeError(f"the server's key request asks for {why}")
        os.makedirs(path)
        self.client.keygen(keyspec.decode(), os.path.join(path, "keys.pckb"), threads=8)

    def accept(self, manifest: bytes) -> None:
        m = json.loads(manifest)
        if self.track == "decode":
            self.level = int(m["input"]["level"])
        else:
            self.metas = [(int(lv), int(dg)) for lv, dg in m["inputs"]]

    def encrypt(self, rows) -> list[bytes]:
        import numpy as np
        d = self.d
        if self.track == "decode":
            return [bytes(self.client.encrypt(self.pack_rows([r], d.N, d.hid, d.dim), self.level)) for r in rows]
        from perseus.impl.blocked import Blocks, blocked_slots
        from perseus.impl.prefill import Chunks
        B = Blocks(d, self.r1.NCH)
        vals = [v for blk in blocked_slots(np.asarray(rows), Chunks(len(rows), B.t), B) for v in blk]
        if len(vals) != len(self.metas):
            raise RuntimeError(f"{len(vals)} blocked inputs, the server takes {len(self.metas)}")
        return [self.r1.pack([bytes(self.client.encrypt(v, lv, dg)) for v, (lv, dg) in zip(vals, self.metas)])]

    def decrypt(self, outputs: list[bytes]) -> list[tuple]:
        import numpy as np
        from perseus.examples.gpt2.head import decode_logits, decode_z
        shim = SimpleNamespace(dims=self.d,
                               ops=SimpleNamespace(decrypt_slots_complex=lambda f: self.client.decrypt(f)[0]))
        out = []
        for payload in outputs:
            lg, z = self.r1.unpack(payload, most=2)
            z = decode_z(shim, [z], VOCAB, self.d.N)
            out.append((decode_logits(shim, [lg], VOCAB, self.d.N), int(np.argmax(z))))
        return out


Backend = PerseusSpChain
