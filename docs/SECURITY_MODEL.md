# Security model

What is protected, who holds what, and what is *not* claimed. This is the contract the
`perseus.nn.EncClient` / `EncServer` roles implement; the single-process authoring
surface (`perseus.session()` + `perseus.nn`) holds every role in one process and is for
development, calibration and benchmarking, not for deployment across a trust boundary.

## Parties

| party | holds | can |
|---|---|---|
| **client** (`EncClient`) | the CKKS **secret key**, public key, evaluation keys; the tokenizer and embedding tables (`client.npz` / the HF checkpoint) | generate keys, encrypt inputs, decrypt outputs |
| **server** (`EncServer`) | the **bundle** (context, public key, relinearization and rotation keys, `bundle.json`); the model weights **in the clear** | evaluate the model on ciphertexts; nothing else |

The secret key never leaves the client process. `EncServer` refuses to start on a session
that holds one (`perseus.errors.SecurityError`), and `EncClient.save_bundle` never writes it.
`EncClient.save_secret_key` writes a raw OpenFHE key file with owner-only permissions
(0600); protect it like any other private key (encrypted volume, KMS) — the library adds no
passphrase layer.

## What crosses the wire

- **client → server, once:** the bundle. Context parameters, public key, evaluation keys
  (the rotation-key band is the large part — tens of GB for the GPT-2 band), and
  `bundle.json`: the CKKS parameters, model widths and packing the keys were generated
  under, plus the level the client's fresh encodes land on. The server rebuilds its
  session **from the manifest**, so a server whose environment disagrees with the client
  fails at load with a field-by-field diff (`perseus.errors.BundleError`) instead of
  mid-bootstrap.
- **server → client, once:** the session manifest (`EncServer.session_manifest()`): the
  bundle manifest echoed back with `bootstrap_output_level` set to the level the server's
  context lands fresh ciphertexts on (a GPU-less client can only compute the parameter
  formula; the server probes the real value). Public information (a parameter of the
  context, not of any key or input); `EncClient.accept()` adopts it.
- **client → server, per request:** one serialized ciphertext (the encrypted input
  vector; ~12 MB at logN=16).
- **server → client, per response:** one serialized ciphertext (the encrypted output).

Ciphertexts carry no plaintext metadata beyond CKKS's public parameters (level, scale,
noise degree). The manifest is public information (it is exactly what `CKKSOptions`
exposes).

## What the server learns

- The **model**: weights are encoded as CKKS *plaintexts* on the server (`Inference.set_weight`
  → `encode_weight_matrix`), never encrypted. Only activations are ciphertexts. This library
  provides **input/output privacy for the client**, not model privacy for a model owner
  who is not the server.
- The **shape** of the computation: which modules run, in what order, the packed widths,
  the number of tokens, timing.
- Nothing about the **values** of inputs, activations, or outputs, under the CKKS
  assumption (RLWE hardness at the configured parameters — `HEStd_128_classic` is enforced
  at context creation and the run refuses parameter sets outside it).

## What is not claimed

- **No verifiable computation.** A malicious server can return any ciphertext; the client
  cannot tell a correct evaluation from a wrong one without redundancy of its own.
- **No protection of the decrypted result from the client's own process.** `generate()`
  with `feedback="encrypted"` keeps the *server's* feedback loop inside the ciphertext
  domain, but the single-process harness decrypts every step's logits to report the token
  — deploying autoregressive generation across the boundary is not implemented.
- **Approximate arithmetic.** CKKS is approximate; every nonlinearity is a fitted
  polynomial (see `perseus.calibrate`). Correctness is statistical (the README quotes the
  measured top-1 / KL bands), not bit-exact.
- **Circuit privacy / IND-CPA-D.** CKKS decryption results can leak information about the
  secret key if decrypted values are shared with the server (Li–Micciancio). The client
  must treat decrypted outputs as sensitive with respect to the server, exactly as with
  any CKKS deployment.

## Research taps

`_core.decrypt_slots`, `_core.decode_linear_output` and the `decode_*` helpers decrypt with
the process's secret key; they exist for calibration and diagnostics and are only callable
in a session that holds the key. They are not reachable from an `EncServer` session (no
key), which is the invariant the `SecurityError` check protects.

## What the process writes

Context creation prints the CKKS parameter table to stderr; nothing in it is derived from
the secret key, and no code path in this tree logs or serializes key material.

A **capture run is different**, and it is the one thing to keep off a machine that holds
real inputs. `STAGE=capture` decrypts every intermediate value to record its magnitude,
because the planner prices a refresh from the size of the data it will refresh. Those
magnitudes, one pair of numbers per operation, are written into `graph.json` next to the
op that produced them. That is a profile of the activations the capture ran on. Capture
with the calibration inputs, not with someone's data, and treat a captured graph as
derived from its input rather than as public.
