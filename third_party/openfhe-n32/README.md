# OpenFHE patches for the 32-bit chain

`patches/` is the OpenFHE series the 32-bit composite chain is built from, and the
one every shipped plan was measured against. `scripts/install_deps.sh` applies it with
`git am` when `NATIVE_SIZE=32`; the 64-bit reference chain needs none of it and builds from
OpenFHE plus FIDESlib's own patches.

| field | value |
|---|---|
| upstream | `https://github.com/openfheorg/openfhe-development` |
| base commit | `aa391988d354d4360f390f223a90e0d1b98839d7` |
| series | `patches/0001`–`0012` |

What the twelve patches do, in order of dependence:

1. **0001** compatibility surface FIDESlib needs from the CryptoContext.
2. **0002–0004** mixed-size modulus chains: the bootstrap segment, the EvalMod output
   recovery (`scaleDec`), and composite-style bootstrap constants.
3. **0005–0006** 32-bit native integers: the encode path's coefficient truncation and the
   bootstrap DFT encoding under composite scaling.
4. **0007** seed expansion for the random half of a key-switching key, the OpenFHE side of
   the in-kernel regeneration (its header must stay byte-equal to FIDESlib's
   `KskSeedExpand.cuh`; `install_deps.sh` checks this and refuses to build otherwise).
5. **0008–0009** exact arithmetic for moduli up to 2^31 and the matching `DoubleInteger`
   width cap.
6. **0010–0012** value and norm probes around q0 and the StC/CtS diagonals, off unless
   `OPENFHE_Q0_DIAG` is set. They are part of the series the measured binary was built
   from, so they stay in it.

Reconstruct the patched source by hand:

```bash
git clone https://github.com/openfheorg/openfhe-development openfhe-src
cd openfhe-src && git checkout -b perseus-n32 aa391988d354d4360f390f223a90e0d1b98839d7
git am /path/to/third_party/openfhe-n32/patches/*.patch
```

These patches are surgery for this chain's measurements, not a proposed upstream
contribution.
