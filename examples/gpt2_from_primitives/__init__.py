"""GPT-2 encrypted decode / generate rebuilt from the leaf primitives of ``perseus._core``.

The C++ runtime (src/model/gpt2, src/algorithms, src/packing/cachemir) is re-expressed in
Python on top of ``Context.{add,sub,mult,square,negate,rotate,rotate_and_sum,conjugate,
inplace_add,bootstrap,bootstrap_hint}`` and the plaintext operands of ``Inference``: the cachemir packing,
the BSGS linear, the K/V cache, the THOR softmax, the Remez/Newton LayerNorm, the THOR GELU,
the tiled LM head, the CutMax argmax and the encrypted feedback embedding.  Every module cites
the C++ it ports so the two can be read side by side, and ``ref.py`` mirrors the same
approximations in numpy so FHE noise and approximation error stay separable.

Modules (the model-independent layer lives in ``perseus.impl``: ops protocol, kernels,
layout, linear, norm, attention, activation, config, Rt, ImplModel, fake)
    weights     export zip -> padded / folded / rearranged matrices -> EncodedLinear
    block       transformer_block()
    head        final_ln, lm_head, decode_logits, cutmax_argmax, feedback_embed
    model       Gpt2Primitives(ImplModel): stages, per-step masks, decode_token /
                argmax_encrypted / feedback / run_decode / generate
    ref         the plaintext mirror (norm_ref, gelu_ref, softmax_ref, attention_ref, cutmax_ref)
    env         the session env (the C++ n32 decode preset, folded ln_1/ln_2) and factory

Packing. The default (``--packing cachemir_complex`` on the complex payload,
``CKKS_COMPLEX=1``) carries two real payloads per ciphertext wherever the decode has a pair:
the fused K + iV projection refreshed by ONE bootstrap, K groups paired into complex buckets
and V lanes into pair buckets (``complex_qkt`` / ``complex_softmax_v``), the output-packed up-
and down-projections and the paired LM-head tile. Every multiply by +-i is the level-free
monomial (``Context.mult_i``), and an output-packed linear carries its unpack's 1/2 in the
weights, so no lane construction costs a level. ``--packing cachemir`` is the C++ decode's
configuration: real weights and linears, the complex payload only in the K/V push
(``cache_kv_push_pair``), CutMax (``cutmax_argmax_packed``) and the feedback tile.
``--payload real`` runs on real slots only. The shipped capture and plan are the default
packing; any other configuration needs its own.

Fused reductions. Off by default (``FUSED_SM_DEN=0``, ``FUSED_LN_VAR=0``): the softmax
denominator and the LayerNorm variance are reduced by their rotation ladders. ``=1`` finishes
the ladder inside a sparse "fold" bootstrap instead (``head_reduce_sum`` / ``norm``; the C++
decode's default for the softmax). Both change the op stream, so a plan is bound to them.

Running (GPU 3 must be free of other users' processes; the drivers refuse otherwise)::

    source scripts/local_env.sh
    .venv/bin/python -m examples.gpt2_from_primitives.run_decode --tokens 16 [--argmax]
    .venv/bin/python -m examples.gpt2_from_primitives.run_generate --prompt 1 --tokens 4

The three modes, the same loop the C++ / EncGPT2 run:

    eager     reactive bootstraps plus the C++ hint sites (default)
    capture   ``--capture graphs/gpt2_decode_python_n32`` writes block_<b>/graph.json for token 0
              (blocks 0..11, 12 = final LN + LM head, 13 = CutMax, 14 = feedback) and the
              capture contract; ``bash examples/gpt2_from_primitives/make_plan.sh
              graphs/gpt2_decode_python_n32 gpt2_decode_python_n32`` plans it (CPU)
    planned   ``--plan bootstrap_placements/gpt2_decode_python_n32`` installs block_<b>_placement.json
              live per block (strict: the op sequence must match the capture); CutMax is planned
              too when its plan is present; ``--no-plan-argmax`` runs it eager

Planning CutMax (block 13) is a second round, ``make_plan.sh <graph> <name> argmax``, from
the same eager capture. Its entry is taken from the tail plan's exit (level and degree)
rather than from the capture: what enters CutMax is what the planned tail produces, and a
plan made from the capture's own levels fails the strict level check at its first op. Its
hints are dissolved, which makes the refresh envelope hard: planned live they fire at the
top of the chain, where a bootstrap returns garbage silently. The eager capture's only
inexactness is the site magnitudes, which are the eager tail's: against a capture taken
under the block plans the 51 sites and their routes come out identical and 5 correction
factors differ by one or two. The feedback (block 14) is not planned; it runs eager.

Tests: ``tests/test_gpt2_primitives_cpu.py`` (numpy fake, no GPU) and
``tests/gpu/test_gpt2_primitives.py`` (each op against its C++ composite on one session).
"""
