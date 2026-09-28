"""``perseus.impl``: build encrypted models from the leaf primitives of ``perseus._core``.

The C++ runtime ships composites (``_core.linear``, ``_core.attention_softmax_thor``, ...);
this package is the same algorithms written in Python on ``Context.{add, sub, mult, square,
negate, rotate, rotate_and_sum, conjugate, bootstrap, bootstrap_hint}`` and the plaintext
operands of ``Inference``, plus the runtime services a model needs to run at C++ speed: the
residency ring for weights, the encode cache with worker staging for masks, graph capture and
planned bootstrapping. A model is a ``ImplModel`` subclass that names its stages, its
per-step masks and its weights; examples/gpt2_from_primitives is the reference model.

    ops         the ops protocol: FheOps (session or fake), NumpyOps (plaintext mirror)
    poly        polynomial / iterative kernels (Chebyshev, power-sum, Remez, Goldschmidt,
                Newton, odd powers, ladders, 2-iteration bootstrap)
    layout      the cachemir geometry (Dims, cm_params, interleave), weight / bias slot
                vectors, every selection mask
    linear      the BSGS linear (prepare + apply, EncodedLinear)
    norm        norm(), ln_affine(), ln_shift(), layer_norm(), norm_step_masks()
    attention   KVCache, cache pushes, qkt, head_reduce_sum, softmax_thor, softmax_v, mha(),
                attention_step_masks()
    activation  THOR composite GELU
    config      configs.json -> NormCfg / SoftmaxCfg / GeluCfg / CutMaxCfg
    rt          Rt: ops + dims + step scopes + the mask discipline (cache, stage, evict)
    driver      ImplModel: modes (eager / capture / planned), the residency ring, per-token
                bookkeeping
    profile     TimedOps: per-primitive and per-step wall time
    fake        a numpy fake of the _core surface (level accounting, strict mode) for CPU tests
"""
