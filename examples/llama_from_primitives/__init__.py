"""Llama-3 (8B) encrypted decode on ``perseus.impl``: the GPT-2 port's kernels, packing and
runtime levers (examples/gpt2_from_primitives: cachemir_complex packing, sparse routes, SparseB,
the softmax / norm switches, capture -> plan -> planned) applied to the Llama architecture.

What is Llama-specific, and where it lives
    RMSNorm     perseus.impl.norm.rms_norm: the LayerNorm pipeline minus the mean (centering
                scale, all-slot ladder, Chebyshev seed + Newton inverse sqrt); gamma / inv_out_scale
                folded into the next linears (weights.py)
    RoPE        perseus.impl.attention.Rope: in the head-interleaved layout a lane's rotary
                partner is N/2 slots away, so q and K get x cos + rotate(x, N/2) sin' with cos /
                sin' folded into the selection masks they are multiplied by anyway: one rotation,
                no level. The masks are per position (step masks)
    GQA         the 8 KV heads are expanded to the 32 query heads in the K / V weights
                (weights.expand_kv): the attention core is the MHA one
    SwiGLU      gate and up from one prepared input (output-packed), SiLU as the THOR composite
                x (p2(p1(x / xmax)) + 1/2) of perseus.impl.activation with sigmoid coefficients
                (1/xmax folded into the gate), then the product and down
    BOS prefix  position 0 carries Llama's massive activations; its K / V are public constants,
                computed exactly in plaintext and written into the caches as encryptions of
                public vectors (prefix.py); the encrypted decode starts at position 1
    weights     ~4.6k plaintexts per block, never resident: their coefficient rows (the plaintext
                polynomials / scale, float32, built once next to the slot-vector cache) sit in pinned
                host memory (~40 GB) and every baby-step sum is encoded and multiplied on the GPU in
                batches (Inference.mult_coeff_sum, FIDESlib CoeffRows); the slot vectors stay for the
                capture and the fake (weights.LlamaWeights, model.block_weights)
    routes      the attention block is tH = t H = 8 x 32 = 256 slots (GPT-2's is 512): sparse routes
                {256, 1}, the periodic softmax at every length (SM_PERIODIC=2, the Gp x 256-slot
                route built as the cache grows: model.routes_for)

Modules
    hf          the HF checkpoint, text windows, per-site statistics, the fp32 oracle logits
    calibrate   configs.json: RMSNorm seeds, SiLU composites, THOR softmax with data-sized
                divisions (configs/model/approximation/llama31_8b_n32)
    config      configs.json -> perseus.impl config structs
    ref         the plaintext mirror (HF model with the approximations patched in): KL of the
                approximations alone
    weights / prefix / block / model / env / run_decode

Running (GPU 3 free of other users' processes)::

    source scripts/local_env.sh
    W=<token window .npy>; G=graphs/llama31_8b_decode_python_n32; P=llama31_8b_decode_python_n32
    # capture (structure only) -> magnitudes from the numpy fake -> plan -> planned decode
    .venv/bin/python -m examples.llama_from_primitives.run_decode --tokens 1 --window $W --kv-offload \
        --set KV_ARENA_GB=56 --set FIDESLIB_BTS_RAISE_DROPS=1,3,5 --capture $G --structure-only
    .venv/bin/python -m examples.llama_from_primitives.capture_mags $G --window $W
    bash examples/llama_from_primitives/make_plan.sh $G $P
    .venv/bin/python -m examples.llama_from_primitives.run_decode --tokens 8 --window $W --kv-offload \
        --set KV_ARENA_GB=56 --plan bootstrap_placements/$P

The planned decode runs ~19.5 s/token (Blackwell PRO 6000, 4 tokens, KL <= 1.6e-7, top-1 4/4).
"""
