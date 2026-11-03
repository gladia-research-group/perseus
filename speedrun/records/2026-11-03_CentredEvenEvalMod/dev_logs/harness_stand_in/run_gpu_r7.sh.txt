#!/bin/bash
# run_gpu_r7.sh: the A/B of the even EvalMod entry against Decode-8 record 7 (FlippingToTokenPerSecond, speedrun
# e805f8d, 0.968 s/token) on a rented RTX PRO 6000. The entry = artefatos/r7/perseus-r7-even.bundle, branch em-gelu-even: record 7
# + the even EvalMod series (PC_EM_EVEN=1) + plan _gnabd7_ee served by speedrun/server.py (same placements, landings
# and keys as record 7). Both arms run with record 6's client (perseus_spchain_client.py pinned to ea62bcc), as record 7
# does. Everything lands in $OUT; nothing is pushed anywhere. Run it inside tmux.
#   BUNDLE=/path/perseus-r7-even.bundle NOSANDBOX=1 bash run_gpu_r7.sh [STEP]
# STEP: preflight | setup | bts | smoke | decode | prefill | abrunner | all (each step stops the script on failure)
# NOSANDBOX=1 on a container rental (RunPod): the harness's clock, protocol, KL and score.py through gpu/ab/sandbox.sh,
# NOT harness runs (run.json harness: null, score.py --official refuses them). NOT RUN YET (written on a CPU-only box).
set -euo pipefail
STEP=${1:-all}
ROOT=${ROOT:-$HOME/perseus-entry}
# NOSANDBOX=1: the A/B on a container rental (RunPod) through an out-of-repo copy of the runner whose sandbox.sh is
# gpu/ab/sandbox.sh (no isolation): the harness's own clock, protocol, KL and score.py, but NOT harness runs (their
# run.json has no harness commit, so score.py --official refuses them); results go to a separate out-ab/
NOSANDBOX=${NOSANDBOX:-0}
AB_SANDBOX=${AB_SANDBOX:-$(cd "$(dirname "$0")" && pwd)/ab/sandbox.sh}
[ "$NOSANDBOX" = 1 ] && OUT=${OUT:-$ROOT/out-ab}
OUT=${OUT:-$ROOT/out}
GPU=${GPU:-0}
PAIRS=${PAIRS:-6}                 # Decode-8 record/entry pairs (CLAUDE.md: alternate on the same idle GPU)
REQUESTS=${REQUESTS:-4}           # dev prompts per run; pair i takes prompts (i-1)*REQUESTS .. i*REQUESTS-1 (32 in dev)
RECORD_SHA=e805f8d6f4529197788d1b5a98a1af88c3895622        # origin/speedrun: Decode-8 record 7, Prefill-128 record 6
ENTRY_SHA=${ENTRY_SHA:-83e8f991eb12bbbe14be31a5a7d3e930cef22ac5}   # em-gelu-even: the entry the bundle carries
BRANCH=${BRANCH:-em-gelu-even}
CLIENT_SHA=89bc830017fe12ab967ddb10c24feae65d8af6df        # record 1's client (exports the weights)
LEAN_SHA=ea62bcc3bd7f052d8005e9039c74cc476f19f364          # record 6's alternating-chain client (records 6-7, both tracks)
WEIGHTS_REL=../perseus-client/.cache/models/openai-community/gpt2/classic/weights.bin.zip
export HF_HOME=${HF_HOME:-$ROOT/hf} RUNS=$OUT/runs
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
export PC_NVCC_JOBS=${PC_NVCC_JOBS:-$(( $(nproc) > 4 ? $(nproc) / 2 : 2 ))}
mkdir -p "$OUT" "$RUNS"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$OUT/run_gpu.log"; }
die() { log "FAILED: $*"; exit 1; }
MEMPID=
trap '[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null; true' EXIT

preflight() {
    # everything the harness needs, checked for real before an hour of builds (relatorio B5)
    log "preflight"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
    local mib
    mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ "$mib" -ge 48000 ] || die "GPU $GPU has $mib MiB; the record needs >= 48 GB"
    command -v nvcc >/dev/null || die "nvcc missing (the CUDA 13.x toolkit)"
    nvcc --version | /usr/bin/grep -q 'release 13\.' || die "nvcc is not CUDA 13.x: $(nvcc --version | tail -1)"
    case "$(readlink -f "$(command -v nvcc)")" in
        /usr/*) ;;
        *) die "nvcc at $(readlink -f "$(command -v nvcc)"): the harness sandbox mounts /usr only (CUDA must live under /usr)" ;;
    esac
    if [ "$NOSANDBOX" = 1 ]; then
        log "preflight: NOSANDBOX=1, the A/B outside the harness's sandbox (gpu/ab/sandbox.sh): NOT harness runs"
        command -v git >/dev/null && command -v curl >/dev/null && command -v python3 >/dev/null || die "git / curl / python3 missing"
        log "preflight ok (A/B): $(head -1 "$OUT/gpu.txt"), nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
        return
    fi
    command -v bwrap >/dev/null || die "bwrap missing"
    # the same mechanisms speedrun/runner/sandbox.sh uses: a systemd user SERVICE with a memory cap, and inside it bwrap
    # with every namespace unshared (user namespace included); a Docker-style container usually fails one of the two
    bwrap --unshare-all --unshare-user --disable-userns --cap-drop ALL --die-with-parent --ro-bind / / --dev /dev \
        --proc /proc true || die "bwrap cannot unshare namespaces here (container? unprivileged user namespaces off?)"
    systemd-run --user --wait --pipe --collect -q -p MemoryMax=1G -p MemorySwapMax=0 true \
        || die "no working systemd user service manager (systemd-run --user --wait)"
    command -v git >/dev/null && command -v curl >/dev/null && command -v python3 >/dev/null || die "git / curl / python3 missing"
    log "preflight ok: $(head -1 "$OUT/gpu.txt"), nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
}

setup() {
    log "setup in $ROOT"
    export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v rustup >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    cd "$ROOT"
    [ -d perseus ] || git clone -q -b speedrun https://github.com/gladia-research-group/perseus
    cd perseus
    git fetch -q origin
    git cat-file -e "$RECORD_SHA" || die "origin lacks $RECORD_SHA"
    # re-runnable (relatorio B3): fetch the bundle's branch to FETCH_HEAD, never into the checked-out branch
    git fetch -q "${BUNDLE:?BUNDLE=perseus-r7-even.bundle}" "$BRANCH"
    if [ "$(git branch --show-current)" = "$BRANCH" ]; then
        git merge -q --ff-only FETCH_HEAD || die "$BRANCH has local commits the bundle lacks"
    else
        git branch -f "$BRANCH" FETCH_HEAD && git switch -q "$BRANCH"
    fi
    [ "$(git rev-parse HEAD)" = "$ENTRY_SHA" ] || die "entry HEAD $(git rev-parse HEAD), expected $ENTRY_SHA"
    git merge-base --is-ancestor "$RECORD_SHA" HEAD || die "the entry is not on top of record 7 ($RECORD_SHA)"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "the entry checkout has local changes"
    log "entry at $(git rev-parse --short HEAD), on record 7"
    [ -d ../perseus-client ] || git worktree add -q ../perseus-client "$CLIENT_SHA"
    [ -d ../perseus-spchain-client ] || git worktree add -q ../perseus-spchain-client "$LEAN_SHA"
    [ -d ../perseus-record ] || git worktree add -q --detach ../perseus-record "$RECORD_SHA"
    local d
    for d in . ../perseus-client ../perseus-spchain-client ../perseus-record; do
        log "build $d"
        (cd "$d" && { [ -d .venv ] || uv venv -q --python 3.12 .venv; } &&
         uv pip install -q --python .venv/bin/python -e ".[hf,examples,dev]" &&
         .venv/bin/maturin develop --release > "$OUT/build_$(basename "$(pwd)").log" 2>&1) || die "build of $d (see $OUT/build_*.log)"
    done
    [ -f "$WEIGHTS_REL" ] || (cd ../perseus-client && .venv/bin/perseus-export --model openai-community/gpt2 \
        --out .cache/models/openai-community/gpt2) || die "weights export"
    [ "$NOSANDBOX" = 1 ] && ab_runner
    log "setup done"
}

ab_runner() {
    # NOSANDBOX=1: an out-of-repo copy of record 7's speedrun/{runner,harness,prompts} (the entry's harness is the same
    # code) with gpu/ab/sandbox.sh over runner/sandbox.sh, and run.py's stop() made to end the process itself (it
    # stops a systemd unit, which the stand-in never creates). The checkouts' own speedrun/ stays untouched.
    local src=$ROOT/perseus-record/speedrun dst=$ROOT/ab/speedrun
    [ -f "$AB_SANDBOX" ] || die "no A/B sandbox stand-in at $AB_SANDBOX"
    mkdir -p "$dst"
    cp -rT "$src/runner" "$dst/runner" && cp -rT "$src/harness" "$dst/harness" && cp -rT "$src/prompts" "$dst/prompts"
    cp "$AB_SANDBOX" "$dst/runner/sandbox.sh"
    python3 - "$dst/runner/run.py" <<'PY' || die "patching the A/B runner's stop()"
import sys
p = sys.argv[1]; s = open(p).read()
old = '''def stop(unit, proc):
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    proc.wait()'''
new = '''def stop(unit, proc):
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    # A/B stand-in (gpu/ab): no systemd unit exists, so end the process here
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()'''
assert s.count(old) == 1, "run.py stop() changed upstream"
open(p, "w").write(s.replace(old, new))
PY
    log "A/B runner at $dst (sandbox stand-in, NOT the harness's isolation)"
}

bts() {
    # the bootstraps the served decode runs, shipped EvalMod against the even one, on record 7's chain and switches:
    # n32sp, fused relin-rescales, SparseRot merge, StC-first over P'', SparseB on the K = 16 table (pc-bench has no
    # EvalMod level keys: the decode A/B below has them). ms per stage (evalmod row) and per route, K = 1 and 2; then the
    # dense route's precision (bts-prec, n32) for both series.
    cd "$ROOT/perseus"
    export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}
    cargo build -q --release -p pc-tools --bin pc-bench || die "pc-bench build"   # incremental: never a stale binary
    local ev name
    for ev in 0 1; do
        [ "$ev" = 1 ] && name=even || name=shipped
        log "pc-bench bts --all (n32sp, served switches), EvalMod $name"
        CUDA_VISIBLE_DEVICES=$GPU PC_EM_EVEN=$ev PC_BENCH_CHAIN=n32sp PC_EM_FUSE_RESCALE=1 PC_BENCH_MERGE=1 \
            PC_BENCH_STC_PS=6 PC_BENCH_SPARSE_B_K=16 target/release/pc-bench bts --all --keys packed --k 1,2 --reps 20 \
            --routes stcf,256,512 > "$OUT/bts_all_n32sp_$name.log" 2>&1 || die "bts --all $name (see $OUT/bts_all_n32sp_$name.log)"
        log "pc-bench bts-prec dense (n32), EvalMod $name"
        CUDA_VISIBLE_DEVICES=$GPU PC_EM_EVEN=$ev PC_EM_FUSE_RESCALE=1 target/release/pc-bench bts-prec --route dense \
            > "$OUT/bts_prec_dense_$name.log" 2>&1 || die "bts-prec $name (see $OUT/bts_prec_dense_$name.log)"
    done
    log "bts done: compare $OUT/bts_all_n32sp_{shipped,even}.log (evalmod ms) and bts_prec_dense_{shipped,even}.log (bits)"
}

run_one() {   # run_one record|entry decode|prefill OFFSET -> prints the new run.json; returns 1 if the run failed
    # (inside $(...) bash drops errexit, so every failure is returned explicitly: relatorio B1)
    local who=$1 t=$2 off=$3 dir before after
    [ "$who" = record ] && dir=../perseus-record || dir=.
    before=$(/usr/bin/ls -1d "$RUNS"/*/ 2>/dev/null | sort || true)
    local entry=speedrun/runner/entry.sh
    [ "$NOSANDBOX" = 1 ] && entry=$ROOT/ab/speedrun/runner/entry.sh
    if ! CLIENT=../perseus-spchain-client WEIGHTS=$WEIGHTS_REL bash "$entry" perseus-spchain "$dir" "$t" "$GPU" \
            --requests "$REQUESTS" --offset "$off" >> "$OUT/${t}_$who.log" 2>&1; then
        return 1
    fi
    after=$(comm -13 <(echo "$before") <(/usr/bin/ls -1d "$RUNS"/*/ 2>/dev/null | sort) | tail -1)
    [ -n "$after" ] && [ -f "${after%/}/run.json" ] || return 1
    echo "${after%/}/run.json"
}

score() {   # score TRACK OUTNAME --entry ... --record ...: never aborts the script (relatorio B2); logs the verdict
    local t=$1 name=$2 rc=0
    shift 2
    python3 speedrun/harness/score.py --track "$t" "$@" --json "$OUT/$name.json" > "$OUT/$name.txt" 2>&1 || rc=$?
    cat "$OUT/$name.txt"
    case $rc in
        0) log "$name: a new record that passes (score.py exit 0)" ;;
        1) log "$name: does NOT beat the record / fails a check (score.py exit 1)" ;;
        *) log "$name: runs refused (score.py exit $rc)" ;;
    esac
}

smoke() {
    # one entry run alone before the pairs: the first time the even EvalMod serves
    cd "$ROOT/perseus"
    log "smoke: one Decode-8 entry run"
    local j
    j=$(run_one entry decode 0) || die "smoke: the entry's Decode-8 run failed (see $OUT/decode_entry.log)"
    log "smoke ok: $j"
}

track() {   # track decode|prefill N: N record/entry pairs alternated (record, entry, entry, record, ...)
    local t=$1 n=$2 rec=() ent=() i who order j off
    cd "$ROOT/perseus"
    # the GPU's used memory every 0.5 s (nvidia-smi itself in the background; the EXIT trap kills it: relatorio B8)
    nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/gpu_mem_$t.csv" &
    MEMPID=$!
    for i in $(seq 1 "$n"); do
        [ $((i % 2)) = 1 ] && order="record entry" || order="entry record"
        off=$(( (i - 1) * REQUESTS ))
        for who in $order; do
            log "$t pair $i/$n: $who (prompts $off..$((off + REQUESTS - 1)))"
            j=$(run_one "$who" "$t" "$off") || die "$t $who run $i (see $OUT/${t}_$who.log)"
            [ "$who" = record ] && rec+=("$j") || ent+=("$j")
        done
    done
    kill "$MEMPID" 2>/dev/null || true
    MEMPID=
    log "$t peak GPU memory $(cut -d, -f2 "$OUT/gpu_mem_$t.csv" | sort -n | tail -1) MiB"
    score "$t" "score_$t" --entry "${ent[@]}" --record "${rec[@]}"
}

case "$STEP" in
    preflight) preflight ;;
    setup) preflight; setup ;;
    bts) bts ;;
    smoke) smoke ;;
    decode) track decode "$PAIRS" ;;
    abrunner) ab_runner ;;          # NOSANDBOX=1: (re)build only the A/B runner copy (setup does it too)
    prefill) track prefill 2 ;;     # must still run on this commit: PC_EM_EVEN is off for Prefill-128
    all) preflight; setup; bts; smoke; track decode "$PAIRS"; track prefill 2 ;;
    *) echo "step: preflight | setup | bts | smoke | decode | prefill | abrunner | all"; exit 2 ;;
esac
log "step $STEP done; results in $OUT"
