"""Wave-2 numerics (impl.solvers, impl.matfun, impl.cluster, impl.learn, impl.signal) as planned stages, gated
against float64. The runner is examples/mathlib's (mirror / twin capture / eager / planned / --auto), driving this
package's tasks:

    python -m perseus.examples.wave2.run --task solve --set method=cg kappa=10 --mirror
    python -m perseus.examples.wave2.run --task solve --set method=chebyshev kappa=100 --auto out/cheb
    python -m perseus.examples.wave2.run --task logreg --set data=breast_cancer epochs=4 --auto out/logreg

`--auto DIR`: twin capture (DIR/graph) -> make_plan.sh (hard envelope, soft fallback; every refresh priced to
PLAN_ERR_TARGET=1e-3) -> the planned run on the device. One JSON line per run (`[wave2] RESULT {...}`)."""
from __future__ import annotations

import sys

from perseus.examples.mathlib import run as base

from . import tasks as T

ME = "perseus.examples.wave2.run"


def _report(a, task, out, mode, extra):
    import json
    met = task.metrics(out)
    ref = task.metrics(task.mirror()) if mode != "mirror" else met
    row = {"task": task.name, "mode": mode, "tag": a.tag, **task.describe(), **met,
           **{"mirror_" + k: v for k, v in ref.items()}, **extra}
    print("[wave2] RESULT " + json.dumps(row, default=float), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
    return row


def run_auto(a, argv):
    """base.run_auto with this module as the program (the capture and the run re-enter here), plus: `--capture-noise
    EPS` captures with the twin's refresh-noise model (FAKECAP_BTS_NOISE: magnitudes downstream of a refresh as the
    device will see them, e.g. a refinement pass's residual, which is ~0 on an exact twin), and a plan that does not
    compile (a refresh the final simulation finds past the envelope, P3c drift) is re-planned with PLAN_DRIFT_RECUT=1."""
    import os
    import subprocess
    from perseus._paths import ASSETS, PKG
    d = os.path.abspath(a.auto)
    graph, plan = os.path.join(d, "graph"), os.path.join(d, "plan")
    os.makedirs(d, exist_ok=True)
    rest = list(argv)
    i = rest.index("--auto")
    del rest[i:i + 2]
    for flag in ("--sparse", "--capture-noise"):
        if flag in rest:
            j = rest.index(flag)
            del rest[j:j + 2]
    me = [sys.executable, "-m", ME]
    cpu = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cap = [x for x in rest if x != "--compile-only"]
    if "--seeds" in cap:
        j = cap.index("--seeds")
        k = j + 1
        while k < len(cap) and not cap[k].startswith("--"):
            k += 1
        cap[j:k] = ["--seeds", cap[j + 1]] if k > j + 1 else []
    noise = ["--env", f"FAKECAP_BTS_NOISE={a.capture_noise}"] if a.capture_noise else []
    subprocess.run([*me, *cap, *noise, "--fake", "--capture", graph], check=True, env=cpu)
    bp = os.path.join(ASSETS, "bootstrap_placements")
    name = os.path.relpath(plan, bp)
    pe = os.environ.get("PLAN_ERR_TARGET", "1e-3")
    band = os.environ.get("PLAN_DELIBERATE_BAND_CF", "1")     # code refreshes past the default CF's band: priced
    caps = [os.environ["PLAN_HARD_ENV_CAP"]] if "PLAN_HARD_ENV_CAP" in os.environ else ["1", "0"]
    for recut in (os.environ.get("PLAN_DRIFT_RECUT", "0"), "1"):
        for cap_ in caps:
            r = subprocess.run(["bash", os.path.join(PKG, "examples", "gpt2", "make_plan.sh"), graph, name],
                               env=dict(cpu, PLAN_SPARSE_SLOTS=a.sparse, PYTHON=sys.executable, PLAN_HARD_ENV_CAP=cap_,
                                        PLAN_ERR_TARGET=pe, PLAN_DRIFT_RECUT=recut, PLAN_DELIBERATE_BAND_CF=band))
            if r.returncode == 0:
                break
        else:
            continue
        if subprocess.run([*me, *rest, "--plan", plan, "--compile-only"], env=cpu).returncode == 0:
            break
    else:
        raise SystemExit(f"make_plan.sh: no plan of {graph} compiles")
    subprocess.run([*me, *rest, "--plan", plan], check=True)


def main(argv=None):
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--capture-noise", type=float, default=0.0)
    extra, argv = pre.parse_known_args(argv)

    def auto(a, av):
        a.capture_noise = extra.capture_noise
        return run_auto(a, av)
    base.T = T                       # the runner's task table (--task choices), report line and --auto
    base._report = _report
    base.run_auto = auto
    return base.main(argv)


if __name__ == "__main__":
    sys.exit(main())
