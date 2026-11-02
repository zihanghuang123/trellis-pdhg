"""PDHG solver for the planar-pushing trellis (plan sections 7-12). Milestone M0 provides the batched-eigh benchmark of plan 8.2 and the environment record; the solver sections follow from M3.

    python run_trajopt.py --bench_eigh                 # writes runs/bench/eigh.json
"""
import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


# ===== CONFIG =====
# The same loader as push_model.py (plan section 2 prefers a copy over a shared module).

def deep_merge(base, over):
    """Recursively merges `over` into a copy of `base`: dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def parse_override(text):
    """Turns 'a.b.c=value' into {'a': {'b': {'c': value}}}, with the value parsed as YAML."""
    key, sep, val = text.partition("=")
    if not sep:
        raise SystemExit(f"--set expects key=value, got {text!r}")
    node = yaml.safe_load(val)
    for part in reversed(key.split(".")):
        node = {part: node}
    return node


def load_config(run=None, sets=(), base=ROOT / "configs" / "base.yaml"):
    """Base config merged with a run file (a name in configs/runs/ or a path) and --set overrides."""
    cfg = yaml.safe_load(Path(base).read_text())
    if run:
        path = Path(run) if Path(run).suffix else ROOT / "configs" / "runs" / f"{run}.yaml"
        cfg = deep_merge(cfg, yaml.safe_load(path.read_text()))
    for s in sets:
        cfg = deep_merge(cfg, parse_override(s))
    return cfg


# ===== ENVIRONMENT =====

def environment_info():
    """JAX, jaxlib, CUDA, driver and device versions plus XLA flags, for timing.json and benchmark files (plan 3)."""
    import jax
    import jaxlib
    dev = jax.devices()[0]
    info = dict(python=platform.python_version(), jax=jax.__version__, jaxlib=jaxlib.__version__,
                backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                device_kind=getattr(dev, "device_kind", "?"), xla_flags=os.environ.get("XLA_FLAGS", ""),
                x64=bool(getattr(jax.config, "jax_enable_x64", False)))
    try:
        info["platform_version"] = dev.client.platform_version
    except Exception as exc:
        info["platform_version"] = f"unavailable: {exc}"
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception as exc:
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


# ===== EDGE_SOLVERS =====
# Plan 8.2: the PSD projection is a batched eigendecomposition. Some GPU eigh paths loop over
# large matrices instead of batching them, so M0 measures the per-matrix time at each moment
# matrix size (16 for kappa = 1, 136 for kappa = 2, 816 for kappa = 3) and two batch sizes.

def bench_eigh(cfg):
    """Times jit-compiled batched jnp.linalg.eigh at the configured [size, batch] pairs, in float64 and float32."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    reps = int(cfg["bench"].get("eigh_repeats", 5))
    eigh = jax.jit(jnp.linalg.eigh)
    results = []
    for n, batch in cfg["bench"]["eigh"]:
        for dtype in (jnp.float64, jnp.float32):
            a = jax.random.normal(jax.random.PRNGKey(n * 7919 + batch), (batch, n, n), dtype=dtype)
            a = 0.5 * (a + jnp.swapaxes(a, -1, -2))
            t0 = time.perf_counter()
            fn = eigh.lower(a).compile()
            compile_s = time.perf_counter() - t0
            w, v = fn(a)
            jax.block_until_ready((w, v))
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                w, v = fn(a)
                jax.block_until_ready((w, v))
                times.append(time.perf_counter() - t0)
            resid = jnp.max(jnp.abs(jnp.einsum("bij,bjk->bik", a, v) - v * w[:, None, :])) / jnp.max(jnp.abs(a))
            ortho = jnp.max(jnp.abs(jnp.einsum("bji,bjk->bik", v, v) - jnp.eye(n, dtype=dtype)))
            row = dict(n=int(n), batch=int(batch), dtype=jnp.dtype(dtype).name, compile_s=round(compile_s, 3),
                       median_s=float(np.median(times)), min_s=float(np.min(times)),
                       per_matrix_us=float(np.median(times)) / batch * 1e6,
                       residual=float(resid), orthogonality=float(ortho))
            results.append(row)
            print(f"eigh n={n:4d} batch={batch:6d} {row['dtype']:8s} median {row['median_s'] * 1e3:9.3f} ms "
                  f"({row['per_matrix_us']:9.3f} us/matrix), compile {compile_s:6.2f} s, "
                  f"residual {row['residual']:.1e}, orthogonality {row['orthogonality']:.1e}", flush=True)
            del a, w, v, fn
    # For each size, compare the two batches: time_ratio well below batch_ratio means the smaller
    # batch left the GPU idle (batching works); time_ratio close to batch_ratio means either the
    # GPU is saturated or the kernel loops over matrices. The per-matrix times tell which.
    scaling = {}
    for r in results:
        same = [q for q in results if q["n"] == r["n"] and q["dtype"] == r["dtype"]]
        if len(same) >= 2:
            small, large = min(same, key=lambda q: q["batch"]), max(same, key=lambda q: q["batch"])
            scaling[f"n{r['n']}_{r['dtype']}"] = dict(batch_ratio=large["batch"] / small["batch"],
                                                     time_ratio=large["median_s"] / small["median_s"])
    out = ROOT / cfg["output"]["root"] / "bench" / "eigh.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(env=environment_info(), results=results, scaling=scaling), indent=2))
    print(f"saved {out}", flush=True)


# ===== PDHG =====
# Milestone M3 (plan section 7).

# ===== SKIPPING =====
# Milestone M5 (plan section 9).

# ===== CERTIFICATE =====
# Milestones M3 and M9 (plan section 10).

# ===== TREES =====
# Milestones M6-M8 (plan section 11).

# ===== MAIN =====

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="run config: a name in configs/runs/ or a path")
    ap.add_argument("--set", action="append", default=[], help="override one key, e.g. --set trellis.N=16")
    ap.add_argument("--bench_eigh", action="store_true", help="time batched eigh at the configured sizes (M0)")
    args = ap.parse_args()
    cfg = load_config(args.run, args.set)
    if args.bench_eigh:
        bench_eigh(cfg)
        return
    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
