"""PDHG solver for the planar-pushing trellis (plan sections 7-12). So far: the M0 benchmarks of the GPU primitives a round will use, each checked for true batching (plan section 0); the solver sections follow from M3.

    python run_trajopt.py --bench_psd       # PSD projection per round, orders 1-3: exact eigh vs low-rank tracking
    python run_trajopt.py --bench_linalg    # Cholesky/triangular solves, tree-merge pieces, coupling scatter/gather
    python run_trajopt.py --bench_eigh      # the original M0 eigh scaling test
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

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)
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


def swap(x):
    return jnp.swapaxes(x, -1, -2)


# ===== EDGE_SOLVERS: PSD PROJECTION =====
# Plan 8.2. The projection onto the PSD cone is Pi(X) = (X + |X|) / 2 = V max(w, 0) V^T.
# Measured in M0 (A100): the GPU library kernels batch only up to 32 x 32 (cuSOLVER's batched
# Jacobi). Above that eigh, eigvalsh and QR solve the matrices one after another (~45 us each at
# 136 x 136, flat with the batch). So nothing here calls them above 32 x 32: the low-rank tracker
# uses matrix products plus eigendecompositions of at most 2p x 2p. Inexact projections are
# allowed to a tolerance (Rasch-Chambolle), and the certificate stays valid for any duals.

def psd_eigh(x):
    """Exact projection via eigh: the reference, and the method of choice at 16 x 16 where eigh batches."""
    w, v = jnp.linalg.eigh(x)
    return (v * jnp.maximum(w, 0.0)[..., None, :]) @ swap(v)


def psd_newton_schulz(x, iters):
    """(X + |X|) / 2 with |X| = X sign(X), sign(X) by Newton-Schulz on X / ||X||_F: matrix products only.

    Truly batched (30 % of FP64 peak at 136, 76 % at 816 in M0) but heavy: 40 iterations reach
    1e-8 at about 18x the arithmetic of an eigendecomposition. Kept as the batched exact fallback
    for the matrices where low-rank tracking fails its residual check.
    """
    s = jnp.sqrt(jnp.sum(x * x, axis=(-2, -1), keepdims=True)) + 1e-300
    eye = jnp.eye(x.shape[-1], dtype=x.dtype)
    y = jax.lax.fori_loop(0, iters, lambda _, y: 0.5 * y @ (3.0 * eye - y @ y), x / s)
    a = x @ y
    return 0.5 * (x + 0.5 * (a + swap(a)))


def orthonormalize(y, floor):
    """Orthonormal basis of span(y) (batch, n, k) from its k x k Gram matrix: one small eigh, which batches. Directions with singular value <= floor become zero columns."""
    g = swap(y) @ y
    s, u = jnp.linalg.eigh(0.5 * (g + swap(g)))
    keep = s > floor ** 2
    return y @ (u * jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, s, 1.0)), 0.0)[..., None, :])


def psd_lowrank(x, q, tol=1e-8):
    """Positive part of X tracked from the previous round's orthonormal basis q (batch, n, p), with matrix products and eigendecompositions no larger than 2p x 2p.

    One block-Krylov Rayleigh-Ritz step on span[q, Xq]: exact whenever X's positive eigenvectors
    lie in that span, which is what tracking a slowly changing, near-low-rank matrix gives. Cost
    O(n^2 p) per matrix instead of O(n^3). Returns (projection, next basis, largest relative
    residual ||X v - theta v|| / ||X||_F over the positive Ritz pairs); the residual lets the
    solver check the result without an exact reference and send failures to a fallback.
    """
    p = q.shape[-1]
    scale = jnp.sqrt(jnp.sum(x * x, axis=(-2, -1)))[..., None]
    xq = x @ q
    r = xq - q @ (swap(q) @ xq)
    r = r - q @ (swap(q) @ r)                                   # block Gram-Schmidt against q, twice
    z = orthonormalize(r, tol * scale)                          # new directions; negligible ones become zero
    z = z - q @ (swap(q) @ z)
    z = orthonormalize(z, 0.5)                                  # second pass: unit columns stay, zero columns stay zero
    w = jnp.concatenate([q, z], axis=-1)
    xw = jnp.concatenate([xq, x @ z], axis=-1)
    t = swap(w) @ xw
    th, u = jnp.linalg.eigh(0.5 * (t + swap(t)))
    v, xv = w @ u, xw @ u                                       # Ritz vectors, and X applied to them
    real = jnp.sum(v * v, axis=-2) > 0.5                        # zero columns of w give zero Ritz vectors
    pos = jnp.where(real, jnp.maximum(th, 0.0), 0.0)
    proj = (v * pos[..., None, :]) @ swap(v)
    res = jnp.sqrt(jnp.sum((xv - v * th[..., None, :]) ** 2, axis=-2))
    res_max = jnp.max(jnp.where(pos > 0.0, res, 0.0), axis=-1) / scale[..., 0]
    idx = jnp.argsort(jnp.where(real, th, -jnp.inf), axis=-1)[..., -p:]
    q_next = jnp.take_along_axis(v, jnp.broadcast_to(idx[..., None, :], v.shape[:-1] + (p,)), axis=-1)
    return proj, q_next, res_max


# ===== BENCH =====
ORDER_OF_SIZE = {16: 1, 136: 2, 816: 3}


def timed(fn, *args, reps=5):
    """Ahead-of-time compiles fn for args and times it: (median seconds, compile seconds, output). Slow calls get fewer repeats."""
    t0 = time.perf_counter()
    comp = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = comp(*args)
    jax.block_until_ready(out)
    first = time.perf_counter() - t0
    reps = 1 if first > 5.0 else 2 if first > 1.0 else reps
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = comp(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), compile_s, out


def make_runner(rows, peak, bandwidth, reps):
    """A run(group, name, n, batch, fn, args, flops, check, traffic, **tags) that times one case, prints it and appends it to rows.

    flops and traffic are per item. traffic is a lower bound on the bytes one item reads and writes
    in GPU memory: a kernel near the memory bandwidth is saturated, even when its FLOP share is low.
    """
    def run(group, name, n, batch, fn, args, flops, check=None, traffic=None, **tags):
        try:
            sec, comp_s, out = timed(fn, *args, reps=reps)
            row = dict(group=group, name=name, n=n, batch=batch, seconds=sec, compile_s=round(comp_s, 3),
                       per_item_us=sec / batch * 1e6, **tags)
            if flops:
                row.update(gflops=flops * batch / sec / 1e9, roofline=flops * batch / sec / peak)
            if traffic:
                row.update(bandwidth_share=traffic * batch / sec / bandwidth)
            if check is not None:
                row.update(check(out))
            rows.append(row)
            extra = "".join(f" {k} {row[k]:.1e}" for k in ("err_max", "res_max", "residual") if k in row)
            print(f"{group:8s} {name:20s} n={n:5d} batch={batch:6d} {sec * 1e3:10.3f} ms {row['per_item_us']:10.3f} us/item"
                  + (f" {100 * row['roofline']:6.2f}% of peak" if flops else "")
                  + (f" {100 * row['bandwidth_share']:6.2f}% of bw" if traffic else "") + extra
                  + "".join(f" {k}={v}" for k, v in tags.items()), flush=True)
        except Exception as exc:
            rows.append(dict(group=group, name=name, n=n, batch=batch, error=f"{type(exc).__name__}: {exc}"[:400], **tags))
            print(f"{group:8s} {name:20s} n={n:5d} batch={batch:6d} FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    return run


def psd_test_set(key, batch, n, r_pos, gap, eps, pmax, n_bases=64):
    """Symmetric test matrices with r_pos eigenvalues in [0.5, 2] and the rest in [-2, -gap] (a near-low-rank moment matrix plus a negative semidefinite dual), perturbed by ~eps to give the next round's matrix.

    Returns (next-round matrices, the previous round's top-pmax eigenvectors). Eigenvectors come
    from n_bases random orthogonal matrices reused across the batch to keep setup cheap; every
    method treats each matrix independently, so the reuse does not affect timings.
    """
    kq, kp, kn, ke = jax.random.split(key, 4)
    qs, _ = jnp.linalg.qr(jax.random.normal(kq, (min(n_bases, batch), n, n)))
    pick = jnp.arange(batch) % qs.shape[0]
    pos = jnp.sort(jax.random.uniform(kp, (batch, r_pos), minval=0.5, maxval=2.0), axis=-1)
    neg = -jnp.sort(jax.random.uniform(kn, (batch, n - r_pos), minval=gap, maxval=2.0), axis=-1)[..., ::-1]
    lam = jnp.concatenate([neg, pos], axis=-1)                  # ascending: the top eigenvectors are the last columns
    q = qs[pick]
    x0 = (q * lam[:, None, :]) @ swap(q)
    e = jax.random.normal(ke, x0.shape)
    x1 = x0 + eps * 0.5 * (e + swap(e)) / np.sqrt(n)
    return x1, qs[:, :, -pmax:][pick]


def projection_check(exact, x):
    """Relative error ||Pi - Pi_exact||_F / ||X||_F of a projection, plus the method's own residual estimate when it reports one."""
    den = jnp.sqrt(jnp.sum(x * x, axis=(-2, -1)))

    def check(out):
        proj = out[0] if isinstance(out, tuple) else out
        r = jnp.sqrt(jnp.sum((proj - exact) ** 2, axis=(-2, -1))) / den
        d = dict(err_median=float(jnp.median(r)), err_max=float(jnp.max(r)))
        if isinstance(out, tuple) and len(out) > 2:
            d["res_max"] = float(jnp.max(out[2]))
        return d
    return check


def bench_psd(cfg):
    """The PSD projection per solver round at the order-1, -2 and -3 moment-matrix sizes: exact eigh against low-rank tracking (plan 8.2)."""
    pc, bc = cfg["bench"]["psd"], cfg["bench"]
    peak, per_round = float(bc["peak_tflops_fp64"]) * 1e12, int(bc["edges_per_round"])
    rows = []
    run = make_runner(rows, peak, float(bc["peak_bandwidth_gbs"]) * 1e9, int(pc.get("repeats", 5)))
    for n, batches in pc["sizes"]:
        ps = sorted({min(int(p), n // 2) for p in pc["lowrank_p"]})
        for batch in batches:
            x1, basis = psd_test_set(jax.random.PRNGKey(31 * n + batch), batch, n, int(pc["rank"]), float(pc["gap"]),
                                     float(pc["perturbation"]), max(ps))
            w1, v1 = jax.jit(jnp.linalg.eigh)(x1)
            exact = (v1 * jnp.maximum(w1, 0.0)[..., None, :]) @ swap(v1)
            npos = jnp.sum(w1 > 0.0, axis=-1)
            del w1, v1
            check = projection_check(exact, x1)
            tags = dict(order=ORDER_OF_SIZE.get(n), positives=f"{int(npos.min())}-{int(npos.max())}")
            if batch * n * n <= float(pc["eigh_max_elements"]):
                run("psd", "eigh", n, batch, psd_eigh, (x1,), 9 * n ** 3, check, traffic=40 * n * n, **tags)
            for p in ps:
                run("psd", f"lowrank_p{p}", n, batch, psd_lowrank, (x1, basis[..., -p:]),
                    8 * n * n * p + 40 * n * p * p, check, traffic=8 * (3 * n * n + 24 * n * p), **tags)
            del x1, basis, exact
    verdicts = batching_verdicts(rows)
    print_verdicts(verdicts)
    print(f"\nPROJECTION TIME PER ROUND at {per_round} edges (one matrix per edge; from the per-item time at the largest batch run)")
    for v in verdicts:
        order = ORDER_OF_SIZE.get(v["n"])
        print(f"  order {order} ({v['n']}x{v['n']}): {v['name']:12s} {v['per_item_us'][-1] * per_round / 1e3:10.3f} ms"
              + ("" if v["err_max"] is None else f"   err_max {v['err_max']:.1e}"), flush=True)
    save_bench(cfg, "psd.json", rows, verdicts)


def chol_solve(a, b):
    """Batched SPD solve by Cholesky and two triangular solves (the IPM's normal equations, the tree's pivots)."""
    l = jnp.linalg.cholesky(a)
    y = jax.scipy.linalg.solve_triangular(l, b, lower=True)
    return jax.scipy.linalg.solve_triangular(l, y, lower=True, trans="T")


def bench_linalg(cfg):
    """Cholesky and triangular solves at the IPM and tree sizes, the low-rank tree merge's small eigh and orthonormalization, and the coupling scatter/gather (plan 8.1, 11, 6.3)."""
    lc, bc = cfg["bench"]["linalg"], cfg["bench"]
    rows = []
    run = make_runner(rows, float(bc["peak_tflops_fp64"]) * 1e12, float(bc["peak_bandwidth_gbs"]) * 1e9, int(lc.get("repeats", 5)))
    for n, batches in lc["cholesky"]:
        for batch in batches:
            k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7 * n + batch), 3)
            g = jax.random.normal(k1, (batch, n, n))
            a = g @ swap(g) / n + jnp.eye(n)
            bv = jax.random.normal(k2, (batch, n, 1))
            bm = jax.random.normal(k3, (batch, n, n))
            del g

            def residual(out, a=a, bv=bv):
                r = jnp.linalg.norm(a @ out - bv, axis=(-2, -1)) / jnp.linalg.norm(bv, axis=(-2, -1))
                return dict(residual=float(jnp.max(r)))
            run("chol", "cholesky", n, batch, jnp.linalg.cholesky, (a,), n ** 3 / 3, traffic=12 * n * n)
            run("chol", "cholesky_solve_vec", n, batch, chol_solve, (a, bv), n ** 3 / 3 + 2 * n ** 2, residual, traffic=20 * n * n + 32 * n)
            run("chol", "cholesky_solve_mat", n, batch, chol_solve, (a, bm), n ** 3 / 3 + 2 * n ** 3, traffic=52 * n * n)
            run("chol", "matmul", n, batch, lambda a: a @ a, (a,), 2 * n ** 3, traffic=24 * n * n)
            del a, bv, bm
    for k, batches in lc["merge_eigh"]:
        for batch in batches:
            x = jax.random.normal(jax.random.PRNGKey(11 * k + batch), (batch, k, k))
            run("merge", "eigh", k, batch, jnp.linalg.eigh, (0.5 * (x + swap(x)),), 9 * k ** 3, traffic=16 * k * k)
    for m, k, batches in lc["merge_orth"]:
        for batch in batches:
            x = jax.random.normal(jax.random.PRNGKey(13 * k + batch), (batch, m, k))
            run("merge", f"qr_{m}x{k}", k, batch, lambda x: jnp.linalg.qr(x)[0], (x,), 4 * m * k * k, traffic=16 * m * k)
            run("merge", f"gram_orth_{m}x{k}", k, batch, lambda x: orthonormalize(x, 1e-10), (x,), 4 * m * k * k, traffic=32 * m * k)
    q = int(lc["coupling"]["q"])
    for e in lc["coupling"]["edges"]:
        nv = max(8, e // 4)
        rng = np.random.default_rng(e)
        tails, heads = jnp.asarray(rng.integers(0, nv, e)), jnp.asarray(rng.integers(0, nv, e))
        seg = jnp.concatenate([tails, heads])
        vals = jax.random.normal(jax.random.PRNGKey(e), (e, 2, q))
        prices = jax.random.normal(jax.random.PRNGKey(e + 1), (nv, q))
        run("coupling", "K_scatter", q, e,
            lambda v: jax.ops.segment_sum(jnp.concatenate([-v[:, 0], v[:, 1]]), seg, num_segments=nv), (vals,), None)
        run("coupling", "Kt_gather", q, e, lambda p: jnp.stack([-p[tails], p[heads]], axis=1), (prices,), None)
    verdicts = batching_verdicts(rows)
    print_verdicts(verdicts)
    save_bench(cfg, "linalg.json", rows, verdicts)


def batching_verdicts(rows):
    """Per primitive and size: per-item time at the largest batch over the smallest (falls when batching helps), the shares of the FLOP and memory-bandwidth rooflines, and a verdict.

    A flat per-item time means either a saturated kernel or a loop over the batch. The rooflines
    tell them apart: a kernel at a fair share of either one is saturated, not looping.
    """
    groups = {}
    for r in rows:
        if "error" not in r:
            groups.setdefault((r["group"], r["name"], r["n"]), []).append(r)
    out = []
    for (group, name, n), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["batch"])
        small, large = rs[0], rs[-1]
        ratio = large["per_item_us"] / small["per_item_us"] if len(rs) > 1 else None
        roof, bw = large.get("roofline"), large.get("bandwidth_share")
        if (ratio is not None and ratio <= 0.8) or (roof is not None and roof >= 0.2) or (bw is not None and bw >= 0.2):
            verdict = "batched"
        elif ratio is None:
            verdict = "one batch size only"
        else:
            verdict = "NOT BATCHED?"
        out.append(dict(group=group, name=name, n=n, batches=[r["batch"] for r in rs],
                        per_item_us=[r["per_item_us"] for r in rs], per_item_ratio=ratio, roofline_at_largest=roof,
                        bandwidth_at_largest=bw, seconds_at_largest=large["seconds"], err_max=large.get("err_max"), verdict=verdict))
    return out


def print_verdicts(verdicts):
    print("\nBATCHING VERDICTS (per-item time at the largest batch / at the smallest; <= 0.8, or >= 20% of the FP64 roofline or of memory bandwidth, counts as batched)")
    for v in verdicts:
        ratio = "   -  " if v["per_item_ratio"] is None else f"{v['per_item_ratio']:6.2f}"
        roof = "   -   " if v["roofline_at_largest"] is None else f"{100 * v['roofline_at_largest']:6.2f}%"
        bw = "   -   " if v["bandwidth_at_largest"] is None else f"{100 * v['bandwidth_at_largest']:6.2f}%"
        print(f"  {v['group']:8s} {v['name']:20s} n={v['n']:5d} batches {str(v['batches']):14s} ratio {ratio} "
              f"roofline {roof}  bandwidth {bw}  {v['seconds_at_largest'] * 1e3:10.3f} ms at largest  {v['verdict']}", flush=True)


def save_bench(cfg, name, rows, verdicts):
    out = ROOT / cfg["output"]["root"] / "bench" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(env=environment_info(), rows=rows, verdicts=verdicts), indent=2))
    print(f"saved {out}", flush=True)


def bench_eigh(cfg):
    """Times jit-compiled batched jnp.linalg.eigh at the configured [size, batch] pairs, in float64 and float32."""
    reps = int(cfg["bench"].get("eigh_repeats", 5))
    eigh = jax.jit(jnp.linalg.eigh)
    results = []
    for n, batch in cfg["bench"]["eigh"]:
        for dtype in (jnp.float64, jnp.float32):
            a = jax.random.normal(jax.random.PRNGKey(n * 7919 + batch), (batch, n, n), dtype=dtype)
            a = 0.5 * (a + swap(a))
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
    out = ROOT / cfg["output"]["root"] / "bench" / "eigh.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(env=environment_info(), results=results), indent=2))
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
    ap.add_argument("--bench_psd", action="store_true", help="PSD projection per round, orders 1-3 (M0)")
    ap.add_argument("--bench_linalg", action="store_true", help="Cholesky, tree-merge pieces and coupling (M0, before M3)")
    ap.add_argument("--bench_eigh", action="store_true", help="the original batched-eigh scaling test (M0)")
    args = ap.parse_args()
    cfg = load_config(args.run, args.set)
    if args.bench_psd:
        bench_psd(cfg)
    if args.bench_linalg:
        bench_linalg(cfg)
    if args.bench_eigh:
        bench_eigh(cfg)
    if not (args.bench_psd or args.bench_linalg or args.bench_eigh):
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
