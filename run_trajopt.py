"""PDHG solver for the planar-pushing trellis (plan sections 7-12): the PDHG round at order 1, its certificate and stopping test (M3), exact skipping with compaction (M5), the exact segment tree behind its safeguard (M6), and the benchmarks of the GPU primitives a round uses, each checked for true batching (plan section 0). The moment matrix's PSD constraint sits inside each edge step by default (solver.psd_coupling: edge, plan 8.4); `split` keeps the earlier copies S and duals Lam. The face cache and the compressed trees are still to come.

    python run_trajopt.py --selftest                 # milestones M3, M5 and M6, on the tiny instance
    python run_trajopt.py --selftest_m6              # milestone M6 alone
    python run_trajopt.py --run A --task T3_translate_rotate   # solve one run with the exact tree, building its problem.npz if needed
    python run_trajopt.py --run D --task T3_translate_rotate   # the same with plain PDHG
    python run_trajopt.py --bench_chol               # Cholesky at the edge QP's real sizes: library vs hand-written
    python run_trajopt.py --bench_psd                # PSD projection per round, orders 1-3: exact eigh vs low-rank tracking
    python run_trajopt.py --bench_linalg             # Cholesky/triangular solves, tree-merge pieces, coupling scatter/gather
    python run_trajopt.py --bench_eigh               # the original M0 eigh scaling test
"""
import argparse
import bisect
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
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


def small_cholesky(a):
    """Cholesky factors of a batch of small matrices as straight-line elementwise code, with no library call: {(row, column): values}. The compiler can fuse it into few kernels, which avoids the library call's fixed overhead."""
    n = a.shape[-1]
    factor = {}
    for j in range(n):
        d = a[..., j, j]
        for k in range(j):
            d = d - factor[j, k] * factor[j, k]
        root = jnp.sqrt(d)
        factor[j, j] = root
        for i in range(j + 1, n):
            v = a[..., i, j]
            for k in range(j):
                v = v - factor[i, k] * factor[j, k]
            factor[i, j] = v / root
    return factor


def small_cholesky_solve(factor, b):
    """Solves L L^T x = b, for factors from small_cholesky and b (..., n), as straight-line elementwise code."""
    n = b.shape[-1]
    y = {}
    for i in range(n):
        v = b[..., i]
        for k in range(i):
            v = v - factor[i, k] * y[k]
        y[i] = v / factor[i, i]
    x = {}
    for i in range(n - 1, -1, -1):
        v = y[i]
        for k in range(i + 1, n):
            v = v - factor[k, i] * x[k]
        x[i] = v / factor[i, i]
    return jnp.stack([x[i] for i in range(n)], axis=-1)


def bench_chol(cfg):
    """Cholesky with one solve at the edge QP's real system sizes (plan 8.1): the library kernels against the hand-written straight-line code, at the two batch sizes of the verdict rule. Both are checked against the right-hand side."""
    sc, bc = cfg["bench"]["small_cholesky"], cfg["bench"]
    rows = []
    run = make_runner(rows, float(bc["peak_tflops_fp64"]) * 1e12, float(bc["peak_bandwidth_gbs"]) * 1e9, int(sc.get("repeats", 5)))
    for n in sc["sizes"]:
        for batch in sc["batches"]:
            n, batch = int(n), int(batch)
            k1, k2 = jax.random.split(jax.random.PRNGKey(17 * n + batch))
            g = jax.random.normal(k1, (batch, n, n))
            a = g @ swap(g) / n + jnp.eye(n)
            b = jax.random.normal(k2, (batch, n))

            def residual(out, a=a, b=b):
                return dict(residual=float(jnp.max(jnp.abs(jnp.einsum("bij,bj->bi", a, out) - b))))

            flops = n ** 3 / 3 + 2 * n ** 2
            run("small", "library", n, batch, lambda a, b: chol_solve(a, b[..., None])[..., 0], (a, b), flops, residual, traffic=20 * n * n + 32 * n)
            run("small", "hand_written", n, batch, lambda a, b: small_cholesky_solve(small_cholesky(a), b), (a, b), flops, residual, traffic=20 * n * n + 32 * n)
            del a, b, g
    verdicts = batching_verdicts(rows)
    print_verdicts(verdicts)
    save_bench(cfg, "chol.json", rows, verdicts)


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


# ===== PROBLEM =====
# The solver reads problem.npz from push_model.py --build, in a separate process, because that script
# forks CPU workers and a fork does not mix with JAX. One padded batch holds all edges (plan 6.2 allows
# a single bucket): a round this small is bound by kernel launches, not arithmetic, so every extra
# bucket would cost more than the padding does. A padded entry of m has no rows and stays at zero.

def run_dir(cfg):
    return ROOT / cfg["output"]["root"] / cfg["run_name"] / cfg["task"] / str(cfg["seed"])


def reference_fingerprint(cfg):
    """A hash of everything the problem and J_rel depend on in the config; the same function as in push_model.py."""
    payload = {k: cfg[k] for k in ("model", "trellis", "relaxation", "costs", "task", "tasks", "task_common")}
    payload["delta"] = cfg["solver"]["delta"]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def ensure_problem(cfg, run, task, sets):
    """Loads the run's problem.npz, first building it when it is missing or came from another config (plan 12.1). The build runs the validity checks of plan 5.4 and stops on a violation."""
    path = run_dir(cfg) / "problem.npz"
    fresh = False
    if path.exists():
        with np.load(path) as z:
            fresh = json.loads(str(z["meta"])).get("fingerprint") == reference_fingerprint(cfg)
    if not fresh:
        cmd = [sys.executable, str(ROOT / "push_model.py"), "--build"] + (["--run", run] if run else []) + (["--task", task] if task else [])
        for s in sets:
            cmd += ["--set", s]
        print(f"building {path}", flush=True)
        subprocess.run(cmd, check=True)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_data(prob, cfg):
    """The solver's arrays over all alive edges, as (arrays for the GPU, plain facts about them). Step sizes are Pock-Chambolle with alpha = 1 and the 0.99 factor (plan 7.3); they are stored for theta = 1, so T scales with theta and Sigma with 1 / theta."""
    meta = json.loads(str(prob["meta"]))
    if str(prob["rel_kind"]) != "sdp" or int(meta["relaxation"]["order"]) != 1 or not meta["relaxation"]["eliminate_linear"]:
        raise NotImplementedError("the solver covers the reduced moment relaxation at order 1 so far")
    split = psd_coupling(cfg) == "split"
    rho, root2 = 0.99, math.sqrt(2.0)
    alive = np.flatnonzero(prob["edge_alive"])
    etype = prob["edge_type"][alive]
    used = np.unique(etype)
    rel_n, nt = prob["rel_n"], prob["rel_n"].size
    n, q = int(rel_n[used].max()), int(prob["rel_q"])
    k_eq, k_ge = prob["rel_eq_rows"], prob["rel_ge_rows"]
    r = max(1, int((k_eq + k_ge)[used].max()))
    sizes = {int(nv): int(prob[f"rel_moment_{int(nv)}_size"]) for nv in np.unique(prob["rel_nv"][used])}
    s = max(sizes.values())

    # per edge type: rows [equalities, inequalities, padding], box, cost, moment matrix map
    R_t, eq_t, ge_t = np.zeros((nt, r, n)), np.zeros((nt, r)), np.zeros((nt, r))
    for key, offset in (("eq", np.zeros(nt, np.int64)), ("ge", k_eq)):
        t = prob[f"rel_{key}_type"]
        keep = np.isin(t, used)
        t = t[keep]
        R_t[t, prob[f"rel_{key}_row"][keep] + offset[t], prob[f"rel_{key}_col"][keep]] = prob[f"rel_{key}_val"][keep]
    lo_t, hi_t, c_t = prob["rel_lo"][:, :n].copy(), prob["rel_hi"][:, :n].copy(), prob["rel_c"][:, :n].copy()
    open_t, col_t = np.zeros((nt, n)), np.zeros((nt, n))
    mat_t, mask_t = np.zeros((nt, s, s), np.int64), np.zeros((nt, s, s))
    adj_t, adj_w_t, trace_t = np.zeros((nt, n), np.int64), np.zeros((nt, n)), np.zeros(nt)
    for t in used:
        eq_t[t, :k_eq[t]] = 1.0
        ge_t[t, k_eq[t]:k_eq[t] + k_ge[t]] = 1.0
        nt_entries = int(rel_n[t])
        if not np.all(lo_t[t, 1:nt_entries] < hi_t[t, 1:nt_entries]):
            raise NotImplementedError(f"edge type {t} has an entry with a zero-width interval, which the edge QP does not handle yet")
        open_t[t, 1:nt_entries] = 1.0
        nv = int(prob["rel_nv"][t])
        row, col, mom = prob[f"rel_moment_{nv}_row"], prob[f"rel_moment_{nv}_col"], prob[f"rel_moment_{nv}_mom"]
        if np.unique(mom).size != mom.size:
            raise NotImplementedError("a moment appears twice in the moment matrix, which only happens above order 1")
        mat_t[t, row, col] = mom
        mat_t[t, col, row] = mom
        mask_t[t, :sizes[nv], :sizes[nv]] = 1.0
        adj_t[t, mom] = row.astype(np.int64) * s + col      # B*(Lam)[k] = sum of Lam over the entries that read m[k]
        adj_w_t[t, mom] = np.where(row == col, 1.0, 2.0)
        col_t[t, mom] += np.where(row == col, 1.0, root2)   # svec rows: sqrt(2) on off-diagonals (plan 7.3)
        trace_t[t] = hi_t[t, mom[row == col]].sum()

    def coupling(key):
        """Each coupled coordinate of an edge type as coef * m[idx]; substitution of single-variable equalities leaves one term per coordinate."""
        t = prob[f"rel_{key}_type"]
        keep = np.isin(t, used)
        t, i = t[keep], prob[f"rel_{key}_row"][keep]
        if np.unique(t.astype(np.int64) * q + i).size != t.size:
            raise NotImplementedError("a coupled coordinate has more than one term")
        idx, coef = np.zeros((nt, q), np.int64), np.zeros((nt, q))
        idx[t, i], coef[t, i] = prob[f"rel_{key}_col"][keep], prob[f"rel_{key}_val"][keep]
        return idx, coef

    tail_idx_t, tail_coef_t = coupling("tail")
    head_idx_t, head_coef_t = coupling("head")

    # per edge
    E = alive.size
    tails, heads = prob["edge_tail"][alive], prob["edge_head"][alive]
    src, dst = int(prob["s"]), int(prob["t"])
    verts = np.unique(np.r_[tails, heads])
    V = verts.size
    vpos = np.full(int(prob["edge_tail"].max() + prob["edge_head"].max() + 2), -1, np.int64)
    vpos[verts] = np.arange(V)
    tail_idx, head_idx = tail_idx_t[etype], head_idx_t[etype]
    tail_coef, head_coef = tail_coef_t[etype].copy(), head_coef_t[etype].copy()
    tail_coef[tails == src, 1:] = 0.0                       # the start and goal keep only their flow row
    head_coef[heads == dst, 1:] = 0.0
    edge = np.arange(E)[:, None]
    head_rows, tail_rows = vpos[heads][:, None] * q + np.arange(q), vpos[tails][:, None] * q + np.arange(q)
    # With the PSD constraint inside the edge step, K is the whole operator, so the PSD coupling's column
    # sums drop out of the primal steps (plan 7.3, 8.4).
    colsum = col_t[etype].copy() if split else np.zeros_like(col_t[etype])
    np.add.at(colsum, (np.broadcast_to(edge, tail_idx.shape), tail_idx), np.abs(tail_coef))
    np.add.at(colsum, (np.broadcast_to(edge, head_idx.shape), head_idx), np.abs(head_coef))
    rowsum = np.zeros(V * q)
    np.add.at(rowsum, head_rows.ravel(), np.abs(head_coef).ravel())
    np.add.at(rowsum, tail_rows.ravel(), np.abs(tail_coef).ravel())
    b = np.zeros((V, q))
    b[vpos[src], 0], b[vpos[dst], 0] = -1.0, 1.0
    is_flow = np.zeros((1, q), bool)
    is_flow[0, 0] = True
    weight = np.full((s, s), root2)
    np.fill_diagonal(weight, 1.0)
    layer = prob["edge_layer"][alive]
    layers = [(np.flatnonzero(layer == k), vpos[tails[layer == k]], vpos[heads[layer == k]]) for k in np.unique(layer)]
    dist0 = np.full(V, np.inf)
    dist0[vpos[src]] = 0.0
    # the tree's layout (plan 11.2): a vertex's prices sit at its layer, at slot * q + coordinate, over 2^k + 1 layers
    vlayer = prob["vert_layer"][verts]
    slot = np.zeros(V, np.int64)
    for k in np.unique(vlayer):
        slot[vlayer == k] = np.arange(np.sum(vlayer == k))
    tree_layers = 2 ** math.ceil(math.log2(int(vlayer.max()))) + 1
    tree_width = int((slot.max() + 1) * q)
    tree_mask = np.zeros((tree_layers, tree_width))
    tree_mask[vlayer[:, None], slot[:, None] * q + np.arange(q)] = (rowsum.reshape(V, q) > 0.0).astype(float)
    e0 = np.zeros(n)
    e0[0] = 1.0
    R = R_t[etype]
    Rw = R.copy()
    Rw[:, :, 0] = 0.0
    eq, ge = eq_t[etype], ge_t[etype]
    op = open_t[etype]
    arrays = dict(
        c=c_t[etype], lo=lo_t[etype], hi=hi_t[etype], open=op, R=R, Rw=Rw, Ry=R[:, :, 0], row_eq=eq, row_ge=ge, row_pad=1.0 - eq - ge,
        n_ineq=np.sum(ge, axis=1) + 2.0 * np.sum(op, axis=1), e0=e0, wm=1.0 - e0, tail_idx=tail_idx, tail_coef=tail_coef, head_idx=head_idx, head_coef=head_coef,
        head_rows=head_rows, tail_rows=tail_rows, k_rows=np.r_[head_rows.ravel(), tail_rows.ravel()],
        k_cols=np.r_[(edge * n + head_idx).ravel(), (edge * n + tail_idx).ravel()],
        mat=mat_t[etype], psd_mask=mask_t[etype], adj=adj_t[etype], adj_w=adj_w_t[etype], trace_hi=trace_t[etype],
        T=np.where(colsum > 0.0, rho / np.where(colsum > 0.0, colsum, 1.0), 1.0), t_s=np.asarray(rho),
        sig_p=np.where(rowsum > 0.0, rho / np.where(rowsum > 0.0, rowsum, 1.0), 0.0).reshape(V, q),
        sig_l=mask_t[etype] * rho / (weight + 1.0), b=b, is_flow=is_flow, state_mask=1.0 - is_flow.astype(float), delta=np.asarray(float(cfg["solver"]["delta"])),
        layers=layers, dist0=dist0, goal=np.asarray(vpos[dst]), eps_cert=np.asarray(float(cfg["certificate"]["eps_cert"])),
        v_layer=vlayer, v_slot=slot, tail_slot=slot[vpos[tails]], head_slot=slot[vpos[heads]], edge_layer=layer, tree_mask=tree_mask)
    info = dict(E=int(E), V=int(V), q=q, n=n, r=r, s=s, alive=alive, types=int(used.size), meta=meta, tree_layers=tree_layers, tree_width=tree_width,
                psd="split" if split else "edge")
    return jax.tree_util.tree_map(jnp.asarray, arrays), info


def psd_coupling(cfg):
    """Where the moment matrix's PSD constraint lives (plan 8.4): `edge`, inside each edge step, or `split`, as copies S with duals Lam."""
    mode = cfg["solver"].get("psd_coupling", "split")
    if mode not in ("edge", "split"):
        raise ValueError(f"solver.psd_coupling must be edge or split, not {mode!r}")
    return mode


def k_forward(d, m):
    """K m per vertex and coordinate (V, q): head +, tail - (plan 18)."""
    head = d["head_coef"] * jnp.take_along_axis(m, d["head_idx"], axis=1)
    tail = d["tail_coef"] * jnp.take_along_axis(m, d["tail_idx"], axis=1)
    return jax.ops.segment_sum(jnp.concatenate([head.ravel(), -tail.ravel()]), d["k_rows"], num_segments=d["b"].size).reshape(d["b"].shape)


def k_adjoint(d, p):
    """K^T p per edge (E, n)."""
    flat = p.ravel()
    head, tail = d["head_coef"] * flat[d["head_rows"]], d["tail_coef"] * flat[d["tail_rows"]]
    return jax.ops.segment_sum(jnp.concatenate([head.ravel(), -tail.ravel()]), d["k_cols"], num_segments=d["c"].size).reshape(d["c"].shape)


def k_adjoint_size(d, p):
    """|K|^T |p| per edge (E, n): the sizes of the terms k_adjoint adds up, before any cancel."""
    flat = jnp.abs(p).ravel()
    head, tail = jnp.abs(d["head_coef"]) * flat[d["head_rows"]], jnp.abs(d["tail_coef"]) * flat[d["tail_rows"]]
    return jax.ops.segment_sum(jnp.concatenate([head.ravel(), tail.ravel()]), d["k_cols"], num_segments=d["c"].size).reshape(d["c"].shape)


def psd_forward(d, m):
    """The moment matrices B(m) (E, s, s)."""
    return jnp.take_along_axis(m, d["mat"].reshape(m.shape[0], -1), axis=1).reshape(d["mat"].shape) * d["psd_mask"]


def psd_adjoint(d, lam):
    """B*(Lam) (E, n), with <Lam, B(m)> the Frobenius product. Lam must be symmetric: the off-diagonal entry is read once and counted twice."""
    return jnp.take_along_axis(lam.reshape(lam.shape[0], -1), d["adj"], axis=1) * d["adj_w"]


def sym(a):
    return 0.5 * (a + swap(a))


def chol(a):
    return jax.lax.linalg.cholesky(a, symmetrize_input=False)


def factor_solve(factor, b):
    """(L L^T)^-1 b for a lower-triangular factor L; b has columns."""
    return lower_solve(factor, lower_solve(factor, b), transpose=True)


def pick(mask, a, b):
    """a where mask (E,) holds, else b, broadcast over a's trailing axes."""
    return jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b)


# ===== EDGE_SOLVERS: EDGE QP =====
# Plan 8.1. Each edge step projects a point onto its polyhedral cone in a diagonal metric:
#   min 1/2 x^T diag(h) x + f^T x  over  R_eq x = 0, R_ge x >= 0, lo_i y <= x_i <= hi_i y,  with y = x[0].
# A batched Mehrotra predictor-corrector in the row form. The two-sided bounds stay diagonal in x_i and
# couple only through y, so H = diag(h) + (bound terms) is an arrow matrix with a closed-form inverse,
# and the only factorization is S = R H^-1 R^T + (diagonal), of the few rows after linear elimination.
# y <= 1 is left out: flow conservation implies it at the solution (tex 2.2).
# Multipliers follow the certificate's sign: stationarity is h x + f - R^T u - (bound terms) = 0, with
# u free on equalities and u >= 0 on inequalities, so u goes straight into the free bound.

def mv(a, x):
    return jnp.einsum("ern,en->er", a, x)


def mtv(a, u):
    return jnp.einsum("ern,er->en", a, u)


def spd_factor(s, method):
    return small_cholesky(s) if method == "small" else jax.lax.linalg.cholesky(s, symmetrize_input=False)


def spd_solve(factor, b, method):
    if method == "small":
        return small_cholesky_solve(factor, b)
    y = jax.lax.linalg.triangular_solve(factor, b[..., None], left_side=True, lower=True)
    return jax.lax.linalg.triangular_solve(factor, y, left_side=True, lower=True, transpose_a=True)[..., 0]


SLACKS = ("sg", "zg", "sl", "zl", "su", "zu")
FIELDS = ("x", "nu") + SLACKS


def edge_qp(d, h, f, ipm):
    """The step of every edge: (x (E, n), row multipliers u (E, r), stats). An edge that has converged stops moving while the others finish."""
    R, Rw, Ry = d["R"], d["Rw"], d["Ry"]
    eqf, gef, padf, op, lo, hi, e0, wm, n_ineq = d["row_eq"], d["row_ge"], d["row_pad"], d["open"], d["lo"], d["hi"], d["e0"], d["wm"], d["n_ineq"]
    method, tol = ipm["cholesky"], float(ipm["tol"])
    fnorm = 1.0 + jnp.max(jnp.abs(f), axis=1)
    diag_mask = jnp.eye(R.shape[1])

    def residuals(v):
        y = v["x"][:, :1]
        rx = mv(R, v["x"])
        cb = v["zl"] - v["zu"] + e0 * jnp.sum(hi * v["zu"] - lo * v["zl"], axis=1, keepdims=True)
        rd = h * v["x"] + f - mtv(R, eqf * v["nu"] + gef * v["zg"]) - cb
        ra, rg = eqf * rx, gef * (rx - v["sg"])
        rl, ru = op * (v["x"] - lo * y - v["sl"]), op * (hi * y - v["x"] - v["su"])
        mu = (jnp.sum(gef * v["sg"] * v["zg"], axis=1) + jnp.sum(v["sl"] * v["zl"] + v["su"] * v["zu"], axis=1)) / n_ineq
        prim = jnp.maximum(jnp.max(jnp.abs(ra) + jnp.abs(rg), axis=1), jnp.max(jnp.maximum(jnp.abs(rl), jnp.abs(ru)), axis=1))
        return rd, ra, rg, rl, ru, mu, jnp.maximum(jnp.max(jnp.abs(rd), axis=1) / fnorm, jnp.maximum(prim, mu))

    def to_boundary(v, dv):
        steps = [jnp.min(jnp.where(dv[k] < 0.0, -v[k] / dv[k], jnp.inf), axis=1) for k in SLACKS]
        return jnp.min(jnp.stack(steps), axis=0)

    def body(v):
        rd, ra, rg, rl, ru, mu, metric = residuals(v)
        # Keep each edge's best iterate. Past its attainable accuracy (a nearly pinned entry caps it near
        # 1e-10) an edge would otherwise keep driving mu and its slacks toward underflow, where the
        # Newton systems break down and the residual explodes (1e+14 and beyond in plain PDHG runs).
        better = metric < v["best_metric"]
        best = {k: jnp.where(better[:, None], v[k], v["best"][k]) for k in FIELDS}
        best_metric = jnp.minimum(metric, v["best_metric"])
        sg, zg, sl, zl, su, zu = (v[k] for k in SLACKS)
        zg_safe = zg + (1.0 - gef)
        dl, du = zl / sl, zu / su
        dw = h + dl + du
        a = -(lo * dl + hi * du)
        e = a / dw
        sigma = h[:, 0] + jnp.sum((h * (lo * lo * dl + hi * hi * du) + dl * du * (hi - lo) ** 2) / dw, axis=1)
        t = Ry - mv(Rw, e)
        s = jnp.einsum("ern,esn->ers", Rw / dw[:, None, :], Rw) + t[:, :, None] * t[:, None, :] / sigma[:, None, None]
        s = s + diag_mask * (gef * sg / zg_safe + padf + 1e-14 * eqf)[:, None, :]
        factor = spd_factor(s, method)

        def hb(w):
            """Applies the inverse of the arrow matrix H."""
            uw = wm * w / dw
            dy = (w[:, 0] - jnp.sum(a * uw, axis=1)) / sigma
            return uw - e * dy[:, None] + e0 * dy[:, None]

        def direction(rmg, rml, rmu):
            tl, tu = dl * rl + rml / sl, du * ru + rmu / su
            rhs = -rd - (tl - tu) - e0 * jnp.sum(hi * tu - lo * tl, axis=1, keepdims=True)
            step = spd_solve(factor, -ra - rg - gef * rmg / zg_safe - mv(R, hb(rhs)), method)
            dx = hb(rhs + mtv(R, (eqf + gef) * step))
            dy = dx[:, :1]
            dsl, dsu = op * (dx - lo * dy + rl), op * (hi * dy - dx + ru)
            return dict(x=dx, nu=eqf * step, sg=gef * (mv(R, dx) + rg), zg=gef * step, sl=dsl, zl=op * (-rml - zl * dsl) / sl,
                        su=dsu, zu=op * (-rmu - zu * dsu) / su)

        rmg, rml, rmu = gef * sg * zg, sl * zl, su * zu
        affine = direction(rmg, rml, rmu)
        a_aff = jnp.minimum(1.0, to_boundary(v, affine))[:, None]
        mu_aff = (jnp.sum(gef * (sg + a_aff * affine["sg"]) * (zg + a_aff * affine["zg"]), axis=1)
                  + jnp.sum((sl + a_aff * affine["sl"]) * (zl + a_aff * affine["zl"]) + (su + a_aff * affine["su"]) * (zu + a_aff * affine["zu"]), axis=1)) / n_ineq
        center = (jnp.minimum(1.0, mu_aff / jnp.maximum(mu, 1e-300)) ** 3 * mu)[:, None]
        full = direction(rmg + gef * (affine["sg"] * affine["zg"] - center), rml + affine["sl"] * affine["zl"] - op * center,
                         rmu + affine["su"] * affine["zu"] - op * center)
        alpha = jnp.minimum(1.0, 0.995 * to_boundary(v, full))
        sane = jnp.all(jnp.isfinite(full["x"]), axis=1) & jnp.all(jnp.isfinite(full["nu"]), axis=1) & jnp.all(jnp.isfinite(full["zg"]), axis=1)
        # An edge stops once it meets tol, once mu is far below tol (more iterations then only push the
        # slacks toward underflow), or once its residual has grown far past its best (it is diverging).
        go = (metric > tol) & sane & (mu > 1e-3 * tol) & (metric <= 1e3 * best_metric)
        move = jnp.where(go, alpha, 0.0)
        new = {k: v[k] + move[:, None] * jnp.where(jnp.isfinite(full[k]), full[k], 0.0) for k in full}
        new.update(it=v["it"] + 1, active=go, best=best, best_metric=best_metric)
        return new

    x0 = 0.5 * (lo + hi) * op + e0
    half = jnp.where(op > 0.0, 0.5 * (hi - lo), 1.0)
    mu0 = jnp.maximum(jnp.max(jnp.abs(h * x0 + f) * 2.0 * half, axis=1, keepdims=True), 1e-2)
    rx0 = mv(R, x0)
    sg0 = gef * jnp.maximum(rx0, 0.1 * (1.0 + jnp.abs(rx0))) + (1.0 - gef)
    v = dict(x=x0, nu=jnp.zeros_like(rx0), sg=sg0, zg=gef * mu0 / sg0, sl=half, zl=op * mu0 / half, su=half, zu=op * mu0 / half,
             it=jnp.zeros((), jnp.int64), active=jnp.ones(R.shape[0], bool))
    v.update(best={k: v[k] for k in FIELDS}, best_metric=jnp.full(R.shape[0], jnp.inf))
    v = ipm_loop(body, v, ipm)
    last = residuals(v)[-1]
    keep_last = last <= v["best_metric"]
    v = dict(v, **{k: jnp.where(keep_last[:, None], v[k], v["best"][k]) for k in FIELDS})
    metric = jnp.minimum(last, v["best_metric"])
    x = jnp.where(jnp.all(jnp.abs(v["x"]) < 1e-13, axis=1, keepdims=True), 0.0, v["x"])
    return x, eqf * v["nu"] + gef * v["zg"], dict(ipm_res=jnp.max(metric), metric=metric, ipm_it=v["it"], ipm_state=v)


def ipm_loop(body, v, ipm):
    """Runs an IPM's iterations: min_iters of them unconditionally, then until every edge has converged or iters is reached (adaptive), or iters in all. The adaptive loop reads one flag per iteration, which costs a host sync on a GPU (plan 15.5), so the iterations nearly every edge needs run without it. An edge that has converged stops moving, so extra iterations change nothing."""
    iters = int(ipm["iters"])
    first = min(int(ipm.get("min_iters", 0)), iters)
    if first > 0:                                   # each loop traces the body once more, which costs compile time
        v = jax.lax.fori_loop(0, first, lambda _, c: body(c), v)
    if ipm["adaptive"]:
        return jax.lax.while_loop(lambda c: (c["it"] < iters) & jnp.any(c["active"]), body, v)
    return jax.lax.fori_loop(first, iters, lambda _, c: body(c), v)


# ===== EDGE_SOLVERS: EDGE SDP =====
# Plan 8.4 (solver.psd_coupling: edge). The edge step takes the moment matrix's PSD constraint inside:
#   min 1/2 x^T diag(h) x + f^T x  over  edge_qp's rows and bounds  and  B(x) >= 0.
# The split form (copies S, duals Lam) left the PSD coupling to the outer PDHG, and on the tiny
# instance that coupling held most of the fixed-point residual, which the tree's vertex prices cannot
# move. Here the outer problem couples edges through K alone, which is what the tree models.
#
# A batched Mehrotra predictor-corrector with the HKM direction (Helmberg-Rendl-Vanderbei-Wolkowicz,
# Kojima-Shindoh-Hara, Monteiro), in the same sign conventions as edge_qp: stationarity is
#   h x + f - R^T u - (bound terms) - B*(Z) = 0,  X = B(x) + Pad,  X, Z >= 0,  X Z = mu I.
# Linearizing X Z = mu I and symmetrizing gives dZ = sym(X^-1 Rc) - sym(X^-1 dX Z), so the PSD block
# adds M[k, l] = tr(B_k X^-1 B_l Z) to the Newton matrix: symmetric and positive definite, but dense,
# so the arrow structure of edge_qp is lost and the matrix is factored whole (n x n, at most 91 at
# order 1). The rows then go through the same row form as edge_qp. The matrix padding of a smaller
# edge type is an identity block in both X and Z that never moves, so it adds nothing to B*(Z), to the
# complementarity, or to the step length.

SDP_FIELDS = FIELDS + ("X", "Z")


def sdp_maps(d):
    """Where each entry of m sits in the moment matrix (plan 5.1): one-hot rows (E, n, s) for the entry's row a and column b, zero for entries outside the edge's matrix, and the weight c (1/2 on the diagonal, 1 off it) with B(x) = sum_k x_k c_k (E_ab + E_ba)."""
    E, n = d["adj"].shape
    s = d["mat"].shape[1]
    a, b = d["adj"] // s, d["adj"] % s
    inside = ((d["adj_w"] > 0.0) & (jnp.take_along_axis(d["psd_mask"].reshape(E, -1), d["adj"], axis=1) > 0.0)).astype(jnp.float64)
    ua = jax.nn.one_hot(a, s, dtype=jnp.float64) * inside[..., None]
    ub = jax.nn.one_hot(b, s, dtype=jnp.float64) * inside[..., None]
    return ua, ub, jnp.where(a == b, 0.5, 1.0) * inside


def psd_newton(maps, P, Q):
    """M[k, l] = tr(B_k P B_l Q) in the coordinates of m, for symmetric P and Q: with (a, b) the position of entry k and (c, e) that of entry l, c_k c_l (P_bc Q_ea + P_be Q_ca + P_ac Q_eb + P_ae Q_cb). With P = X^-1 and Q = Z it is the HKM term of the Newton matrix."""
    ua, ub, c = maps
    pa, pb, qa, qb = ua @ P, ub @ P, ua @ Q, ub @ Q
    p_aa, p_ab, p_bb = pa @ swap(ua), pa @ swap(ub), pb @ swap(ub)
    q_aa, q_ab, q_bb = qa @ swap(ua), qa @ swap(ub), qb @ swap(ub)
    return c[:, :, None] * c[:, None, :] * (swap(p_ab) * q_ab + p_bb * q_aa + p_aa * q_bb + p_ab * swap(q_ab))


def sdp_newton(d, h, v, maps):
    """The edge SDP's Newton matrix at an iterate v (E, n, n): diag(h) plus the bounds' arrow part (edge_qp's H, written out) plus the PSD term. Also returns the Cholesky factor of X, X^-1 and the bounds' weights."""
    lo, hi, e0 = d["lo"], d["hi"], d["e0"]
    n, s = h.shape[1], v["X"].shape[-1]
    dl, du = v["zl"] / v["sl"], v["zu"] / v["su"]
    arrow = -(lo * dl + hi * du)                    # coupling of entry i to y = x[0]; zero at i = 0, whose bounds are closed
    corner = jnp.sum(lo * lo * dl + hi * hi * du, axis=1)
    H = (jnp.eye(n) * (h + dl + du)[:, None, :] + e0[None, :, None] * arrow[:, None, :] + arrow[:, :, None] * e0[None, None, :]
         + (e0[:, None] * e0[None, :]) * corner[:, None, None])
    lx = chol(v["X"])
    linv = lower_solve(lx, jnp.broadcast_to(jnp.eye(s), v["X"].shape))
    xinv = swap(linv) @ linv
    return H + psd_newton(maps, xinv, v["Z"]), lx, xinv, dl, du


def edge_sdp(d, h, f, ipm):
    """The edge mode's step of every edge: (x (E, n), row multipliers u (E, r), PSD multiplier Z (E, s, s), stats). Same rows, bounds, signs and stopping rules as edge_qp, plus B(x) >= 0. An edge that has converged stops moving while the others finish."""
    R = d["R"]
    eqf, gef, padf, op, lo, hi, e0, n_ineq = d["row_eq"], d["row_ge"], d["row_pad"], d["open"], d["lo"], d["hi"], d["e0"], d["n_ineq"]
    tol = float(ipm["tol"])
    E, n = f.shape
    s = d["mat"].shape[1]
    pm = d["psd_mask"]
    act = jnp.diagonal(pm, axis1=1, axis2=2)          # the rows of the edge's own matrix
    eye_s = jnp.eye(s)
    pad = eye_s * (1.0 - act)[:, None, :]
    n_all = n_ineq + jnp.sum(act, axis=1)
    maps = sdp_maps(d)
    fnorm = 1.0 + jnp.max(jnp.abs(f), axis=1)
    diag_r = jnp.eye(R.shape[1])

    def residuals(v):
        x, X, Z = v["x"], v["X"], v["Z"]
        y = x[:, :1]
        rx = mv(R, x)
        cb = v["zl"] - v["zu"] + e0 * jnp.sum(hi * v["zu"] - lo * v["zl"], axis=1, keepdims=True)
        rd = h * x + f - mtv(R, eqf * v["nu"] + gef * v["zg"]) - cb - psd_adjoint(d, pm * Z)
        ra, rg = eqf * rx, gef * (rx - v["sg"])
        rl, ru = op * (x - lo * y - v["sl"]), op * (hi * y - x - v["su"])
        rX = psd_forward(d, x) + pad - X
        mu = (jnp.sum(gef * v["sg"] * v["zg"], axis=1) + jnp.sum(v["sl"] * v["zl"] + v["su"] * v["zu"], axis=1) + jnp.sum(pm * X * Z, axis=(1, 2))) / n_all
        prim = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(ra) + jnp.abs(rg), axis=1), jnp.max(jnp.maximum(jnp.abs(rl), jnp.abs(ru)), axis=1)), jnp.max(jnp.abs(rX), axis=(1, 2)))
        return rd, ra, rg, rl, ru, rX, mu, jnp.maximum(jnp.max(jnp.abs(rd), axis=1) / fnorm, jnp.maximum(prim, mu))

    def mat_step(factor, dm):
        """The longest step a with M + a dM >= 0, for M = L L^T: -1 / lambda_min(L^-1 dM L^-T), or inf."""
        a = lower_solve(factor, dm)
        lowest = jnp.linalg.eigvalsh(sym(lower_solve(factor, swap(a))))[:, 0]
        return jnp.where(lowest < 0.0, -1.0 / lowest, jnp.inf)

    def to_boundary(v, dv, lx, lz):
        steps = [jnp.min(jnp.where(dv[k] < 0.0, -v[k] / dv[k], jnp.inf), axis=1) for k in SLACKS]
        return jnp.min(jnp.stack(steps + [mat_step(lx, dv["X"]), mat_step(lz, dv["Z"])]), axis=0)

    def body(v):
        rd, ra, rg, rl, ru, rX, mu, metric = residuals(v)
        better = metric < v["best_metric"]                 # best iterate, as in edge_qp
        best = {k: pick(better, v[k], v["best"][k]) for k in SDP_FIELDS}
        best_metric = jnp.minimum(metric, v["best_metric"])
        sg, zg, sl, zl, su, zu, X, Z = (v[k] for k in SLACKS + ("X", "Z"))
        zg_safe = zg + (1.0 - gef)
        H, lx, xinv, dl, du = sdp_newton(d, h, v, maps)
        lh, lz = chol(H), chol(Z)
        hirt = factor_solve(lh, swap(R))                                     # H^-1 R^T (E, n, r)
        factor = chol(R @ hirt + diag_r * (gef * sg / zg_safe + padf + 1e-14 * eqf)[:, None, :])
        carry = xinv @ rX @ Z                                              # X^-1 rX Z, from the PSD block's own residual

        def direction(rmg, rml, rmu, rc):
            tl, tu = dl * rl + rml / sl, du * ru + rmu / su
            t = xinv @ rc
            rhs = (-rd - (tl - tu) - e0 * jnp.sum(hi * tu - lo * tl, axis=1, keepdims=True)
                   + psd_adjoint(d, pm * sym(t)) - psd_adjoint(d, pm * sym(carry)))
            hr = factor_solve(lh, rhs[..., None])[..., 0]
            step = factor_solve(factor, (-ra - rg - gef * rmg / zg_safe - mv(R, hr))[..., None])[..., 0]
            dx = hr + jnp.einsum("enr,er->en", hirt, (eqf + gef) * step)
            dy = dx[:, :1]
            dsl, dsu = op * (dx - lo * dy + rl), op * (hi * dy - dx + ru)
            dX = psd_forward(d, dx) + rX
            return dict(x=dx, nu=eqf * step, sg=gef * (mv(R, dx) + rg), zg=gef * step, sl=dsl, zl=op * (-rml - zl * dsl) / sl,
                        su=dsu, zu=op * (-rmu - zu * dsu) / su, X=dX, Z=pm * sym(t - xinv @ dX @ Z))

        rmg, rml, rmu = gef * sg * zg, sl * zl, su * zu
        xz = X @ Z
        affine = direction(rmg, rml, rmu, -pm * xz)
        a_aff = jnp.minimum(1.0, to_boundary(v, affine, lx, lz))
        a2, a3 = a_aff[:, None], a_aff[:, None, None]
        mu_aff = (jnp.sum(gef * (sg + a2 * affine["sg"]) * (zg + a2 * affine["zg"]), axis=1)
                  + jnp.sum((sl + a2 * affine["sl"]) * (zl + a2 * affine["zl"]) + (su + a2 * affine["su"]) * (zu + a2 * affine["zu"]), axis=1)
                  + jnp.sum(pm * (X + a3 * affine["X"]) * (Z + a3 * affine["Z"]), axis=(1, 2))) / n_all
        center = jnp.minimum(1.0, mu_aff / jnp.maximum(mu, 1e-300)) ** 3 * mu
        c2 = center[:, None]
        full = direction(rmg + gef * (affine["sg"] * affine["zg"] - c2), rml + affine["sl"] * affine["zl"] - op * c2,
                         rmu + affine["su"] * affine["zu"] - op * c2, pm * (center[:, None, None] * eye_s - xz - affine["X"] @ affine["Z"]))
        alpha = jnp.minimum(1.0, 0.995 * to_boundary(v, full, lx, lz))
        finite = [full[k] for k in ("x", "nu", "zg", "X", "Z")] + [lx, lz]       # a failed factor of X or Z would hide the cone's boundary
        sane = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a), axis=tuple(range(1, a.ndim))) for a in finite]), axis=0)
        go = (metric > tol) & sane & (mu > 1e-3 * tol) & (metric <= 1e3 * best_metric)    # edge_qp's stopping rules
        move = jnp.where(go, alpha, 0.0)
        new = {k: v[k] + move.reshape((-1,) + (1,) * (full[k].ndim - 1)) * jnp.where(jnp.isfinite(full[k]), full[k], 0.0) for k in full}
        new.update(it=v["it"] + 1, active=go, best=best, best_metric=best_metric)
        return new

    # Start: edge_qp's point and slacks; X is B(x0) shifted to be safely positive definite (an infeasible
    # start, so rX starts nonzero) and Z = mu0 X^-1, which puts the PSD block on the central path.
    x0 = 0.5 * (lo + hi) * op + e0
    half = jnp.where(op > 0.0, 0.5 * (hi - lo), 1.0)
    mu0 = jnp.maximum(jnp.max(jnp.abs(h * x0 + f) * 2.0 * half, axis=1, keepdims=True), 1e-2)
    rx0 = mv(R, x0)
    sg0 = gef * jnp.maximum(rx0, 0.1 * (1.0 + jnp.abs(rx0))) + (1.0 - gef)
    b0 = psd_forward(d, x0) + pad
    lowest = jnp.linalg.eigvalsh(b0)[:, 0]
    top = jnp.max(jnp.abs(jnp.diagonal(b0, axis1=1, axis2=2)) * act, axis=1)
    shift = jnp.maximum(-lowest, 0.0) + 0.1 * jnp.maximum(top, 1.0)
    X0 = b0 + (shift[:, None] * act)[:, None, :] * eye_s
    linv0 = lower_solve(chol(X0), jnp.broadcast_to(eye_s, X0.shape))
    Z0 = pm * (mu0[:, :, None] * (swap(linv0) @ linv0)) + pad
    v = dict(x=x0, nu=jnp.zeros_like(rx0), sg=sg0, zg=gef * mu0 / sg0, sl=half, zl=op * mu0 / half, su=half, zu=op * mu0 / half, X=X0, Z=Z0,
             it=jnp.zeros((), jnp.int64), active=jnp.ones(E, bool))
    v.update(best={k: v[k] for k in SDP_FIELDS}, best_metric=jnp.full(E, jnp.inf))
    v = ipm_loop(body, v, ipm)
    last = residuals(v)[-1]
    keep_last = last <= v["best_metric"]
    v = dict(v, **{k: pick(keep_last, v[k], v["best"][k]) for k in SDP_FIELDS})
    metric = jnp.minimum(last, v["best_metric"])
    x = jnp.where(jnp.all(jnp.abs(v["x"]) < 1e-13, axis=1, keepdims=True), 0.0, v["x"])
    return x, eqf * v["nu"] + gef * v["zg"], pm * v["Z"], dict(ipm_res=jnp.max(metric), metric=metric, ipm_it=v["it"], ipm_state=v)


# ===== CERTIFICATE =====
# Plan 10.1, tex Thm 5.5. A valid bound for any state prices and any multipliers in their cones. On an
# edge, with m_0 = 1 and feasible m: g^T m >= r^T m for r = g - R^T u - B*(Gamma), since R_eq m = 0,
# u_ge >= 0 with R_ge m >= 0, and Gamma >= 0 with B(m) >= 0. The box then bounds r^T m from below.
# Gamma is the negative part of the matrix the PSD step just decomposed, divided by its step. The plan
# says Pi(-Lam): any PSD matrix is valid, this one tends to -Lam at convergence, and it needs no second
# eigendecomposition. It is built from the negative eigenvalues alone, so it is PSD to rounding relative
# to its own size, whatever the size of the positive part. In the edge mode (plan 8.4) there is no PSD
# step: Gamma is the edge step's own PSD multiplier Z, projected onto the PSD cone the same way. Flow
# prices are left out, since they cancel along a start-goal path. The zero test of plan 9.1 is the same
# bound at the edge step's own data, with the multipliers of the edge's last solve.

def cone_bound(d, g, size, u, gamma):
    """A lower bound on min g^T m over each edge's outer set at m_0 = 1, for any multipliers u (>= 0 on inequality rows) and PSD gamma, with the forward-error allowance of tex Prop signed error: eps_cert times the size of every term, taken before any cancel, since cancellation is where roundoff hides. `size` holds the sizes of g's own terms. The same allowance covers the sum along a path, since it exceeds N u |bound|. A bound that is not finite says nothing, so it becomes -inf, which is valid."""
    r = g - mtv(d["R"], u) - psd_adjoint(d, gamma)
    terms = jnp.minimum(r * d["lo"], r * d["hi"]) * d["wm"]
    size = size + mtv(jnp.abs(d["R"]), jnp.abs(u)) + psd_adjoint(d, jnp.abs(gamma))
    reach = jnp.maximum(jnp.abs(d["lo"]), jnp.abs(d["hi"])) * d["wm"]
    slack = d["eps_cert"] * (1.0 + size[:, 0] + jnp.sum(size * reach, axis=1) + jnp.sqrt(jnp.sum(gamma * gamma, axis=(1, 2))) * d["trace_hi"])
    lb = r[:, 0] + jnp.sum(terms, axis=1) - slack
    return jnp.where(jnp.isfinite(lb), lb, -jnp.inf)


def edge_bounds(d, p, u, gamma):
    """(each edge's bound, the state prices): cone_bound at g = c + K^T (0, lambda), with no center term and no Lam term (plan 10.1)."""
    lam = p * d["state_mask"]
    return cone_bound(d, d["c"] + k_adjoint(d, lam), jnp.abs(d["c"]) + k_adjoint_size(d, lam), u, gamma), lam


def free_bound(d, p, u, gamma):
    """(L_0, L_delta): the shortest path over edge bounds, without and with the delta term."""
    lb, lam = edge_bounds(d, p, u, gamma)
    dist = d["dist0"]
    for idx, tails, heads in d["layers"]:
        dist = jnp.minimum(dist, jax.ops.segment_min(dist[tails] + lb[idx], heads, num_segments=dist.size))
    l0 = dist[d["goal"]]
    return l0, l0 - d["delta"] * jnp.sum(jnp.abs(lam)) * (1.0 + 100.0 * d["eps_cert"])


# ===== PDHG =====
# Plan 7. One round is Chambolle-Pock on the saddle problem with the term <p, K m - b> and, in the split
# mode, the Lagrangian term <Lam, B(m) - S> (at a solution -Lam is PSD). In the edge mode (plan 8.4) the
# PSD constraint is part of each edge's set, so z = (m, p) and K is the whole operator. theta is the
# primal/dual balance: T scales with it and Sigma with its inverse, so it is a value in the state and
# changing it needs no recompile.
#
# A round also reads `aux`, the multipliers of each edge's last solve (u, and gamma in the edge mode).
# The zero test uses them to skip edges whose step is exactly zero (plan 9.1), so they change which
# edges are solved and never the result: any multipliers in their cones give a valid test.

def scalar(x):
    return jnp.asarray(x, dtype=jnp.float64)


def step_data(d, z, theta):
    """The edge step's data at z: (h = 1 / t, the linear term f = c + K^T p (+ B*(Lam)) - m / t, and the sizes of f's terms before they cancel, for the zero test's rounding allowance)."""
    t = d["T"] * theta
    g = d["c"] + k_adjoint(d, z["p"])
    size = jnp.abs(d["c"]) + k_adjoint_size(d, z["p"]) + jnp.abs(z["m"]) / t
    if "Lam" in z:
        g = g + psd_adjoint(d, z["Lam"])
        size = size + psd_adjoint(d, jnp.abs(z["Lam"]))
    return 1.0 / t, g - z["m"] / t, size


def pdhg_step(d, z, aux, theta, ss, response=False):
    """One round from z = (m, p), or (m, S, Lam, p) in the split mode: (next z, the multipliers and stats of the round). With response, also each edge's K_e J_e K_e^T for the tree."""
    sig_p = d["sig_p"] / theta
    h, f, size = step_data(d, z, theta)
    step = edge_step(d, h, f, size, aux, ss, response)
    m = step["x"]
    m_bar = 2.0 * m - z["m"]
    moved = z["p"] + sig_p * (k_forward(d, m_bar) - d["b"])
    p = jnp.where(d["is_flow"], moved, jnp.sign(moved) * jnp.maximum(jnp.abs(moved) - sig_p * d["delta"], 0.0))
    extra = dict(u=step["u"], gamma=step["gamma"], ipm_res=step["res"], ipm_it=step["it"], solved=step["solved"], response=step["response"])
    if ss["psd"] == "edge":
        return dict(m=m, p=p), extra
    t_s, sig_l = d["t_s"] * theta, d["sig_l"] / theta
    x = z["S"] + t_s * z["Lam"]
    w, vec = jnp.linalg.eigh(x)
    s = (vec * jnp.maximum(w, 0.0)[:, None, :]) @ swap(vec)
    lam = z["Lam"] + sig_l * (psd_forward(d, m_bar) - (2.0 * s - z["S"]))
    extra["gamma"] = (vec * jnp.maximum(-w, 0.0)[:, None, :]) @ swap(vec) / t_s
    return dict(m=m, S=s, Lam=lam, p=p), extra


def p_inner(d, a, b, theta):
    """<a, b>_P in the metric that makes a round firmly nonexpansive (plan 11.6, tex Lemma 4.3), with P = [T^-1, -A^T; -A, Sigma^-1] and A the whole operator: K m, plus B(m) - S in the split mode. A row with no coefficients has Sigma = 0 and its dual never moves, so it counts for nothing."""
    t, sig_p = d["T"] * theta, d["sig_p"] / theta
    plain = jnp.sum(a["m"] * b["m"] / t) + jnp.sum(jnp.where(sig_p > 0.0, a["p"] * b["p"] / jnp.where(sig_p > 0.0, sig_p, 1.0), 0.0))
    cross = jnp.sum(k_forward(d, a["m"]) * b["p"]) + jnp.sum(k_forward(d, b["m"]) * a["p"])
    if "S" in a:
        t_s, sig_l = d["t_s"] * theta, d["sig_l"] / theta
        plain = plain + jnp.sum(a["S"] * b["S"]) / t_s + jnp.sum(jnp.where(sig_l > 0.0, a["Lam"] * b["Lam"] / jnp.where(sig_l > 0.0, sig_l, 1.0), 0.0))
        cross = cross + jnp.sum((psd_forward(d, a["m"]) - a["S"]) * b["Lam"]) + jnp.sum((psd_forward(d, b["m"]) - b["S"]) * a["Lam"])
    return plain - cross


def fixed_point_residual(d, z, aux, theta, ss):
    """||z - T(z)||_P: the KKT error used for restarts (plan 7.4). It is zero exactly at a saddle point, and it needs no projection onto the normal cones of the edge polytopes."""
    next_z, _ = pdhg_step(d, z, aux, theta, ss)
    res = {k: z[k] - next_z[k] for k in z}
    return jnp.sqrt(jnp.maximum(p_inner(d, res, res, theta), 0.0))


def candidate_stats(d, z):
    """A candidate's flow objective and its feasibility violations for the stopping test (plan 10.5)."""
    m = z["m"]
    r = k_forward(d, m) - d["b"]
    rows = mv(d["R"], m)
    y = m[:, :1]
    box = jnp.maximum(d["open"] * (d["lo"] * y - m), d["open"] * (m - d["hi"] * y))
    b_m = psd_forward(d, m)
    norm = jnp.sqrt(jnp.sum(b_m * b_m, axis=(1, 2)))
    return dict(U=jnp.sum(d["c"] * m), flow=jnp.max(jnp.abs(r) * d["is_flow"]), state=jnp.max(jnp.abs(r) * d["state_mask"]),
                rows=jnp.maximum(jnp.max(jnp.abs(rows) * d["row_eq"]), jnp.maximum(jnp.max(-rows * d["row_ge"]), jnp.max(box))),
                psd=jnp.max(-jnp.linalg.eigvalsh(b_m)[:, 0] / (1.0 + norm)))


def initial_state(d, theta, ss):
    """Cold start: zeros (plan 12.1), with no stored multipliers. Zero multipliers are valid, so the first zero test uses the bounds alone."""
    E, r = d["c"].shape[0], d["R"].shape[1]
    zero = dict(m=jnp.zeros_like(d["c"]), p=jnp.zeros_like(d["b"]))
    if ss["psd"] == "split":
        zero.update(S=jnp.zeros_like(d["sig_l"]), Lam=jnp.zeros_like(d["sig_l"]))
    return dict(z=zero, avg=zero, count=scalar(0.0), theta=scalar(theta), l0_best=scalar(-np.inf), ld_best=scalar(-np.inf), ipm_res=scalar(0.0),
                ipm_it=jnp.zeros((), jnp.int64), solved=scalar(0.0), u=jnp.zeros((E, r)), gamma=jnp.zeros_like(d["sig_l"]))


def aux_of(state):
    """The multipliers a round reads: from each edge's last solve, for the zero test, and the free bound's between rounds."""
    return dict(u=state["u"], gamma=state["gamma"])


def solver_settings(cfg, d):
    """What a round needs from the config as plain values (static under jit): where the PSD constraint lives, both IPMs' settings, and skipping (plan 9) with its bucket sizes for d's edge count."""
    sc, sk = cfg["solver"], cfg["skipping"]
    E = int(d["c"].shape[0])
    buckets = sorted({0, E} | {int(b) for b in sk.get("buckets", []) if 0 < int(b) < E})
    return dict(psd=psd_coupling(cfg), ipm=dict(sc["ipm"]), ipm_sdp=dict(sc.get("ipm_sdp", sc["ipm"])), zero_test=bool(sk["zero_test"]),
                lb_tol=float(sk["lb_tol"]), compaction=str(sk["compaction"]), buckets=tuple(buckets))


def check_settings_built(cfg):
    """Stops a run that asks for something the solver does not do yet. It would otherwise run something else under that name, and comparisons against it would mean nothing."""
    psd_coupling(cfg)
    if cfg["solver"].get("theta_psd", "same") != "same":
        raise NotImplementedError("solver.theta_psd is not wired into the step sizes: every round uses theta_pd for the PSD rows too")
    proj = cfg["solver"]["psd_proj"]
    if proj not in ("auto", "eigh"):
        raise NotImplementedError(f"solver.psd_proj {proj!r} is not wired into the solver yet: the split mode uses eigh, and the edge mode has no projection (plan 8.2, 8.4)")
    sk = cfg["skipping"]
    asked = [k for k in ("face_cache", "gershgorin_skip") if sk[k]]
    if asked:
        raise NotImplementedError(f"skipping.{asked} is not built: the zero test and compaction are (plan 9). The face cache is exact only for polyhedral steps, "
                                  "so it cannot serve the edge mode's SDP step, and the Gershgorin skip only saves eigh calls in the split mode")
    if sk["compaction"] not in ("none", "bucketed"):
        raise ValueError(f"skipping.compaction must be none or bucketed, not {sk['compaction']!r}")
    if int(cfg["certificate"]["free_every"]) < 1:
        raise ValueError("certificate.free_every must be at least 1")


def compile_solver(cfg, d):
    """Ahead-of-time compiled pieces (plan 7.5): one scan block, the block-end diagnostics, the restart residual and, with a tree, its direction and trial. A tree runs between blocks, so a block is then tree.every rounds. The free bound runs every certificate.free_every rounds, which must divide a block. Returns (pieces, seconds)."""
    check_settings_built(cfg)
    sc = cfg["solver"]
    ss = solver_settings(cfg, d)
    tree = tree_settings(cfg)
    rounds = int(cfg["tree"]["every"]) if tree else int(sc["rounds_per_scan"])
    every = int(cfg["certificate"]["free_every"])
    if rounds % every:
        raise ValueError(f"certificate.free_every {every} must divide the {rounds} rounds of a scan block")

    def block_fn(data, state):
        def one(st, _):
            z, extra = pdhg_step(data, st["z"], aux_of(st), st["theta"], ss)
            count = st["count"] + 1.0
            avg = jax.tree_util.tree_map(lambda a, b: a + (b - a) / count, st["avg"], z)
            return dict(st, z=z, avg=avg, count=count, u=extra["u"], gamma=extra["gamma"], ipm_res=jnp.maximum(st["ipm_res"], extra["ipm_res"]),
                        ipm_it=jnp.maximum(st["ipm_it"], extra["ipm_it"]), solved=st["solved"] + extra["solved"]), None

        def group(st, _):
            # Any prices and multipliers give a valid bound, so the running maximum over every
            # free_every-th round is valid too; it costs N sequential steps, so not every round.
            st, _ = jax.lax.scan(one, st, None, length=every)
            l0, ld = free_bound(data, st["z"]["p"], st["u"], st["gamma"])
            return dict(st, l0_best=jnp.maximum(st["l0_best"], l0), ld_best=jnp.maximum(st["ld_best"], ld)), None

        return jax.lax.scan(group, state, None, length=rounds // every)[0]

    def diag_fn(data, state):
        aux = aux_of(state)
        return dict(cur=candidate_stats(data, state["z"]), avg=candidate_stats(data, state["avg"]),
                    e_cur=fixed_point_residual(data, state["z"], aux, state["theta"], ss), e_avg=fixed_point_residual(data, state["avg"], aux, state["theta"], ss))

    state = initial_state(d, 1.0, ss)
    aux = aux_of(state)
    t0 = time.perf_counter()
    pieces = dict(block=jax.jit(block_fn).lower(d, state).compile(), diag=jax.jit(diag_fn).lower(d, state).compile(),
                  residual=jax.jit(lambda data, z, a, th: fixed_point_residual(data, z, a, th, ss)).lower(d, state["z"], aux, state["theta"]).compile(), block_rounds=rounds)
    if tree:
        pieces.update(direction=jax.jit(lambda data, z, a, th: tree_direction(data, z, a, th, ss, tree)).lower(d, state["z"], aux, state["theta"]).compile(),
                      trial=jax.jit(lambda data, z, a, dp, tau, th: tree_trial(data, z, a, dp, tau, th, ss)).lower(d, state["z"], aux, state["z"]["p"], state["theta"], state["theta"]).compile())
    return pieces, time.perf_counter() - t0


STOPPED = ("converged_eta", "plateau")     # the statuses that count as a finished solve (plan 10.5)


def run_pdhg(cfg, d, pieces, theta, rounds_max, time_budget, stall_rounds, eta, log_path=None, progress=True, plateau=True):
    """Rounds until a stop or a safety stop (plan 10.5, 12.1), with restarts to the better of the current iterate and the average (plan 7.4) and, when compiled with a tree, a tree pass after each block (plan 11.6). Host work happens only between scan blocks. Returns (result dict, final state)."""
    sc = cfg["solver"]
    stop = sc["stop"]
    strict, flat = stop["strict"], stop["plateau"]
    delta, eps_flow, eps_psd, eps_row = float(sc["delta"]), float(stop["eps_flow"]), float(stop["eps_psd"]), float(stop["eps_row"])
    below = float(stop.get("below_rel", 0.01))
    beta_s, beta_n, beta_a = float(sc["restarts"]["beta_sufficient"]), float(sc["restarts"]["beta_necessary"]), float(sc["restarts"]["beta_artificial"])
    adapt = sc["primal_weight_update"] == "pdlp"
    state = initial_state(d, theta, solver_settings(cfg, d))
    E = int(d["c"].shape[0])
    z_restart = state["z"]
    err_restart = float(pieces["residual"](d, state["z"], aux_of(state), state["theta"]))
    first_err, prev_candidate, restarts = err_restart, np.inf, 0
    rounds, epoch_start, status, best_u, last_print = 0, 0, None, np.inf, 0.0
    best_seen, best_round = (-np.inf, np.inf), 0
    below_l, ipm_worst, ipm_most = 0.0, 0.0, 0
    seen_rounds, seen_bounds = [], []
    guard = new_guard(err_restart) if "direction" in pieces else None
    block_rounds = int(pieces["block_rounds"])
    blocks = max(1, int(sc["rounds_per_scan"]) // block_rounds)
    passes = lambda c, flow, psd, row: c["flow"] <= flow and c["state"] <= delta + flow and c["rows"] <= row and c["psd"] <= psd
    log = open(log_path, "w") if log_path else None
    tree_log = open(Path(log_path).with_name("tree.jsonl"), "w") if log_path and guard is not None else None
    t0 = time.perf_counter()
    while status is None:
        ipm_res, ipm_it, solved = 0.0, 0, 0.0
        for _ in range(blocks):
            state = dict(state, ipm_res=scalar(0.0), ipm_it=jnp.zeros((), jnp.int64), solved=scalar(0.0))
            state = pieces["block"](d, state)
            rounds += block_rounds
            solved += float(state["solved"])           # edge steps the zero test left to the IPM, ordinary rounds only (plan 9.4)
            if guard is not None:
                state, used = tree_attempt(cfg["tree"]["safeguard"], d, pieces, state, guard)
                rounds += used
                if tree_log:
                    tree_log.write(json.dumps(dict(round=rounds, **guard["last"])) + "\n")
            ipm_res, ipm_it = max(ipm_res, float(state["ipm_res"])), max(ipm_it, int(state["ipm_it"]))
        solved_share = solved / (blocks * block_rounds * E)
        jax.block_until_ready(state["z"]["m"])
        diag = jax.device_get(pieces["diag"](d, state))
        now = time.perf_counter() - t0
        l0, ld = float(state["l0_best"]), float(state["ld_best"])
        ipm_worst, ipm_most = max(ipm_worst, ipm_res), max(ipm_most, ipm_it)
        candidates = (diag["cur"], diag["avg"])
        passing = [c for c in candidates if passes(c, eps_flow, eps_psd, eps_row)]
        u = min((float(c["U"]) for c in passing), default=np.inf)
        gap = (u - ld) / max(abs(u), 1e-12) if passing else np.inf
        # A candidate that costs clearly less than the certified bound is infeasible beyond what the
        # tolerances admit, so it neither stops the solve nor counts as the reported flow cost.
        # U_flow is the latest plausible cost, not the minimum over history: early flows pass the loose
        # tolerances while costing far below the bound that later rounds certify.
        plausible = passing and gap >= -below
        if plausible:
            best_u = u
        exact = [float(c["U"]) for c in candidates if passes(c, float(strict["flow"]), float(strict["psd"]), float(strict["row"]))]
        if exact:
            below_l = max(below_l, ld - min(exact))
            if ld - min(exact) > 1e-5 * max(1.0, abs(ld)):
                raise RuntimeError(f"round {rounds}: a feasible flow costs {min(exact)} but the certified bound is {ld}: a bug in the bound or the checks (plan 10.5)")
        u_seen = u if plausible else np.inf
        if ld > best_seen[0] + 1e-6 * max(1.0, abs(ld)) or u_seen < best_seen[1] - 1e-6 * max(1.0, abs(u_seen)):
            best_seen, best_round = (max(best_seen[0], ld), min(best_seen[1], u_seen)), rounds
        seen_rounds.append(rounds)
        seen_bounds.append(ld)
        earlier = bisect.bisect_left(seen_rounds, (1.0 - float(flat["window"])) * rounds)
        flattened = (plateau and rounds >= int(flat["min_rounds"]) and ld > 0.0 and earlier < len(seen_rounds) - 1
                     and ld - seen_bounds[earlier] < float(flat["gain"]) * ld)     # all costs are nonnegative, so a bound at or below zero says nothing yet
        if not math.isfinite(float(diag["cur"]["U"])):
            status = "diverged"
        elif eta >= 0.0 and plausible and gap <= eta:     # U above L, or below it only by `below`: crossing from below is not convergence
            status = "converged_eta"
        elif flattened:
            status = "plateau"
        elif rounds >= rounds_max:
            status = "round_cap"
        elif now >= time_budget:
            status = "time_cap"
        elif rounds - best_round >= stall_rounds:
            status = "stalled"
        e_cur, e_avg = float(diag["e_cur"]), float(diag["e_avg"])
        use_avg = e_avg < e_cur
        candidate_err = min(e_cur, e_avg)
        restarted = False
        # The KKT error cannot rise for the current iterate, since a round is nonexpansive, so the second
        # test rarely fires on its own. The third is PDLP's artificial restart, which keeps restarts coming.
        due = (candidate_err <= beta_s * err_restart or (candidate_err <= beta_n * err_restart and candidate_err > prev_candidate)
               or rounds - epoch_start >= beta_a * rounds)
        if status is None and due:
            z = state["avg"] if use_avg else state["z"]
            th = state["theta"]
            if adapt:
                dx = math.sqrt(sum(float(jnp.sum((z[k] - z_restart[k]) ** 2)) for k in ("m", "S") if k in z))
                dy = math.sqrt(sum(float(jnp.sum((z[k] - z_restart[k]) ** 2)) for k in ("p", "Lam") if k in z))
                if dx > 1e-10 and dy > 1e-10:
                    th = scalar(1.0 / math.exp(0.5 * math.log(dy / dx) + 0.5 * math.log(1.0 / float(th))))
            state = dict(state, z=z, avg=z, count=scalar(0.0), theta=th)
            z_restart = z
            err_restart = float(pieces["residual"](d, z, aux_of(state), th)) if adapt else candidate_err
            prev_candidate, restarted, epoch_start = np.inf, True, rounds
            restarts += 1
            if guard is not None:
                guard["r_safe"] = err_restart
        else:
            prev_candidate = candidate_err
        line = dict(round=rounds, seconds=now, L_delta=ld, L_0=l0, U_flow=u, gap=gap, kkt_cur=e_cur, kkt_avg=e_avg, err_restart=err_restart, restarted=restarted, restarts=restarts,
                    theta=float(state["theta"]), ipm_res=ipm_res, ipm_it=ipm_it, solved_share=solved_share, cur={k: float(v) for k, v in diag["cur"].items()}, avg={k: float(v) for k, v in diag["avg"].items()}, tree=guard)
        if log:
            log.write(json.dumps(line) + "\n")
        if progress and (now - last_print >= float(sc["log_every_s"]) or status):
            last_print = now
            tree_note = ""
            if guard is not None:
                last = guard["last"]
                ratios = "/".join(f"{t['ratio']:.2f}" for t in last["trials"]) or "no tries"
                tree_note = (f"  tree {guard['attempts']} tried, {guard['accepted_k1']} + {guard['accepted_k2']} kept, {guard['fallback']} fell back, {guard['rejected_ipm']} tries rejected;"
                             f" last step {last['d_ratio']:.1f} x residual, largest price change {last['dp_max']:.1e}, residual ratio per try {ratios}")
            print(f"  round {rounds:7d} {now:8.1f} s  {1e3 * now / rounds:8.3f} ms/round  L_delta {ld:.6f}  U_cur {float(diag['cur']['U']):.6f}  U_ok {u:.6f}  flow {float(diag['cur']['flow']):.1e}  state {float(diag['cur']['state']):.1e}"
                  f"  psd {float(diag['cur']['psd']):.1e}  kkt {candidate_err:.2e}  restarts {restarts}  ipm its {ipm_it} res {ipm_res:.1e}  solved {solved_share:.0%}" + tree_note, flush=True)
    if log:
        log.close()
    if tree_log:
        tree_log.close()
    seconds = time.perf_counter() - t0
    current = diag["cur"]
    score = math.sqrt(max(float(current["U"]) - ld, ld - float(current["U"])) ** 2 + float(current["flow"]) ** 2 + max(float(current["state"]) - delta, 0.0) ** 2 + max(float(current["psd"]), 0.0) ** 2)
    return dict(status=status, rounds=rounds, seconds=seconds, ms_per_round=1e3 * seconds / rounds, L_delta=ld, L_0=l0, U_flow=None if not math.isfinite(best_u) else best_u,
                U_current=float(current["U"]), gap=None if not math.isfinite(gap) else gap, restarts=restarts, first_kkt=first_err, last_restart_kkt=err_restart, theta=float(state["theta"]),
                ipm_res=ipm_worst, ipm_it=ipm_most, largest_bound_excess=below_l, score=score if math.isfinite(score) else float("inf"), tree=guard), state


def choose_theta(cfg, d, pieces, label):
    """The primal/dual balance (plan 7.3): a number from the config, or the grid value with the smallest error after the same short run, stored per configuration, PSD coupling and task (the two couplings have different step sizes, so one's value says nothing about the other)."""
    sc = cfg["solver"]
    if sc["theta_pd"] != "auto":
        return float(sc["theta_pd"])
    path = ROOT / cfg["output"]["root"] / "tuned" / f"{label}_{psd_coupling(cfg)}_{cfg['task']}.yaml"
    if path.exists():
        return float(yaml.safe_load(path.read_text())["theta_pd"])
    t0 = time.perf_counter()
    scores = {}
    for theta in sc["tune"]["grid"]:
        result, _ = run_pdhg(cfg, d, pieces, float(theta), int(sc["tune"]["rounds"]), float(sc["time_budget_s"]), 10 ** 12, -1.0, progress=False, plateau=False)
        scores[float(theta)] = result["score"]
        print(f"  tune theta {float(theta):7.4f}: score {result['score']:.3e}  L_delta {result['L_delta']:.6f}  U_cur {result['U_current']:.6f}  {result['ms_per_round']:.3f} ms/round", flush=True)
    best = min(scores, key=lambda k: scores[k] if math.isfinite(scores[k]) else np.inf)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(theta_pd=best, scores=scores, seconds=time.perf_counter() - t0)))
    print(f"  chose theta {best} in {time.perf_counter() - t0:.1f} s; saved {path}", flush=True)
    return best


def component_times(cfg, d, state):
    """One-off timings of a round's and a tree pass's pieces at a real state, for finding the hot spot: each compiled and run alone, so launch overhead is not shared as it is in the scan. edge_step is the edges' work with skipping as configured; edge_solve_all solves every edge, which is what skipping saves against."""
    ss = solver_settings(cfg, d)
    z, theta, aux = state["z"], state["theta"], aux_of(state)
    h, f, size = step_data(d, z, theta)
    pieces = dict(edge_step=(lambda a, b, c: edge_step(d, a, b, c, aux, ss)["x"], (h, f, size)), edge_solve_all=(lambda a, b: edge_solve(d, a, b, ss, False)["x"], (h, f)),
                  k_forward=(lambda m: k_forward(d, m), (z["m"],)), k_adjoint=(lambda p: k_adjoint(d, p), (z["p"],)),
                  free_bound=(lambda p, u, gamma: free_bound(d, p, u, gamma), (z["p"], state["u"], state["gamma"])), round=(lambda st: pdhg_step(d, st, aux, theta, ss)[0], (z,)))
    if "Lam" in z:
        pieces.update(eigh=(jnp.linalg.eigh, (z["S"] + d["t_s"] * theta * z["Lam"],)), psd_adjoint=(lambda lam: psd_adjoint(d, lam), (z["Lam"],)))
    times = {name: timed(fn, *args)[0] * 1e3 for name, (fn, args) in pieces.items()}
    tree = tree_settings(cfg)
    if tree:
        wall, _, extra = timed(lambda st: pdhg_step(d, st, aux, theta, ss, response=True)[1], z)
        times["round_with_response"] = wall * 1e3
        wall, _, (D, E) = timed(lambda a: tree_assemble(d, a), extra["response"])
        times["tree_assemble"] = wall * 1e3
        mean = jnp.maximum(jnp.mean(jnp.diagonal(D, axis1=1, axis2=2)), 1e-300)
        wall, _, factors = timed(tree_factor, D + jnp.eye(D.shape[1]) * tree["reg"] * mean, E)
        times["tree_factor"] = wall * 1e3
        times["tree_solve"] = timed(tree_solve, factors, tree_write(d, z["p"]))[0] * 1e3
        times["tree_direction"] = timed(lambda a, b: tree_direction(d, a, aux, b, ss, tree), z, theta)[0] * 1e3
    return times


# ===== SKIPPING =====
# Milestone M5 (plan section 9). The zero test is exact: with stored multipliers it bounds the edge's
# pricing value pi_e(f) from below (cone_bound), and pi_e(f) >= 0 means f lies in the dual cone of the
# edge's cone, where the step is exactly zero. The multipliers are those of the edge's last solve; when
# that solve ended at the apex they certify f with a positive margin as long as f is interior to the
# dual cone, so the test keeps passing while prices drift by less than that margin. In the split mode
# the step's set has no PSD block (Lam is in f instead), so its test uses no gamma: a PSD multiplier
# would certify the smaller cone and could skip a step that is not zero.
#
# Compaction (plan 9.3): the edges the test leaves go to the IPM in the smallest bucket that holds them,
# chosen on the device by lax.switch over buckets compiled ahead of time; padding slots repeat the first
# edge in need and their results are dropped. With compaction none, every edge is solved and the zero edges are
# overwritten, which is the plan's verification mode.

EDGE_KEYS = ("R", "Rw", "Ry", "row_eq", "row_ge", "row_pad", "open", "lo", "hi", "n_ineq", "mat", "psd_mask", "adj", "adj_w", "tail_idx", "head_idx", "tail_coef", "head_coef")


def edge_subset(d, idx):
    """d with the edge step's per-edge arrays taken at the edges idx."""
    return dict(d, **{k: d[k][idx] for k in EDGE_KEYS})


def edge_response_of(d, h, v, ss):
    return edge_response_sdp(d, h, v) if ss["psd"] == "edge" else edge_response(d, h, v)


def edge_solve(d, h, f, ss, response):
    """The IPM on every edge of d: the split mode's QP or the edge mode's SDP, with the multipliers the zero test and the certificate read (gamma is the PSD multiplier projected onto its cone, edge mode only), per-edge residuals, and with response the tree's K_e J_e K_e^T."""
    if ss["psd"] == "edge":
        x, u, Z, st = edge_sdp(d, h, f, ss["ipm_sdp"])
        gamma = psd_eigh(Z)
    else:
        x, u, st = edge_qp(d, h, f, ss["ipm"])
        gamma = None
    g = edge_response_of(d, h, st["ipm_state"], ss) if response else None
    return dict(x=x, u=u, gamma=gamma, response=g, metric=st["metric"], it=st["ipm_it"], state=st["ipm_state"])


def solve_bucketed(d, h, f, need, aux, ss, response):
    """Plan 9.3: the IPM on the smallest compiled bucket that holds every edge in need, scattered back. Edges outside need get x = 0, keep their stored multipliers and respond with zero."""
    E = f.shape[0]
    edge = ss["psd"] == "edge"
    shape = (E, 2 * d["tail_idx"].shape[1], 2 * d["tail_idx"].shape[1])

    def branch(B):
        def run(h, f, need, u_old, gamma_old):
            x = jnp.zeros_like(f)
            g = jnp.zeros(shape) if response else None
            if B == 0:
                return dict(x=x, u=u_old, gamma=gamma_old, response=g, res=scalar(0.0), it=jnp.zeros((), jnp.int64))
            idx = jnp.nonzero(need, size=B, fill_value=E)[0]
            keep = jnp.where(idx < E, idx, idx[0])        # padding repeats an edge in need, so it never keeps the adaptive loop running longer
            sol = edge_solve(edge_subset(d, keep), h[keep], f[keep], ss, response)
            return dict(x=x.at[idx].set(sol["x"], mode="drop"), u=u_old.at[idx].set(sol["u"], mode="drop"),
                        gamma=gamma_old.at[idx].set(sol["gamma"], mode="drop") if edge else gamma_old,
                        response=g.at[idx].set(sol["response"], mode="drop") if response else None,
                        res=jnp.max(jnp.where(idx < E, sol["metric"], 0.0)), it=jnp.asarray(sol["it"], jnp.int64))
        return run

    index = jnp.searchsorted(jnp.asarray(ss["buckets"]), jnp.sum(need))
    return jax.lax.switch(index, [branch(B) for B in ss["buckets"]], h, f, need, aux["u"], aux["gamma"])


def edge_step(d, h, f, size, aux, ss, response=False):
    """Every edge's step (plan 8, 9): the zero test with the stored multipliers aux, then the IPM on the edges it leaves. Returns x, the multipliers to store (u, and gamma in the edge mode; the split mode returns aux's gamma unchanged), the largest IPM residual over solved edges, the iteration count, the number of edges solved, and with response the tree's edge blocks (zero for skipped edges, whose step stays zero under small price changes)."""
    E = f.shape[0]
    edge = ss["psd"] == "edge"
    if ss["zero_test"]:
        zero = cone_bound(d, f, size, aux["u"], aux["gamma"] if edge else jnp.zeros_like(aux["gamma"])) >= ss["lb_tol"]
    else:
        zero = jnp.zeros(E, bool)
    need = ~zero
    if ss["compaction"] == "bucketed":
        out = solve_bucketed(d, h, f, need, aux, ss, response)
    else:
        sol = edge_solve(d, h, f, ss, response)
        out = dict(x=pick(zero, jnp.zeros_like(sol["x"]), sol["x"]), u=pick(zero, aux["u"], sol["u"]),
                   gamma=pick(zero, aux["gamma"], sol["gamma"]) if edge else aux["gamma"],
                   response=pick(zero, jnp.zeros_like(sol["response"]), sol["response"]) if response else None,
                   res=jnp.max(jnp.where(need, sol["metric"], 0.0)), it=jnp.asarray(sol["it"], jnp.int64))
    return dict(out, solved=jnp.sum(need).astype(jnp.float64))


# ===== TREES =====
# Plan 11.1, 11.2, 11.6 and tex 4 (tree pass, safeguard): the exact tree, milestone M6. The compressed
# trees follow at M7-M8.
#
# With faces frozen, an edge step is affine in its prices, so the flow and state rows respond through
# S = sum_e K_e J_e K_e^T, which is block tridiagonal by layer. A tree pass solves S dp = r for a change
# of prices, and a real round then checks that change. The tree never writes L: any prices give a valid
# bound, so a bad direction costs time and nothing else.
#
# J_e comes from the edge step's last IPM iterate, as the derivative of the step with respect to its
# linear term. An eigendecomposition of the frozen rows would run matrix by matrix above 32 x 32. This is
# the arrow matrix the IPM already factors, plus a Cholesky of the few rows after elimination, and as the
# barrier goes to zero it tends to the frozen-face response of the tex: an active bound's weight grows
# without limit, an inactive bound's goes to zero, and at a cone's apex every entry is pinned, so J_e -> 0.
# In the edge mode (plan 8.4) the step's Newton matrix is dense, with the PSD block's HKM term in it, and
# J_e comes from its Cholesky factor the same way (edge_response_sdp). Edges the zero test skipped
# respond with zero: their step stays exactly zero under price changes smaller than the test's margin.
#
# Prices sit in a padded layout (L, w): the layer, then slot * q + coordinate. A missing vertex, a padding
# layer or a row with no coefficients becomes an identity row with a zero right-hand side.

def tree_settings(cfg):
    """The exact tree's settings as plain values, or None with no tree. Anything else raises, so a run cannot silently use less than its config asks for."""
    tc = cfg["tree"]
    if tc["type"] == "none":
        return None
    if tc["type"] != "exact":
        raise NotImplementedError(f"tree.type {tc['type']!r} comes at M7-M8: use none or exact")
    sg = tc["safeguard"]
    if tc["jacobian"] != "one_step" or tc["primal_predict"] or sg["type"] != "supermann" or float(sg["c0"]) != 0.0:
        raise NotImplementedError("the exact tree covers tree.jacobian one_step, tree.primal_predict false, and the SuperMann safeguard with c0 = 0 so far")
    return dict(reg=float(tc["reg"]), floor=float(tc["floor"]), filter=bool(tc["filter"]), d_max=float(sg["d_max"]))


def lower_solve(a, b, transpose=False):
    """L^-1 b, or L^-T b, for lower-triangular a; b has columns."""
    return jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True, transpose_a=transpose)


def edge_response(d, h, v):
    """K_e J_e K_e^T of every edge from its final IPM iterate v, as (E, 2q, 2q) with the tail's rows first (plan 11.1). J = H^-1 - H^-1 R^T S^-1 R H^-1 with H^-1 = diag(1 / dw) + zeta zeta^T / sigma, the arrow inverse of edge_qp. The rows are scaled to a unit diagonal before the small shift that handles rows the frozen face makes redundant: an inactive row's weight grows without limit as the IPM converges, so a shift set by the largest diagonal would swamp the active rows. An edge whose factorization still fails responds with zero, and the safeguard judges the result."""
    R, Rw, lo, hi, gef = d["R"], d["Rw"], d["lo"], d["hi"], d["row_ge"]
    E, r = R.shape[:2]
    dl, du = v["zl"] / v["sl"], v["zu"] / v["su"]
    dw = h + dl + du
    e = -(lo * dl + hi * du) / dw
    zeta = d["e0"] - e
    sigma = h[:, 0] + jnp.sum((h * (lo * lo * dl + hi * hi * du) + dl * du * (hi - lo) ** 2) / dw, axis=1)
    t = d["Ry"] - mv(Rw, e)
    s = jnp.einsum("ern,esn->ers", Rw / dw[:, None, :], Rw) + t[:, :, None] * t[:, None, :] / sigma[:, None, None]
    s = s + jnp.eye(r) * (gef * v["sg"] / jnp.maximum(v["zg"] + (1.0 - gef), 1e-300) + d["row_pad"])[:, None, :]
    unit = 1.0 / jnp.sqrt(jnp.maximum(jnp.diagonal(s, axis1=1, axis2=2), 1e-300))
    factor = jax.lax.linalg.cholesky(unit[:, :, None] * s * unit[:, None, :] + jnp.eye(r) * 1e-12, symmetrize_input=False)
    cols = jnp.concatenate([d["tail_idx"], d["head_idx"]], axis=1)
    kc = jnp.concatenate([-d["tail_coef"], d["head_coef"]], axis=1)
    rows = Rw / dw[:, None, :] + t[:, :, None] * zeta[:, None, :] / sigma[:, None, None]
    x = lower_solve(factor, unit[:, :, None] * jnp.take_along_axis(rows, jnp.broadcast_to(cols[:, None, :], (E, r, cols.shape[1])), axis=2) * kc[:, None, :])
    z = kc * jnp.take_along_axis(zeta, cols, axis=1)
    free = jnp.take_along_axis(d["wm"] / dw, cols, axis=1)
    g = ((cols[:, :, None] == cols[:, None, :]) * kc[:, :, None] * kc[:, None, :] * free[:, :, None]
         + z[:, :, None] * z[:, None, :] / sigma[:, None, None] - swap(x) @ x)
    return jnp.where(jnp.all(jnp.isfinite(g), axis=(1, 2))[:, None, None], g, 0.0)


def edge_response_sdp(d, h, v):
    """edge_response for the edge mode (plan 8.4): the same J = H^-1 - H^-1 R^T S^-1 R H^-1, with H now the SDP step's dense Newton matrix at its final iterate, which edge_sdp factors anyway. Its PSD term tr(B_k X^-1 B_l Z) holds the frozen face's PSD part: it pins directions that leave the null space of X, as X^-1 grows there, and on the rest it keeps the curvature term that lets a low-rank moment matrix rotate, i.e. its state change. As the barrier goes to zero this tends to the derivative of the exact step where strict complementarity and nondegeneracy hold. Rows are scaled to a unit diagonal before the small shift, as in edge_response."""
    R, gef = d["R"], d["row_ge"]
    E, r, n = R.shape
    H = sdp_newton(d, h, v, sdp_maps(d))[0]
    factor = chol(H)
    cols = jnp.concatenate([d["tail_idx"], d["head_idx"]], axis=1)
    kc = jnp.concatenate([-d["tail_coef"], d["head_coef"]], axis=1)
    pick_cols = jax.nn.one_hot(cols, n, dtype=jnp.float64) * kc[..., None]              # K_e restricted to its coupled coordinates (E, 2q, n)
    a1, a2 = lower_solve(factor, swap(pick_cols)), lower_solve(factor, swap(R))          # L^-1 K_e^T, L^-1 R^T
    s = swap(a2) @ a2 + jnp.eye(r) * (gef * v["sg"] / jnp.maximum(v["zg"] + (1.0 - gef), 1e-300) + d["row_pad"])[:, None, :]
    unit = 1.0 / jnp.sqrt(jnp.maximum(jnp.diagonal(s, axis1=1, axis2=2), 1e-300))
    rows = chol(unit[:, :, None] * s * unit[:, None, :] + jnp.eye(r) * 1e-12)
    x = lower_solve(rows, unit[:, :, None] * (swap(a2) @ a1))
    g = swap(a1) @ a1 - swap(x) @ x
    return jnp.where(jnp.all(jnp.isfinite(g), axis=(1, 2))[:, None, None], g, 0.0)


def tree_read(d, y):
    """From the layout (L, w) to (V, q)."""
    return y[d["v_layer"][:, None], d["v_slot"][:, None] * d["b"].shape[1] + jnp.arange(d["b"].shape[1])]


def tree_write(d, x):
    """From (V, q) to the layout (L, w)."""
    return jnp.zeros(d["tree_mask"].shape).at[d["v_layer"][:, None], d["v_slot"][:, None] * d["b"].shape[1] + jnp.arange(d["b"].shape[1])].set(x)


def tree_assemble(d, g):
    """The blocks of S (plan 11.1): diagonal D (L, w, w) and E (L - 1, w, w) with E[k] = S[layer k, layer k + 1]. Two vertices of a layer share no edge, so at this point D is block diagonal by vertex."""
    q = d["b"].shape[1]
    L, w = d["tree_mask"].shape
    k = d["edge_layer"][:, None, None]
    tail, head = d["tail_slot"][:, None] * q + jnp.arange(q), d["head_slot"][:, None] * q + jnp.arange(q)
    D = jnp.zeros((L, w, w)).at[k, tail[:, :, None], tail[:, None, :]].add(g[:, :q, :q]).at[k + 1, head[:, :, None], head[:, None, :]].add(g[:, q:, q:])
    return D, jnp.zeros((L - 1, w, w)).at[k, tail[:, :, None], head[:, None, :]].add(g[:, :q, q:])


def tree_matvec(D, E, x):
    """S x on the layout."""
    out = jnp.einsum("kij,kj->ki", D, x)
    return out.at[:-1].add(jnp.einsum("kij,kj->ki", E, x[1:])).at[1:].add(jnp.einsum("kji,kj->ki", E, x[:-1]))


def tree_factor(D, E):
    """Odd-even block cyclic reduction (plan 11.2, tex Thm cyclic reduction is exact), for 2^k + 1 layers. Each level eliminates every odd layer as one batched call, then the same on the even layers that remain, until two layers are left for the root."""
    levels = []
    while D.shape[0] > 2:
        factor = jax.lax.linalg.cholesky(D[1::2], symmetrize_input=False)
        left, right = lower_solve(factor, swap(E[0::2])), lower_solve(factor, E[1::2])     # L^-1 E[j - 1]^T and L^-1 E[j] of odd layer j
        levels.append((factor, left, right))
        D, E = D[0::2].at[:-1].add(-swap(left) @ left).at[1:].add(-swap(right) @ right), -swap(left) @ right
    last = jax.lax.linalg.cholesky(D[1], symmetrize_input=False)
    top = lower_solve(last, swap(E[0]))
    return levels, (jax.lax.linalg.cholesky(D[0] - swap(top) @ top, symmetrize_input=False), last, top)


def tree_solve(factors, rho):
    """Solves with tree_factor's result, up the levels and back down; rho has the layout (L, w), and a level's factors serve both passes."""
    levels, (first, last, top) = factors
    kept = []
    for factor, left, right in levels:
        w = lower_solve(factor, rho[1::2, :, None])[..., 0]
        kept.append(w)
        rho = rho[0::2].at[:-1].add(-jnp.einsum("kji,kj->ki", left, w)).at[1:].add(-jnp.einsum("kji,kj->ki", right, w))
    w = lower_solve(last, rho[1][:, None])[:, 0]
    x0 = lower_solve(first, lower_solve(first, (rho[0] - swap(top) @ w)[:, None]), transpose=True)[:, 0]
    x = jnp.stack([x0, lower_solve(last, (w - top @ x0)[:, None], transpose=True)[:, 0]])
    for (factor, left, right), w in zip(reversed(levels), reversed(kept)):
        odd = lower_solve(factor, (w - jnp.einsum("kij,kj->ki", left, x[:-1]) - jnp.einsum("kij,kj->ki", right, x[1:]))[..., None], transpose=True)[..., 0]
        x = jnp.zeros((2 * x.shape[0] - 1, x.shape[1])).at[0::2].set(x).at[1::2].set(odd)
    return x


def tree_prices(d, D, E, p, m_plus, theta, tree):
    """The change of prices at (p, m_plus) (plan 11.1, tex Prop exact proximal step). Flow rows and state rows at their delta bound are equalities. A state row inside its bound has a zero price at the solution, so it goes straight there. A row that barely responds stays put: its edges carry almost no flow, or its coordinate sits at a bound, and a frozen face says nothing about how far it could move. The yardstick is the row's own response with no constraint active, the diagonal of K T K^T, which bounds the real one from above and stays meaningful when every flow is near zero.

    Redundant rows (a pinned coordinate at a contact vertex, the flow rows' constant) leave S singular, and a plain solve would multiply any inconsistency in r by up to 1 / reg. Solving twice, dp = (S + eps)^-1 S (S + eps)^-1 r, has the factor lambda / (lambda + eps)^2 on an eigenvalue lambda: 1 / lambda on the rows that matter and 0 on an exact null direction, and it stays symmetric, positive semidefinite and zero at r = 0. eps is tree.reg times each row's dual metric 1 / sigma, so in that metric the step is at most 1 / (4 reg) ordinary dual steps: a frozen face only predicts small changes, and far beyond that the edge steps saturate or fail.
    """
    w = d["tree_mask"].shape[1]
    t = d["T"] * theta
    unconstrained = jax.ops.segment_sum(jnp.concatenate([(d["head_coef"] ** 2 * jnp.take_along_axis(t, d["head_idx"], axis=1)).ravel(), (d["tail_coef"] ** 2 * jnp.take_along_axis(t, d["tail_idx"], axis=1)).ravel()]),
                                        d["k_rows"], num_segments=d["b"].size).reshape(d["b"].shape)
    scale = jnp.maximum(jnp.sum(unconstrained) / jnp.maximum(jnp.sum(d["tree_mask"]), 1.0), 1e-300)
    sig = d["sig_p"] / theta
    r0 = k_forward(d, m_plus) - d["b"]
    moved = p + sig * r0
    outside = jnp.abs(moved) > sig * d["delta"]
    responds = (tree_read(d, jnp.diagonal(D, axis1=1, axis2=2)) > tree["floor"] * unconstrained) & (sig > 0.0)
    solved = tree_write(d, (responds & (d["is_flow"] | outside)).astype(jnp.float64))
    pinned = tree_write(d, jnp.where(responds & ~d["is_flow"] & ~outside, -p, 0.0))
    r = tree_write(d, jnp.where(d["is_flow"], r0, r0 - d["delta"] * jnp.sign(moved)))
    rho = solved * (r - tree_matvec(D, E, pinned)) + (1.0 - solved) * scale * pinned
    eye = jnp.eye(w)
    Dm = D * solved[:, :, None] * solved[:, None, :] + eye * ((1.0 - solved) * scale)[:, :, None]
    Em = E * solved[:-1, :, None] * solved[1:, None, :]
    eps = solved * tree_write(d, tree["reg"] / jnp.where(sig > 0.0, sig, 1.0)) + (1.0 - solved) * tree["reg"] * scale
    factors = tree_factor(Dm + eye * eps[:, :, None], Em)
    x = tree_solve(factors, rho)
    if tree["filter"]:
        x = tree_solve(factors, tree_matvec(Dm, Em, x))
    dp = tree_read(d, x)
    return jnp.where(jnp.all(jnp.isfinite(dp)), dp, 0.0)


def tree_direction(d, z, aux, theta, ss, tree):
    """The ordinary round T(z), its residual norm and bound, the tree's price change at z, capped at d_max times the residual (tex Algorithm 1), and the numbers logged per pass. An edge with flow whose response is exactly zero had a failed factorization; an edge the zero test skipped responds with zero by design and carries no flow."""
    next_z, extra = pdhg_step(d, z, aux, theta, ss, response=True)
    res = {k: z[k] - next_z[k] for k in z}
    r_norm = jnp.sqrt(jnp.maximum(p_inner(d, res, res, theta), 0.0))
    g = extra["response"]
    D, E = tree_assemble(d, g)
    dp = tree_prices(d, D, E, z["p"], next_z["m"], theta, tree)
    direction = jax.tree_util.tree_map(jnp.zeros_like, z)
    d_norm = jnp.sqrt(jnp.maximum(p_inner(d, dict(direction, p=dp), dict(direction, p=dp), theta), 0.0))
    shrink = jnp.minimum(1.0, tree["d_max"] * r_norm / jnp.maximum(d_norm, 1e-300))
    l0, ld = free_bound(d, next_z["p"], extra["u"], extra["gamma"])
    return dict(next=next_z, dp=shrink * dp, r_norm=r_norm, d_norm=shrink * d_norm, d_ratio=d_norm / jnp.maximum(r_norm, 1e-300), dp_max=jnp.max(jnp.abs(shrink * dp)),
                response_failed=jnp.sum(jnp.all(g == 0.0, axis=(1, 2)) & (next_z["m"][:, 0] > 1e-9)), l0=l0, ld=ld, u=extra["u"], gamma=extra["gamma"], ipm_res=extra["ipm_res"], ipm_it=extra["ipm_it"],
                solved=extra["solved"])


def tree_trial(d, z, aux, dp, tau, theta, ss):
    """A real round at w = z + tau d, with d changing the prices only (plan 11.1): T(w), the residual R(w), its norm squared and its product with d. These edge steps also give a valid bound at w's prices. Skipping is exact, so the stored multipliers aux change which edges are solved and nothing else."""
    w = dict(z, p=z["p"] + tau * dp)
    next_w, extra = pdhg_step(d, w, aux, theta, ss)
    res = {k: w[k] - next_w[k] for k in w}
    direction = dict(jax.tree_util.tree_map(jnp.zeros_like, z), p=dp)
    l0, ld = free_bound(d, w["p"], extra["u"], extra["gamma"])
    return dict(next=next_w, res=res, rw2=p_inner(d, res, res, theta), inner=p_inner(d, res, direction, theta), l0=l0, ld=ld, u=extra["u"], gamma=extra["gamma"], ipm_res=extra["ipm_res"], ipm_it=extra["ipm_it"])


def new_guard(r_safe):
    """The safeguard's state and counters (plan 11.6), and the last pass's numbers."""
    return dict(r_safe=r_safe, attempts=0, accepted_k1=0, accepted_k2=0, accepted_at_tau={}, fallback=0, trial_rounds=0, rejected_ipm=0, derivative_evals=0, last=None)


def tree_attempt(sg, d, pieces, state, guard):
    """One tree pass behind the SuperMann test (plan 11.6, tex Algorithm 1), with the blind step off. Both tests rest on T being the exact, firmly nonexpansive round, so a round whose IPM did not converge can support neither: a failed T(z) ends the pass, and a failed trial is rejected whatever its numbers say. Accepting w keeps T(w), which the trial already computed, as the next iterate: that is the accepted step followed by an ordinary round. A kept step starts a new average, because the old one averages points on another path. Returns (next state, rounds of edge work used)."""
    z, theta, aux = state["z"], state["theta"], aux_of(state)
    out = pieces["direction"](d, z, aux, theta)
    r_norm, ipm_tol = float(out["r_norm"]), float(sg["ipm_tol"])
    guard["attempts"] += 1
    guard["derivative_evals"] += int(round(float(out["solved"])))     # skipped edges respond with zero and need none
    l0, ld = max(float(state["l0_best"]), float(out["l0"])), max(float(state["ld_best"]), float(out["ld"]))
    next_z, source, outcome, used, tau = out["next"], out, "fallback", 1, 1.0
    trials = []
    if float(out["ipm_res"]) <= ipm_tol and float(out["d_norm"]) > 0.0:
        for _ in range(int(sg["max_trials"])):
            t = pieces["trial"](d, z, aux, out["dp"], scalar(tau), theta)
            used += 1
            guard["trial_rounds"] += 1
            l0, ld = max(l0, float(t["l0"])), max(ld, float(t["ld"]))
            rw2 = max(float(t["rw2"]), 0.0)
            rw = math.sqrt(rw2)
            trial_res = float(t["ipm_res"])
            trials.append(dict(tau=tau, ratio=rw / max(r_norm, 1e-300), ipm_res=trial_res))
            if not trial_res <= ipm_tol:
                guard["rejected_ipm"] += 1
            elif r_norm <= guard["r_safe"] and rw <= float(sg["c1"]) * r_norm:
                guard["r_safe"] = rw + float(sg["q"]) ** guard["accepted_k1"]
                guard["accepted_k1"] += 1
                next_z, source, outcome = t["next"], t, "k1"
                break
            else:
                rho = rw2 - tau * float(t["inner"])
                if rw > 0.0 and rho >= float(sg["sigma"]) * rw * r_norm:
                    step = float(sg["relax"]) * rho / rw2
                    guard["accepted_k2"] += 1
                    next_z, source, outcome = {k: z[k] - step * t["res"][k] for k in z}, t, "k2"
                    break
            tau *= float(sg["beta"])
    guard["last"] = dict(outcome=outcome, r_norm=r_norm, d_ratio=float(out["d_ratio"]), dp_max=float(out["dp_max"]), ipm_res=float(out["ipm_res"]), ipm_it=int(out["ipm_it"]), response_failed=int(out["response_failed"]), trials=trials)
    if outcome != "fallback":
        guard["accepted_at_tau"][f"{tau:g}"] = guard["accepted_at_tau"].get(f"{tau:g}", 0) + 1
        avg, count = next_z, scalar(0.0)
    else:
        guard["fallback"] += 1
        count = state["count"] + 1.0
        avg = jax.tree_util.tree_map(lambda a, b: a + (b - a) / count, state["avg"], next_z)
    return dict(state, z=next_z, avg=avg, count=count, l0_best=scalar(l0), ld_best=scalar(ld), u=source["u"], gamma=source["gamma"], ipm_res=jnp.maximum(state["ipm_res"], source["ipm_res"]),
                ipm_it=jnp.maximum(state["ipm_it"], source["ipm_it"])), used


# ===== SELFTEST =====
# Milestones M3, M5 and M6 (plan 17) on the tiny instance, against the J_rel that Clarabel saved in reference.json.

def run_checks(checks):
    ok_all, results = True, []
    for name, fn in checks:
        t0 = time.perf_counter()
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        results.append(dict(name=name, ok=bool(ok), msg=msg, seconds=round(time.perf_counter() - t0, 3)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({results[-1]['seconds']:.1f}s): {msg}", flush=True)
        ok_all &= bool(ok)
    return ok_all, results


def random_primal(d, key):
    """A random m and S on the active entries only, so that padding stays zero."""
    k1, k2 = jax.random.split(key)
    keep = jnp.maximum(d["open"], d["e0"])
    m = jax.random.normal(k1, d["c"].shape) * keep
    g = jax.random.normal(k2, d["sig_l"].shape) * d["psd_mask"]
    return m, 0.5 * (g + swap(g))


def check_operators(d, ss):
    """The adjoints are true adjoints, and the step sizes give ||Sigma^(1/2) A T^(1/2)|| < 1 (plan 7.3), by power iteration on the whole operator: (K m, B(m) - S) in the split mode, K m in the edge mode."""
    key = jax.random.PRNGKey(5)
    k1, k2, k3 = jax.random.split(key, 3)
    m, lam = random_primal(d, k1)
    p = jax.random.normal(k2, d["b"].shape)
    err_k = abs(float(jnp.sum(k_forward(d, m) * p) - jnp.sum(m * k_adjoint(d, p))))
    err_b = abs(float(jnp.sum(psd_forward(d, m) * lam) - jnp.sum(m * psd_adjoint(d, lam))))
    keep = jnp.maximum(d["open"], d["e0"])
    split = ss["psd"] == "split"
    rt, rs, rp, rl = jnp.sqrt(d["T"]) * keep, jnp.sqrt(d["t_s"]), jnp.sqrt(d["sig_p"]), jnp.sqrt(d["sig_l"]) * split

    def forward(x, s):
        return rp * k_forward(d, rt * x), rl * (psd_forward(d, rt * x) - rs * s)

    def adjoint(a, b):
        return rt * (k_adjoint(d, rp * a) + psd_adjoint(d, rl * b)), -rs * rl * b

    x, s = random_primal(d, k3)
    s = s * split
    norm = 0.0
    for _ in range(300):
        a, b = forward(x, s)
        x, s = adjoint(a, b)
        size = jnp.sqrt(jnp.sum(x * x) + jnp.sum(s * s))
        x, s = x / size, s / size
        norm = math.sqrt(float(size))
    ok = err_k <= 1e-9 and err_b <= 1e-9 and norm < 1.0
    return ok, f"K adjoint error {err_k:.1e}, B adjoint error {err_b:.1e}; step-scaled operator norm {norm:.4f} over {'(K, B - S)' if split else 'K'} (must be below 1)"


def clarabel_edge_qp(d, e, h, f, psd=False):
    """One edge's step through Clarabel, an independent solver: the same rows on the entries in use, and with psd the moment matrix's PSD cone in push_model's triangle convention (the one the reference solve uses)."""
    import clarabel
    import scipy.sparse as sp
    keep = np.flatnonzero(np.maximum(np.asarray(d["open"][e]), np.asarray(d["e0"])))
    lo, hi, R = np.asarray(d["lo"][e])[keep], np.asarray(d["hi"][e])[keep], np.asarray(d["R"][e])[:, keep]
    eq, ge = np.asarray(d["row_eq"][e]) > 0.0, np.asarray(d["row_ge"][e]) > 0.0
    n = keep.size
    bounds = np.zeros((2 * (n - 1), n))
    for i in range(1, n):
        bounds[2 * (i - 1), i], bounds[2 * (i - 1), 0] = 1.0, -lo[i]
        bounds[2 * (i - 1) + 1, i], bounds[2 * (i - 1) + 1, 0] = -1.0, hi[i]
    rows = sp.csc_matrix(np.vstack([R[eq], -R[ge], -bounds]))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    cones = ([clarabel.ZeroConeT(int(eq.sum()))] if eq.any() else []) + [clarabel.NonnegativeConeT(int(ge.sum()) + bounds.shape[0])]
    if psd:
        import push_model as pm
        place = np.full(h.shape[1], -1)
        place[keep] = np.arange(n)
        size = int(round(float(np.trace(np.asarray(d["psd_mask"][e])))))
        iu, ju = np.triu_indices(size)
        cols = place[np.asarray(d["mat"][e])[iu, ju]]
        if np.any(cols < 0):
            raise ValueError(f"edge {e}: a moment matrix entry reads an entry of m that is not in use")
        tri = sp.csc_matrix((-np.where(iu == ju, 1.0, math.sqrt(2.0)), (pm.svec_position(iu, ju, size, "clarabel"), cols)), shape=(size * (size + 1) // 2, n))
        rows = sp.vstack([rows, tri], format="csc")         # s = -tri x = svec(B(x)) in the PSD cone
        cones.append(clarabel.PSDTriangleConeT(size))
    sol = clarabel.DefaultSolver(sp.diags(np.asarray(h[e])[keep], format="csc"), np.asarray(f[e])[keep], rows, np.zeros(rows.shape[0]), cones, settings).solve()
    out = np.zeros(h.shape[1])
    out[keep] = np.asarray(sol.x)
    return out


def check_edge_step(d, cfg, draws=3):
    """The batched IPM of the configured mode (edge_qp, or edge_sdp in the edge mode) against Clarabel on sampled edges, with random data that puts solutions on the boundary, and at the cone's apex, where a large positive flow price makes the exact step zero. It compares objectives and checks feasibility directly, PSD included in the edge mode: a solver's gap tolerance of 1e-10 pins x only to about its square root, so x is held to the looser limit that this allows (3x looser for the SDP, whose curvature along the cone's boundary is weaker)."""
    st = cfg["selftest"]["m3"]
    ss = dict(solver_settings(cfg, d), zero_test=False, compaction="none")
    edge = ss["psd"] == "edge"
    solve = jax.jit(lambda a, b: edge_solve(d, a, b, ss, False))
    E = d["c"].shape[0]
    picks = np.linspace(0, E - 1, min(int(st["qp_edges"]), E)).astype(int)
    worst_x, worst_obj, worst_feas, worst_psd, res, its = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for i in range(draws):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(100 + i), 3)
        keep = jnp.maximum(d["open"], d["e0"])
        h = jnp.exp(jax.random.normal(k1, d["c"].shape)) * keep + (1.0 - keep)
        center = (0.5 * (d["lo"] + d["hi"]) * d["open"] + d["e0"]) * jax.random.uniform(k2, (E, 1), minval=0.0, maxval=2.0)
        f = (jax.random.normal(k3, d["c"].shape) - h * center) * keep
        out = solve(h, f)
        x = out["x"]
        res, its = max(res, float(jnp.max(out["metric"]))), max(its, int(out["it"]))
        rows, y = mv(d["R"], x), x[:, :1]
        worst_feas = max(worst_feas, float(jnp.max(jnp.abs(rows) * d["row_eq"])), float(jnp.max(-rows * d["row_ge"])), float(jnp.max(d["open"] * (d["lo"] * y - x))), float(jnp.max(d["open"] * (x - d["hi"] * y))))
        if edge:
            worst_psd = max(worst_psd, float(candidate_stats(d, dict(m=x))["psd"]))
        for e in picks:
            xe, other, he, fe = np.asarray(x[e]), clarabel_edge_qp(d, int(e), h, f, psd=edge), np.asarray(h[e]), np.asarray(f[e])
            value = lambda a: 0.5 * np.sum(he * a * a) + fe @ a
            worst_obj = max(worst_obj, abs(value(xe) - value(other)) / (1.0 + abs(value(other))))
            worst_x = max(worst_x, float(np.max(np.abs(xe - other))))
    out = solve(jnp.ones_like(d["c"]), 10.0 * d["e0"] * jnp.ones_like(d["c"]))
    apex = float(jnp.max(jnp.abs(out["x"])))
    limit_x = 3e-4 if edge else 1e-4
    ok = worst_obj <= 1e-8 and worst_feas <= 1e-9 and worst_psd <= 1e-9 and worst_x <= limit_x and res <= 1e-8 and apex <= 1e-8
    return ok, (f"{'edge SDP' if edge else 'edge QP'}, {picks.size} edges x {draws} draws: largest objective difference from Clarabel {worst_obj:.1e}, x difference {worst_x:.1e}, row or box violation {worst_feas:.1e}"
                + (f", PSD violation {worst_psd:.1e}" if edge else "") + f"; worst IPM residual {res:.1e}, most iterations {its}; apex step size {apex:.1e} in {int(out['it'])} iterations")


def check_bound_allowance():
    """The edge bound stays at or below its exact value when large dual terms cancel, which is where roundoff hides (tex Prop signed error). One made-up edge: cost 1e-4 on m_0, with a state price and an equality multiplier of 1e8 that cancel there, so the exact value is 1e-4. Negative control: the sum alone rounds above that."""
    cost, big = 1e-4, 1e8
    d = dict(state_mask=jnp.array([[0.0, 1.0]]), c=jnp.array([[cost, 0.0]]), head_coef=jnp.array([[0.0, 1.0]]), tail_coef=jnp.zeros((1, 2)), head_rows=jnp.array([[2, 3]]), tail_rows=jnp.array([[0, 1]]),
             k_cols=jnp.zeros(4, jnp.int64), R=jnp.array([[[1.0, 0.0]]]), adj=jnp.zeros((1, 2), jnp.int64), adj_w=jnp.zeros((1, 2)), lo=jnp.zeros((1, 2)), hi=jnp.ones((1, 2)), wm=jnp.array([0.0, 1.0]),
             trace_hi=jnp.ones(1), eps_cert=jnp.asarray(1e-12))
    lb = float(edge_bounds(d, jnp.array([[0.0, 0.0], [0.0, big]]), jnp.array([[big]]), jnp.zeros((1, 1, 1)))[0][0])
    rounded = cost + big - big
    return rounded > cost and lb <= cost, f"the sum alone gives {rounded:.12f}, above the exact {cost:.12f}; with its allowance the bound is {lb:.6e}"


def check_hand_push(cfg, d, info, ref):
    """The GPU arrays hold the hand push: a true trajectory from push_model.py is flow-conserving, satisfies every row, box and PSD block, and costs J_hand. The free bound stays below J_hand and J_rel for random prices and multipliers in their cones (tex Thm 5.5), which it must for any of them."""
    import push_model as pm
    plain = deep_merge(cfg, {"build": {"workers": 1}})
    md, modes, task, tr = pm.build_instance(plain)
    if not np.array_equal(np.flatnonzero(tr["edge_alive"]), info["alive"]):
        return False, "push_model.py and problem.npz disagree on the alive edges"
    place = {int(e): i for i, e in enumerate(info["alive"])}
    m = np.zeros(d["c"].shape)
    J_hand = 0.0
    steps = pm.hand_push(plain, md, task, modes)
    for (kind, tail, head, v), e in zip(steps, pm.path_edge_ids(tr, modes, steps)):
        rel = pm.relax_edge_type(plain, md, task, modes, tr, int(tr["edge_type"][e]))
        m[place[int(e)], :rel["n"]] = pm.lift_points(rel, md, task, kind, tail, head, v[None])[0]
        J_hand += pm.poly_eval(pm.edge_cost(plain, tail, head), v)
    z = dict(m=jnp.asarray(m))
    stats = {k: float(v) for k, v in candidate_stats(d, z).items()}
    ok = stats["flow"] <= 1e-12 and stats["state"] <= 1e-12 and stats["rows"] <= 1e-9 and stats["psd"] <= 1e-10 and abs(stats["U"] - J_hand) <= 1e-12
    worst_l0, worst_ld = -np.inf, -np.inf
    bound = jax.jit(lambda p, u, gamma: free_bound(d, p, u, gamma))
    for i in range(int(cfg["selftest"]["m3"]["bound_draws"])):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(200 + i), 3)
        scale = 10.0 ** (i % 4 - 2)
        p = scale * jax.random.normal(k1, d["b"].shape)
        raw = scale * jax.random.normal(k2, d["R"].shape[:2])
        u = d["row_eq"] * raw + d["row_ge"] * jnp.abs(raw)
        g = scale * jax.random.normal(k3, d["sig_l"].shape) * d["psd_mask"]
        l0, ld = bound(p, u, g @ swap(g))
        worst_l0, worst_ld = max(worst_l0, float(l0)), max(worst_ld, float(ld))
    J_rel = float(ref["J"])
    tol = float(cfg["selftest"]["m3"]["tol_j_rel"])
    ok = ok and worst_l0 <= J_hand + 1e-9 and worst_ld <= J_rel + tol
    return ok, (f"hand push: flow {stats['flow']:.1e}, state {stats['state']:.1e}, rows {stats['rows']:.1e}, psd {stats['psd']:.1e}, cost {stats['U']:.6f} vs J_hand {J_hand:.6f}; "
                f"best random bound L_0 {worst_l0:.3e}, L_delta {worst_ld:.3e}")


def tiny_setup(cfg, sets):
    """The tiny instance's arrays and its saved reference value, or None with a message when that value is missing or stale."""
    ref_path = run_dir(cfg) / "reference.json"
    if not ref_path.exists():
        print(f"no reference value at {ref_path}; run: python push_model.py --run ref_tiny --reference", flush=True)
        return None
    ref = json.loads(ref_path.read_text())
    if ref.get("fingerprint") != reference_fingerprint(cfg):
        print(f"{ref_path} came from another config; rerun: python push_model.py --run ref_tiny --reference", flush=True)
        return None
    d, info = build_data(ensure_problem(cfg, "ref_tiny", None, sets), cfg)
    print(f"tiny instance: {info['E']} edges, {info['V']} vertices, {info['types']} edge types; m {info['n']}, rows {info['r']}, moment matrix {info['s']}, q {info['q']}, tree width {info['tree_width']} over {info['tree_layers']} layers; J_rel {ref['J']:.6f} ({ref['status']}), J_hand {ref['J_hand']:.6f}", flush=True)
    return d, info, ref


def selftest(sets):
    """Milestone M3 (plan 17) on the tiny instance: run D's solver reaches a stop close to J_rel from below, and restarts work."""
    t0 = time.perf_counter()
    enable_compile_cache(load_config("ref_tiny", sets))
    base = load_config("ref_tiny", sets)
    st = base["selftest"]["m3"]
    cfg = deep_merge(base, {"tree": {"type": "none"}, "solver": {"tune": {"rounds": int(st["tune_rounds"])}, "stop": {"eta_rel": float(st["eta_rel"])}}})
    setup = tiny_setup(cfg, sets)
    if setup is None:
        return False
    d, info, ref = setup
    J_rel = float(ref["J"])
    tol = float(st["tol_j_rel"])
    solved = {}

    def solve_tiny():
        pieces, compile_s = compile_solver(cfg, d)
        print(f"  compiled in {compile_s:.1f} s", flush=True)
        theta = choose_theta(cfg, d, pieces, "selftest_m3")
        out = run_dir(cfg) / "selftest_m3"
        out.mkdir(parents=True, exist_ok=True)
        result, state = run_pdhg(cfg, d, pieces, theta, int(st["rounds_max"]), float(st["time_budget_s"]), int(st["stall_rounds"]), float(st["eta_rel"]), log_path=out / "log.jsonl")
        solved.update(result)
        times = component_times(cfg, d, state)
        print("  one-off piece times, ms: " + ", ".join(f"{k} {v:.3f}" for k, v in times.items()), flush=True)
        solved["component_ms"] = times
        return result["status"] in STOPPED, f"{result['status']} at theta {theta}: {result['rounds']} rounds in {result['seconds']:.1f} s ({result['ms_per_round']:.3f} ms/round); L_delta {result['L_delta']:.6f}, U_flow {result['U_flow']}, gap {result['gap']}"

    def check_close():
        return stop_is_close(solved, J_rel, tol, float(st["close_rel"]))

    def check_bound_below():
        L = solved.get("L_delta", np.inf)
        return L <= J_rel + tol, f"best L_delta over all rounds {L:.7f}, J_rel {J_rel:.7f}: the best is a running maximum, so this covers every round"

    def check_restarts():
        fired, first, last = solved.get("restarts", 0), solved.get("first_kkt", np.inf), solved.get("last_restart_kkt", np.inf)
        return fired >= 1 and last < first, f"{fired} restarts; KKT error at the start {first:.2e}, at the last restart {last:.2e}"

    ok, checked = run_checks([
        ("M3 operators: adjoints and step sizes", lambda: check_operators(d, solver_settings(cfg, d))),
        ("M3 edge step against Clarabel", lambda: check_edge_step(d, cfg)),
        ("M3 arrays hold the hand push, and the free bound stays valid", lambda: check_hand_push(cfg, d, info, ref)),
        ("M3 the bound's allowance covers terms that cancel", check_bound_allowance),
        ("M5 skipping changes no iterate, and the zero test never skips a nonzero step", lambda: check_skipping(cfg, d)),
        ("M3 run D reaches a stop on the tiny instance", solve_tiny),
        ("M3 the stop is close to J_rel", check_close),
        ("M3 every logged bound stays below J_rel", check_bound_below),
        ("M3 restarts fire and reduce the KKT error", check_restarts),
    ])
    out = ROOT / cfg["output"]["root"] / "selftest" / "m3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, results=checked, solve=solved, J_rel=J_rel, env=environment_info()), indent=2))
    print(f"\n{'ALL M3 CHECKS PASSED' if ok else 'SOME M3 CHECKS FAILED'} in {time.perf_counter() - t0:.0f}s; details in {out}", flush=True)
    return ok


def stop_is_close(solved, J_rel, tol, close):
    """A loose stop can't show that the solver converges to J_rel, so this asks less: the bound at the stop is valid and within `close` of J_rel. One long run with a tighter stop is what shows the rest (plan 17)."""
    if solved.get("status") not in STOPPED:
        return False, "the solve did not reach a stop"
    L, U = solved["L_delta"], solved["U_flow"]
    below = (J_rel - L) / abs(J_rel)
    note = "no candidate passed the feasibility tolerances" if U is None else f"U_flow {U:.7f}, {(U - J_rel) / abs(J_rel):+.1%} from J_rel"
    return L <= J_rel + tol and below <= close, f"L_delta {L:.7f} is {below:.1%} below J_rel {J_rel:.7f} (limit {close:.0%}); {note}"


def short_run(cfg, d, theta=0.3, rounds=200):
    """A mid-run state from plain rounds, for the checks that need realistic faces and flows."""
    ss = solver_settings(cfg, d)
    th = scalar(theta)
    step = jax.jit(lambda z, aux: pdhg_step(d, z, aux, th, ss))
    state = initial_state(d, theta, ss)
    z, aux = state["z"], aux_of(state)
    for _ in range(rounds):
        z, extra = step(z, aux)
        aux = dict(u=extra["u"], gamma=extra["gamma"])
    jax.block_until_ready(z["m"])
    return z, th


def tight_settings(cfg, d):
    """Solver settings for checks that need the edge steps solved hard: no skipping, and tighter IPMs."""
    ss = solver_settings(cfg, d)
    return dict(ss, zero_test=False, compaction="none", ipm=dict(ss["ipm"], tol=1e-13, iters=60, min_iters=0),
                ipm_sdp=dict(ss["ipm_sdp"], tol=1e-11, iters=80, min_iters=0))


def check_skipping(cfg, d, theta=0.3, rounds=300, every=50, clarabel_edges=8):
    """Milestone M5 (plan 9.1, 9.3).

    Compaction: with the zero test on, bucketed compaction gives the same iterates as the verification mode, which solves every edge and overwrites the settled ones, to 1e-8 at every 50th round; only batched rounding may differ.

    Zero test: at the end, every edge it settles has a zero step. An IPM cannot show that by |x|: at the apex its iterate sits about n mu / pi from zero (pi the edge's pricing margin), so with tol 1e-10 and a small margin a zero step comes back as 1e-7 or more. The step's optimal value can: it is exactly 0 at a zero step and -1/2 x*^T H x* < 0 otherwise, so on the settled edges neither the tight IPM nor Clarabel may find a value below -1e-9 (1 + |f|) (-1e-8 for Clarabel, whose gap tolerance is 1e-10). For the same reason, comparing iterates with a run that has no zero test would measure the IPM's error at the apex, not the skip."""
    base = solver_settings(cfg, d)
    edge = base["psd"] == "edge"
    variants = dict(on=dict(base, zero_test=True, compaction="bucketed"), verify=dict(base, zero_test=True, compaction="none"))
    th = scalar(theta)
    E = int(d["c"].shape[0])
    kept, last, skipped = {}, {}, 0.0
    for name, ss in variants.items():
        step = jax.jit(lambda z, aux, ss=ss: pdhg_step(d, z, aux, th, ss))
        state = initial_state(d, theta, ss)
        z, aux = state["z"], aux_of(state)
        kept[name] = []
        for k in range(rounds):
            z, extra = step(z, aux)
            aux = dict(u=extra["u"], gamma=extra["gamma"])
            if name == "on":
                skipped += E - float(extra["solved"])
            if (k + 1) % every == 0:
                kept[name].append(jax.device_get(z))
        last[name] = (z, aux)
    worst = max(float(np.max(np.abs(a[key] - b[key]))) for a, b in zip(kept["on"], kept["verify"]) for key in a)
    z, aux = last["on"]
    h, f, size = step_data(d, z, th)
    zero = cone_bound(d, f, size, aux["u"], aux["gamma"] if edge else jnp.zeros_like(aux["gamma"])) >= base["lb_tol"]
    x = jax.jit(lambda a, b: edge_solve(d, a, b, tight_settings(cfg, d), False)["x"])(h, f)
    scale = 1.0 + jnp.max(jnp.abs(f), axis=1)
    value = (0.5 * jnp.sum(h * x * x, axis=1) + jnp.sum(f * x, axis=1)) / scale
    lowest = float(jnp.min(jnp.where(zero, value, jnp.inf)))
    settled = np.flatnonzero(np.asarray(zero))
    lowest_c = np.inf
    hn, fn, sn = np.asarray(h), np.asarray(f), np.asarray(scale)
    for e in settled[np.unique(np.linspace(0, settled.size - 1, min(clarabel_edges, settled.size)).astype(int))] if settled.size else []:
        xc = clarabel_edge_qp(d, int(e), h, f, psd=edge)
        lowest_c = min(lowest_c, float((0.5 * np.sum(hn[e] * xc * xc) + fn[e] @ xc) / sn[e]))
    ok = worst <= 1e-8 and lowest >= -1e-9 and lowest_c >= -1e-8
    return ok, (f"{rounds} rounds, {skipped / (rounds * E):.0%} of edge steps skipped: largest iterate difference between bucketed compaction and the verification mode {worst:.1e}; "
                f"at the end the test settles {settled.size} of {E} edges, whose steps have optimal value at least {lowest:.1e} (tight IPM) and {lowest_c:.1e} (Clarabel, {min(clarabel_edges, settled.size)} edges), "
                f"relative to 1 + |f|: zero up to solver accuracy")


def dense_blocks(D, E):
    """The whole matrix of a block tridiagonal system."""
    D, E = np.asarray(D), np.asarray(E)
    L, w = D.shape[:2]
    out = np.zeros((L * w, L * w))
    for k in range(L):
        out[k * w:(k + 1) * w, k * w:(k + 1) * w] = D[k]
    for k in range(L - 1):
        out[k * w:(k + 1) * w, (k + 1) * w:(k + 2) * w] = E[k]
        out[(k + 1) * w:(k + 2) * w, k * w:(k + 1) * w] = E[k].T
    return out


def check_cyclic_reduction(d, cfg, z, theta):
    """Cyclic reduction against a dense solve (plan 11.2): on random systems, one with padding layers, and on the instance's own S. Its S is nearly singular by design, so that one is judged by the residual of the system and the difference is only reported."""
    worst = 0.0
    for i, (L, w, pad) in enumerate(((3, 4, 0), (5, 7, 0), (9, 12, 3), (33, 5, 0))):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(30 + i), 3)
        a = jax.random.normal(k1, (L, w, w))
        D = a @ swap(a) + 2.0 * w * jnp.eye(w)
        E = 0.5 * jax.random.normal(k2, (L - 1, w, w))
        if pad:
            D, E = D.at[L - pad:].set(jnp.eye(w)), E.at[L - 1 - pad:].set(0.0)
        rho = jax.random.normal(k3, (L, w))
        x = np.asarray(tree_solve(tree_factor(D, E), rho)).ravel()
        exact = np.linalg.solve(dense_blocks(D, E), np.asarray(rho).ravel())
        worst = max(worst, float(np.linalg.norm(x - exact) / np.linalg.norm(exact)))
    ss = solver_settings(cfg, d)
    aux = aux_of(initial_state(d, float(theta), ss))

    def own_blocks(y, th):
        return tree_assemble(d, pdhg_step(d, y, aux, th, ss, response=True)[1]["response"])

    D, E = jax.jit(own_blocks)(z, theta)
    mean = jnp.sum(jnp.diagonal(D, axis1=1, axis2=2) * d["tree_mask"]) / jnp.sum(d["tree_mask"])
    D = D + jnp.eye(D.shape[1]) * float(cfg["tree"]["reg"]) * mean
    rho = jax.random.normal(jax.random.PRNGKey(41), d["tree_mask"].shape)
    x = tree_solve(tree_factor(D, E), rho)
    residual = float(jnp.linalg.norm(tree_matvec(D, E, x) - rho) / jnp.linalg.norm(rho))
    exact = np.linalg.solve(dense_blocks(D, E), np.asarray(rho).ravel())
    gap = float(np.linalg.norm(np.asarray(x).ravel() - exact) / np.linalg.norm(exact))
    return worst <= 1e-10 and residual <= 1e-10, f"largest error against a dense solve on random systems {worst:.1e}; on the instance's S, relative residual {residual:.1e} and difference from dense {gap:.1e}"


def check_edge_response(d, cfg, z, theta):
    """The edge response against finite differences of the real edge step (plan 17): on edges with flow whose active set does not change, K_e (m+(p + dp) - m+(p)) = -K_e J_e K_e^T dp. In the edge mode the face includes the moment matrix's rank. An IPM pins a narrow coordinate only to about the square root of its tolerance, which is noise for a finite difference, so this runs the IPM tighter and reads the median and the 90th percentile."""
    ss = tight_settings(cfg, d)
    t = d["T"] * theta
    h, base = 1.0 / t, d["c"] - z["m"] / t + (psd_adjoint(d, z["Lam"]) if "Lam" in z else 0.0)
    solve = jax.jit(lambda p: edge_solve(d, h, base + k_adjoint(d, p), ss, False))
    dp = 1e-4 * jax.random.normal(jax.random.PRNGKey(7), d["b"].shape) * (d["sig_p"] > 0.0)
    s0, s1 = solve(z["p"]), solve(z["p"] + dp)
    x0, x1, v0, v1 = s0["x"], s1["x"], s0["state"], s1["state"]
    g = jax.jit(lambda v: edge_response_of(d, h, v, ss))(v0)
    cols = jnp.concatenate([d["tail_idx"], d["head_idx"]], axis=1)
    kc = jnp.concatenate([-d["tail_coef"], d["head_coef"]], axis=1)
    rows = dp.ravel()[jnp.concatenate([d["tail_rows"], d["head_rows"]], axis=1)]
    actual = np.asarray(kc * jnp.take_along_axis(x1 - x0, cols, axis=1))
    predicted = np.asarray(-jnp.einsum("eab,eb->ea", g, rows))

    def faces(v):
        out = [v["zl"] > v["sl"], v["zu"] > v["su"], v["zg"] > v["sg"]]
        if "X" in v:                                       # the moment matrix's rank: eigenvalues of X well above its smallest scale
            w = jnp.linalg.eigvalsh(v["X"])
            out.append(w > 1e-6 * (1.0 + w[:, -1:]))
        return jnp.concatenate(out, axis=1)

    use = np.asarray(jnp.all(faces(v0) == faces(v1), axis=1) & (x0[:, 0] > 1e-4))
    if not use.any():
        return False, "no edge with flow kept its active set, so nothing was compared"
    errors = np.linalg.norm(actual[use] - predicted[use], axis=1) / (np.linalg.norm(actual[use], axis=1) + 1e-12)
    median, high = float(np.median(errors)), float(np.quantile(errors, 0.9))
    return median <= 1e-2 and high <= 0.1, f"{int(use.sum())} of {use.size} edges compared: relative error median {median:.1e}, 90th percentile {high:.1e}, largest {float(errors.max()):.1e}"


def check_zero_direction(d, cfg, z, theta):
    """A zero residual gives exactly zero correction (plan 11.1): b is moved so that the current edge step conserves every row, with delta = 0, which makes z's prices a fixed point of the dual step."""
    ss, tree = solver_settings(cfg, d), tree_settings(cfg)
    aux = aux_of(initial_state(d, float(theta), ss))

    def at_rest(y, th):
        next_y, extra = pdhg_step(d, y, aux, th, ss, response=True)
        D, E = tree_assemble(d, extra["response"])
        return tree_prices(dict(d, b=k_forward(d, next_y["m"]), delta=jnp.zeros_like(d["delta"])), D, E, y["p"], next_y["m"], th, tree)

    largest = float(jnp.max(jnp.abs(jax.jit(at_rest)(z, theta))))
    return largest == 0.0, f"largest price change {largest:.1e}"


def check_fallback(cfg, d, pieces, z, theta):
    """A fallback is the plain PDHG step (plan 11.6): with no trials allowed, a tree pass must return T(z). Compiled separately the two may differ in rounding, so this reports the difference and allows only rounding."""
    ss = solver_settings(cfg, d)
    state = dict(initial_state(d, float(theta), ss), z=z)
    guard = new_guard(np.inf)
    new, used = tree_attempt(dict(cfg["tree"]["safeguard"], max_trials=0), d, pieces, state, guard)
    plain = jax.jit(lambda y, a, th: pdhg_step(d, y, a, th, ss)[0])(z, aux_of(state), theta)
    worst = max(float(jnp.max(jnp.abs(new["z"][k] - plain[k]))) for k in plain)
    return worst <= 1e-9 and guard["fallback"] == 1 and used == 1, f"largest difference from the plain step {worst:.1e}, {guard['fallback']} fallback, {used} round of edge work"


def check_firm_nonexpansive(d, cfg, z, theta):
    """What the safeguard's kept steps rest on (plan 11.6, tex Lemma halfspace): for any states a, b, <R(a) - R(b), a - b>_P >= ||R(a) - R(b)||_P^2. With b a fixed point this is the plan's Fejer property, and it needs no saved solution. A wrong sign in the metric would break it by a large margin."""
    ss = tight_settings(cfg, d)
    aux = aux_of(initial_state(d, float(theta), ss))
    step = jax.jit(lambda y: pdhg_step(d, y, aux, theta, ss)[0])
    keep = jnp.maximum(d["open"], d["e0"])

    def shifted(key, size):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        out = dict(m=z["m"] + size * jax.random.normal(k3, d["c"].shape) * keep, p=z["p"] + size * jax.random.normal(k4, d["b"].shape) * (d["sig_p"] > 0.0))
        if "S" in z:
            lam, s = jax.random.normal(k1, d["sig_l"].shape) * d["psd_mask"], jax.random.normal(k2, d["sig_l"].shape) * d["psd_mask"]
            out.update(S=z["S"] + size * 0.5 * (s + swap(s)), Lam=z["Lam"] + size * 0.5 * (lam + swap(lam)))
        return out

    tightest = np.inf
    for i in range(6):
        k1, k2 = jax.random.split(jax.random.PRNGKey(60 + i))
        a, b = shifted(k1, 10.0 ** (-(i % 3) - 1)), shifted(k2, 10.0 ** (-(i % 3) - 1))
        ta, tb = step(a), step(b)
        dr = {k: (a[k] - ta[k]) - (b[k] - tb[k]) for k in a}
        dz = {k: a[k] - b[k] for k in a}
        size = float(p_inner(d, dr, dr, theta))
        if size < 0.0:
            return False, f"the metric is not positive on a residual: {size:.3e}"
        tightest = min(tightest, (float(p_inner(d, dr, dz, theta)) - size) / max(size, 1e-30))
    return tightest >= -1e-3, f"smallest margin, relative to ||dR||^2: {tightest:+.2e} (must not be below -1e-3)"


def check_failed_trial(cfg, d, pieces, z, theta):
    """A trial whose edge steps failed is never kept, whatever its numbers say (plan 11.6): a stub reports a failed IPM on real trials, so every try must be rejected and the pass must fall back."""
    sg = dict(cfg["tree"]["safeguard"], ipm_tol=1e-2)
    stub = dict(pieces, trial=lambda *args: dict(pieces["trial"](*args), ipm_res=scalar(1.0)))
    guard = new_guard(np.inf)
    tree_attempt(sg, d, stub, dict(initial_state(d, float(theta), solver_settings(cfg, d)), z=z), guard)
    tries = int(sg["max_trials"])
    return guard["rejected_ipm"] == tries and guard["fallback"] == 1 and guard["accepted_k1"] + guard["accepted_k2"] == 0, f"{guard['rejected_ipm']} of {tries} tries rejected, {guard['accepted_k1'] + guard['accepted_k2']} kept, {guard['fallback']} fallback"


def check_frozen_face(d, cfg):
    """One tree step on a linear problem (plan 17): boxes wide enough never to be active and no rows keep every edge on one face, so the edge step is exactly affine in the prices, and b is set so that a known small price vector solves every row. The Newton step must then cut the row residual by at least 100 x. This runs the whole path: edge response, assembly, cyclic reduction, right-hand side and a real trial. In the edge mode the PSD block is switched off here (an empty moment matrix), since a PSD constraint is never affine; check_edge_response covers its response. It uses a tiny tree.reg, since it tests the solve and not how cautious the default step is."""
    base, tree = solver_settings(cfg, d), dict(tree_settings(cfg), reg=1e-8, d_max=1e9)
    ss = dict(base, ipm=dict(base["ipm"], iters=80), ipm_sdp=dict(base["ipm_sdp"], iters=80))
    no_rows = jnp.zeros_like(d["R"])
    wide = dict(d, lo=-100.0 * d["open"], hi=100.0 * d["open"], R=no_rows, Rw=no_rows, Ry=no_rows[:, :, 0], row_eq=jnp.zeros_like(d["row_eq"]), row_ge=jnp.zeros_like(d["row_ge"]),
                row_pad=jnp.ones_like(d["row_pad"]), n_ineq=2.0 * jnp.sum(d["open"], axis=1), delta=jnp.zeros_like(d["delta"]))
    if ss["psd"] == "edge":
        wide["psd_mask"] = jnp.zeros_like(d["psd_mask"])
    theta = scalar(1.0)
    state = initial_state(d, 1.0, ss)
    aux = aux_of(state)
    start = dict(state["z"], m=d["e0"] * jnp.ones_like(d["c"]))
    known = 1e-2 * jax.random.normal(jax.random.PRNGKey(9), d["b"].shape) * (d["sig_p"] > 0.0)
    target = jax.jit(lambda: pdhg_step(wide, dict(start, p=known), aux, theta, ss)[0]["m"])()
    linear = dict(wide, b=k_forward(wide, target))
    out = jax.jit(lambda y, th: tree_direction(linear, y, aux, th, ss, tree))(start, theta)
    after = jax.jit(lambda y, dp, th: tree_trial(linear, y, aux, dp, scalar(1.0), th, ss)["next"]["m"])(start, out["dp"], theta)
    size = lambda m: float(jnp.max(jnp.abs(k_forward(linear, m) - linear["b"])))
    before, left = size(out["next"]["m"]), size(after)
    return before >= 100.0 * left, f"largest row residual {before:.2e} before the step, {left:.2e} after: {before / max(left, 1e-300):.0f} x"


def selftest_m6(sets):
    """Milestone M6 (plan 17) on the tiny instance: the exact tree's parts, then a tree solve that reaches a stop close to J_rel from below."""
    t0 = time.perf_counter()
    enable_compile_cache(load_config("ref_tiny", sets))
    base = load_config("ref_tiny", sets)
    st = base["selftest"]["m3"]
    cfg = deep_merge(base, {"tree": {"type": "exact"}, "solver": {"tune": {"rounds": int(st["tune_rounds"])}, "stop": {"eta_rel": float(st["eta_rel"])}}})
    setup = tiny_setup(cfg, sets)
    if setup is None:
        return False
    d, info, ref = setup
    J_rel = float(ref["J"])
    tol = float(st["tol_j_rel"])
    pieces, compile_s = compile_solver(cfg, d)
    print(f"  compiled in {compile_s:.1f} s", flush=True)
    z, theta = short_run(cfg, d)
    solved = {}

    def solve_tiny():
        chosen = choose_theta(cfg, d, pieces, "selftest_m6")
        out = run_dir(cfg) / "selftest_m6"
        out.mkdir(parents=True, exist_ok=True)
        result, state = run_pdhg(cfg, d, pieces, chosen, int(st["rounds_max"]), float(st["time_budget_s"]), int(st["stall_rounds"]), float(st["eta_rel"]), log_path=out / "log.jsonl")
        solved.update(result)
        times = component_times(cfg, d, state)
        print("  one-off piece times, ms: " + ", ".join(f"{k} {v:.3f}" for k, v in times.items()), flush=True)
        solved["component_ms"] = times
        tree = result["tree"]
        return result["status"] in STOPPED, (f"{result['status']} at theta {chosen}: {result['rounds']} rounds in {result['seconds']:.1f} s ({result['ms_per_round']:.3f} ms/round); L_delta {result['L_delta']:.6f}, U_flow {result['U_flow']}, gap {result['gap']}; "
                                             f"{tree['attempts']} tree passes: {tree['accepted_k1']} kept whole, {tree['accepted_k2']} kept by projection, {tree['fallback']} fell back, {tree['trial_rounds']} trial rounds, {tree['rejected_ipm']} tries rejected for a failed IPM, kept at tau {tree['accepted_at_tau']}")

    def check_close():
        return stop_is_close(solved, J_rel, tol, float(st["close_rel"]))

    def check_bound_below():
        L = solved.get("L_delta", np.inf)
        return L <= J_rel + tol, f"best L_delta over all rounds and trials {L:.7f}, J_rel {J_rel:.7f}"

    ok, checked = run_checks([
        ("M6 cyclic reduction against a dense solve", lambda: check_cyclic_reduction(d, cfg, z, theta)),
        ("M6 edge response against finite differences", lambda: check_edge_response(d, cfg, z, theta)),
        ("M6 a zero residual gives zero correction", lambda: check_zero_direction(d, cfg, z, theta)),
        ("M6 a fallback is the plain PDHG step", lambda: check_fallback(cfg, d, pieces, z, theta)),
        ("M6 a trial whose edge steps failed is never kept", lambda: check_failed_trial(cfg, d, pieces, z, theta)),
        ("M6 a round is firmly nonexpansive in the metric", lambda: check_firm_nonexpansive(d, cfg, z, theta)),
        ("M6 one tree step on a frozen-face problem", lambda: check_frozen_face(d, cfg)),
        ("M6 the tree solve reaches a stop on the tiny instance", solve_tiny),
        ("M6 the stop is close to J_rel", check_close),
        ("M6 every bound stays below J_rel", check_bound_below),
    ])
    out = ROOT / cfg["output"]["root"] / "selftest" / "m6.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, results=checked, solve=solved, J_rel=J_rel, env=environment_info()), indent=2))
    print(f"\n{'ALL M6 CHECKS PASSED' if ok else 'SOME M6 CHECKS FAILED'} in {time.perf_counter() - t0:.0f}s; details in {out}", flush=True)
    return ok


# ===== MAIN =====

def enable_compile_cache(cfg):
    """A persistent compilation cache, so that jobs with the same shapes reuse compiled code (plan 3)."""
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser(str(cfg["solver"]["compile"]["cache_dir"])))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def solve_run(cfg, run, task, sets):
    """One run from config to outputs (plan 12.1, 12.4). So far: order 1, with plain PDHG or the exact tree."""
    check_settings_built(cfg)
    tree_settings(cfg)
    enable_compile_cache(cfg)
    prob = ensure_problem(cfg, run, task, sets)
    d, info = build_data(prob, cfg)
    print(f"{cfg['run_name']} on {cfg['task']}: {info['E']} edges, {info['V']} vertices, {info['types']} edge types; m {info['n']}, rows {info['r']}, moment matrix {info['s']}, q {info['q']}; "
          f"PSD {info['psd']}; tree {cfg['tree']['type']}, width {info['tree_width']} over {info['tree_layers']} layers", flush=True)
    pieces, compile_s = compile_solver(cfg, d)
    print(f"compiled in {compile_s:.1f} s", flush=True)
    theta = choose_theta(cfg, d, pieces, cfg["run_name"])
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    result, state = run_pdhg(cfg, d, pieces, theta, int(cfg["solver"]["rounds_max"]), float(cfg["solver"]["time_budget_s"]), int(cfg["solver"]["stall_rounds"]),
                             float(cfg["solver"]["stop"]["eta_rel"]), log_path=out / "log.jsonl")
    times = component_times(cfg, d, state)
    print("one-off piece times, ms: " + ", ".join(f"{k} {v:.3f}" for k, v in times.items()), flush=True)
    (out / "cert.json").write_text(json.dumps(result, indent=2))
    (out / "timing.json").write_text(json.dumps(dict(env=environment_info(), compile_s=compile_s, seconds=result["seconds"], rounds=result["rounds"], ms_per_round=result["ms_per_round"], component_ms=times), indent=2))
    z = jax.device_get(state["z"])
    lb = jax.device_get(jax.jit(lambda p, u, gamma: edge_bounds(d, p, u, gamma)[0])(state["z"]["p"], state["u"], state["gamma"]))
    np.savez(out / "final_state.npz", **z, alive=info["alive"], lb=lb)      # m and p, plus S and Lam in the split mode
    print(json.dumps(result, indent=2))
    print(f"saved {out}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="run config: a name in configs/runs/ or a path")
    ap.add_argument("--task", default=None, help="task name, overriding the config")
    ap.add_argument("--set", action="append", default=[], help="override one key, e.g. --set trellis.N=16")
    ap.add_argument("--selftest", action="store_true", help="milestones M3, M5 and M6 on the tiny instance")
    ap.add_argument("--selftest_m6", action="store_true", help="milestone M6 alone")
    ap.add_argument("--bench_chol", action="store_true", help="Cholesky at the edge QP's real sizes: library vs hand-written")
    ap.add_argument("--bench_psd", action="store_true", help="PSD projection per round, orders 1-3 (M0)")
    ap.add_argument("--bench_linalg", action="store_true", help="Cholesky, tree-merge pieces and coupling (M0, before M3)")
    ap.add_argument("--bench_eigh", action="store_true", help="the original batched-eigh scaling test (M0)")
    args = ap.parse_args()
    if args.selftest:
        passed = [selftest(args.set), selftest_m6(args.set)]
        sys.exit(0 if all(passed) else 1)
    if args.selftest_m6:
        sys.exit(0 if selftest_m6(args.set) else 1)
    cfg = load_config(args.run, args.set)
    if args.task:
        cfg["task"] = args.task
    if args.bench_chol:
        bench_chol(cfg)
    if args.bench_psd:
        bench_psd(cfg)
    if args.bench_linalg:
        bench_linalg(cfg)
    if args.bench_eigh:
        bench_eigh(cfg)
    if args.bench_chol or args.bench_psd or args.bench_linalg or args.bench_eigh:
        return
    if not args.run:
        ap.print_help()
        sys.exit(1)
    result = solve_run(cfg, args.run, args.task, args.set)
    sys.exit(0 if result["status"] in STOPPED else 1)


if __name__ == "__main__":
    main()
