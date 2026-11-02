"""PDHG solver for the planar-pushing trellis (plan sections 7-12). So far: the M0 benchmarks of the GPU primitives a round will use, each checked for true batching (plan section 0); the solver sections follow from M3.

    python run_trajopt.py --bench_eigh          # batched eigh at the moment-matrix sizes -> runs/bench/eigh.json
    python run_trajopt.py --bench_primitives    # every round primitive, batching verdicts -> runs/bench/primitives.json
"""
import argparse
import copy
import functools
import json
import math
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
# Plan 8.2. The projection onto the PSD cone is Pi(X) = (X + |X|) / 2. jnp.linalg.eigh batches
# only up to 32 x 32 on the GPU (cuSOLVER's batched Jacobi); above that it solves the matrices
# one after another (M0: 136 x 136 took ~45 us each, flat with the batch). Every candidate below
# runs as batched array operations instead. Inexact projections are allowed to a tolerance: PDHG
# tolerates summable errors (Rasch-Chambolle), and the certificate stays valid for any duals.

def psd_eigh(x):
    """Exact projection via eigh: V max(w, 0) V^T (the library kernel; loops above 32 x 32 on GPU)."""
    w, v = jnp.linalg.eigh(x)
    return (v * jnp.maximum(w, 0.0)[..., None, :]) @ swap(v)


def psd_newton_schulz(x, iters):
    """(X + |X|) / 2 with |X| = X sign(X), sign(X) by Newton-Schulz on X / ||X||_F: matrix products only.

    Eigenvalues with |lambda| / ||X||_F below ~1.5^-iters are not yet converged; their error in
    the projection is at most |lambda|, so a few dozen iterations reach working accuracy.
    """
    s = jnp.sqrt(jnp.sum(x * x, axis=(-2, -1), keepdims=True)) + 1e-300
    eye = jnp.eye(x.shape[-1], dtype=x.dtype)
    y = jax.lax.fori_loop(0, iters, lambda _, y: 0.5 * y @ (3.0 * eye - y @ y), x / s)
    a = x @ y
    return 0.5 * (x + 0.5 * (a + swap(a)))


def psd_qdwh(x):
    """(X + |X|) / 2 with |X| the positive factor of the QDWH polar decomposition (jax.scipy.linalg.polar)."""
    p = jax.vmap(lambda a: jax.scipy.linalg.polar(a, method="qdwh")[1])(x)
    return 0.5 * (x + 0.5 * (p + swap(p)))


def jacobi_tables(n):
    """Round-robin pairings for parallel Jacobi: n - 1 rounds, each a permutation whose consecutive entries are n/2 disjoint pairs; every pair occurs once per sweep."""
    if n % 2:
        raise ValueError("parallel Jacobi needs an even size")
    others = list(range(1, n))
    perms = []
    for r in range(n - 1):
        arr = [0] + others[r:] + others[:r]
        perms.append([idx for i in range(n // 2) for idx in (arr[i], arr[n - 1 - i])])
    perms = np.array(perms, np.int32)
    return jnp.asarray(perms), jnp.asarray(np.argsort(perms, axis=1).astype(np.int32))


def jacobi_eigh(a, v, perms, invs, sweeps):
    """Batched parallel cyclic Jacobi: each round applies all n/2 disjoint rotations of every matrix at once. Returns (unsorted eigenvalues, eigenvectors V with A = V diag(w) V^T)."""
    h = a.shape[-1] // 2
    ev, od = jnp.arange(0, 2 * h, 2), jnp.arange(1, 2 * h, 2)

    def rot_cols(m, c, s):
        # M J for the block rotation J = [[c, s], [-s, c]] on every pair of columns (2i, 2i+1)
        mp, mq = m[..., ev], m[..., od]
        cc, ss = c[:, None, :], s[:, None, :]
        return jnp.stack([cc * mp - ss * mq, ss * mp + cc * mq], axis=-1).reshape(m.shape)

    def one_round(carry, pi):
        a, v = carry
        perm, inv = pi
        ap = a[:, perm][:, :, perm]
        app, aqq, apq = ap[:, ev, ev], ap[:, od, od], ap[:, ev, od]
        zero = apq == 0.0
        tau = (aqq - app) / (2.0 * jnp.where(zero, 1.0, apq))
        t = jnp.where(tau >= 0.0, 1.0, -1.0) / (jnp.abs(tau) + jnp.sqrt(1.0 + tau * tau))
        t = jnp.where(zero, 0.0, t)
        c = 1.0 / jnp.sqrt(1.0 + t * t)
        s = t * c
        ap = swap(rot_cols(swap(rot_cols(ap, c, s)), c, s))      # J^T A J
        vp = rot_cols(v[:, :, perm], c, s)                         # V J
        return (ap[:, inv][:, :, inv], vp[:, :, inv]), None

    def sweep(_, carry):
        return jax.lax.scan(one_round, carry, (perms, invs))[0]

    a, v = jax.lax.fori_loop(0, sweeps, sweep, (a, v))
    return jnp.diagonal(a, axis1=-2, axis2=-1), v


def psd_jacobi(x, perms, invs, sweeps):
    """Projection by batched Jacobi from scratch (V = I)."""
    w, v = jacobi_eigh(x, jnp.broadcast_to(jnp.eye(x.shape[-1], dtype=x.dtype), x.shape), perms, invs, sweeps)
    return (v * jnp.maximum(w, 0.0)[..., None, :]) @ swap(v)


def psd_jacobi_warm(x, v0, perms, invs, sweeps):
    """Projection by batched Jacobi warm-started from the previous round's eigenvectors v0."""
    w, v = jacobi_eigh(swap(v0) @ x @ v0, v0, perms, invs, sweeps)
    return (v * jnp.maximum(w, 0.0)[..., None, :]) @ swap(v)


def psd_lowrank(x, q, steps):
    """Positive part of X tracked from a warm orthonormal basis q (batch, n, p): block-Krylov Rayleigh-Ritz on span[q, Xq].

    Exact when the positive eigenvectors lie in the span; costs O(n^2 p) per matrix instead of
    O(n^3). Moment matrices approach low rank as the relaxation converges, which is what this
    exploits. Returns (projection, new basis).
    """
    p = q.shape[-1]
    th = z = None
    for _ in range(steps):
        w, _ = jnp.linalg.qr(jnp.concatenate([q, x @ q], axis=-1))
        t = swap(w) @ (x @ w)
        th, y = jnp.linalg.eigh(0.5 * (t + swap(t)))
        z = w @ y
        q = z[..., -p:]
    return (z * jnp.maximum(th, 0.0)[..., None, :]) @ swap(z), q


# ===== BENCH =====

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


def structured_symmetric(key, batch, n, r_pos):
    """Symmetric test matrices with r_pos eigenvalues in [0.5, 2] and the rest negative, log-spread over [1e-6, 2]: a near-low-rank moment matrix plus a negative semidefinite dual, the shape the projection sees near convergence."""
    k1, k2, k3 = jax.random.split(key, 3)
    q, _ = jnp.linalg.qr(jax.random.normal(k1, (batch, n, n)))
    pos = jax.random.uniform(k2, (batch, r_pos), minval=0.5, maxval=2.0)
    neg = -jnp.exp(jax.random.uniform(k3, (batch, n - r_pos), minval=math.log(1e-6), maxval=math.log(2.0)))
    lam = jnp.concatenate([pos, neg], axis=-1)
    return (q * lam[:, None, :]) @ swap(q)


def perturb(key, x, eps):
    """x plus a symmetric random change of spectral size ~eps: the next round's matrix, for warm starts."""
    e = jax.random.normal(key, x.shape)
    return x + eps * 0.5 * (e + swap(e)) / math.sqrt(x.shape[-1])


def chol_solve(a, b):
    """Batched SPD solve by Cholesky and two triangular solves (the IPM's normal equations, the tree's pivots)."""
    l = jnp.linalg.cholesky(a)
    y = jax.scipy.linalg.solve_triangular(l, b, lower=True)
    return jax.scipy.linalg.solve_triangular(l, y, lower=True, trans="T")


def bench_primitives(cfg):
    """Times every GPU primitive a PDHG round will use at its real batch sizes and judges whether it truly batches (plan section 0)."""
    pc = cfg["bench"]["primitives"]
    peak = float(cfg["bench"]["peak_tflops_fp64"]) * 1e12
    reps = int(pc.get("repeats", 5))
    rows = []

    def run(group, name, n, batch, fn, args, flops, check=None, **tags):
        try:
            sec, comp_s, out = timed(fn, *args, reps=reps)
            row = dict(group=group, name=name, n=n, batch=batch, seconds=sec, compile_s=round(comp_s, 3),
                       per_item_us=sec / batch * 1e6, **tags)
            if flops:
                row.update(gflops=flops * batch / sec / 1e9, roofline=flops * batch / sec / peak)
            if check is not None:
                row.update(check(out))
            rows.append(row)
            extra = "".join(f" {k} {v:.1e}" for k, v in row.items() if k in ("err_median", "err_max", "residual"))
            print(f"{group:8s} {name:22s} n={n:5d} batch={batch:6d} {sec * 1e3:10.3f} ms {row['per_item_us']:10.3f} us/item"
                  + (f" {100 * row['roofline']:6.2f}% of peak" if flops else "") + extra
                  + "".join(f" {k}={v}" for k, v in tags.items()), flush=True)
        except Exception as exc:
            rows.append(dict(group=group, name=name, n=n, batch=batch, error=f"{type(exc).__name__}: {exc}"[:400], **tags))
            print(f"{group:8s} {name:22s} n={n:5d} batch={batch:6d} FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    def rel_err(exact, x):
        den = jnp.sqrt(jnp.sum(x * x, axis=(-2, -1)))

        def check(out):
            proj = out[0] if isinstance(out, tuple) else out
            r = jnp.sqrt(jnp.sum((proj - exact) ** 2, axis=(-2, -1))) / den
            return dict(err_median=float(jnp.median(r)), err_max=float(jnp.max(r)))
        return check

    # 1. PSD projection at the order-2 and order-3 moment-matrix sizes (order 1 already batches, M0)
    p_low = int(pc["lowrank"]["p"])
    for n, batches in pc["psd"]:
        perms, invs = jacobi_tables(n)
        for r_pos in pc["psd_rank"]:
            for batch in batches:
                ka, kb = jax.random.split(jax.random.PRNGKey(1000 * n + 10 * r_pos + batch))
                x0 = structured_symmetric(ka, batch, n, r_pos)
                x1 = perturb(kb, x0, float(pc["psd_perturbation"]))
                _, v0 = jax.jit(jnp.linalg.eigh)(x0)
                exact = jax.jit(psd_eigh)(x1)
                check = rel_err(exact, x1)
                tag = dict(rank=r_pos)
                run("psd", "eigh", n, batch, psd_eigh, (x1,), 9 * n ** 3, check, **tag)
                run("psd", "eigvalsh", n, batch, jnp.linalg.eigvalsh, (x1,), 4 * n ** 3, None, **tag)
                for k in pc["ns_iters"]:
                    run("psd", f"newton_schulz_{k}", n, batch, functools.partial(psd_newton_schulz, iters=k), (x1,),
                        (4 * k + 2) * n ** 3, check, **tag)
                run("psd", "qdwh_polar", n, batch, psd_qdwh, (x1,), None, check, **tag)
                if batch * n * n <= float(pc["jacobi_max_elements"]):
                    if n <= int(pc["jacobi_cold_max_n"]):
                        for s in pc["jacobi_cold_sweeps"]:
                            run("psd", f"jacobi_cold_{s}", n, batch,
                                functools.partial(psd_jacobi, perms=perms, invs=invs, sweeps=s), (x1,), 9 * s * n ** 3, check, **tag)
                    for s in pc["jacobi_warm_sweeps"]:
                        run("psd", f"jacobi_warm_{s}", n, batch,
                            functools.partial(psd_jacobi_warm, perms=perms, invs=invs, sweeps=s), (x1, v0),
                            9 * s * n ** 3 + 4 * n ** 3, check, **tag)
                for st in pc["lowrank"]["steps"]:
                    run("psd", f"lowrank_p{p_low}_s{st}", n, batch, functools.partial(psd_lowrank, steps=st),
                        (x1, v0[..., -p_low:]), (6 * st + 4) * n * n * p_low, check, **tag)
                del x0, x1, v0, exact

    # 2. Cholesky and triangular solves: the IPM's normal equations (small n, all edges) and the
    #    cyclic-reduction pivots (block width w, a few blocks per level); matmul is the roofline reference
    for n, batches in pc["cholesky"]:
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
            run("chol", "cholesky", n, batch, jnp.linalg.cholesky, (a,), n ** 3 / 3)
            run("chol", "cholesky_solve_vec", n, batch, chol_solve, (a, bv), n ** 3 / 3 + 2 * n ** 2, residual)
            run("chol", "cholesky_solve_mat", n, batch, chol_solve, (a, bm), n ** 3 / 3 + 2 * n ** 3)
            run("chol", "matmul", n, batch, lambda a: a @ a, (a,), 2 * n ** 3)
            del a, bv, bm

    # 3. The low-rank tree merge (plan 11.3): k x k eigh and thin QR of the boundary directions
    for k, batches in pc["merge_eigh"]:
        for batch in batches:
            x = jax.random.normal(jax.random.PRNGKey(11 * k + batch), (batch, k, k))
            run("merge", "eigh", k, batch, jnp.linalg.eigh, (0.5 * (x + swap(x)),), 9 * k ** 3)
    for m, k, batches in pc["merge_qr"]:
        for batch in batches:
            x = jax.random.normal(jax.random.PRNGKey(13 * k + batch), (batch, m, k))
            run("merge", f"qr_{m}x{k}", k, batch, lambda x: jnp.linalg.qr(x)[0], (x,), 4 * m * k * k)

    # 4. The coupling operator K (plan 6.3): scatter-add edge coordinates into vertex prices and gather back
    q = int(pc["coupling"]["q"])
    for e in pc["coupling"]["edges"]:
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
    out = ROOT / cfg["output"]["root"] / "bench" / "primitives.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(env=environment_info(), peak_tflops_fp64=peak / 1e12, rows=rows, verdicts=verdicts), indent=2))
    print_verdicts(verdicts)
    print(f"saved {out}", flush=True)


def batching_verdicts(rows):
    """Per primitive and size: per-item time at the largest batch over the smallest (falls when batching helps), the share of the FLOP roofline, and a verdict."""
    groups = {}
    for r in rows:
        if "error" not in r:
            groups.setdefault((r["group"], r["name"], r["n"], r.get("rank")), []).append(r)
    out = []
    for (group, name, n, rank), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["batch"])
        small, large = rs[0], rs[-1]
        ratio = large["per_item_us"] / small["per_item_us"] if len(rs) > 1 else None
        roof = large.get("roofline")
        if (ratio is not None and ratio <= 0.8) or (roof is not None and roof >= 0.2):
            verdict = "batched"
        elif ratio is None:
            verdict = "one batch size only"
        else:
            verdict = "NOT BATCHED?"
        out.append(dict(group=group, name=name, n=n, rank=rank, batches=[r["batch"] for r in rs],
                        per_item_us=[r["per_item_us"] for r in rs], per_item_ratio=ratio, roofline_at_largest=roof,
                        seconds_at_largest=large["seconds"], err_max=large.get("err_max"), verdict=verdict))
    return out


def print_verdicts(verdicts):
    print("\nBATCHING VERDICTS (per-item time at the largest batch / at the smallest; <= 0.8, or >= 20% of the FP64 roofline, counts as batched)")
    for v in verdicts:
        ratio = "   -  " if v["per_item_ratio"] is None else f"{v['per_item_ratio']:6.2f}"
        roof = "   -   " if v["roofline_at_largest"] is None else f"{100 * v['roofline_at_largest']:6.2f}%"
        err = "" if v["err_max"] is None else f"  err_max {v['err_max']:.1e}"
        rank = "" if v["rank"] is None else f" rank {v['rank']}"
        print(f"  {v['group']:8s} {v['name']:22s} n={v['n']:5d}{rank:8s} batches {str(v['batches']):14s} ratio {ratio} "
              f"roofline {roof}  {v['seconds_at_largest'] * 1e3:10.3f} ms at largest  {v['verdict']}{err}", flush=True)


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
    ap.add_argument("--bench_primitives", action="store_true", help="time every round primitive and judge its batching (M0)")
    args = ap.parse_args()
    cfg = load_config(args.run, args.set)
    if args.bench_eigh:
        bench_eigh(cfg)
    if args.bench_primitives:
        bench_primitives(cfg)
    if not (args.bench_eigh or args.bench_primitives):
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
