"""Planar pushing on a time-layered trellis (plan sections 4-6): geometry and modes, the true per-edge constraints as one polynomial table, the convex outer models, the true-model step, trellis assembly with pruning, the CPU reference solve, and the milestone self-tests.

Math: docs/trajopt_detailed.tex (source of truth). Implementation: IMPLEMENTATION_PLAN.md.

    python push_model.py --selftest                  # milestone checks implemented so far (M0, M1, M2)
    python push_model.py --run A --build             # build, prune, validate and save one run's edge models
    python push_model.py --run B --band_report       # the Taylor band widths and the region sizing table
    python push_model.py --run A --reference         # the whole relaxation on the CPU, for J_rel
"""
import argparse
import collections
import copy
import itertools
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

# Build-time CPU work runs in forked workers (pool_map): one BLAS thread each.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import yaml
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent


# ===== CONFIG =====
# configs/base.yaml holds every default. A run file in configs/runs/ overrides some keys, and
# --set a.b.c=value overrides single keys on the command line.

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


def resolve_task(cfg, name=None):
    """Start and goal slider poses (x, y, theta), the pusher's start (body frame) and its start region."""
    name = name or cfg["task"]
    spec, common = cfg["tasks"][name], cfg.get("task_common", {})
    return dict(name=name,
                start=np.asarray(spec["start"], float),
                goal=np.asarray(spec["goal"], float),
                pusher_start=np.asarray(spec.get("pusher_start", common.get("pusher_start")), float),
                start_region=spec.get("start_region", common.get("start_region")))


# ===== WORKERS =====
# Build-time CPU work runs in forked processes; nothing here touches the GPU, and this script never
# imports JAX. Workers run every HiGHS LP and the parent never does: HiGHS keeps a thread pool per
# process, and a fork of a process that has started one inherits a scheduler whose threads are gone.

_POOL_FN = None


def _pool_call(item):
    return _POOL_FN(item)


def pool_map(fn, items, build):
    """[fn(x) for x in items] in forked workers, one item at a time because edge types differ in cost. With one worker, or no fork on this platform, it runs in-process and never forks."""
    global _POOL_FN
    items = list(items)
    if not items:
        return []
    workers = min(int(build["workers"]), os.cpu_count() or 1)
    if workers <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        return [fn(x) for x in items]
    _POOL_FN = fn
    with multiprocessing.get_context("fork").Pool(min(workers, len(items))) as pool:
        return pool.map_async(_pool_call, items, chunksize=1).get(timeout=float(build["timeout_s"]))


# ===== MODEL =====
# Slider body frame: faces j = 0, 1, 2, 3 are +x, +y, -x, -y. Face j has outward normal n_j,
# counter-clockwise tangent t_j = R(+90 deg) n_j (so the 2-D cross product n_j x t_j = +1),
# center c_j = (half-size along n_j) n_j and half-length l_j. The pusher is a disk of radius
# r_p; r is its center in the slider body frame. The slider's body twist follows the
# ellipsoidal quasi-static limit surface: v = c_v f / f_max^2, omega = c_v tau / tau_max^2.

FACE_NORMALS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
CONTACT_TYPES = ("stick", "slide_pos", "slide_neg")
REGION_NAMES = ("strip_0", "strip_1", "strip_2", "strip_3", "corner_0", "corner_1", "corner_2", "corner_3")
EDGE_KINDS = ("start", "mid", "goal")


def mean_distance_rectangle(ax, ay, n):
    """Mean distance to the centroid over a uniform-pressure rectangle, on an n x n midpoint grid."""
    u = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y = np.meshgrid(u * ax, u * ay, indexing="ij")
    return float(np.mean(np.hypot(x, y)))


def build_model(cfg):
    """Geometry, limit surface and limits of plan 4.1 as plain floats and arrays, plus the region counts of plan 5.3."""
    m, tr = cfg["model"], cfg["trellis"]
    ax, ay = float(m["slider"]["half_x"]), float(m["slider"]["half_y"])
    normals = FACE_NORMALS.copy()
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    centers = normals * np.array([ax, ay])
    half_len = np.array([ay, ax, ay, ax])
    rp, mu_c, fn_max = float(m["pusher"]["radius"]), float(m["pusher"]["mu_contact"]), float(m["force"]["fn_max"])
    f_max = float(m["slider"]["mu_table"]) * float(m["slider"]["mass"]) * float(m.get("gravity", 9.81))
    rbar = mean_distance_rectangle(ax, ay, int(m["limit_surface"].get("rbar_grid", 1000)))
    # Two bounds on |tau|, both valid on every true point; the box keeps the smaller.
    # Geometry: on face j, tau = f_t |c_j| + lambda f_n with lambda = t_j^T r_c in [-l_j, l_j] and
    # |f_t| <= mu_c f_n. Rotation: unit norms make c_k s_{k+1} - s_k c_{k+1} = sin(dtheta), and the
    # step bound gives |dtheta| <= theta_max, so |dt c_v tau / tau_max^2| <= sin(theta_max).
    tau_max = f_max * rbar
    c_v, dt = float(m["limit_surface"]["c_v"]), float(tr["dt"])
    theta_step_max = math.radians(float(m["step"]["theta_step_max_deg"]))
    tau_bound = np.minimum(fn_max * (mu_c * np.linalg.norm(centers, axis=1) + half_len),
                           math.sin(theta_step_max) * tau_max ** 2 / (dt * c_v))
    return dict(ax=ax, ay=ay, normals=normals, tangents=tangents, centers=centers, half_len=half_len,
                rp=rp, mu_c=mu_c, v_max=float(m["pusher"]["v_max_step"]),
                fn_max=fn_max, ft_max=mu_c * fn_max, tau_bound=tau_bound,
                f_max=f_max, rbar=rbar, tau_max=tau_max, c_v=c_v, dt=dt, theta_step_max=theta_step_max,
                p_box=np.asarray(m["workspace"]["p_box"], float), r_ws=float(m["workspace"]["r_ws"]),
                theta_splits=int(cfg["relaxation"]["regions"]["theta_splits"]),
                lambda_splits=int(cfg["relaxation"]["regions"]["lambda_splits"]))


def build_modes(cfg):
    """Contact modes (configured faces x friction types) first, then the configured non-contact regions (plan 4.3), each copied per region (plan 5.3).

    A copy keeps its parent's name in 'base'. 'theta' and 'lam' are its region indices, or None
    when that split is off.
    """
    tr = cfg["trellis"]
    regions_cfg = cfg["relaxation"]["regions"]
    n_theta, n_lam = int(regions_cfg["theta_splits"]), int(regions_cfg["lambda_splits"])
    faces = [int(j) for j in tr.get("faces", [0, 1, 2, 3])]
    use_all = tr.get("regions", "all") == "all"
    regions = list(REGION_NAMES) if use_all else list(tr["regions"])
    unknown = [r for r in regions if r not in REGION_NAMES] + [c for c in tr["contact_types"] if c not in CONTACT_TYPES]
    if unknown:
        raise ValueError(f"unknown regions or contact types: {unknown}")
    if use_all and int(tr.get("noncontact_regions", 8)) != 8:
        raise ValueError("trellis.noncontact_regions must be 8 when trellis.regions is 'all'")
    parents = [dict(name=f"c{j}_{ct}", contact=True, face=j, ctype=ct, region=None)
               for j in faces for ct in tr["contact_types"]]
    parents += [dict(name=r, contact=False, face=None, ctype=None, region=r) for r in regions]
    modes = []
    for p in parents:
        for lam in (range(n_lam) if p["contact"] and n_lam > 1 else [None]):
            for theta in (range(n_theta) if n_theta > 1 else [None]):
                suffix = ("" if lam is None else f"_lam{lam}") + ("" if theta is None else f"_th{theta}")
                modes.append(dict(p, name=p["name"] + suffix, base=p["name"], lam=lam, theta=theta))
    return modes


def regions_adjacent(a, b):
    """Corner j sits between faces j and j+1, so strip j borders corners j-1 and j."""
    (ka, ja), (kb, jb) = [(s.split("_")[0], int(s.split("_")[1])) for s in (a, b)]
    if ka == kb:
        return False
    strip, corner = (ja, jb) if ka == "strip" else (jb, ja)
    return corner in (strip, (strip - 1) % 4)


def transition_allowed(m, m2):
    """Mode graph of plan 4.4, the same at every layer."""
    if m["contact"] and m2["contact"]:
        return m["face"] == m2["face"]
    if m["contact"]:
        return m2["region"] == f"strip_{m['face']}"
    if m2["contact"]:
        return m["region"] == f"strip_{m2['face']}"
    return m["region"] == m2["region"] or regions_adjacent(m["region"], m2["region"])


def knot_from_pose(pose, r):
    """Knot state (p_x, p_y, c, s, r_x, r_y) from a slider pose (x, y, theta) and a pusher position."""
    return np.array([pose[0], pose[1], math.cos(pose[2]), math.sin(pose[2]), r[0], r[1]], float)


# Regions as modes (plan 5.3). theta_splits > 1 copies every mode per angle interval over [-pi, pi);
# lambda_splits > 1 copies the contact modes per cell of the contact position along the face. A copy
# refines its parent (tex Prop 3.7): same true model and same cost, on a smaller set. To keep every
# true step, label each knot by the half-open interval holding its angle: a static step then keeps
# its label, and a contact step's labels are no further apart than the rotation step bound.

def theta_interval(splits, i):
    """Closed angle interval (lo, hi) of theta region i."""
    width = 2.0 * math.pi / splits
    return -math.pi + i * width, -math.pi + (i + 1) * width


def theta_label(theta, splits):
    """The theta region holding an angle, with half-open intervals."""
    return min(int(((theta + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi / splits)), splits - 1)


def chart_near(theta, center):
    """theta + 2 pi k closest to center; theta may be an array."""
    return theta + 2.0 * math.pi * np.round((center - theta) / (2.0 * math.pi))


def sector_box(splits, i):
    """Ranges of cos and sin over theta region i: (c_lo, c_hi, s_lo, s_hi)."""
    a, b = theta_interval(splits, i)
    angles = [a, b] + [k * 0.5 * math.pi for k in range(-4, 5) if a <= k * 0.5 * math.pi <= b]
    c, s = np.cos(angles), np.sin(angles)
    return float(c.min()), float(c.max()), float(s.min()), float(s.max())


def step_allowed(md, m, m2):
    """Mode graph plus the theta-region rule (plan 5.2.2): a non-contact step keeps its region since the slider is static, and a contact step reaches regions within the rotation step bound."""
    if not transition_allowed(m, m2):
        return False
    if m.get("theta") is None:
        return True
    d = abs(m["theta"] - m2["theta"])
    d = min(d, md["theta_splits"] - d)
    if not m["contact"]:
        return d == 0
    return (d - 1) * 2.0 * math.pi / md["theta_splits"] <= md["theta_step_max"]


def goal_allowed(md, m, task):
    """Whether a goal edge from mode m can hold a true step: the goal angle lies in m's theta region, or for a contact step within the rotation step bound of it."""
    if m.get("theta") is None:
        return True
    a, b = theta_interval(md["theta_splits"], m["theta"])
    g = chart_near(task["goal"][2], 0.5 * (a + b))
    return max(a - g, g - b, 0.0) <= (md["theta_step_max"] if m["contact"] else 1e-12)


def start_mode_index(md, modes, task):
    """The start vertex's mode: the start region, in the theta region holding the start angle."""
    label = theta_label(task["start"][2], md["theta_splits"]) if md["theta_splits"] > 1 else None
    for i, m in enumerate(modes):
        if m["base"] == task["start_region"] and m["theta"] == label and not m["contact"]:
            return i
    raise ValueError(f"start region {task['start_region']!r} is not a configured non-contact region")


# ===== CONSTRAINTS =====
# The true, nonconvex model of every edge (plan 4.5) as ONE table of polynomials of degree <= 2
# over the edge vector v. The true-model checks, pruning and both relaxation builders (M1) all
# read this table, so they cannot disagree about the model.
#
# Edge vector v (15 entries, SDP layout): knot k, knot k+1, then forces. A knot is
# (p_x, p_y, c, s, r_x, r_y): slider position (world), rotation (c, s) = (cos, sin), pusher
# center (slider body frame). Non-contact edges pin the forces to zero, so all edges share one layout.
PX0, PY0, C0, S0, RX0, RY0, PX1, PY1, C1, S1, RX1, RY1, FN, FT, TAU = range(15)
NV = 15
VAR_NAMES = ("px0", "py0", "c0", "s0", "rx0", "ry0", "px1", "py1", "c1", "s1", "rx1", "ry1", "fn", "ft", "tau")
KNOT0 = (PX0, PY0, C0, S0, RX0, RY0)
KNOT1 = (PX1, PY1, C1, S1, RX1, RY1)

# A polynomial is a dict {monomial: coefficient}. A monomial is a sorted tuple of variable
# indices with repetition: (C0, C0) is c_k^2, () is the constant.


def const(c):
    return {(): float(c)} if c else {}


def var(i, c=1.0):
    return {(i,): float(c)}


def lin(coeffs, c0=0.0):
    """Affine polynomial sum_i coeffs[i] v_i + c0."""
    p = {(int(i),): float(a) for i, a in coeffs.items() if a}
    if c0:
        p[()] = float(c0)
    return p


def poly_add(*ps):
    out = {}
    for p in ps:
        for k, c in p.items():
            out[k] = out.get(k, 0.0) + c
    return {k: c for k, c in out.items() if c != 0.0}


def poly_scale(p, a):
    return {k: a * c for k, c in p.items()} if a else {}


def poly_mul(p, q, a=1.0):
    """a * p * q."""
    out = {}
    for k1, c1 in p.items():
        for k2, c2 in q.items():
            k = tuple(sorted(k1 + k2))
            out[k] = out.get(k, 0.0) + a * c1 * c2
    return {k: c for k, c in out.items() if c != 0.0}


def poly_degree(p):
    return max((len(k) for k in p), default=0)


def poly_eval(p, v):
    return sum(c * math.prod(v[i] for i in k) for k, c in p.items())


def region_rows(md, name, ix, iy):
    """Linear rows g >= 0 that put the pusher center (v[ix], v[iy]) in a non-contact region, at least r_p from the box.

    strip_j: in front of face j, within its extent. corner_j: beyond faces j and j+1, with the
    corner cut by a diagonal so every point is at least r_p from the box's corner.
    """
    n, t, c, l, rp = md["normals"], md["tangents"], md["centers"], md["half_len"], md["rp"]
    kind, j = name.split("_")
    j = int(j)
    if kind == "strip":
        return [lin({ix: n[j][0], iy: n[j][1]}, -(n[j] @ c[j] + rp)),
                lin({ix: -t[j][0], iy: -t[j][1]}, l[j]),
                lin({ix: t[j][0], iy: t[j][1]}, l[j])]
    a, b = j, (j + 1) % 4
    d = n[a] + n[b]
    return [lin({ix: n[a][0], iy: n[a][1]}, -(n[a] @ c[a])),
            lin({ix: n[b][0], iy: n[b][1]}, -(n[b] @ c[b])),
            lin({ix: d[0], iy: d[1]}, -(n[a] @ c[a] + n[b] @ c[b] + math.sqrt(2.0) * rp))]


def contact_state_rows(md, j, ix, iy, tag):
    """Pusher center on face j's contact line (offset r_p from the face) and within the face's extent."""
    n, t, c, l = md["normals"][j], md["tangents"][j], md["centers"][j], md["half_len"][j]
    along = lin({ix: t[0], iy: t[1]}, -(t @ c))
    return [("eq", lin({ix: n[0], iy: n[1]}, -(n @ c + md["rp"])), f"contact_line_{tag}"),
            ("ge", poly_add(const(l), poly_scale(along, -1.0)), f"contact_extent_hi_{tag}"),
            ("ge", poly_add(const(l), along), f"contact_extent_lo_{tag}")]


def theta_rows(md, theta, ic, js, tag):
    """Angular sector of a theta region as linear rows in (c, s), valid for unit-norm points in the sector (plan 5.3): two half-planes and the chord cut."""
    a, b = theta_interval(md["theta_splits"], theta)
    mid, h = 0.5 * (a + b), 0.5 * (b - a)
    clean = lambda x: 0.0 if abs(x) < 1e-15 else x
    return [("ge", lin({ic: clean(-math.sin(a)), js: clean(math.cos(a))}), f"theta_lo_{tag}"),
            ("ge", lin({ic: clean(math.sin(b)), js: clean(-math.cos(b))}), f"theta_hi_{tag}"),
            ("ge", lin({ic: clean(math.cos(mid)), js: clean(math.sin(mid))}, -clean(math.cos(h))), f"theta_chord_{tag}")]


def lambda_rows(md, mode, ix, iy, tag):
    """Contact position along the face in the mode's lambda cell (plan 5.2.2)."""
    j = mode["face"]
    t, c, l = md["tangents"][j], md["centers"][j], md["half_len"][j]
    width = 2.0 * l / md["lambda_splits"]
    lo = -l + mode["lam"] * width
    along = lin({ix: t[0], iy: t[1]}, -(t @ c))
    return [("ge", poly_add(along, const(-lo)), f"lambda_lo_{tag}"),
            ("ge", poly_add(const(lo + width), poly_scale(along, -1.0)), f"lambda_hi_{tag}")]


def contact_step_rows(md, mode):
    """Rows 3-6 of plan 4.5 for one step in contact on mode['face'] with friction type mode['ctype']."""
    j = mode["face"]
    n, t = md["normals"][j], md["tangents"][j]
    rows = []
    # body-frame force on the slider: f = f_n (-n_j) + f_t t_j
    fbx = lin({FN: -n[0], FT: t[0]})
    fby = lin({FN: -n[1], FT: t[1]})
    # 3. translation p_{k+1} - p_k = dt R(c_k, s_k) v, v = c_v f / f_max^2 (two bilinear equalities)
    k_tr = md["dt"] * md["c_v"] / md["f_max"] ** 2
    rows.append(("eq", poly_add(var(PX1), var(PX0, -1.0),
                                poly_scale(poly_add(poly_mul(var(C0), fbx), poly_mul(var(S0), fby, -1.0)), -k_tr)), "trans_x"))
    rows.append(("eq", poly_add(var(PY1), var(PY0, -1.0),
                                poly_scale(poly_add(poly_mul(var(S0), fbx), poly_mul(var(C0), fby)), -k_tr)), "trans_y"))
    # 4. torque tau = r_c x f, with contact point r_c = r_k - r_p n_j (bilinear in r and forces)
    rcx = lin({RX0: 1.0}, -md["rp"] * n[0])
    rcy = lin({RY0: 1.0}, -md["rp"] * n[1])
    rows.append(("eq", poly_add(var(TAU), poly_mul(rcx, fby, -1.0), poly_mul(rcy, fbx)), "torque"))
    # 5. contact geometry at both knots
    rows += contact_state_rows(md, j, RX0, RY0, "k0") + contact_state_rows(md, j, RX1, RY1, "k1")
    # 6. friction. f_t is the tangential force the pusher applies TO THE SLIDER. Friction on the
    # pusher opposes the pusher's sliding relative to the slider, so by Newton's third law the
    # force on the slider points ALONG that sliding: in slide_pos the pusher slides toward +t_j
    # and drags the slider toward +t_j, so f_t = +mu f_n (check_friction_sign tests this).
    mu = md["mu_c"]
    rows.append(("ge", var(FN), "fn_nonneg"))
    rows.append(("ge", lin({FN: -1.0}, md["fn_max"]), "fn_max"))
    slide = lin({RX1: t[0], RX0: -t[0], RY1: t[1], RY0: -t[1]})          # t_j^T (r_{k+1} - r_k)
    if mode["ctype"] == "stick":
        rows.append(("ge", lin({FN: mu, FT: -1.0}), "stick_hi"))
        rows.append(("ge", lin({FN: mu, FT: 1.0}), "stick_lo"))
        rows.append(("eq", lin({RX1: 1.0, RX0: -1.0}), "stick_rx"))
        rows.append(("eq", lin({RY1: 1.0, RY0: -1.0}), "stick_ry"))
    elif mode["ctype"] == "slide_pos":
        rows.append(("eq", lin({FT: 1.0, FN: -mu}), "slide_force"))
        rows.append(("ge", slide, "slide_dir"))
    elif mode["ctype"] == "slide_neg":
        rows.append(("eq", lin({FT: 1.0, FN: mu}), "slide_force"))
        rows.append(("ge", poly_scale(slide, -1.0), "slide_dir"))
    else:
        raise ValueError(f"unknown contact type {mode['ctype']!r}")
    return rows


def free_step_rows(md, mode):
    """Row 7 of plan 4.5 for one non-contact step: static slider, zero forces, pusher in the tail region at knot k."""
    rows = [("eq", lin({a: 1.0, b: -1.0}), f"static_{VAR_NAMES[b][:-1]}")
            for a, b in ((PX1, PX0), (PY1, PY0), (C1, C0), (S1, S0))]
    rows += [("eq", var(i), f"zero_{VAR_NAMES[i]}") for i in (FN, FT, TAU)]
    rows += [("ge", p, f"tail_region_{i}") for i, p in enumerate(region_rows(md, mode["region"], RX0, RY0))]
    return rows


def edge_constraints(md, task, kind, tail, head):
    """True constraints of one edge type (plan 4.5) as rows ('eq' | 'ge', polynomial, tag): h(v) = 0 or g(v) >= 0.

    kind is 'start', 'mid' or 'goal'. The step runs in the tail mode (the start region for start
    edges); head is the mode at knot k+1, or None for goal edges. task is used by start and goal
    edges only.
    """
    rows = []
    # 1. unit norm of the rotation at both knots
    for ic, js, k in ((C0, S0, 0), (C1, S1, 1)):
        rows.append(("eq", poly_add(poly_mul(var(ic), var(ic)), poly_mul(var(js), var(js)), const(-1.0)), f"unit_norm_{k}"))
    # 2. relative rotation c_k s_{k+1} - s_k c_{k+1} = dt omega, omega = c_v tau / tau_max^2. This
    # is sin(theta_{k+1} - theta_k) and stays consistent with both unit norms; do not use the Euler
    # update c_{k+1} = c_k - dt omega s_k, which contradicts them. Step bound: cos(dtheta) >= cos(max).
    k_rot = md["dt"] * md["c_v"] / md["tau_max"] ** 2
    rows.append(("eq", poly_add(poly_mul(var(C0), var(S1)), poly_mul(var(S0), var(C1), -1.0), var(TAU, -k_rot)), "rot_rel"))
    rows.append(("ge", poly_add(poly_mul(var(C0), var(C1)), poly_mul(var(S0), var(S1)),
                                const(-math.cos(md["theta_step_max"]))), "rot_step"))
    rows += contact_step_rows(md, tail) if tail["contact"] else free_step_rows(md, tail)
    # regions (plan 5.3): the tail's copy holds knot k and the head's copy holds knot k+1
    if tail.get("theta") is not None:
        rows += theta_rows(md, tail["theta"], C0, S0, "k0")
    if head is not None and head.get("theta") is not None:
        rows += theta_rows(md, head["theta"], C1, S1, "k1")
    if tail.get("lam") is not None:
        rows += lambda_rows(md, tail, RX0, RY0, "k0")
    if head is not None and head.get("lam") is not None:
        rows += lambda_rows(md, head, RX1, RY1, "k1")
    # 7. pusher speed limit on every edge (body frame, infinity norm)
    for i1, i0, a in ((RX1, RX0, "x"), (RY1, RY0, "y")):
        rows.append(("ge", lin({i1: -1.0, i0: 1.0}, md["v_max"]), f"v_max_{a}_hi"))
        rows.append(("ge", lin({i1: 1.0, i0: -1.0}, md["v_max"]), f"v_max_{a}_lo"))
    # 8. head-mode membership at knot k+1 (two-sided convention)
    if head is not None:
        if head["contact"]:
            if not (tail["contact"] and tail["face"] == head["face"]):
                rows += contact_state_rows(md, head["face"], RX1, RY1, "head")
        else:
            rows += [("ge", p, f"head_region_{i}") for i, p in enumerate(region_rows(md, head["region"], RX1, RY1))]
    # 10. start: the whole knot k is fixed; goal: the slider pose at knot k+1 is fixed, the pusher is free
    if kind == "start":
        for i, val in zip(KNOT0, knot_from_pose(task["start"], task["pusher_start"])):
            rows.append(("eq", lin({i: 1.0}, -val), f"start_{VAR_NAMES[i]}"))
    elif kind == "goal":
        g = task["goal"]
        for i, val in zip((PX1, PY1, C1, S1), (g[0], g[1], math.cos(g[2]), math.sin(g[2]))):
            rows.append(("eq", lin({i: 1.0}, -val), f"goal_{VAR_NAMES[i]}"))
    return rows


def edge_box(md, tail, head=None):
    """Variable box of plan 4.5 item 9, before LP tightening; it holds every true point of the edge. A theta region narrows (c, s) at its knot (plan 5.3)."""
    lo, hi = np.empty(NV), np.empty(NV)
    for off, mode in ((0, tail), (6, head)):
        lo[off + 0], hi[off + 0] = md["p_box"][0]
        lo[off + 1], hi[off + 1] = md["p_box"][1]
        lo[off + 2:off + 4], hi[off + 2:off + 4] = -1.0, 1.0
        if mode is not None and mode.get("theta") is not None:
            lo[off + 2], hi[off + 2], lo[off + 3], hi[off + 3] = sector_box(md["theta_splits"], mode["theta"])
        lo[off + 4:off + 6], hi[off + 4:off + 6] = -md["r_ws"], md["r_ws"]
    if tail["contact"]:
        tb = md["tau_bound"][tail["face"]]
        lo[FN:], hi[FN:] = (0.0, -md["ft_max"], -tb), (md["fn_max"], md["ft_max"], tb)
    else:
        lo[FN:], hi[FN:] = 0.0, 0.0
    return lo, hi


def mode_switch(tail, head):
    """Whether a step changes mode. Moving between region copies of one mode is no switch (plan 5.3): a true trajectory does not pay for it, so charging it would make the bound invalid."""
    return head is not None and head.get("base", head["name"]) != tail.get("base", tail["name"])


def edge_cost(cfg, tail, head):
    """Quadratic edge cost of plan 4.6 as a polynomial over v (the same for all runs)."""
    w = cfg["costs"]
    diff = lambda a, b: lin({a: 1.0, b: -1.0})
    sq = lambda a, b: poly_mul(diff(a, b), diff(a, b))
    switch = mode_switch(tail, head)
    return poly_add(poly_scale(poly_add(sq(PX1, PX0), sq(PY1, PY0)), w["w_p"]),
                    poly_scale(poly_add(const(2.0), poly_mul(var(C0), var(C1), -2.0), poly_mul(var(S0), var(S1), -2.0)), w["w_theta"]),
                    poly_scale(poly_add(sq(RX1, RX0), sq(RY1, RY0)), w["w_r"]),
                    poly_scale(poly_add(poly_mul(var(FN), var(FN)), poly_mul(var(FT), var(FT))), w["w_f"]),
                    const(w["w_switch"] if switch else 0.0))


# ===== RELAXATIONS =====
# Convex outer models of every edge type (plan section 5), both read from the constraint table above:
# the order-kappa moment relaxation (tex 3.3) and the Taylor model with certified bands (tex 3.2).
#
# Both produce the same record, homogeneous in m, with m[0] = y_e the edge flow:
#   Aeq m = 0 and G m >= 0: sparse rows. The constant of a polynomial row multiplies m[0], so a
#       moment m_alpha of the constant monomial is the flow and homogenization needs no extra step;
#   lo m[0] <= m <= hi m[0]: interval bounds per unit flow, with 0 <= m[0] <= 1 on top;
#   psd: blocks B_b(m) >= 0, each a sparse upper-triangular map (row, col, moment index, coefficient);
#   soc: run B with cost_model soc only, entries z y >= scale (form m)^2;
#   c: the linear cost; tail_map, head_map: sparse maps from m to the coordinates coupled at the tail
#       and head vertices (tex 2.1 allows any linear map that is consistent on true points).
# An SDP record also holds `free` and `sub`: the variables that survive linear elimination (plan 5.1.7)
# and every edge variable as a polynomial in them.

# ----- Moment relaxation (plan 5.1) -----

_BASES = {}


def moment_basis(kappa, nv=NV):
    """Monomials of degree <= 2 kappa in nv variables, ordered by degree then lexicographically, with their index maps (plan 5.1.1). Cached per order and variable count.

    monos: the monomials; index: monomial -> position; upto[d]: how many have degree <= d;
    times[i, u]: position of v_i * monomial u (deg u < 2 kappa); mat[a, b]: position of monomial a *
    monomial b (deg a, deg b <= kappa); parent, last: monomial = parent * v_last; exp: exponents.
    """
    if (kappa, nv) in _BASES:
        return _BASES[(kappa, nv)]
    monos, upto = [()], [1]
    for d in range(1, 2 * kappa + 1):
        monos += list(itertools.combinations_with_replacement(range(nv), d))
        upto.append(len(monos))
    index = {m: i for i, m in enumerate(monos)}
    n, s, low = len(monos), upto[kappa], upto[2 * kappa - 1]
    times = np.empty((nv, low), np.int32)
    for u in range(low):
        for i in range(nv):
            times[i, u] = index[tuple(sorted(monos[u] + (i,)))]
    mat = np.empty((s, s), np.int32)
    for a in range(s):
        for b in range(a, s):
            mat[a, b] = mat[b, a] = index[tuple(sorted(monos[a] + monos[b]))]
    parent, last, exp = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros((n, nv), np.int32)
    for i in range(1, n):
        parent[i], last[i] = index[monos[i][:-1]], monos[i][-1]
        exp[i] = exp[parent[i]]
        exp[i, last[i]] += 1
    basis = dict(kappa=kappa, nv=nv, monos=monos, index=index, upto=upto, times=times, mat=mat, parent=parent, last=last, exp=exp)
    basis["moment_map"] = localizing_map(basis, const(1.0), kappa)
    _BASES[(kappa, nv)] = basis
    return basis


def mono_times(basis, k, u):
    """Positions of monomial k times each monomial in the index array u."""
    for i in k:
        u = basis["times"][i, u]
    return u


def rank_one_moments(basis, V):
    """Moment vectors (n, monomials) of the points V (n, nv): every monomial's value at each point."""
    M = np.empty((V.shape[0], len(basis["monos"])))
    M[:, 0] = 1.0
    upto = basis["upto"]
    for d in range(1, len(upto)):
        idx = np.arange(upto[d - 1], upto[d])
        M[:, idx] = M[:, basis["parent"][idx]] * V[:, basis["last"][idx]]
    return M


def moment_box(basis, lo, hi):
    """Interval bounds of every monomial over the variable box, per unit flow (plan 5.1.4): interval arithmetic with exact powers, so an even power of a variable that straddles zero starts at zero."""
    exp = basis["exp"]
    mlo, mhi = np.ones(exp.shape[0]), np.ones(exp.shape[0])
    for i in range(basis["nv"]):
        nz = np.flatnonzero(exp[:, i])
        e = exp[nz, i]
        a, b = float(lo[i]) ** e, float(hi[i]) ** e
        pl, ph = np.minimum(a, b), np.maximum(a, b)
        if lo[i] < 0.0 < hi[i]:
            pl[e % 2 == 0] = 0.0
        cand = np.stack([mlo[nz] * pl, mlo[nz] * ph, mhi[nz] * pl, mhi[nz] * ph])
        mlo[nz], mhi[nz] = cand.min(axis=0), cand.max(axis=0)
    return mlo, mhi


def lift_rows(basis, polys, all_multipliers, degrees=None):
    """Moment rows L_m(p u) of the polynomials: every monomial u with deg(p u) <= 2 kappa when all_multipliers (equalities, plan 5.1.2), else u = 1 only. degrees gives each polynomial's degree before substitution, which sets its multipliers. Returns (csr matrix, source polynomial per row, multiplier monomial per row)."""
    n, top = len(basis["monos"]), 2 * basis["kappa"]
    rows, cols, vals, owner, mult = [], [], [], [], []
    base = 0
    for pi, p in enumerate(polys):
        deg = poly_degree(p) if degrees is None else degrees[pi]
        us = np.arange(basis["upto"][top - deg] if all_multipliers else 1)
        for k, c in p.items():
            rows.append(base + np.arange(us.size))
            cols.append(mono_times(basis, k, us))
            vals.append(np.full(us.size, float(c)))
        owner.append(np.full(us.size, pi))
        mult.append(us)
        base += us.size
    if not base:
        return sp.csr_matrix((0, n)), np.zeros(0, np.int64), np.zeros(0, np.int64)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(base, n))
    return A, np.concatenate(owner), np.concatenate(mult)


def independent_rows(A, tol=1e-10):
    """A maximal independent subset of the rows of a small sparse matrix, by pivoted QR on unit-norm rows (plan 5.1.7): the IPM's normal equations need full row rank."""
    if A.shape[0] == 0:
        return np.zeros(0, np.int64)
    d = A.toarray()
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r, piv = scipy.linalg.qr(d.T, mode="r", pivoting=True)
    diag = np.abs(np.diag(r))
    return np.sort(piv[:int(np.sum(diag > tol * diag[0]))])


def unique_rows(A):
    """One row per class of rows equal up to scale, by two random projections of the rows scaled to a unit first entry (plan 5.1.7). A hash collision would drop a row and only enlarge the outer set, so it cannot break validity."""
    if A.shape[0] == 0:
        return np.zeros(0, np.int64)
    A = A.copy()
    A.sort_indices()
    lead = A.data[A.indptr[:-1]]
    key = np.round((A @ np.random.default_rng(0).normal(size=(A.shape[1], 2))) / lead[:, None], 9)
    return np.sort(np.unique(key, axis=0, return_index=True)[1])


def localizing_map(basis, p, order):
    """Sparse upper-triangular map of the localizing matrix M_order(p m), with entry (a, b) = L_m(p u_a u_b): (row, col, moment index, coefficient) as parallel arrays. p = 1 gives the moment matrix."""
    s = basis["upto"][order]
    iu, ju = np.triu_indices(s)
    pairs = basis["mat"][iu, ju]
    moms = [mono_times(basis, k, pairs) for k in p]
    return dict(size=s, row=np.tile(iu, len(p)).astype(np.int32), col=np.tile(ju, len(p)).astype(np.int32),
                mom=np.concatenate(moms).astype(np.int32), coef=np.concatenate([np.full(iu.size, float(c)) for c in p.values()]))


def coupling_map(basis, sub, knot, second_moments):
    """Sparse map from m to a knot's coupled coordinates: flow, first moments, then second moments (plan 5.1.5), each read through the substitution. The same order at both knots, so a head's coordinates meet the next tail's."""
    polys = [const(1.0)] + [sub[i] for i in knot]
    if second_moments:
        polys += [poly_mul(sub[a], sub[b]) for pos, a in enumerate(knot) for b in knot[pos:]]
    return lift_rows(basis, polys, False)[0]


# ----- Linear elimination (plan 5.1.7) -----
# Most equalities of an edge are linear: r_{k+1} = r_k, f_t = mu f_n, zero forces on a non-contact step.
# Lifted, each one becomes a row per monomial and forces every moment matrix to be singular, so the PSD
# blocks have no interior. Substituting v = a + Z w first removes those rows and variables instead, and
# keeps the outer set: with the linear rows in force, a full moment vector is m = T m_w, the moment
# matrix is W M_w W^T with W of full column rank, and every other row is the image of a reduced row.
# The elimination happens at build time, and each edge type gets its own smaller m.

def linear_elimination(rows):
    """Solves a table's linear equalities: (free variables, {variable: polynomial over the free variables' positions}). Each row eliminates its highest remaining variable, so the tail knot and f_n stay free."""
    a = np.array([[p.get((i,), 0.0) for i in range(NV)] + [p.get((), 0.0)]
                  for sense, p, _ in rows if sense == "eq" and poly_degree(p) <= 1]).reshape(-1, NV + 1)
    pivots, rank = {}, 0
    for col in range(NV - 1, -1, -1):
        if rank == a.shape[0]:
            break
        best = rank + int(np.argmax(np.abs(a[rank:, col])))
        if abs(a[best, col]) < 1e-12:
            continue
        a[[rank, best]] = a[[best, rank]]
        a[rank] /= a[rank, col]
        others = a[:, col] != 0.0
        others[rank] = False
        a[others] -= a[others, col][:, None] * a[rank]
        pivots[col] = rank
        rank += 1
    if np.max(np.abs(a[rank:, NV]), initial=0.0) > 1e-9:
        raise ValueError("the linear equalities of this edge type are inconsistent, so it should have been pruned")
    free = [i for i in range(NV) if i not in pivots]
    sub = {v: var(j) for j, v in enumerate(free)}
    for col, r in pivots.items():
        terms = {(j,): -float(a[r, v]) for j, v in enumerate(free) if abs(a[r, v]) > 1e-14}
        if abs(a[r, NV]) > 1e-14:
            terms[()] = -float(a[r, NV])
        sub[col] = terms
    return free, sub


def poly_substitute(p, sub):
    """p with every variable replaced by its polynomial in sub; terms that are only rounding are dropped."""
    out = {}
    for k, c in p.items():
        term = {(): c}
        for i in k:
            term = poly_mul(term, sub[i])
        for m, val in term.items():
            out[m] = out.get(m, 0.0) + val
    return {m: val for m, val in out.items() if abs(val) > 1e-13}


def row_key(p, any_sign):
    """Hashable form of a row up to scale: any nonzero scale for an equality, positive scale for an inequality."""
    items = sorted(p.items())
    lead = items[-1][1]
    scale = lead if any_sign else abs(lead)
    return tuple((k, round(c / scale, 10)) for k, c in items)


def implied_by_equalities(g, eqs):
    """Whether an inequality is a multiple of an equality plus a positive constant, e.g. the step bound of a static slider, c^2 + s^2 >= cos(max), given c^2 + s^2 = 1."""
    top = max(g, key=len)
    for h, _, _ in eqs:
        if top in h:
            rest = poly_add(g, poly_scale(h, -g[top] / h[top]))
            if not any(k for k, c in rest.items() if abs(c) > 1e-12) and rest.get((), 0.0) > 0.0:
                return True
    return False


def simplify_rows(eqs, ges, lo, hi, tol=1e-9):
    """Cleans rows after substitution without changing the outer set (plan 5.1.7): drops rows that became constants, duplicates, and inequalities implied by the equalities. A single-variable inequality is a variable bound: it tightens the box and loses its row, but keeps its localizing block at kappa >= 2. Rows are (poly, tag, degree before substitution); returns (eqs, ges with an is-bound flag, lo, hi)."""
    lo, hi = lo.copy(), hi.copy()
    constant = lambda p: not any(k for k in p)
    out_eq, seen = [], set()
    for p, tag, deg in eqs:
        if constant(p):
            if abs(p.get((), 0.0)) > tol:
                raise ValueError(f"equality {tag} reduces to the nonzero constant {p[()]}, so this edge type holds no true step")
            continue
        key = row_key(p, True)
        if key not in seen:
            seen.add(key)
            out_eq.append((p, tag, deg))
    out_ge, seen = [], set()
    for p, tag, deg in ges:
        if constant(p):
            if p.get((), 0.0) < -tol:
                raise ValueError(f"inequality {tag} reduces to the negative constant {p[()]}, so this edge type holds no true step")
            continue
        if implied_by_equalities(p, out_eq):
            continue
        key = row_key(p, False)
        if key in seen:
            continue
        seen.add(key)
        terms = [k for k in p if k]
        is_bound = len(terms) == 1 and len(terms[0]) == 1
        if is_bound:
            j, b = terms[0][0], p[terms[0]]
            edge = -p.get((), 0.0) / b
            if b > 0.0:
                lo[j] = max(lo[j], edge)
            else:
                hi[j] = min(hi[j], edge)
        out_ge.append((p, tag, deg, is_bound))
    crossed = lo > hi                           # only by the LP boxes' outward pad
    lo[crossed] = hi[crossed] = 0.5 * (lo[crossed] + hi[crossed])
    return out_eq, out_ge, lo, hi


def relax_sdp(cfg, md, task, kind, tail, head, lo, hi):
    """Order-kappa moment relaxation of one edge type (plan 5.1, tex Def 3.4), in the record described at the top of this section."""
    rx = cfg["relaxation"]
    kappa = int(rx["order"])
    eliminate = bool(rx["eliminate_linear"])
    rows = edge_constraints(md, task, kind, tail, head)
    free, sub = linear_elimination(rows) if eliminate else (list(range(NV)), {i: var(i) for i in range(NV)})
    basis = moment_basis(kappa, len(free))
    eqs, ges = [], []
    for sense, p, tag in rows:
        deg = poly_degree(p)
        if sense == "ge":
            ges.append((poly_substitute(p, sub), tag, deg))
        elif not (eliminate and deg <= 1):
            eqs.append((poly_substitute(p, sub), tag, deg))
    if kappa >= 2 and rx["ball"]:
        squares = [poly_mul(var(i), var(i), -1.0) for i in range(NV)]
        ges.append((poly_substitute(poly_add(const(np.sum(np.maximum(lo ** 2, hi ** 2))), *squares), sub), "ball", 2))
    wlo, whi = lo[free], hi[free]
    if eliminate:
        eqs, ges, wlo, whi = simplify_rows(eqs, ges, wlo, whi)
    else:
        ges = [(p, tag, deg, False) for p, tag, deg in ges]
    psd = [dict(basis["moment_map"], poly=const(1.0), tag="moment")]
    for p, tag, deg, _ in ges:
        order = kappa - math.ceil(deg / 2)
        if order >= 1:
            psd.append(dict(localizing_map(basis, p, order), poly=p, tag=tag))
    ge_rows = [(p, tag) for p, tag, _, is_bound in ges if not is_bound]
    if rx["rlt"]:
        linear = [(p, tag) for p, tag, deg, _ in ges if deg == 1]
        for pos, (p, ta) in enumerate(linear):
            for q, tb in linear[pos:]:
                ge_rows.append((poly_mul(p, q), f"rlt_{ta}_{tb}"))
    raw, owner, _ = lift_rows(basis, [e[0] for e in eqs], True, [e[2] for e in eqs])
    keep = independent_rows(raw) if kappa == 1 else unique_rows(raw)
    mlo, mhi = moment_box(basis, wlo, whi)
    n = len(basis["monos"])
    c = np.zeros(n)
    for k, w in poly_substitute(edge_cost(cfg, tail, head), sub).items():
        c[basis["index"][k]] += w
    second = bool(rx["couple_second_moments"])
    return dict(kind="sdp", kappa=kappa, n=n, free=free, sub=sub, Aeq=raw[keep], eq_owner=owner[keep], eq_tags=[e[1] for e in eqs],
                eq_polys=[e[0] for e in eqs], eq_degrees=[e[2] for e in eqs], eq_rows_raw=raw.shape[0],
                G=lift_rows(basis, [r[0] for r in ge_rows], False)[0], ge_tags=[r[1] for r in ge_rows], ge_polys=[r[0] for r in ge_rows],
                lo=mlo, hi=mhi, c=c, psd=psd, soc=[], bands={},
                tail_map=coupling_map(basis, sub, KNOT0, second), head_map=coupling_map(basis, sub, KNOT1, second))


# ----- Taylor model with certified bands (plan 5.2, tex Prop 3.1) -----
# Run B's edge vector: knot k and knot k+1 with an angle in place of (c, s), the forces, and one
# epigraph variable per cost term. m = (y, y v), the degree-1 moments of these 20 variables. Rows over
# them use the same polynomial dicts as above, and a row's constant multiplies y.
TPX0, TPY0, TTH0, TRX0, TRY0, TPX1, TPY1, TTH1, TRX1, TRY1, TFN, TFT, TTAU = range(13)
TZ = {name: 13 + i for i, name in enumerate(("dpx", "dpy", "dth", "drx", "dry", "fn", "ft"))}
NT = 13 + len(TZ)
TAYLOR_OF_SDP = {PX0: TPX0, PY0: TPY0, RX0: TRX0, RY0: TRY0, PX1: TPX1, PY1: TPY1, RX1: TRX1, RY1: TRY1, FN: TFN, FT: TFT, TAU: TTAU}
# Table rows that the angle variable makes redundant. Rows tagged theta_* become the angle's box.
TAYLOR_IMPLIED = {"unit_norm_0", "unit_norm_1", "static_s", "start_s0", "goal_s1"}


def taylor_charts(md, task, kind, tail, head):
    """Angle charts of one Taylor edge (plan 5.2.2): each vertex keeps theta in its own region's chart, and `wrap` (a multiple of 2 pi) makes theta_1 + wrap - theta_0 the true step when the regions straddle the +-pi cut. A goal edge keeps the goal angle in the tail's chart."""
    splits = md["theta_splits"]
    if splits < 3:
        raise ValueError("the Taylor model needs relaxation.regions.theta_splits >= 3, so that a pair of regions has one way around the circle")
    lo0, hi0 = theta_interval(splits, tail["theta"])
    th0, h = 0.5 * (lo0 + hi0), 0.5 * (hi0 - lo0)
    if kind == "goal":
        th1 = float(chart_near(task["goal"][2], th0))
        return dict(th0=th0, h=h, lo0=lo0, hi0=hi0, th1=th1, lo1=th1, hi1=th1, wrap=0.0)
    lo1, hi1 = theta_interval(splits, head["theta"])
    th1 = 0.5 * (lo1 + hi1)
    return dict(th0=th0, h=h, lo0=lo0, hi0=hi0, th1=th1, lo1=lo1, hi1=hi1, wrap=float(chart_near(th1, th0) - th1))


def taylor_dtheta(ch):
    """The angle step theta_1 + wrap - theta_0 as a polynomial."""
    return poly_add(var(TTH1), var(TTH0, -1.0), const(ch["wrap"]))


def force_box(md, tail, lo, hi):
    """Body-frame force (f_x, f_y) on a contact edge as linear forms over (f_n, f_t), with its center and radius over the edge box."""
    j = tail["face"]
    n, t = md["normals"][j], md["tangents"][j]
    fn_c, fn_r = 0.5 * (lo[FN] + hi[FN]), 0.5 * (hi[FN] - lo[FN])
    ft_c, ft_r = 0.5 * (lo[FT] + hi[FT]), 0.5 * (hi[FT] - lo[FT])
    return dict(fx=lin({TFN: -n[0], TFT: t[0]}), fy=lin({TFN: -n[1], TFT: t[1]}),
                fx_c=-n[0] * fn_c + t[0] * ft_c, fx_r=abs(n[0]) * fn_r + abs(t[0]) * ft_r,
                fy_c=-n[1] * fn_c + t[1] * ft_c, fy_r=abs(n[1]) * fn_r + abs(t[1]) * ft_r)


def translation_bands(f, th, h):
    """Bands of (cos f_x - sin f_y, sin f_x + cos f_y) around their first-order model at the region center: the remainder h^2 / 2 times the force bound, plus the cross term h times the force radius. The cross term is first order in h (tex Remark 3.2)."""
    c, s = abs(math.cos(th)), abs(math.sin(th))
    rem = 0.5 * h * h * (abs(f["fx_c"]) + f["fx_r"] + abs(f["fy_c"]) + f["fy_r"])
    return rem + h * (s * f["fx_r"] + c * f["fy_r"]), rem + h * (c * f["fx_r"] + s * f["fy_r"])


def mccormick(a, alo, ahi, b, blo, bhi, a_err=0.0):
    """McCormick under- and over-estimators of a * b, for linear forms b in [blo, bhi] and a in [alo, ahi], as affine polynomials. a may be a first-order model of the true factor with error at most a_err: the error enters through the bound of b that multiplies it."""
    under = [poly_add(poly_scale(b, alo), poly_scale(a, blo), const(-alo * blo - abs(blo) * a_err)),
             poly_add(poly_scale(b, ahi), poly_scale(a, bhi), const(-ahi * bhi - abs(bhi) * a_err))]
    over = [poly_add(poly_scale(b, alo), poly_scale(a, bhi), const(-alo * bhi + abs(bhi) * a_err)),
            poly_add(poly_scale(b, ahi), poly_scale(a, blo), const(-ahi * blo + abs(blo) * a_err))]
    return under, over


def taylor_translation(md, tail, ch, lo, hi, mode):
    """Rows of p_1 - p_0 = k R(theta_0) f. Returns (bands as (tag, affine, width), plain ge rows as (poly, tag))."""
    f = force_box(md, tail, lo, hi)
    k_tr = md["dt"] * md["c_v"] / md["f_max"] ** 2
    th, h = ch["th0"], ch["h"]
    c, s = math.cos(th), math.sin(th)
    dx, dy = lin({TPX1: 1.0, TPX0: -1.0}), lin({TPY1: 1.0, TPY0: -1.0})
    d0 = lin({TTH0: 1.0}, -th)
    if mode == "band":
        phi_x = poly_add(poly_scale(f["fx"], c), poly_scale(f["fy"], -s), poly_scale(d0, -s * f["fx_c"] - c * f["fy_c"]))
        phi_y = poly_add(poly_scale(f["fx"], s), poly_scale(f["fy"], c), poly_scale(d0, c * f["fx_c"] - s * f["fy_c"]))
        ex, ey = translation_bands(f, th, h)
        return [("trans_x", poly_add(dx, poly_scale(phi_x, -k_tr)), k_tr * ex),
                ("trans_y", poly_add(dy, poly_scale(phi_y, -k_tr)), k_tr * ey)], []
    c_lo, c_hi, s_lo, s_hi = sector_box(md["theta_splits"], tail["theta"])
    cos_lin, sin_lin = poly_add(const(c), poly_scale(d0, -s)), poly_add(const(s), poly_scale(d0, c))
    err = 0.5 * h * h
    fx_lo, fx_hi, fy_lo, fy_hi = f["fx_c"] - f["fx_r"], f["fx_c"] + f["fx_r"], f["fy_c"] - f["fy_r"], f["fy_c"] + f["fy_r"]
    cfx_u, cfx_o = mccormick(cos_lin, c_lo, c_hi, f["fx"], fx_lo, fx_hi, err)
    sfy_u, sfy_o = mccormick(sin_lin, s_lo, s_hi, f["fy"], fy_lo, fy_hi, err)
    sfx_u, sfx_o = mccormick(sin_lin, s_lo, s_hi, f["fx"], fx_lo, fx_hi, err)
    cfy_u, cfy_o = mccormick(cos_lin, c_lo, c_hi, f["fy"], fy_lo, fy_hi, err)
    minus = lambda a, b: poly_add(a, poly_scale(b, -1.0))
    rows = []
    for a, b in itertools.product(cfx_u, sfy_o):       # phi_x = C f_x - S f_y >= under - over
        rows.append((minus(dx, poly_scale(minus(a, b), k_tr)), "trans_x_mc_lo"))
    for a, b in itertools.product(cfx_o, sfy_u):       # phi_x <= over - under
        rows.append((minus(poly_scale(minus(a, b), k_tr), dx), "trans_x_mc_hi"))
    for a, b in itertools.product(sfx_u, cfy_u):       # phi_y = S f_x + C f_y >= under + under
        rows.append((minus(dy, poly_scale(poly_add(a, b), k_tr)), "trans_y_mc_lo"))
    for a, b in itertools.product(sfx_o, cfy_o):       # phi_y <= over + over
        rows.append((minus(poly_scale(poly_add(a, b), k_tr), dy), "trans_y_mc_hi"))
    return [], rows


def taylor_torque(md, tail, lo, hi, mode):
    """Rows of the torque. On face j, with contact position lambda = t_j^T (r_c - c_j) and r_c on the face, tau = |c_j| f_t + lambda f_n: exactly one bilinear term. Returns (bands, plain ge rows) as in taylor_translation."""
    j = tail["face"]
    t, c = md["tangents"][j], md["centers"][j]
    lam = lin({TRX0: t[0], TRY0: t[1]}, -(t @ c))
    ends = [(a * lo[i], a * hi[i]) for a, i in ((t[0], RX0), (t[1], RY0))]
    lam_lo, lam_hi = sum(min(e) for e in ends) - t @ c, sum(max(e) for e in ends) - t @ c
    w = lin({TTAU: 1.0, TFT: -float(np.linalg.norm(c))})           # w = tau - |c_j| f_t = lambda f_n
    if mode == "band":
        lam_c, lam_r = 0.5 * (lam_lo + lam_hi), 0.5 * (lam_hi - lam_lo)
        fn_c, fn_r = 0.5 * (lo[FN] + hi[FN]), 0.5 * (hi[FN] - lo[FN])
        return [("torque", poly_add(w, var(TFN, -lam_c), poly_scale(lam, -fn_c), const(lam_c * fn_c)), lam_r * fn_r)], []
    under, over = mccormick(lam, lam_lo, lam_hi, var(TFN), lo[FN], hi[FN])
    return [], [(poly_add(w, poly_scale(u, -1.0)), "torque_mc_lo") for u in under] + [(poly_add(o, poly_scale(w, -1.0)), "torque_mc_hi") for o in over]


def taylor_rows(cfg, md, task, kind, tail, head, lo, hi):
    """Rows of one Taylor edge type: (eq rows, ge rows, band widths), rows as (poly, tag). It reads the constraint table row by row, so the model cannot drift from it, and a row with no Taylor form raises."""
    mode = cfg["relaxation"]["taylor_mode"]
    if mode not in ("band", "mccormick"):
        raise ValueError(f"unknown relaxation.taylor_mode {mode!r}")
    ch = taylor_charts(md, task, kind, tail, head)
    theta_max = md["theta_step_max"]
    dth = taylor_dtheta(ch)
    eq, ge, bands = [], [], {}

    def add_bands(items):
        for tag, affine, width in items:
            ge.append((poly_add(affine, const(width)), f"{tag}_lo"))
            ge.append((poly_add(poly_scale(affine, -1.0), const(width)), f"{tag}_hi"))
            bands[tag] = float(width)

    for sense, p, tag in edge_constraints(md, task, kind, tail, head):
        if tag in TAYLOR_IMPLIED or tag.startswith("theta_") or tag == "trans_y":
            continue                                               # trans_y comes with trans_x
        if tag == "rot_rel":                                       # |dtheta - dt omega| <= max |asin u - u|
            k_rot = md["dt"] * md["c_v"] / md["tau_max"] ** 2
            add_bands([(tag, poly_add(dth, var(TTAU, -k_rot)), theta_max - math.sin(theta_max))])
        elif tag == "rot_step":
            ge.append((poly_add(const(theta_max), dth), "rot_step_lo"))
            ge.append((poly_add(const(theta_max), poly_scale(dth, -1.0)), "rot_step_hi"))
        elif tag == "static_c":
            eq.append((lin({TTH1: 1.0, TTH0: -1.0}), tag))
        elif tag == "start_c0":
            eq.append((lin({TTH0: 1.0}, -float(chart_near(task["start"][2], ch["th0"]))), tag))
        elif tag == "goal_c1":
            eq.append((lin({TTH1: 1.0}, -ch["th1"]), tag))
        elif tag == "trans_x":
            items, plain = taylor_translation(md, tail, ch, lo, hi, mode)
            add_bands(items)
            ge += plain
        elif tag == "torque":
            items, plain = taylor_torque(md, tail, lo, hi, mode)
            add_bands(items)
            ge += plain
        else:
            mapped = {}
            for k, c in p.items():
                if len(k) > 1 or (k and k[0] not in TAYLOR_OF_SDP):
                    raise ValueError(f"table row {tag!r} has no Taylor form")
                mapped[(TAYLOR_OF_SDP[k[0]],) if k else ()] = c
            (eq if sense == "eq" else ge).append((mapped, tag))
    return eq, ge, bands


def taylor_cost(cfg, md, tail, lo, hi, dth):
    """Epigraph rows of run B's cost (plan 5.2.4): tangent planes of each quadratic term, or SOC entries with cost_model soc. Returns (ge rows, soc entries, upper bound per term, weight per term)."""
    w, rx = cfg["costs"], cfg["relaxation"]
    if rx["cost_model"] not in ("tangent_planes", "soc"):
        raise ValueError(f"unknown relaxation.cost_model {rx['cost_model']!r}")
    use_soc, planes = rx["cost_model"] == "soc", int(rx["cost_planes"])
    dp_max, dth_max = 0.0, 0.0
    if tail["contact"]:                                            # |dp_x|, |dp_y| <= k ||f||, since R is a rotation
        f = force_box(md, tail, lo, hi)
        dp_max = md["dt"] * md["c_v"] / md["f_max"] ** 2 * math.hypot(abs(f["fx_c"]) + f["fx_r"], abs(f["fy_c"]) + f["fy_r"])
        dth_max = md["theta_step_max"]
    vm = md["v_max"]
    squares = [("dpx", lin({TPX1: 1.0, TPX0: -1.0}), -dp_max, dp_max, w["w_p"]), ("dpy", lin({TPY1: 1.0, TPY0: -1.0}), -dp_max, dp_max, w["w_p"]),
               ("drx", lin({TRX1: 1.0, TRX0: -1.0}), -vm, vm, w["w_r"]), ("dry", lin({TRY1: 1.0, TRY0: -1.0}), -vm, vm, w["w_r"]),
               ("fn", var(TFN), lo[FN], hi[FN], w["w_f"]), ("ft", var(TFT), lo[FT], hi[FT], w["w_f"])]
    ge, soc, ub, weight = [], [], {}, {}
    for name, x, a, b, wt in squares:
        ub[name], weight[name] = max(a * a, b * b), wt
        if a == 0.0 and b == 0.0:
            continue                                               # pinned at zero by the box
        if use_soc:
            soc.append(dict(z=TZ[name], form=x, scale=1.0))
        else:                                                      # z >= 2 g x - g^2
            ge += [(poly_add(var(TZ[name]), poly_scale(x, -2.0 * g), const(g * g)), f"plane_{name}") for g in np.linspace(a, b, planes)]
    # 2 (1 - cos dtheta) is convex on |dtheta| <= pi / 2, which covers the step bound
    ub["dth"], weight["dth"] = 2.0 * (1.0 - math.cos(dth_max)), w["w_theta"]
    if dth_max > 0.0:
        if use_soc:                                                # (1 - cos x) / x^2 falls with |x|, so this underestimates
            soc.append(dict(z=TZ["dth"], form=dth, scale=ub["dth"] / dth_max ** 2))
        else:                                                      # z >= f(g) + f'(g) (dtheta - g)
            ge += [(poly_add(var(TZ["dth"]), poly_scale(dth, -2.0 * math.sin(g)), const(-2.0 * (1.0 - math.cos(g)) + 2.0 * g * math.sin(g))), "plane_dth")
                   for g in np.linspace(-dth_max, dth_max, planes)]
    return ge, soc, ub, weight


def taylor_matrix(polys):
    """Sparse rows over m = (y, v): variable i is column i + 1, and a row's constant multiplies y (column 0)."""
    if not polys:
        return sp.csr_matrix((0, NT + 1))
    rows, cols, vals = [], [], []
    for r, p in enumerate(polys):
        for k, c in p.items():
            rows.append(r)
            cols.append(k[0] + 1 if k else 0)
            vals.append(float(c))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(polys), NT + 1))


def relax_taylor(cfg, md, task, kind, tail, head, lo, hi):
    """Polyhedral outer model of one edge type (plan 5.2), in the record described at the top of this section."""
    eq, ge, bands = taylor_rows(cfg, md, task, kind, tail, head, lo, hi)
    ch = taylor_charts(md, task, kind, tail, head)
    planes, soc, ub, weight = taylor_cost(cfg, md, tail, lo, hi, taylor_dtheta(ch))
    ge = ge + planes
    raw = taylor_matrix([p for p, _ in eq])
    keep = independent_rows(raw)
    n = NT + 1
    mlo, mhi = np.ones(n), np.ones(n)
    for s, t in TAYLOR_OF_SDP.items():
        mlo[t + 1], mhi[t + 1] = lo[s], hi[s]
    pad = float(cfg["pruning"]["box_margin"])
    mlo[TTH0 + 1], mhi[TTH0 + 1] = ch["lo0"] - pad, ch["hi0"] + pad
    mlo[TTH1 + 1], mhi[TTH1 + 1] = ch["lo1"] - pad, ch["hi1"] + pad
    c = np.zeros(n)
    c[0] = cfg["costs"]["w_switch"] if mode_switch(tail, head) else 0.0
    for name, z in TZ.items():
        mlo[z + 1], mhi[z + 1] = 0.0, ub[name]
        c[z + 1] = weight[name]
    coupling = lambda ids: sp.csr_matrix((np.ones(len(ids) + 1), (np.arange(len(ids) + 1), [0] + [i + 1 for i in ids])), shape=(len(ids) + 1, n))
    return dict(kind="taylor", n=n, Aeq=raw[keep], eq_owner=keep, eq_tags=[t for _, t in eq], eq_rows_raw=raw.shape[0],
                G=taylor_matrix([p for p, _ in ge]), ge_tags=[t for _, t in ge], ge_polys=[p for p, _ in ge], lo=mlo, hi=mhi, c=c, psd=[],
                soc=[dict(s, form=taylor_matrix([s["form"]])) for s in soc], bands=bands,
                tail_map=coupling((TPX0, TPY0, TTH0, TRX0, TRY0)), head_map=coupling((TPX1, TPY1, TTH1, TRX1, TRY1)))


def taylor_points(md, task, kind, tail, head, V):
    """Edge-layout points V as Taylor vectors at y = 1: the angle in the edge's charts and each epigraph variable at its true value."""
    ch = taylor_charts(md, task, kind, tail, head)
    M = np.zeros((V.shape[0], NT + 1))
    M[:, 0] = 1.0
    for s, t in TAYLOR_OF_SDP.items():
        M[:, t + 1] = V[:, s]
    M[:, TTH0 + 1] = chart_near(np.arctan2(V[:, S0], V[:, C0]), ch["th0"])
    M[:, TTH1 + 1] = chart_near(np.arctan2(V[:, S1], V[:, C1]), ch["th1"])
    true = dict(dpx=(V[:, PX1] - V[:, PX0]) ** 2, dpy=(V[:, PY1] - V[:, PY0]) ** 2,
                dth=2.0 * (1.0 - V[:, C0] * V[:, C1] - V[:, S0] * V[:, S1]),
                drx=(V[:, RX1] - V[:, RX0]) ** 2, dry=(V[:, RY1] - V[:, RY0]) ** 2, fn=V[:, FN] ** 2, ft=V[:, FT] ** 2)
    for name, z in TZ.items():
        M[:, z + 1] = true[name]
    return M


def worst_bands(md, theta_splits, lambda_splits):
    """Largest Taylor bands at given split counts, over every face and theta region on the full force box: (translation in m, torque in N m, rotation in rad). The rotation band does not depend on the regions."""
    k_tr = md["dt"] * md["c_v"] / md["f_max"] ** 2
    h = math.pi / theta_splits
    e_tr, e_tau = 0.0, 0.0
    for j in range(4):
        tail = dict(contact=True, face=j)
        lo, hi = edge_box(md, tail)
        f = force_box(md, tail, lo, hi)
        for i in range(theta_splits):
            e_tr = max(e_tr, k_tr * max(translation_bands(f, sum(theta_interval(theta_splits, i)) * 0.5, h)))
        e_tau = max(e_tau, md["half_len"][j] / lambda_splits * 0.5 * (hi[FN] - lo[FN]))
    return e_tr, float(e_tau), md["theta_step_max"] - math.sin(md["theta_step_max"])


def size_regions(md, target):
    """Smallest split counts whose bands meet the targets (plan 5.3, tex Cor 3.3), computed offline: (theta_splits, lambda_splits), with None where max_splits is not enough."""
    limit = int(target["max_splits"])
    theta = next((s for s in range(3, limit + 1) if worst_bands(md, s, 1)[0] <= float(target["trans"])), None)
    lam = next((s for s in range(1, limit + 1) if worst_bands(md, 3, s)[1] <= float(target["torque"])), None)
    return theta, lam


def band_widths(cfg, md, modes, task, tr):
    """Every band of the instance's alive edge types, per tag: {tag: list of widths}."""
    widths = {}
    for ti in np.flatnonzero(tr["type_alive"]):
        tail, head = type_modes(tr, modes, ti)
        bands = taylor_rows(cfg, md, task, EDGE_KINDS[int(tr["type_kind"][ti])], tail, head, tr["type_lo"][ti], tr["type_hi"][ti])[2]
        for tag, width in bands.items():
            widths.setdefault(tag, []).append(width)
    return widths


def band_report(cfg, md, modes, task, tr):
    """Prints every band width of the run's edge types and the region sizing table (plan 5.2.3, 5.3). Halving h halves the translation band: the bilinear cross term is first order in the region size (tex Remark 3.2)."""
    print(f"\nTaylor bands at theta_splits {md['theta_splits']}, lambda_splits {md['lambda_splits']} (width over the alive edge types)")
    for tag, w in sorted(band_widths(cfg, md, modes, task, tr).items()):
        print(f"  {tag:10s} min {min(w):.3e}  max {max(w):.3e}  ({len(w)} edge types)")
    print("region sizing, worst case over faces and regions on the full force box")
    prev = None
    for s in (4, 8, 16, 32, 64, 128, 256):
        e = worst_bands(md, s, 1)[0]
        print(f"  theta_splits {s:4d}: translation band {e:.3e} m" + ("" if prev is None else f"   (x{e / prev:.2f} of the previous row)"))
        prev = e
    for s in (1, 2, 4, 8, 16):
        print(f"  lambda_splits {s:3d}: torque band {worst_bands(md, 3, s)[1]:.3e} N m")
    print(f"  rotation band {worst_bands(md, 3, 1)[2]:.3e} rad at any split count")
    target = cfg["relaxation"]["size_regions"]
    print(f"  smallest splits for trans <= {target['trans']} m and torque <= {target['torque']} N m: {size_regions(md, target)}", flush=True)


def relax_edge_type(cfg, md, task, modes, tr, ti):
    """The outer model of edge type ti under cfg's relaxation."""
    tail, head = type_modes(tr, modes, ti)
    builder = {"sdp": relax_sdp, "taylor": relax_taylor}[cfg["relaxation"]["type"]]
    return builder(cfg, md, task, EDGE_KINDS[int(tr["type_kind"][ti])], tail, head, tr["type_lo"][ti], tr["type_hi"][ti])


def lift_points(rel, md, task, kind, tail, head, V):
    """Lifted vectors m (n, rel['n']) at y = 1 of edge-layout points V (n, NV)."""
    if rel["kind"] == "sdp":
        return rank_one_moments(moment_basis(rel["kappa"], len(rel["free"])), V[:, rel["free"]])
    return taylor_points(md, task, kind, tail, head, V)


def full_points(rel, W):
    """Edge-layout points (n, NV) of an SDP record's free-variable points W (n, nv), through its substitution."""
    return np.column_stack([poly_eval_batch(rel["sub"][i], W) for i in range(NV)])


def affine_error(rel, V):
    """How far points V are from an SDP record's affine set: the largest violation of the linear equalities it eliminated (0 for a record without a substitution)."""
    if "sub" not in rel:
        return 0.0
    return float(np.max(np.abs(full_points(rel, V[:, rel["free"]]) - V)))


def prebuild_bases(cfg, md, modes, task, tr):
    """Builds an SDP instance's moment bases once, before any fork."""
    if cfg["relaxation"]["type"] != "sdp":
        return
    kappa = int(cfg["relaxation"]["order"])
    counts = {NV}
    if cfg["relaxation"]["eliminate_linear"]:
        counts = {len(linear_elimination(type_rows(md, task, modes, tr, ti))[0]) for ti in np.flatnonzero(tr["type_alive"])}
    for nv in counts:
        moment_basis(kappa, nv)


# ===== TRUE MODEL =====

def true_contact_step(md, x, face, fn, ft, slide):
    """Next knot of one contact step on `face` under the true quasi-static model; slide is the pusher's move along t_j (body frame). Returns (x_next, tau)."""
    n, t = md["normals"][face], md["tangents"][face]
    f = -fn * n + ft * t
    rc = x[4:6] - md["rp"] * n
    tau = rc[0] * f[1] - rc[1] * f[0]
    sin_step = md["dt"] * md["c_v"] * tau / md["tau_max"] ** 2
    if abs(sin_step) > math.sin(md["theta_step_max"]):
        raise ValueError(f"rotation step asin({sin_step:.3f}) exceeds the step bound")
    c0, s0 = x[2], x[3]
    th1 = math.atan2(s0, c0) + math.asin(sin_step)
    v = md["c_v"] * f / md["f_max"] ** 2
    dp = md["dt"] * np.array([c0 * v[0] - s0 * v[1], s0 * v[0] + c0 * v[1]])
    r1 = x[4:6] + slide * t
    return np.array([x[0] + dp[0], x[1] + dp[1], math.cos(th1), math.sin(th1), r1[0], r1[1]]), tau


def edge_vector(x0, x1, fn=0.0, ft=0.0, tau=0.0):
    return np.concatenate([x0, x1, [fn, ft, tau]])


def violations(rows, v, lo=None, hi=None):
    """Largest violation per tag of a constraint table (and optionally a box) at point v; 0 means satisfied."""
    out = {}
    for kind, p, tag in rows:
        val = poly_eval(p, v)
        out[tag] = max(out.get(tag, 0.0), abs(val) if kind == "eq" else max(0.0, -val))
    if lo is not None:
        out["box"] = float(max(np.max(lo - v), np.max(v - hi), 0.0))
    return out


def poly_eval_batch(p, V):
    """Values of a polynomial at points V (n, nv)."""
    out = np.zeros(V.shape[0])
    for k, c in p.items():
        term = c
        for i in k:
            term = term * V[:, i]
        out = out + term
    return out


def biased_uniform(rng, n, a, b, p_end=0.2):
    """Uniform draws on [a, b], with a share exactly at the ends so that samples also test rows at their boundary."""
    x = rng.uniform(a, b, n)
    ends = rng.random(n) < p_end
    x[ends] = np.where(rng.random(int(ends.sum())) < 0.5, a, b)
    return x


def contact_wrench_batch(md, face, r0, fn, ft):
    """Body-frame force on the slider and its torque about the centroid, for pusher centers r0 (n, 2) on `face`: (f_x, f_y, tau)."""
    n, t = md["normals"][face], md["tangents"][face]
    fx, fy = -fn * n[0] + ft * t[0], -fn * n[1] + ft * t[1]
    rcx, rcy = r0[:, 0] - md["rp"] * n[0], r0[:, 1] - md["rp"] * n[1]
    return fx, fy, rcx * fy - rcy * fx


def sample_contact_forces(md, tail, r0, rng):
    """Forces of true contact steps at pusher centers r0, scaled down where the rotation step would pass its bound (half of those land exactly on it). Returns (f_n, f_t, tau, f_x, f_y, dtheta)."""
    n = r0.shape[0]
    mu, limit = md["mu_c"], math.sin(md["theta_step_max"])
    k_rot = md["dt"] * md["c_v"] / md["tau_max"] ** 2
    fn = biased_uniform(rng, n, 0.0, md["fn_max"])
    if tail["ctype"] == "stick":
        ft = mu * fn * biased_uniform(rng, n, -1.0, 1.0)
    else:
        ft = mu * fn * (1.0 if tail["ctype"] == "slide_pos" else -1.0)
    fx, fy, tau = contact_wrench_batch(md, tail["face"], r0, fn, ft)
    u = np.abs(k_rot * tau)
    over = u > limit
    scale = np.ones(n)
    scale[over] = limit / u[over] * np.where(rng.random(int(over.sum())) < 0.5, 1.0, rng.random(int(over.sum())))
    fn, ft, tau, fx, fy = fn * scale, ft * scale, tau * scale, fx * scale, fy * scale
    return fn, ft, tau, fx, fy, np.arcsin(np.clip(k_rot * tau, -limit, limit))


PUSHER_VARS = (RX0, RY0, RX1, RY1)


def pusher_polytope_vertices(rows, lo, hi, tol=1e-10):
    """All vertices of the polytope of (r_k, r_{k+1}) cut out by the table's linear pusher rows and the box, or None if it is empty. Enumerated directly: a 4-D polytope with a few dozen rows has few candidate corners."""
    pos = {v: i for i, v in enumerate(PUSHER_VARS)}
    a_eq, b_eq, a_ub, b_ub = [], [], [], []
    for sense, p, _ in rows:
        terms = [k for k in p if k]
        if poly_degree(p) != 1 or any(k[0] not in pos for k in terms):
            continue
        a = np.zeros(4)
        for k in terms:
            a[pos[k[0]]] = p[k]
        if sense == "eq":
            a_eq.append(a)
            b_eq.append(-p.get((), 0.0))
        else:                                   # a x + c0 >= 0  <=>  -a x <= c0
            a_ub.append(-a)
            b_ub.append(p.get((), 0.0))
    for i, v in enumerate(PUSHER_VARS):
        e = np.eye(4)[i]
        a_ub += [e, -e]
        b_ub += [hi[v], -lo[v]]
    a_ub, b_ub = np.array(a_ub), np.array(b_ub)
    x0, z = np.zeros(4), np.eye(4)              # x = x0 + z y satisfies the equalities
    if a_eq:
        a_eq, b_eq = np.array(a_eq), np.array(b_eq)
        x0 = np.linalg.lstsq(a_eq, b_eq, rcond=None)[0]
        if np.max(np.abs(a_eq @ x0 - b_eq)) > 1e-9:
            return None
        _, s, vt = np.linalg.svd(a_eq)
        z = vt[int(np.sum(s > 1e-10)):].T
    if z.shape[1] == 0:
        return x0[None] if np.all(a_ub @ x0 <= b_ub + tol) else None
    ay, by = a_ub @ z, b_ub - a_ub @ x0
    moving = np.linalg.norm(ay, axis=1) > 1e-12
    if np.any(by[~moving] < -tol):
        return None
    subsets = np.array(list(itertools.combinations(np.flatnonzero(moving), z.shape[1])))
    systems = ay[subsets]
    good = np.abs(np.linalg.det(systems)) > 1e-10
    y = np.linalg.solve(systems[good], by[subsets[good]][..., None])[..., 0]
    y = y[np.all(y @ ay.T <= by + tol, axis=1)]
    if not y.shape[0]:
        return None
    x = x0 + y @ z.T
    return x[np.unique(np.round(x, 10), axis=0, return_index=True)[1]]


def sample_true_steps(md, task, kind, tail, head, n, rng, max_rounds=60):
    """True points of one edge type (plan 5.4), as an array (m, NV) with m <= n.

    Pusher paths are mixtures of the vertices of the table's pusher polytope, with a share at the
    vertices themselves. Forces, rotation and translation come from the true model, not from the
    polynomial table, so the table and its relaxations are checked against independent values.
    Draws that miss the edge's theta regions or the workspace are rejected.
    """
    rows = edge_constraints(md, task, kind, tail, head)
    lo, hi = edge_box(md, tail, head)
    verts = pusher_polytope_vertices(rows, lo, hi)
    if verts is None:
        return np.zeros((0, NV))
    splits = md["theta_splits"]
    labelled = tail.get("theta") is not None
    a0, b0 = theta_interval(splits, tail["theta"]) if labelled else (-math.pi, math.pi)
    center = 0.5 * (a0 + b0)
    a1, b1 = -np.inf, np.inf
    if head is not None and head.get("theta") is not None:      # the head's interval, in the tail's chart
        a1, b1 = theta_interval(splits, head["theta"])
        shift = float(chart_near(0.5 * (a1 + b1), center)) - 0.5 * (a1 + b1)
        a1, b1 = a1 + shift, b1 + shift
    k_tr = md["dt"] * md["c_v"] / md["f_max"] ** 2
    px, py = md["p_box"]
    tol = 1e-12
    got, count = [], 0
    for _ in range(max_rounds):
        need = n - count
        if need <= 0:
            break
        w = rng.dirichlet(np.full(verts.shape[0], 0.3), need)
        pure = rng.random(need) < 0.1
        w[pure] = np.eye(verts.shape[0])[rng.integers(verts.shape[0], size=int(pure.sum()))]
        r = w @ verts
        if tail["contact"]:
            fn, ft, tau, fx, fy, dth = sample_contact_forces(md, tail, r[:, :2], rng)
        else:
            fn = ft = tau = fx = fy = dth = np.zeros(need)
        if kind == "start":
            th0 = np.full(need, float(chart_near(task["start"][2], center)))
        elif kind == "goal":
            th0 = float(chart_near(task["goal"][2], center)) - dth
        else:
            low, up = np.maximum(a0, a1 - dth), np.minimum(b0, b1 - dth)
            th0 = low + (up - low) * rng.random(need)
        th1 = th0 + dth
        ok = (th1 >= a1 - tol) & (th1 <= b1 + tol)
        if labelled:
            ok &= (th0 >= a0 - tol) & (th0 <= b0 + tol)
        c0, s0, c1, s1 = np.cos(th0), np.sin(th0), np.cos(th1), np.sin(th1)
        dpx, dpy = k_tr * (c0 * fx - s0 * fy), k_tr * (s0 * fx + c0 * fy)
        if kind == "start":
            p0x, p0y = np.full(need, task["start"][0]), np.full(need, task["start"][1])
            p1x, p1y = p0x + dpx, p0y + dpy
        elif kind == "goal":
            p1x, p1y = np.full(need, task["goal"][0]), np.full(need, task["goal"][1])
            p0x, p0y = p1x - dpx, p1y - dpy
        else:
            p0x, p0y = biased_uniform(rng, need, px[0], px[1]), biased_uniform(rng, need, py[0], py[1])
            p1x, p1y = p0x + dpx, p0y + dpy
        ok &= (p0x >= px[0]) & (p0x <= px[1]) & (p0y >= py[0]) & (p0y <= py[1])
        ok &= (p1x >= px[0]) & (p1x <= px[1]) & (p1y >= py[0]) & (p1y <= py[1])
        got.append(np.column_stack([p0x, p0y, c0, s0, r[:, 0], r[:, 1], p1x, p1y, c1, s1, r[:, 2], r[:, 3], fn, ft, tau])[ok])
        count += got[-1].shape[0]
    return np.vstack(got)[:n] if got else np.zeros((0, NV))


def hand_push(cfg, md, task, modes):
    """Scripted start->goal push for a straight push along +x (task T1): approach face 2 at its center inside the start strip, then stick-push with a constant normal force.

    Returns the steps as (kind, tail mode, head mode, v). Every knot comes from the true model,
    so the path is a true trajectory whenever its rows check out.
    """
    N = int(cfg["trellis"]["N"])
    s0, g = task["start"], task["goal"]
    if abs(g[1] - s0[1]) > 1e-12 or abs(g[2]) > 1e-12 or abs(s0[2]) > 1e-12 or g[0] <= s0[0]:
        raise ValueError("the hand push is scripted only for straight pushes along +x at zero rotation (T1)")
    if task["start_region"] != "strip_2":
        raise ValueError("the hand push starts in strip_2")
    # with regions: the copies holding zero rotation and the face's center
    label = theta_label(s0[2], md["theta_splits"]) if md["theta_splits"] > 1 else None
    cell = md["lambda_splits"] // 2 if md["lambda_splits"] > 1 else None
    pick = lambda base, lam: next(m for m in modes if m["base"] == base and m["theta"] == label and m["lam"] == lam)
    strip, stick, j = pick("strip_2", None), pick("c2_stick", cell), 2
    contact_r = md["centers"][j] + md["rp"] * md["normals"][j]
    r0 = task["pusher_start"]
    n_app = max(1, math.ceil(float(np.max(np.abs(contact_r - r0))) / md["v_max"] - 1e-12))
    n_push = N - n_app
    if n_push < 1:
        raise ValueError("horizon too short for the scripted push")
    fn = (g[0] - s0[0]) / n_push * md["f_max"] ** 2 / (md["dt"] * md["c_v"])
    steps, x = [], knot_from_pose(s0, r0)
    for k in range(n_app):
        x1 = x.copy()
        x1[4:6] = r0 + (contact_r - r0) * (k + 1) / n_app
        steps.append(("start" if k == 0 else "mid", strip, stick if k == n_app - 1 else strip, edge_vector(x, x1)))
        x = x1
    for k in range(n_app, N):
        x1, tau = true_contact_step(md, x, j, fn, 0.0, 0.0)
        steps.append(("goal" if k == N - 1 else "mid", stick, None if k == N - 1 else stick, edge_vector(x, x1, fn, 0.0, tau)))
        x = x1
    return steps


# ===== VALIDITY =====
# Plan 5.4. Every true point must satisfy every row, bound and PSD block of its edge's outer model.
# A violation is a bug that would make the certified bound invalid, so a build stops on one.

def relaxation_violations(rel, M):
    """Largest violation per row class at lifted points M (n, len m), as {class: (value, where)}; 0 means satisfied."""
    out = {}
    if rel["Aeq"].shape[0]:
        r = np.max(np.abs(rel["Aeq"] @ M.T), axis=1)
        i = int(np.argmax(r))
        out["eq"] = (float(r[i]), rel["eq_tags"][int(rel["eq_owner"][i])])
    if rel["G"].shape[0]:
        r = np.max(-(rel["G"] @ M.T), axis=1)
        i = int(np.argmax(r))
        out["ge"] = (max(float(r[i]), 0.0), rel["ge_tags"][i])
    y = M[:, :1]
    r = np.maximum(np.max(rel["lo"] * y - M, axis=0), np.max(M - rel["hi"] * y, axis=0))
    i = int(np.argmax(r))
    out["box"] = (max(float(r[i]), 0.0), f"m[{i}]")
    for s in rel["soc"]:
        form = (s["form"] @ M.T).ravel()
        value = float(np.max(s["scale"] * form ** 2 - M[:, s["z"] + 1] * M[:, 0]))
        if value > out.get("soc", (0.0, ""))[0]:
            out["soc"] = (value, f"z[{s['z']}]")
    return out


def block_operator(block, n):
    """(sparse operator from moment vectors to the block's upper-triangle entries, entry rows, entry columns)."""
    s = int(block["size"])
    uniq, inv = np.unique(block["row"].astype(np.int64) * s + block["col"], return_inverse=True)
    rows, cols = np.divmod(uniq, s)
    return sp.csr_matrix((block["coef"], (inv.ravel(), block["mom"])), shape=(uniq.size, n)), rows, cols


def block_identity_error(block, op, M, V):
    """Largest entry error, per point, of the rank-one identity B(m(v)) = g(v) w w^T, with w the block's monomials. Also returns the entries (n, entries) and g(v)."""
    mat, rows, cols = op
    vals = (mat @ M.T).T
    g = poly_eval_batch(block["poly"], V)
    return np.max(np.abs(vals - g[:, None] * M[:, rows] * M[:, cols]), axis=1), vals, g


def psd_violation(block, op, M, V, eig_count):
    """How far one PSD block's smallest eigenvalue can fall below zero at true points: (bound on every point by Weyl's inequality, from the rank-one identity and g(v); deficit computed directly on the first eig_count points)."""
    s = int(block["size"])
    err, vals, g = block_identity_error(block, op, M, V)
    bound = s * err + np.maximum(-g, 0.0) * np.sum(M[:, :s] ** 2, axis=1)
    deficit = 0.0
    if eig_count:
        _, rows, cols = op
        B = np.zeros((eig_count, s, s))
        B[:, rows, cols] = vals[:eig_count]
        B[:, cols, rows] = vals[:eig_count]
        deficit = max(float(-np.min(np.linalg.eigvalsh(B)[:, 0])), 0.0)
    return float(np.max(bound)), deficit


def coupling_error(rel, M, V):
    """Largest difference between the coupled coordinates read from lifted points and the same coordinates computed from the point itself (tex 2.1, consistency)."""
    worst = 0.0
    for mapping, knot in ((rel["tail_map"], KNOT0), (rel["head_map"], KNOT1)):
        got, x = (mapping @ M.T).T, V[:, list(knot)]
        if rel["kind"] == "sdp":
            want = [np.ones(x.shape[0])] + [x[:, a] for a in range(6)]
            if got.shape[1] > 7:
                want += [x[:, a] * x[:, b] for a in range(6) for b in range(a, 6)]
            worst = max(worst, float(np.max(np.abs(got - np.column_stack(want)))))
        else:                                   # flow, p, theta, r: theta through (c, s), since V has no chart
            want = np.column_stack([np.ones(x.shape[0]), x[:, 0], x[:, 1], x[:, 4], x[:, 5]])
            worst = max(worst, float(np.max(np.abs(got[:, [0, 1, 2, 4, 5]] - want))),
                        float(np.max(np.abs(np.cos(got[:, 3]) - x[:, 2]))), float(np.max(np.abs(np.sin(got[:, 3]) - x[:, 3]))))
    return worst


def validate_type(cfg, md, task, modes, tr, ti, samples):
    """Plan 5.4 for one edge type: {type, name, samples, worst: {class: (value, where)}}. Blocks up to 32 x 32 get a direct eigenvalue on every point, larger ones on a few (relaxation.validity)."""
    vc = cfg["relaxation"]["validity"]
    tail, head = type_modes(tr, modes, ti)
    kind = EDGE_KINDS[int(tr["type_kind"][ti])]
    V = sample_true_steps(md, task, kind, tail, head, samples, np.random.default_rng([int(vc["seed"]), int(ti)]))
    out = dict(type=int(ti), name=f"{kind} {tail['name']} -> {head['name'] if head else 'goal'}", samples=int(V.shape[0]), worst={})
    if not V.shape[0]:
        return out

    def note(key, value, where=""):
        if key not in out["worst"] or value > out["worst"][key][0]:
            out["worst"][key] = (float(value), where)

    for sense, p, tag in edge_constraints(md, task, kind, tail, head):
        val = poly_eval_batch(p, V)
        note("table", np.max(np.abs(val)) if sense == "eq" else max(float(np.max(-val)), 0.0), tag)
    rel = relax_edge_type(cfg, md, task, modes, tr, ti)
    cost = edge_cost(cfg, tail, head)
    ops = [block_operator(b, rel["n"]) for b in rel["psd"]]
    chunk = 1000 if rel["n"] < 10000 else 50
    for start in range(0, V.shape[0], chunk):
        v = V[start:start + chunk]
        M = lift_points(rel, md, task, kind, tail, head, v)
        for key, (value, where) in relaxation_violations(rel, M).items():
            note(key, value, where)
        note("affine", affine_error(rel, v))
        note("cost", np.max(np.abs(M @ rel["c"] - poly_eval_batch(cost, v))))
        note("coupling", coupling_error(rel, M, v))
        w = v[:, rel["free"]] if rel["kind"] == "sdp" else v
        for block, op in zip(rel["psd"], ops):
            size = int(block["size"])
            few = int(vc["eig_samples"]) if size <= 200 else int(vc["eig_samples_large"])
            count = v.shape[0] if size <= 32 else (min(few, v.shape[0]) if start == 0 else 0)
            bound, deficit = psd_violation(block, op, M, w, count)
            note("psd_bound", bound, block["tag"])
            note("psd_eig", deficit, block["tag"])
    return out


def validity_samples(cfg):
    """True samples per edge type for this relaxation."""
    rx, vc = cfg["relaxation"], cfg["relaxation"]["validity"]
    return int(vc["samples_k3"]) if rx["type"] == "sdp" and int(rx["order"]) >= 3 else int(vc["samples"])


def validate_relaxation(cfg, md, modes, task, tr, kinds=None):
    """Runs validate_type on the alive edge types of one instance (optionally only some kinds), in forked workers."""
    prebuild_bases(cfg, md, modes, task, tr)
    use = tr["type_alive"].copy()
    if kinds is not None:
        use &= np.isin(tr["type_kind"], [EDGE_KINDS.index(k) for k in kinds])
    samples = validity_samples(cfg)
    return pool_map(lambda ti: validate_type(cfg, md, task, modes, tr, int(ti), samples), np.flatnonzero(use), cfg["build"])


def validity_summary(results, vc):
    """(ok, one-line message, worst per class) over results of validate_type."""
    worst = {}
    for r in results:
        for key, (value, where) in r["worst"].items():
            if key not in worst or value > worst[key][0]:
                worst[key] = (value, f"{r['name']}: {where}".rstrip(": "))
    limit = lambda key: float(vc["tol_psd"]) if key.startswith("psd") else float(vc["tol_rows"])
    bad = {k: v for k, v in worst.items() if v[0] > limit(k)}
    counts = [r["samples"] for r in results]
    empty = [r["name"] for r in results if r["samples"] == 0]
    sampled = [c for c in counts if c > 0]
    msg = (f"{len(results)} edge types, {sum(counts)} true samples (fewest on a sampled type: {min(sampled) if sampled else 0}); worst "
           + ", ".join(f"{k} {v[0]:.1e}" for k, v in sorted(worst.items())))
    if empty:
        msg += f"; NO TRUE POINT FOUND on {len(empty)} types, e.g. {empty[:6]}"
    if bad:
        msg += "; VIOLATIONS: " + "; ".join(f"{k} {v[0]:.2e} at {v[1]}" for k, v in sorted(bad.items()))
    return not bad and bool(results), msg, worst


# ===== ASSEMBLY =====
# Vertices: start s (layer 0), then (k, mode) for k = 1..N-1, then goal t (layer N). Edge k joins
# layer k to layer k+1, and edges are listed layer by layer, i.e. in topological order. Flow
# convention (tex Def 2.2): sum over edges into v minus sum over edges out of v = beta_v, with
# beta_s = -1, beta_t = +1: the start supplies one unit and the goal absorbs it. In the coupling
# operator K an edge enters its head with sign + and its tail with sign -. The Python loops here
# run once at build time; the solver itself never loops over edges.

def build_trellis(cfg, md, task, modes):
    """Vertices, edges and edge types of plan 6.1, before pruning."""
    N, M = int(cfg["trellis"]["N"]), len(modes)
    if N < 2:
        raise ValueError("N >= 2: the trellis needs a start step and a goal step")
    sm = start_mode_index(md, modes, task)
    nv = 2 + (N - 1) * M
    s, t = 0, nv - 1
    vid = lambda k, m: 1 + (k - 1) * M + m
    vert_layer, vert_mode = np.zeros(nv, np.int64), np.full(nv, -1, np.int64)
    vert_layer[t] = N
    for k in range(1, N):
        vert_layer[vid(k, 0):vid(k, 0) + M] = k
        vert_mode[vid(k, 0):vid(k, 0) + M] = np.arange(M)
    allow = np.array([[step_allowed(md, a, b) for b in modes] for a in modes])
    types, tails, heads, etype, elayer = {}, [], [], [], []

    def add(k, tv, hv, kind, tm, hm):
        tails.append(tv)
        heads.append(hv)
        elayer.append(k)
        etype.append(types.setdefault((kind, tm, hm), len(types)))

    for m2 in np.flatnonzero(allow[sm]):
        add(0, s, vid(1, int(m2)), "start", sm, int(m2))
    for k in range(1, N - 1):
        for m in range(M):
            for m2 in np.flatnonzero(allow[m]):
                add(k, vid(k, m), vid(k + 1, int(m2)), "mid", m, int(m2))
    for m in range(M):
        if goal_allowed(md, modes[m], task):
            add(N - 1, vid(N - 1, m), t, "goal", m, -1)
    keys = sorted(types, key=types.get)
    return dict(N=N, M=M, nv=nv, s=s, t=t, start_mode=sm, allow=allow,
                vert_layer=vert_layer, vert_mode=vert_mode,
                edge_tail=np.array(tails, np.int64), edge_head=np.array(heads, np.int64),
                edge_type=np.array(etype, np.int64), edge_layer=np.array(elayer, np.int64),
                type_kind=np.array([EDGE_KINDS.index(k[0]) for k in keys], np.int64),
                type_tail=np.array([k[1] for k in keys], np.int64),
                type_head=np.array([k[2] for k in keys], np.int64))


def type_modes(tr, modes, ti):
    """(tail mode, head mode or None) of edge type ti."""
    h = int(tr["type_head"][ti])
    return modes[int(tr["type_tail"][ti])], (modes[h] if h >= 0 else None)


def type_rows(md, task, modes, tr, ti):
    tail, head = type_modes(tr, modes, ti)
    return edge_constraints(md, task, EDGE_KINDS[int(tr["type_kind"][ti])], tail, head)


def linear_rows(rows):
    """The degree-1 rows of a constraint table in scipy form: A_eq v = b_eq and A_ub v <= b_ub."""
    a_eq, b_eq, a_ub, b_ub = [], [], [], []
    for kind, p, _ in rows:
        if poly_degree(p) > 1:
            continue
        a, c0 = np.zeros(NV), p.get((), 0.0)
        for k, c in p.items():
            if k:
                a[k[0]] += c
        if kind == "eq":
            a_eq.append(a)
            b_eq.append(-c0)
        else:                                   # a v + c0 >= 0  <=>  -a v <= c0
            a_ub.append(-a)
            b_ub.append(c0)
    return (np.array(a_eq).reshape(-1, NV), np.array(b_eq), np.array(a_ub).reshape(-1, NV), np.array(b_ub))


LP_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}


def lp_kwargs(rows, lo, hi):
    a_eq, b_eq, a_ub, b_ub = linear_rows(rows)
    return dict(A_ub=a_ub if len(b_ub) else None, b_ub=b_ub if len(b_ub) else None,
                A_eq=a_eq if len(b_eq) else None, b_eq=b_eq if len(b_eq) else None,
                bounds=list(zip(lo, hi)), method="highs", options=LP_OPTIONS)


def tighten_box(rows, lo, hi, margin):
    """Smallest box implied by the linear rows plus the variable box (two LPs per free variable), padded outward by `margin` relative to absorb LP tolerances.

    Returns (None, 'infeasible') when the linear rows admit no point. That empties the true set
    and every outer model that keeps these rows, so the edge type can be pruned.
    """
    kw = lp_kwargs(rows, lo, hi)
    res = linprog(np.zeros(NV), **kw)
    if res.status == 2:
        return None, "infeasible"
    if res.status != 0:
        return (lo.copy(), hi.copy()), f"lp_status_{res.status}"
    new_lo, new_hi = lo.copy(), hi.copy()
    for i in range(NV):
        if lo[i] == hi[i]:
            continue
        for sign in (1.0, -1.0):
            c = np.zeros(NV)
            c[i] = sign
            r = linprog(c, **kw)
            if r.status != 0:
                continue
            ext = sign * r.fun
            pad = margin * (1.0 + abs(ext))
            if sign > 0:
                new_lo[i] = max(lo[i], ext - pad)
            else:
                new_hi[i] = min(hi[i], ext + pad)
    crossed = new_lo > new_hi
    new_lo[crossed], new_hi[crossed] = lo[crossed], hi[crossed]
    return (new_lo, new_hi), "ok"


def infeasibility_margin(rows, lo, hi):
    """Smallest total violation of the linear rows over the box, by an LP with nonnegative slacks; > 0 means infeasible."""
    a_eq, b_eq, a_ub, b_ub = linear_rows(rows)
    ne, nu = len(b_eq), len(b_ub)
    c = np.r_[np.zeros(NV), np.ones(2 * ne + nu)]
    A_eq = np.hstack([a_eq, -np.eye(ne), np.eye(ne), np.zeros((ne, nu))]) if ne else None
    A_ub = np.hstack([a_ub, np.zeros((nu, 2 * ne)), -np.eye(nu)]) if nu else None
    bounds = list(zip(lo, hi)) + [(0.0, None)] * (2 * ne + nu)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub if nu else None, A_eq=A_eq, b_eq=b_eq if ne else None,
                  bounds=bounds, method="highs", options=LP_OPTIONS)
    if res.status != 0:
        raise RuntimeError(f"slack LP failed with status {res.status}: {res.message}")
    return float(res.fun)


def reachable(nv, s, t, tails, heads, alive):
    """Alive edges that lie on some start->goal path; edges must be in topological order."""
    fwd, bwd = np.zeros(nv, bool), np.zeros(nv, bool)
    fwd[s], bwd[t] = True, True
    for e in range(len(tails)):
        if alive[e] and fwd[tails[e]]:
            fwd[heads[e]] = True
    for e in range(len(tails) - 1, -1, -1):
        if alive[e] and bwd[heads[e]]:
            bwd[tails[e]] = True
    return alive & fwd[tails] & bwd[heads]


def prune(cfg, md, task, modes, tr):
    """Per-type LP pruning and box tightening, then reachability pruning (plan 6.1). Adds the results to tr."""
    margin = float(cfg["pruning"]["box_margin"])
    nt = len(tr["type_kind"])
    type_alive = np.ones(nt, bool)
    lo_t, hi_t, status = np.zeros((nt, NV)), np.zeros((nt, NV)), []
    t0 = time.perf_counter()

    def one_type(ti):
        lo, hi = edge_box(md, *type_modes(tr, modes, ti))
        return (lo, hi) + tighten_box(type_rows(md, task, modes, tr, ti), lo, hi, margin)

    for ti, (lo, hi, box, st) in enumerate(pool_map(one_type, range(nt), cfg["build"])):
        status.append(st)
        if box is None:
            type_alive[ti] = False
            box = (lo, hi)
        lo_t[ti], hi_t[ti] = box
    lp_alive = type_alive[tr["edge_type"]]
    tr.update(type_alive=type_alive, type_lo=lo_t, type_hi=hi_t, type_status=np.array(status),
              edge_lp_alive=lp_alive,
              edge_alive=reachable(tr["nv"], tr["s"], tr["t"], tr["edge_tail"], tr["edge_head"], lp_alive),
              lp_seconds=time.perf_counter() - t0)
    return tr


def build_instance(cfg):
    """Model, modes, task and pruned trellis of one config."""
    md, modes, task = build_model(cfg), build_modes(cfg), resolve_task(cfg)
    return md, modes, task, prune(cfg, md, task, modes, build_trellis(cfg, md, task, modes))


def taylor_feasible(rel):
    """Whether a Taylor edge type's slice y = 1 has a point (an LP, since the model is polyhedral)."""
    A, G = rel["Aeq"].toarray(), rel["G"].toarray()
    res = linprog(np.zeros(rel["n"] - 1), A_ub=-G[:, 1:] if G.shape[0] else None, b_ub=G[:, 0] if G.shape[0] else None,
                  A_eq=A[:, 1:] if A.shape[0] else None, b_eq=-A[:, 0] if A.shape[0] else None,
                  bounds=list(zip(rel["lo"][1:], rel["hi"][1:])), method="highs", options=LP_OPTIONS)
    return res.status != 2


def prune_outer(cfg, md, modes, task, tr):
    """Run B: an edge type whose Taylor slice is empty holds no true step, and its rows, with the bands, see more than the linear rows of the true model do (plan 6.1). Updates tr."""
    alive = np.flatnonzero(tr["type_alive"])
    feasible = np.array(pool_map(lambda ti: taylor_feasible(relax_edge_type(cfg, md, task, modes, tr, ti)), alive, cfg["build"]), bool)
    tr["type_alive"][alive[~feasible]] = False
    lp_alive = tr["type_alive"][tr["edge_type"]]
    tr.update(edge_lp_alive=lp_alive, edge_alive=reachable(tr["nv"], tr["s"], tr["t"], tr["edge_tail"], tr["edge_head"], lp_alive))
    return alive[~feasible]


def build_relaxations(cfg, md, modes, task, tr):
    """The outer model of every alive edge type (None for the others), after the validity check of plan 5.4: a violation raises, so no run can start from an invalid model."""
    if cfg["relaxation"]["type"] == "taylor":
        prune_outer(cfg, md, modes, task, tr)
    ok, msg, _ = validity_summary(validate_relaxation(cfg, md, modes, task, tr), cfg["relaxation"]["validity"])
    print(f"validity (plan 5.4): {msg}", flush=True)
    if not ok:
        raise RuntimeError("the outer models fail plan 5.4, so any bound computed from them would be invalid: " + msg)
    return [relax_edge_type(cfg, md, task, modes, tr, ti) if tr["type_alive"][ti] else None for ti in range(len(tr["type_kind"]))]


def type_group(kind, tail, head):
    """Label of an edge type's group in the size table."""
    if kind == "start":
        return "start"
    t = "contact" if tail["contact"] else "free"
    if kind == "goal":
        return f"goal, {t} tail"
    return f"mid, {t} -> {'contact' if head['contact'] else 'free'}"


def relaxation_stats(cfg, md, task, modes, tr, ti):
    """Sizes of one edge type's outer model."""
    rel = relax_edge_type(cfg, md, task, modes, tr, ti)
    tail, head = type_modes(tr, modes, ti)
    sizes = collections.Counter(int(b["size"]) for b in rel["psd"])
    return dict(group=type_group(EDGE_KINDS[int(tr["type_kind"][ti])], tail, head), n=rel["n"], nv=len(rel.get("free", range(rel["n"] - 1))), free=int(np.sum(rel["lo"] < rel["hi"])),
                eq_raw=rel["eq_rows_raw"], eq=rel["Aeq"].shape[0], ge=rel["G"].shape[0], rows=rel["Aeq"].shape[0] + rel["G"].shape[0],
                nnz=int(rel["Aeq"].nnz + rel["G"].nnz), blocks=" + ".join(f"{c} x {s}" for s, c in sorted(sizes.items(), reverse=True)) or "none")


def relaxation_table(cfg, md, modes, task, tr, label):
    """Prints the sizes the solver will see, per group of edge types (plan 5.1.7). `vars` counts the variables left after elimination, `m` the entries of m, `open` those not pinned by their box, `eq` the equality rows kept, and eq+ge is the size of the IPM's row-form system (plan 8.1)."""
    prebuild_bases(cfg, md, modes, task, tr)
    stats = pool_map(lambda ti: relaxation_stats(cfg, md, task, modes, tr, int(ti)), np.flatnonzero(tr["type_alive"]), cfg["build"])
    groups = {}
    for s in stats:
        groups.setdefault(s["group"], []).append(s)
    print(f"\n{label}: {len(stats)} alive edge types", flush=True)
    for name, rs in sorted(groups.items()):
        span = lambda key: f"{min(r[key] for r in rs)}-{max(r[key] for r in rs)}"
        print(f"  {name:24s} types {len(rs):5d}  vars {span('nv'):>6s}  m {span('n'):>13s}  open {span('free'):>13s}  eq raw {span('eq_raw'):>13s}  eq {span('eq'):>13s}  ge {span('ge'):>9s}  "
              f"eq+ge {span('rows'):>13s}  nnz {span('nnz'):>15s}  psd {sorted({r['blocks'] for r in rs})}", flush=True)
    keys = ("nv", "n", "free", "eq_raw", "eq", "ge", "rows", "nnz")
    return {name: {k: [min(r[k] for r in rs), max(r[k] for r in rs)] for k in keys} for name, rs in groups.items()}


def summary(tr, modes, task):
    """Counts of plan 6.1 for logging and the saved metadata."""
    kinds = {k: int(np.sum(tr["type_kind"][tr["edge_type"]] == i)) for i, k in enumerate(EDGE_KINDS)}
    used_types = np.unique(tr["edge_type"][tr["edge_alive"]])
    return dict(task=task["name"], N=tr["N"], modes=len(modes), transitions_per_layer=int(tr["allow"].sum()),
                vertices=tr["nv"], edges=int(len(tr["edge_tail"])), edges_by_kind=kinds,
                edges_after_lp=int(tr["edge_lp_alive"].sum()), edges_alive=int(tr["edge_alive"].sum()),
                types=int(len(tr["type_kind"])), types_lp_infeasible=int((~tr["type_alive"]).sum()),
                types_used=int(len(used_types)),
                types_lp_trouble=[str(s) for s in tr["type_status"] if s not in ("ok", "infeasible")],
                lp_seconds=round(float(tr["lp_seconds"]), 2))


def pack_relaxations(rels):
    """Per-type outer models as flat arrays for problem.npz (plan 6.1, 6.2): rows and coupling maps as (type, row, col, value), PSD maps as (type, block, row, col, moment, coefficient), vectors padded to the longest m with rel_n[type] entries in use. Block 0 is the moment matrix, which depends only on a type's variable count, so it is stored once per count as rel_moment_<count>_*. Per type rather than per edge: memory scales with the types."""
    alive = [r for r in rels if r is not None]
    first = alive[0]
    nt, longest = len(rels), max(r["n"] for r in alive)
    is_sdp = first["kind"] == "sdp"
    out = dict(rel_kind=first["kind"], rel_q=first["tail_map"].shape[0], rel_n=np.zeros(nt, np.int64),
               rel_lo=np.zeros((nt, longest)), rel_hi=np.zeros((nt, longest)), rel_c=np.zeros((nt, longest)),
               rel_eq_rows=np.zeros(nt, np.int64), rel_ge_rows=np.zeros(nt, np.int64))
    if is_sdp:
        out.update(rel_nv=np.zeros(nt, np.int64), rel_free=np.full((nt, NV), -1, np.int64), rel_sub_const=np.zeros((nt, NV)), rel_sub_coef=np.zeros((nt, NV, NV)))
    coo = dict(eq=[], ge=[], tail=[], head=[])
    psd, psd_sizes, soc = [], [], []
    for ti, rel in enumerate(rels):
        if rel is None:
            continue
        n = rel["n"]
        out["rel_n"][ti] = n
        out["rel_lo"][ti, :n], out["rel_hi"][ti, :n], out["rel_c"][ti, :n] = rel["lo"], rel["hi"], rel["c"]
        out["rel_eq_rows"][ti], out["rel_ge_rows"][ti] = rel["Aeq"].shape[0], rel["G"].shape[0]
        for key, mat in (("eq", rel["Aeq"]), ("ge", rel["G"]), ("tail", rel["tail_map"]), ("head", rel["head_map"])):
            c = mat.tocoo()
            coo[key].append((np.full(c.nnz, ti, np.int32), c.row.astype(np.int32), c.col.astype(np.int32), c.data))
        if is_sdp:
            nv = len(rel["free"])
            out["rel_nv"][ti] = nv
            out["rel_free"][ti, :nv] = rel["free"]
            for i in range(NV):
                for k, value in rel["sub"][i].items():
                    if k:
                        out["rel_sub_coef"][ti, i, k[0]] = value
                    else:
                        out["rel_sub_const"][ti, i] = value
            moment = rel["psd"][0]
            out[f"rel_moment_{nv}_size"], out[f"rel_moment_{nv}_row"] = moment["size"], moment["row"]
            out[f"rel_moment_{nv}_col"], out[f"rel_moment_{nv}_mom"] = moment["col"], moment["mom"]
        for b, block in enumerate(rel["psd"]):
            if b > 0:
                psd.append((np.full(block["row"].size, ti, np.int32), np.full(block["row"].size, b, np.int32),
                            block["row"], block["col"], block["mom"], block["coef"]))
                psd_sizes.append((ti, b, block["size"]))
        soc += [(ti, s["z"], s["scale"]) + tuple(s["form"].toarray().ravel()) for s in rel["soc"]]
    for key, parts in coo.items():
        for name, column in zip(("type", "row", "col", "val"), zip(*parts)):
            out[f"rel_{key}_{name}"] = np.concatenate(column)
    if psd:
        for name, column in zip(("type", "block", "row", "col", "mom", "coef"), zip(*psd)):
            out[f"rel_psd_{name}"] = np.concatenate(column)
        out["rel_psd_sizes"] = np.array(psd_sizes, np.int64)
    if soc:
        out["rel_soc"] = np.array(soc)                          # (type, z, scale, form over m)
    return out


def save_problem(path, cfg, md, modes, task, tr, rels):
    """Writes problem.npz (plan 6.1): the graph and the per-type outer models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(summary=summary(tr, modes, task), modes=[m["name"] for m in modes], var_names=list(VAR_NAMES),
                edge_kinds=list(EDGE_KINDS), task=dict(name=task["name"], start=task["start"].tolist(),
                goal=task["goal"].tolist(), pusher_start=task["pusher_start"].tolist(), start_region=task["start_region"]),
                model={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in md.items()},
                relaxation=cfg["relaxation"])
    arrays = pack_relaxations(rels)
    # compressing a kappa = 3 model takes minutes and saves little that matters
    writer = np.savez_compressed if sum(np.size(a) for a in arrays.values()) < 5e7 else np.savez
    writer(path, meta=json.dumps(meta), **arrays,
           **{k: tr[k] for k in ("vert_layer", "vert_mode", "edge_tail", "edge_head", "edge_type", "edge_layer",
                                 "edge_lp_alive", "edge_alive", "type_kind", "type_tail", "type_head",
                                 "type_alive", "type_lo", "type_hi", "allow")},
           type_status=tr["type_status"].astype("U32"), s=tr["s"], t=tr["t"], N=tr["N"], M=tr["M"])


# ===== REFERENCE SOLVE =====
# Plan 10.4, milestone M2: the whole relaxation P_delta (tex Def 2.2) of a small instance as one conic
# program, solved on the CPU. Its value J_rel is the ceiling every certified bound must stay under at
# M3. The data goes to the solvers directly, not through CVXPY: the rows are already in conic form, and
# this keeps control of the triangle convention, which check_cone_conventions tests on a known answer.
# Both solvers take: minimize c^T x with s = b - A x, s in (zero cone, nonnegative orthant, PSD cones).

def svec_position(row, col, size, solver):
    """Position of entry (row, col), row <= col, in the solver's scaled triangle of a symmetric matrix: Clarabel stacks the upper triangle by columns, SCS the lower triangle by columns. Both scale off-diagonals by sqrt(2)."""
    if solver == "clarabel":
        return col * (col + 1) // 2 + row
    return row * size - row * (row - 1) // 2 + col - row


def svec_to_matrix(v, size, solver):
    """The symmetric matrix of a scaled triangle."""
    iu, ju = np.triu_indices(size)
    vals = v[svec_position(iu, ju, size, solver)] / np.where(iu == ju, 1.0, math.sqrt(2.0))
    out = np.zeros((size, size))
    out[iu, ju] = vals
    out[ju, iu] = vals
    return out


def type_conic_rows(rel, solver, pin_width):
    """One edge type's rows over its own m: equalities (zero), nonnegative rows (nonneg) and the PSD blocks' scaled triangles (psd, with their sizes). An interval bound of at most pin_width becomes an equality, because two facing inequalities leave an interior-point solver no interior."""
    if rel["soc"]:
        raise NotImplementedError("the reference solve has no second-order rows; use cost_model tangent_planes")
    n, lo, hi = rel["n"], rel["lo"], rel["hi"]
    entries = np.arange(1, n)
    tight = hi[1:] - lo[1:] <= pin_width
    pinned, spread = entries[tight], entries[~tight]

    def bound_rows(j, a, b):
        k = np.arange(j.size)
        rows = sp.csr_matrix((np.r_[np.full(j.size, a), b], (np.r_[k, k], np.r_[j, np.zeros(j.size, np.int64)])), shape=(j.size, n))
        rows.eliminate_zeros()
        return rows

    zero = sp.vstack([rel["Aeq"], bound_rows(pinned, 1.0, -0.5 * (lo[pinned] + hi[pinned]))], format="csr")
    zero = zero[independent_rows(zero) if n <= 200 else unique_rows(zero)]
    flow = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n))
    nonneg = sp.vstack([rel["G"], bound_rows(spread, 1.0, -lo[spread]), bound_rows(spread, -1.0, hi[spread]), flow], format="csr")
    rows, cols, vals, sizes, base = [], [], [], [], 0
    for block in rel["psd"]:
        size = int(block["size"])
        r, c = block["row"].astype(np.int64), block["col"].astype(np.int64)
        rows.append(base + svec_position(r, c, size, solver))
        cols.append(block["mom"])
        vals.append(block["coef"] * np.where(r == c, 1.0, math.sqrt(2.0)))
        sizes.append(size)
        base += size * (size + 1) // 2
    psd = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(base, n)) if rows else sp.csr_matrix((0, n))
    return dict(zero=zero.tocoo(), nonneg=nonneg.tocoo(), psd=psd.tocoo(), sizes=sizes)


def assemble_conic(cfg, tr, rels, solver):
    """P_delta of an instance as one conic program over the stacked m of its alive edges: each edge's rows, flow conservation (exact, with beta_s = -1, beta_t = +1, head +, tail -) and the state rows |K m| <= delta at inner vertices. The goal's flow row is left out, since the others imply it."""
    delta = float(cfg["solver"]["delta"])
    alive = np.flatnonzero(tr["edge_alive"])
    if not alive.size:
        raise ValueError("no start->goal path survives pruning")
    types = tr["edge_type"][alive]
    rows_of = {int(ti): type_conic_rows(rels[ti], solver, float(cfg["reference"]["pin_width"])) for ti in np.unique(types)}
    offs = np.concatenate([[0], np.cumsum([rels[ti]["n"] for ti in types])]).astype(np.int64)
    nx = int(offs[-1])

    def stacked(key):
        rows, cols, vals, count = [], [], [], 0
        for off, ti in zip(offs, types):
            coo = rows_of[int(ti)][key]
            rows.append(coo.row.astype(np.int64) + count)
            cols.append(coo.col.astype(np.int64) + off)
            vals.append(coo.data)
            count += coo.shape[0]
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(count, nx))

    c = np.zeros(nx)
    q = rels[types[0]]["tail_map"].shape[0]
    verts = np.unique(np.r_[tr["edge_tail"][alive], tr["edge_head"][alive]])
    vpos = np.full(tr["nv"], -1, np.int64)
    vpos[verts] = np.arange(verts.size)
    krows, kcols, kvals = [], [], []
    for off, e, ti in zip(offs, alive, types):
        rel = rels[ti]
        c[off:off + rel["n"]] = rel["c"]
        for mapping, vertex, sign in ((rel["head_map"], tr["edge_head"][e], 1.0), (rel["tail_map"], tr["edge_tail"][e], -1.0)):
            coo = mapping.tocoo()
            krows.append(vpos[vertex] * q + coo.row.astype(np.int64))
            kcols.append(off + coo.col.astype(np.int64))
            kvals.append(sign * coo.data)
    K = sp.csr_matrix((np.concatenate(kvals), (np.concatenate(krows), np.concatenate(kcols))), shape=(verts.size * q, nx))
    v_s, v_t = int(vpos[tr["s"]]), int(vpos[tr["t"]])
    flow = np.array([i * q for i in range(verts.size) if i != v_t], np.int64)
    inner = np.array([i for i in range(verts.size) if i not in (v_s, v_t)], np.int64)
    state = (inner[:, None] * q + np.arange(1, q)).ravel()
    edge_zero, edge_nonneg, edge_psd = stacked("zero"), stacked("nonneg"), stacked("psd")
    beta = np.where(flow == v_s * q, -1.0, 0.0)
    if delta > 0.0:
        zero = sp.vstack([edge_zero, K[flow]], format="csr")
        b_zero = np.r_[np.zeros(edge_zero.shape[0]), beta]
        nonneg = sp.vstack([-edge_nonneg, K[state], -K[state]], format="csr")
        b_nonneg = np.r_[np.zeros(edge_nonneg.shape[0]), np.full(2 * state.size, delta)]
    else:
        zero = sp.vstack([edge_zero, K[flow], K[state]], format="csr")
        b_zero = np.r_[np.zeros(edge_zero.shape[0]), beta, np.zeros(state.size)]
        nonneg = -edge_nonneg
        b_nonneg = np.zeros(edge_nonneg.shape[0])
    A = sp.vstack([zero, nonneg, -edge_psd], format="csc")
    A.eliminate_zeros()
    A.sort_indices()
    return dict(solver=solver, A=A, b=np.r_[b_zero, b_nonneg, np.zeros(edge_psd.shape[0])], c=c, nz=zero.shape[0], nl=nonneg.shape[0],
                sizes=[s for ti in types for s in rows_of[int(ti)]["sizes"]], alive=alive, offs=offs)


def conic_residuals(prob, x):
    """(largest equality error, largest inequality violation, largest PSD deficit) of a point, computed here rather than read from a solver."""
    s = prob["b"] - prob["A"] @ x
    nz, nl = prob["nz"], prob["nl"]
    eq = float(np.max(np.abs(s[:nz]), initial=0.0))
    ge = max(float(np.max(-s[nz:nz + nl], initial=0.0)), 0.0)
    psd, base = 0.0, nz + nl
    for size in prob["sizes"]:
        k = size * (size + 1) // 2
        psd = max(psd, -float(np.linalg.eigvalsh(svec_to_matrix(s[base:base + k], size, prob["solver"]))[0]))
        base += k
    return eq, ge, psd


def solve_conic(cfg, prob):
    """Solves an assembled program with its solver: a dict with status, solved, infeasible, iters, the dual value, seconds and x."""
    rc = cfg["reference"]
    nz, nl, sizes = prob["nz"], prob["nl"], [int(s) for s in prob["sizes"]]
    t0 = time.perf_counter()
    if prob["solver"] == "clarabel":
        import clarabel
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.time_limit = float(rc["time_limit_s"])
        settings.max_iter = int(rc["clarabel"]["max_iter"])
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = float(rc["clarabel"]["tol"])
        if rc["clarabel"]["direct_solve_method"]:
            settings.direct_solve_method = str(rc["clarabel"]["direct_solve_method"])
        cones = ([clarabel.ZeroConeT(nz)] if nz else []) + ([clarabel.NonnegativeConeT(nl)] if nl else []) + [clarabel.PSDTriangleConeT(s) for s in sizes]
        n = prob["A"].shape[1]
        sol = clarabel.DefaultSolver(sp.csc_matrix((n, n)), prob["c"], prob["A"], prob["b"], cones, settings).solve()
        status = str(sol.status)
        out = dict(status=status, solved=status in ("Solved", "AlmostSolved"), infeasible=status in ("PrimalInfeasible", "AlmostPrimalInfeasible"),
                   iters=int(sol.iterations), J_dual=float(getattr(sol, "obj_val_dual", float("nan"))), x=np.asarray(sol.x, float))
    elif prob["solver"] == "scs":
        import scs
        eps = float(rc["scs"]["eps"])
        sol = scs.SCS(dict(A=prob["A"], b=prob["b"], c=prob["c"]), dict(z=nz, l=nl, s=sizes), verbose=False, eps_abs=eps, eps_rel=eps,
                      max_iters=int(rc["scs"]["max_iters"]), time_limit_secs=float(rc["time_limit_s"])).solve()
        info = sol["info"]
        out = dict(status=str(info["status"]), solved=int(info["status_val"]) == 1, infeasible=int(info["status_val"]) in (-2, -7),
                   iters=int(info["iter"]), J_dual=float(info["dobj"]), x=np.asarray(sol["x"], float))
    else:
        raise ValueError(f"unknown reference solver {prob['solver']!r}")
    out["seconds"] = time.perf_counter() - t0
    return out


def edge_models(cfg, md, task, modes, tr):
    """Outer models of the edge types that alive edges use (None for the others)."""
    used = set(np.unique(tr["edge_type"][tr["edge_alive"]]).tolist())
    return [relax_edge_type(cfg, md, task, modes, tr, ti) if ti in used else None for ti in range(len(tr["type_kind"]))]


def reference_solver(cfg, rels):
    """Clarabel forms a dense KKT block per PSD cone, so it fits small blocks only: 'auto' sends larger ones to SCS, a first-order solver with lower accuracy."""
    rc = cfg["reference"]
    if rc["solver"] != "auto":
        return rc["solver"]
    largest = max((int(b["size"]) for r in rels if r is not None for b in r["psd"]), default=0)
    return "clarabel" if largest <= int(rc["clarabel_max_block"]) else "scs"


def reference_line(r):
    """One printed line of a reference result."""
    if "error" in r:
        return f"  {r['label']:44s} ERROR {r['error']}"
    value = "infeasible" if r["J"] is None else f"{r['J']:.6f}"
    tail = "" if r["J"] is None else f"  residuals eq {r['eq']:.1e} ge {r['ge']:.1e} psd {r['psd']:.1e}  edges with flow {r['flow_edges']}/{r['edges']}"
    return (f"  {r['label']:44s} J_rel {value:>10s}  dual {r['J_dual']:9.6f}  {r['solver']:8s} {r['status']:24s} {r['iters']:7d} its {r['seconds']:8.1f} s"
            f"  x {r['variables']:7d}  rows {r['rows']:7d}  block {r['largest_block']:4d}" + tail)


def reference_job(cfg, inst, label):
    """P_delta of one instance on the CPU (plan 10.4), as a JSON-ready dict. It prints a line when done, since jobs overlap and can take minutes, and it reports an error instead of raising, so that one failure does not hide the other solves."""
    md, modes, task, tr = inst
    try:
        rels = edge_models(cfg, md, task, modes, tr)
        prob = assemble_conic(cfg, tr, rels, reference_solver(cfg, rels))
        out = solve_conic(cfg, prob)
        x = out.pop("x")
        out.update(label=label, task=task["name"], solver=prob["solver"], variables=int(prob["c"].size), rows=int(prob["A"].shape[0]),
                   edges=int(prob["alive"].size), largest_block=max(prob["sizes"], default=0), J=None)
        if not out["infeasible"]:
            out["eq"], out["ge"], out["psd"] = conic_residuals(prob, x)
            out["J"] = float(prob["c"] @ x)
            out["flow_edges"] = int(np.sum(x[prob["offs"][:-1]] > 1e-4))
    except Exception as exc:
        out = dict(label=label, task=task["name"], error=f"{type(exc).__name__}: {exc}", solved=False, infeasible=False, J=None)
    print(reference_line(out), flush=True)
    return out


# ===== SELFTEST =====
# Milestone checks (plan 17). M0: sign conventions, the mode graph, pruning soundness, and a
# hand-scripted true push that must be a valid start->goal path. M1: the outer models hold every true
# point, and elimination does not change them. M2: the reference solve.

def flow_residual(tr, y):
    """Inflow - outflow - beta_v at every vertex, for edge flows y (zero when flow is conserved)."""
    r = np.zeros(tr["nv"])
    np.add.at(r, tr["edge_head"], y)
    np.add.at(r, tr["edge_tail"], -y)
    r[tr["s"]] += 1.0
    r[tr["t"]] -= 1.0
    return r


def path_edge_ids(tr, modes, steps):
    """Edge ids of a scripted path, one edge per step; raises if a step is not an edge of the trellis."""
    names = [m["name"] for m in modes]
    lookup = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(tr["edge_tail"], tr["edge_head"]))}
    vid = lambda k, m: 1 + (k - 1) * tr["M"] + m
    ids = []
    for k, (_, tail, head, _) in enumerate(steps):
        if k > 0 and steps[k - 1][2]["name"] != tail["name"]:
            raise ValueError(f"step {k} starts in {tail['name']} but step {k - 1} ends in {steps[k - 1][2]['name']}")
        tv = tr["s"] if k == 0 else vid(k, names.index(tail["name"]))
        hv = tr["t"] if head is None else vid(k + 1, names.index(head["name"]))
        if (tv, hv) not in lookup:
            raise KeyError(f"step {k} ({tail['name']} -> {head['name'] if head else 'goal'}) is not an edge")
        ids.append(lookup[(tv, hv)])
    return ids


def check_geometry(md, cfg):
    n, t = md["normals"], md["tangents"]
    cross = n[:, 0] * t[:, 1] - n[:, 1] * t[:, 0]
    if not np.allclose(cross, 1.0):
        return False, f"n_j x t_j = {cross}, expected +1 (counter-clockwise tangents)"
    if not np.allclose(md["centers"], n * [md["ax"], md["ay"]]) or not np.allclose(md["half_len"], [md["ay"], md["ax"], md["ay"], md["ax"]]):
        return False, "face centers or half-lengths do not match the box"
    r2 = mean_distance_rectangle(md["ax"], md["ay"], 2 * int(cfg["model"]["limit_surface"].get("rbar_grid", 1000)))
    rel = abs(r2 - md["rbar"]) / r2
    if rel > 1e-5:
        return False, f"rbar grid not converged: {md['rbar']:.7f} vs {r2:.7f} at twice the resolution"
    return True, (f"tangents counter-clockwise; rbar {md['rbar']:.5f} m (changes {rel:.1e} at 2x grid), "
                  f"f_max {md['f_max']:.4f} N, tau_max {md['tau_max']:.5f} N m")


def check_friction_sign(md):
    th0 = 0.3
    rot = np.array([[math.cos(th0), -math.sin(th0)], [math.sin(th0), math.cos(th0)]])
    for j in range(4):
        n, t = md["normals"][j], md["tangents"][j]
        x0 = knot_from_pose([0.0, 0.0, th0], md["centers"][j] + md["rp"] * n)
        fn, dist = 0.02, 0.004
        for ctype, sgn in (("slide_pos", 1.0), ("slide_neg", -1.0)):
            x1, tau = true_contact_step(md, x0, j, fn, sgn * md["mu_c"] * fn, sgn * dist)
            v = edge_vector(x0, x1, fn, sgn * md["mu_c"] * fn, tau)
            mode = dict(name=f"c{j}_{ctype}", contact=True, face=j, ctype=ctype, region=None)
            viol = violations(edge_constraints(md, None, "mid", mode, mode), v)
            if max(viol.values()) > 1e-9:
                worst = max(viol, key=viol.get)
                return False, f"face {j} {ctype}: the true step violates {worst} by {viol[worst]:.2e}"
            drag = (x1[:2] - x0[:2]) @ (rot @ t)
            if sgn * drag <= 0.0:
                return False, f"face {j} {ctype}: slider moved {drag:+.2e} along t_j, expected sign {sgn:+.0f}"
            other = "slide_neg" if ctype == "slide_pos" else "slide_pos"
            wrong = dict(mode, ctype=other, name=f"c{j}_{other}")
            if max(violations(edge_constraints(md, None, "mid", wrong, wrong), v).values()) < 1e-6:
                return False, f"face {j}: the {ctype} step also passes as {other}"
    return True, "slide_pos drags the slider toward +t_j and slide_neg toward -t_j on every face; each fails the opposite type"


def check_torque_sign(md):
    j = 2
    n = md["normals"][j]
    for y, expect in ((0.03, -1.0), (-0.03, 1.0)):
        x0 = knot_from_pose([0.0, 0.0, 0.0], [md["centers"][j][0] + md["rp"] * n[0], y])
        x1, tau = true_contact_step(md, x0, j, 0.02, 0.0, 0.0)
        dth = math.atan2(x1[3], x1[2])
        if np.sign(tau) != expect or np.sign(dth) != expect:
            return False, f"push on face 2 at y={y:+.2f}: tau {tau:+.2e}, dtheta {dth:+.2e}, expected sign {expect:+.0f}"
    return True, "a +x push on face 2 above its center turns the slider clockwise, below it counter-clockwise"


def check_true_steps(md, modes, rng, samples=300):
    """Random true contact and non-contact steps satisfy every row and the box (the table matches the true model)."""
    worst, worst_at, count = 0.0, "", 0
    contact = [m for m in modes if m["contact"]]
    for _ in range(samples):
        m = contact[rng.integers(len(contact))]
        j = m["face"]
        lam = rng.uniform(-1.0, 1.0) * md["half_len"][j]
        x0 = knot_from_pose([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-math.pi, math.pi)],
                            md["centers"][j] + md["rp"] * md["normals"][j] + lam * md["tangents"][j])
        fn = rng.uniform(0.0, 0.02)
        room_pos, room_neg = min(md["v_max"], md["half_len"][j] - lam), min(md["v_max"], md["half_len"][j] + lam)
        ft, slide = {"stick": (rng.uniform(-1, 1) * md["mu_c"] * fn, 0.0),
                     "slide_pos": (md["mu_c"] * fn, rng.uniform(0.0, room_pos)),
                     "slide_neg": (-md["mu_c"] * fn, -rng.uniform(0.0, room_neg))}[m["ctype"]]
        try:
            x1, tau = true_contact_step(md, x0, j, fn, ft, slide)
        except ValueError:
            continue
        viol = violations(edge_constraints(md, None, "mid", m, m), edge_vector(x0, x1, fn, ft, tau), *edge_box(md, m))
        count += 1
        if max(viol.values()) > worst:
            worst, worst_at = max(viol.values()), f"{m['name']} {max(viol, key=viol.get)}"
    free = [m for m in modes if not m["contact"]]
    for m in free:
        rows0 = region_rows(md, m["region"], 0, 1)
        got = 0
        for _ in range(20000):
            r0 = rng.uniform(-md["r_ws"], md["r_ws"], 2)
            r1 = r0 + rng.uniform(-md["v_max"], md["v_max"], 2)
            if min(poly_eval(p, r0) for p in rows0) < 0 or min(poly_eval(p, r1) for p in rows0) < 0 or np.max(np.abs(r1)) > md["r_ws"]:
                continue
            pose = [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-math.pi, math.pi)]
            v = edge_vector(knot_from_pose(pose, r0), knot_from_pose(pose, r1))
            viol = violations(edge_constraints(md, None, "mid", m, m), v, *edge_box(md, m))
            got += 1
            if max(viol.values()) > worst:
                worst, worst_at = max(viol.values()), f"{m['name']} {max(viol, key=viol.get)}"
            if got == 50:
                break
        if got == 0:
            return False, f"no true points sampled in region {m['region']}"
        count += got
    return worst <= 1e-9, f"worst violation over {count} random true steps {worst:.1e}" + (f" ({worst_at})" if worst_at else "")


def check_table(md, task, modes, tr):
    expected = {"stick": {"stick_hi", "stick_lo", "stick_rx", "stick_ry"}, "slide_pos": {"slide_force", "slide_dir"},
                "slide_neg": {"slide_force", "slide_dir"}}
    for ti in range(len(tr["type_kind"])):
        tail, head = type_modes(tr, modes, ti)
        rows = type_rows(md, task, modes, tr, ti)
        tags = {tag for _, _, tag in rows}
        deg = max(poly_degree(p) for _, p, _ in rows)
        cdeg = poly_degree(edge_cost({"costs": {"w_p": 1, "w_theta": 1, "w_r": 1, "w_f": 1, "w_switch": 1}}, tail, head))
        if deg > 2 or cdeg > 2:
            return False, f"type {ti}: degree {deg} rows, degree {cdeg} cost; plan 4.5 needs <= 2"
        need = {"unit_norm_0", "unit_norm_1", "rot_rel", "rot_step", "v_max_x_hi", "v_max_y_lo"}
        need |= ({"trans_x", "trans_y", "torque", "contact_line_k0", "contact_line_k1", "fn_nonneg"} | expected[tail["ctype"]]
                 if tail["contact"] else {"static_px", "static_c", "zero_fn", "zero_tau", "tail_region_0"})
        kind = EDGE_KINDS[int(tr["type_kind"][ti])]
        need |= {f"start_{VAR_NAMES[i]}" for i in KNOT0} if kind == "start" else set()
        need |= {"goal_px1", "goal_py1", "goal_c1", "goal_s1"} if kind == "goal" else set()
        if not need <= tags:
            return False, f"type {ti} ({kind} {tail['name']} -> {head['name'] if head else 'goal'}) lacks rows {sorted(need - tags)}"
    return True, f"{len(tr['type_kind'])} edge types: every row and cost has degree <= 2 and carries the rows of its mode"


def check_mode_graph(modes):
    A = np.array([[transition_allowed(a, b) for b in modes] for a in modes])
    if not np.array_equal(A, A.T):
        return False, "the mode graph is not symmetric"
    for ia, a in enumerate(modes):
        for ib, b in enumerate(modes):
            if a["contact"]:
                expect = (b["contact"] and b["face"] == a["face"]) or (not b["contact"] and b["region"] == f"strip_{a['face']}")
            elif b["contact"]:
                expect = a["region"] == f"strip_{b['face']}"
            else:
                expect = a is b or regions_adjacent(a["region"], b["region"])
            if A[ia, ib] != expect:
                return False, f"{a['name']} -> {b['name']}: allowed {A[ia, ib]}, plan 4.4 says {expect}"
    free = [m for m in modes if not m["contact"]]
    for a in free:
        nbrs = [b["name"] for b in free if b is not a and regions_adjacent(a["region"], b["region"])]
        if len(free) == 8 and len(nbrs) != 2:
            return False, f"{a['name']} borders {nbrs}, expected exactly 2 regions"
    total = int(A.sum())
    if len(modes) == 20 and total != 84:
        return False, f"{total} transitions per layer, expected 36 + 12 + 12 + 24 = 84"
    return True, f"{len(modes)} modes, {total} transitions per layer, symmetric, matching plan 4.4 entry by entry"


def check_trellis_counts(tr, modes):
    M, N = len(modes), tr["N"]
    n_start = int(tr["allow"][tr["start_mode"]].sum())
    expect = n_start + (N - 2) * int(tr["allow"].sum()) + M
    if len(tr["edge_tail"]) != expect or tr["nv"] != 2 + (N - 1) * M:
        return False, f"{len(tr['edge_tail'])} edges and {tr['nv']} vertices, expected {expect} and {2 + (N - 1) * M}"
    if np.any(tr["vert_layer"][tr["edge_head"]] != tr["vert_layer"][tr["edge_tail"]] + 1) or np.any(np.diff(tr["edge_layer"]) < 0):
        return False, "an edge does not join adjacent layers, or edges are not in layer order"
    return True, f"{len(tr['edge_tail'])} edges ({n_start} start, {M} goal), {tr['nv']} vertices; adjacent layers only, in order"


def check_hand_push(cfg, md, modes, task, tr, record):
    steps = hand_push(cfg, md, task, modes)
    worst, at = 0.0, ""
    for k, (kind, tail, head, v) in enumerate(steps):
        viol = violations(edge_constraints(md, task, kind, tail, head), v, *edge_box(md, tail))
        if max(viol.values()) > worst:
            worst, at = max(viol.values()), f"step {k} {max(viol, key=viol.get)}"
    if worst > 1e-9:
        return False, f"the scripted push violates the true model: {at} by {worst:.1e}"
    ids = path_edge_ids(tr, modes, steps)
    dead = [k for k, e in enumerate(ids) if not tr["edge_alive"][e]]
    if dead:
        return False, f"pruning removed steps {dead} of a true trajectory"
    margin = float(cfg["pruning"]["box_margin"])
    for k, (e, step) in enumerate(zip(ids, steps)):
        ti = tr["edge_type"][e]
        tol = 10 * margin * (1.0 + np.abs(step[3]))
        if np.any(step[3] < tr["type_lo"][ti] - tol) or np.any(step[3] > tr["type_hi"][ti] + tol):
            return False, f"step {k}: the tightened box of its edge type excludes a true point"
    if np.max(np.abs(flow_residual(tr, np.isin(np.arange(len(tr["edge_tail"])), ids).astype(float)))) > 1e-12:
        return False, "unit flow along the scripted path is not conserved"
    J = sum(poly_eval(edge_cost(cfg, tail, head), v) for _, tail, head, v in steps)
    record[f"J_hand_N{tr['N']}_{len(modes)}modes"] = J
    seq = [steps[0][1]["name"]] + [s[2]["name"] if s[2] else "goal" for s in steps]
    return True, f"N={tr['N']}, {len(modes)} modes: valid start->goal path, worst residual {worst:.1e}, J_hand = {J:.6f} ({seq[0]} -> {seq[1]} x{seq.count(seq[1])} -> goal)"


def check_flow_signs(cfg, md, modes, task, tr):
    ids = path_edge_ids(tr, modes, hand_push(cfg, md, task, modes))
    y = np.zeros(len(tr["edge_tail"]))
    y[ids] = 1.0
    if np.max(np.abs(flow_residual(tr, y))) > 1e-12:
        return False, "a start->goal path with unit flow does not satisfy inflow - outflow = beta"
    y[ids[len(ids) // 2]] = 0.0
    bad = np.flatnonzero(np.abs(flow_residual(tr, y)) > 0.5)
    if len(bad) != 2:
        return False, f"removing one path edge should break conservation at exactly 2 vertices, got {len(bad)}"
    return True, "unit flow on a path conserves flow with beta_s = -1, beta_t = +1 (head +, tail -); breaking the path breaks it at both ends"


def check_pruning(cfg, md, modes, task, tr):
    gap = float(cfg["pruning"]["infeasible_margin"])
    pruned = np.flatnonzero(~tr["type_alive"])
    misses = pool_map(lambda ti: infeasibility_margin(type_rows(md, task, modes, tr, ti), *edge_box(md, *type_modes(tr, modes, ti))),
                      pruned, cfg["build"])
    for ti, miss in zip(pruned, misses):
        if miss <= gap:
            tail, head = type_modes(tr, modes, ti)
            return False, f"type {ti} ({tail['name']} -> {head['name'] if head else 'goal'}) was pruned but misses feasibility by only {miss:.1e}"
    adj_out, adj_in = {}, {}
    for e in np.flatnonzero(tr["edge_lp_alive"]):
        adj_out.setdefault(int(tr["edge_tail"][e]), []).append(int(tr["edge_head"][e]))
        adj_in.setdefault(int(tr["edge_head"][e]), []).append(int(tr["edge_tail"][e]))

    def flood(start, adj):
        seen, stack = {start}, [start]
        while stack:
            for w in adj.get(stack.pop(), []):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    fwd, bwd = flood(tr["s"], adj_out), flood(tr["t"], adj_in)
    on_path = np.array([tr["edge_lp_alive"][e] and int(tr["edge_tail"][e]) in fwd and int(tr["edge_head"][e]) in bwd
                        for e in range(len(tr["edge_tail"]))])
    if not np.array_equal(on_path, tr["edge_alive"]):
        return False, f"reachability pruning disagrees with an independent search on {int(np.sum(on_path != tr['edge_alive']))} edges"
    trouble = [str(s) for s in tr["type_status"] if s not in ("ok", "infeasible")]
    return True, (f"{len(pruned)} edge types LP-infeasible (each misses feasibility by > {gap:.0e}); "
                  f"{int(tr['edge_lp_alive'].sum() - tr['edge_alive'].sum())} edges off every start->goal path; "
                  f"{int(tr['edge_alive'].sum())} of {len(tr['edge_tail'])} edges kept"
                  + (f"; LP trouble on {len(trouble)} types, boxes left untightened there" if trouble else ""))


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


def check_moment_basis():
    """Sizes of plan 5.1.1, the monomial order, the index tables at random points, and the interval box on a random box."""
    rng = np.random.default_rng(3)
    lines = []
    for kappa, (n, s) in {1: (136, 16), 2: (3876, 136), 3: (54264, 816)}.items():
        b = moment_basis(kappa)
        monos = b["monos"]
        if len(monos) != n or b["upto"][kappa] != s:
            return False, f"kappa {kappa}: {len(monos)} moments and a {b['upto'][kappa]} x {b['upto'][kappa]} moment matrix, plan 5.1.1 says {n} and {s}"
        if any((len(x), x) >= (len(y), y) for x, y in zip(monos, monos[1:])):
            return False, f"kappa {kappa}: monomials are not ordered by degree, then lexicographically"
        if np.unique(b["mat"]).size != n:
            return False, f"kappa {kappa}: not every moment appears in the moment matrix"
        V = rng.uniform(-1.0, 1.0, (4, NV))
        M = rank_one_moments(b, V)
        low = b["times"].shape[1]
        err = max(float(np.max(np.abs(M[:, b["times"][i]] - V[:, [i]] * M[:, :low]))) for i in range(NV))
        err = max(err, float(np.max(np.abs(M[:, b["mat"]] - M[:, :s, None] * M[:, None, :s]))))
        if err > 1e-13:
            return False, f"kappa {kappa}: index tables disagree with monomial values by {err:.1e}"
        lines.append(f"kappa {kappa}: {n} moments, {s} x {s}, tables good to {err:.0e}")
    # the sizes after linear elimination (plan 5.1.7): 12 variables in contact, 8 on a free-to-free step
    for (kappa, nv), (n, s) in {(1, 12): (91, 13), (2, 12): (1820, 91), (3, 12): (18564, 455), (1, 8): (45, 9), (2, 8): (495, 45), (3, 8): (3003, 165)}.items():
        b = moment_basis(kappa, nv)
        if len(b["monos"]) != n or b["upto"][kappa] != s:
            return False, f"kappa {kappa}, {nv} variables: {len(b['monos'])} moments and a matrix of size {b['upto'][kappa]}, expected {n} and {s}"
    b = moment_basis(2)
    lo = rng.uniform(-1.0, 0.5, NV)
    hi = lo + rng.uniform(0.0, 1.0, NV)
    lo[3], hi[3] = 0.0, 0.0
    mlo, mhi = moment_box(b, lo, hi)
    M = rank_one_moments(b, rng.uniform(lo, hi, (2000, NV)))
    if np.any(M < mlo - 1e-12) or np.any(M > mhi + 1e-12):
        return False, "random points in a box leave the interval moment bounds"
    mlo, mhi = moment_box(b, -np.ones(NV), 2.0 * np.ones(NV))
    expect = {(0, 0): (0.0, 4.0), (0, 1): (-2.0, 4.0), (0, 0, 1, 1): (0.0, 16.0), (0, 0, 0): (-1.0, 8.0)}
    for mono, (a, c) in expect.items():
        i = b["index"][mono]
        if (mlo[i], mhi[i]) != (a, c):
            return False, f"interval box of monomial {mono} over [-1, 2]: ({mlo[i]}, {mhi[i]}), expected ({a}, {c})"
    return True, "; ".join(lines) + "; interval box holds on random points and is exact on hand cases"


def check_sampler(inst):
    """Contact samples must reproduce the scalar true step of M0, an independent path through the model."""
    md, modes, task, tr = inst
    rng = np.random.default_rng(5)
    limit = math.sin(md["theta_step_max"])
    k_rot = md["dt"] * md["c_v"] / md["tau_max"] ** 2
    mid = np.flatnonzero(tr["type_alive"] & (tr["type_kind"] == EDGE_KINDS.index("mid")))
    worst, count, seen = 0.0, 0, set()
    for ti in mid:
        tail, head = type_modes(tr, modes, ti)
        if not tail["contact"] or tail["name"] in seen:
            continue
        seen.add(tail["name"])
        for v in sample_true_steps(md, task, "mid", tail, head, 100, rng):
            if abs(k_rot * v[TAU]) > limit * (1.0 - 1e-12):
                continue
            slide = md["tangents"][tail["face"]] @ (v[[RX1, RY1]] - v[[RX0, RY0]])
            x1, tau = true_contact_step(md, v[:6], tail["face"], v[FN], v[FT], slide)
            worst = max(worst, float(np.max(np.abs(x1 - v[6:12]))), abs(tau - v[TAU]))
            count += 1
    return worst <= 1e-12 and count > 0, f"{count} samples over {len(seen)} contact modes match the scalar true step to {worst:.1e}"


def check_region_graph(inst, samples=2000):
    """Region copies: transition counts per layer (plan 5.2.2), and every true contact step has an allowed pair of copies (tex Prop 3.7)."""
    md, modes, task, tr = inst
    splits = md["theta_splits"]
    width = 2.0 * math.pi / splits
    parents = [m for m in modes if m["theta"] == 0]
    free = sum(transition_allowed(a, b) for a in parents if not a["contact"] for b in parents)
    contact = sum(transition_allowed(a, b) for a in parents if a["contact"] for b in parents)
    reach = min(splits, 1 + 2 * (1 + int(md["theta_step_max"] / width)))
    expect = splits * (free + contact * reach)
    if int(tr["allow"].sum()) != expect:
        return False, f"{int(tr['allow'].sum())} transitions per layer, expected {splits} x ({free} + {contact} x {reach}) = {expect}"
    plain = dict(md, theta_splits=1, lambda_splits=1)
    rng = np.random.default_rng(7)
    checked = 0
    for j in range(4):
        mode = dict(contact=True, face=j, ctype="stick", region=None, name=f"c{j}_stick")
        V = sample_true_steps(plain, task, "mid", mode, mode, samples, rng)
        th0, th1 = np.arctan2(V[:, S0], V[:, C0]), np.arctan2(V[:, S1], V[:, C1])
        for a, b in zip(th0, th1):
            la, lb = theta_label(float(a), splits), theta_label(float(b), splits)
            if not step_allowed(md, dict(mode, theta=la), dict(mode, theta=lb)):
                return False, f"a true step from angle {a:.4f} to {b:.4f} has no allowed copy (regions {la} -> {lb})"
            checked += 1
    return True, f"{expect} transitions per layer at {splits} theta regions; {checked} random true steps all have an allowed pair of copies"


def lift_identity_type(cfg, md, task, modes, tr, ti):
    """Largest error of one edge type's substitution, lifted rows, blocks, coupling and cost against their source polynomials at random points of its free variables, plus (rank of the raw equalities, rows kept)."""
    tail, head = type_modes(tr, modes, ti)
    rel = relax_edge_type(cfg, md, task, modes, tr, ti)
    basis = moment_basis(rel["kappa"], len(rel["free"]))
    W = np.random.default_rng(int(ti)).uniform(-1.0, 1.0, (8, len(rel["free"])))
    M = rank_one_moments(basis, W)
    V = full_points(rel, W)
    err = 0.0
    for _, p, _ in type_rows(md, task, modes, tr, ti):
        err = max(err, float(np.max(np.abs(poly_eval_batch(poly_substitute(p, rel["sub"]), W) - poly_eval_batch(p, V)))))
    raw, owner, mult = lift_rows(basis, rel["eq_polys"], True, rel["eq_degrees"])
    if raw.shape[0]:
        values = np.stack([poly_eval_batch(p, W) for p in rel["eq_polys"]])
        err = max(err, float(np.max(np.abs(raw @ M.T - values[owner] * M[:, mult].T))))
    if rel["G"].shape[0]:
        values = np.stack([poly_eval_batch(p, W) for p in rel["ge_polys"]])
        err = max(err, float(np.max(np.abs(rel["G"] @ M.T - values))))
    err = max(err, float(np.max(np.abs(M @ rel["c"] - poly_eval_batch(edge_cost(cfg, tail, head), V)))), coupling_error(rel, M, V))
    for block in rel["psd"]:
        err = max(err, float(np.max(block_identity_error(block, block_operator(block, rel["n"]), M, W)[0])))
    rank = None
    if rel["kappa"] == 1:
        rank = 0
        if raw.shape[0]:
            d = raw.toarray()
            rank = int(np.linalg.matrix_rank(d / np.linalg.norm(d, axis=1, keepdims=True), tol=1e-10))
    return err, rank, rel["Aeq"].shape[0]


def check_lift_identity(cfg, inst, max_types=None):
    """Negative control for the rows (plan 5.4): at random points, where no quadratic row vanishes, each lifted row, block and cost must equal the polynomial it came from. Rows that are trivially zero would pass the validity check but fail here. At kappa 1, the rows kept must span the raw rows."""
    md, modes, task, tr = inst
    prebuild_bases(cfg, md, modes, task, tr)
    alive = np.flatnonzero(tr["type_alive"])
    if max_types:
        alive = alive[np.linspace(0, alive.size - 1, max_types).astype(int)]
    out = pool_map(lambda ti: lift_identity_type(cfg, md, task, modes, tr, int(ti)), alive, cfg["build"])
    worst = max(r[0] for r in out)
    short = sum(1 for _, rank, kept in out if rank is not None and rank != kept)
    if short:
        return False, f"{short} edge types keep equality rows that do not span the raw ones"
    return worst <= 1e-10, f"{len(out)} edge types: lifted rows, blocks and costs match at random points to {worst:.1e}" + (
        "; the rows kept span the raw rows" if int(cfg["relaxation"]["order"]) == 1 else "")


def substitution_map(basis_full, basis, sub):
    """Sparse T with m_full = T m_reduced: each monomial of the full variables written over the reduced ones. It sends a reduced moment vector to the full one that satisfies every eliminated row (plan 5.1.7)."""
    rows, cols, vals = [], [], []
    for a, mono in enumerate(basis_full["monos"]):
        term = {(): 1.0}
        for i in mono:
            term = poly_mul(term, sub[i])
        for k, c in term.items():
            rows.append(a)
            cols.append(basis["index"][k])
            vals.append(c)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(basis_full["monos"]), len(basis["monos"])))


def block_matrix(block, m):
    """A PSD map's symmetric matrix at one vector m."""
    s = int(block["size"])
    out = np.zeros((s, s))
    np.add.at(out, (block["row"], block["col"]), block["coef"] * m[block["mom"]])
    return out + out.T - np.diag(np.diag(out))


def elimination_type(cfg, cfg_full, md, task, modes, tr, ti):
    """One edge type's reduced model against its full one, read through T (plan 5.1.7). Every value should be about zero, except `implied`, which should not be negative."""
    rel, full = relax_edge_type(cfg, md, task, modes, tr, ti), relax_edge_type(cfg_full, md, task, modes, tr, ti)
    basis, basis_full = moment_basis(rel["kappa"], len(rel["free"])), moment_basis(rel["kappa"])
    T = substitution_map(basis_full, basis, rel["sub"])
    out = dict(cost=float(np.max(np.abs(T.T @ full["c"] - rel["c"]))))
    out["coupling"] = max(float(np.max(np.abs((a @ T - b).toarray()))) for a, b in ((full["tail_map"], rel["tail_map"]), (full["head_map"], rel["head_map"])))
    # every full equality, including the eliminated ones, lies in the span of the reduced ones
    mapped = (lift_rows(basis_full, [p for s, p, _ in type_rows(md, task, modes, tr, ti) if s == "eq"], True)[0] @ T).toarray()
    if rel["Aeq"].shape[0]:
        q, r, _ = scipy.linalg.qr(rel["Aeq"].toarray().T, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        q = q[:, :int(np.sum(diag > 1e-10 * diag[0]))]
        mapped = mapped - (mapped @ q) @ q.T
    out["eq_span"] = float(np.max(np.abs(mapped), initial=0.0))
    # B_full(T m) = W B(m) W^T with W of full column rank, at any m, so one block is PSD exactly when the other is
    m = np.random.default_rng(int(ti)).normal(size=rel["n"])
    m_full = T @ m
    blocks = {b["tag"]: b for b in full["psd"]}
    out["blocks"] = 0.0
    for block in rel["psd"]:
        fb = blocks[block["tag"]]
        W = T[:int(fb["size"]), :int(block["size"])]
        out["blocks"] = max(out["blocks"], float(np.max(np.abs(block_matrix(fb, m_full) - W @ (W @ block_matrix(block, m)).T))))
    # the reduced box's image stays in the full box, and the two agree on the monomials they share, up to the LP boxes' pad
    high, low = T.maximum(0) @ rel["hi"] + T.minimum(0) @ rel["lo"], T.maximum(0) @ rel["lo"] + T.minimum(0) @ rel["hi"]
    shared = [basis_full["index"][tuple(rel["free"][j] for j in mono)] for mono in basis["monos"]]
    out["box"] = max(float(np.max(full["lo"] - low)), float(np.max(high - full["hi"])),
                     float(np.max(np.abs(rel["lo"] - full["lo"][shared]))), float(np.max(np.abs(rel["hi"] - full["hi"][shared]))))
    if rel["kappa"] == 1:
        # same dimension (136 - 51 = 91 - 6 on a contact edge), and every full inequality is implied: an LP per row over the reduced rows and box
        out["dim"] = abs((full["n"] - full["Aeq"].shape[0]) - (rel["n"] - rel["Aeq"].shape[0]))
        A, G = rel["Aeq"].toarray(), rel["G"].toarray()
        kw = dict(A_ub=-G[:, 1:] if G.shape[0] else None, b_ub=G[:, 0] if G.shape[0] else None, A_eq=A[:, 1:] if A.shape[0] else None, b_eq=-A[:, 0] if A.shape[0] else None,
                  bounds=list(zip(rel["lo"][1:], rel["hi"][1:])), method="highs", options=LP_OPTIONS)
        out["implied"] = float("inf")
        for g in (full["G"] @ T).toarray():
            res = linprog(g[1:], **kw)
            if res.status != 0:
                raise RuntimeError(f"edge type {ti}: implied-row LP failed with status {res.status}")
            out["implied"] = min(out["implied"], float(res.fun + g[0]))
    return out


def check_elimination(cfg, inst, max_types=None):
    """The reduced model is the full one (plan 5.1.7), checked algebraically, without a solver: same cost and coupling, same equalities, same PSD blocks through W, same box, and at kappa 1 the same dimension and every full inequality implied."""
    md, modes, task, tr = inst
    cfg_full = deep_merge(cfg, {"relaxation": {"eliminate_linear": False}})
    prebuild_bases(cfg, md, modes, task, tr)
    prebuild_bases(cfg_full, md, modes, task, tr)
    alive = np.flatnonzero(tr["type_alive"])
    if max_types:
        alive = alive[np.linspace(0, alive.size - 1, max_types).astype(int)]
    results = pool_map(lambda ti: elimination_type(cfg, cfg_full, md, task, modes, tr, int(ti)), alive, cfg["build"])
    limits = dict(cost=1e-11, coupling=1e-11, eq_span=1e-9, blocks=1e-9, box=1e-5)
    if int(cfg["relaxation"]["order"]) == 1:
        limits["dim"] = 0
    worst = {k: max(r[k] for r in results) for k in limits}
    bad = [k for k in limits if worst[k] > limits[k]]
    msg = f"{len(results)} edge types; worst " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items())
    if int(cfg["relaxation"]["order"]) == 1:
        implied = min(r["implied"] for r in results)
        msg += f", slack of the full inequalities at least {implied:.1e}"
        if implied < -1e-7:
            bad.append("implied")
    return not bad, msg + (f"; OUT OF TOLERANCE: {bad}" if bad else "")


def check_validity(cfg, instances, record, label):
    """Plan 5.4 on one relaxation: every alive edge type of the first task, and the start and goal types of the others (the mid types do not depend on the task)."""
    results = []
    for i, inst in enumerate(instances.values()):
        results += validate_relaxation(cfg, *inst, kinds=None if i == 0 else ("start", "goal"))
    ok, msg, worst = validity_summary(results, cfg["relaxation"]["validity"])
    record[label] = dict(types=len(results), samples=sum(r["samples"] for r in results), worst={k: [v[0], v[1]] for k, v in worst.items()},
                         no_true_point=[r["name"] for r in results if r["samples"] == 0])
    return ok, msg


def check_negative_controls(cfg, inst):
    """The validity check must be able to fail: true samples moved off the true set must be flagged (plan 5.4). An eliminated row shows up as a distance from the edge's affine set."""
    md, modes, task, tr = inst
    mid = np.flatnonzero(tr["type_alive"] & (tr["type_kind"] == EDGE_KINDS.index("mid")))
    tail_of = lambda ti: modes[int(tr["type_tail"][ti])]
    ti_contact = int(next(t for t in mid if tail_of(t)["contact"]))
    ti_free = int(next(t for t in mid if not tail_of(t)["contact"]))
    sdp = cfg["relaxation"]["type"] == "sdp"
    line_col = RX0 if abs(md["normals"][tail_of(ti_contact)["face"]][0]) > 0.5 else RY0
    cases = [("torque", ti_contact, TAU, 1e-3), ("translation", ti_contact, PX1, 0.2), ("contact line", ti_contact, line_col, 1e-3),
             ("friction", ti_contact, FT, 0.5), ("static slider", ti_free, PX1, 1e-3), ("zero force", ti_free, FN, 1e-3)]
    if sdp:
        cases.append(("unit norm", ti_contact, C0, 0.01))
        if int(cfg["relaxation"]["order"]) >= 2:
            cases.append(("localizing block", ti_contact, FN, 1.0))
    rng = np.random.default_rng(13)
    lines = []
    for name, ti, col, delta in cases:
        tail, head = type_modes(tr, modes, ti)
        kind = EDGE_KINDS[int(tr["type_kind"][ti])]
        V = sample_true_steps(md, task, kind, tail, head, 64, rng)
        if not V.shape[0]:
            return False, f"{name}: no true samples on {tail['name']} -> {head['name']}"
        rel = relax_edge_type(cfg, md, task, modes, tr, ti)
        V = V.copy()
        V[:, col] += delta
        M = lift_points(rel, md, task, kind, tail, head, V)
        if name == "localizing block":
            values = dict(psd=max(psd_violation(b, block_operator(b, rel["n"]), M, V[:, rel["free"]], 0)[0] for b in rel["psd"]))
        else:
            values = {k: v[0] for k, v in relaxation_violations(rel, M).items()}
            values["affine"] = affine_error(rel, V)
        key = max(values, key=values.get)
        if values[key] <= 1e-6:
            return False, f"{name}: moving column {VAR_NAMES[col]} by {delta} went unflagged (largest violation {values[key]:.1e})"
        lines.append(f"{name} {values[key]:.1e} ({key})")
    return True, "flagged: " + ", ".join(lines)


def check_path_lift(cfg, inst):
    """Along the hand push: every lifted step satisfies its model, each head's coupled coordinates equal the next tail's, and the linear costs sum to J_hand (tex 2.1, Thm 2.3)."""
    md, modes, task, tr = inst
    steps = hand_push(cfg, md, task, modes)
    worst, viol, J, J_true, prev = 0.0, 0.0, 0.0, 0.0, None
    for (kind, tail, head, v), e in zip(steps, path_edge_ids(tr, modes, steps)):
        rel = relax_edge_type(cfg, md, task, modes, tr, int(tr["edge_type"][e]))
        M = lift_points(rel, md, task, kind, tail, head, v[None])
        viol = max(viol, max(value for value, _ in relaxation_violations(rel, M).values()), affine_error(rel, v[None]))
        if prev is not None:
            worst = max(worst, float(np.max(np.abs(prev - rel["tail_map"] @ M[0]))))
        prev = rel["head_map"] @ M[0]
        J += float(M[0] @ rel["c"])
        J_true += poly_eval(edge_cost(cfg, tail, head), v)
    ok = worst <= 1e-12 and viol <= 1e-9 and abs(J - J_true) <= 1e-12
    return ok, f"{len(steps)} steps, q = {prev.size}: coupling mismatch {worst:.1e}, row violation {viol:.1e}, linear cost {J:.6f} vs J_hand {J_true:.6f}"


def check_bands(cfg, inst):
    """Prints the band report. The bands of the alive types stay under the worst-case formula, the sizing is minimal, and the translation band is first order in the region size (tex Remark 3.2)."""
    md, modes, task, tr = inst
    band_report(cfg, md, modes, task, tr)
    widths = band_widths(cfg, md, modes, task, tr)
    splits, cells = md["theta_splits"], md["lambda_splits"]
    e_tr, e_tau, e_rot = worst_bands(md, splits, cells)
    slack = 1.0 + 1e-5                                          # the LP boxes carry a small outward pad
    top = max(widths["trans_x"] + widths["trans_y"])
    if top > e_tr * slack or max(widths["torque"]) > e_tau * slack or abs(max(widths["rot_rel"]) - e_rot) > 1e-12:
        return False, f"bands exceed the worst-case formula: translation {top:.3e} vs {e_tr:.3e}, torque {max(widths['torque']):.3e} vs {e_tau:.3e}"
    target = cfg["relaxation"]["size_regions"]
    s, l = size_regions(md, target)
    if s is None or l is None:
        return False, f"max_splits cannot meet the sizing targets: {s}, {l}"
    if worst_bands(md, s, 1)[0] > float(target["trans"]) or (s > 3 and worst_bands(md, s - 1, 1)[0] <= float(target["trans"])):
        return False, f"theta_splits {s} is not the smallest that meets the target"
    ratio = worst_bands(md, 512, 1)[0] / worst_bands(md, 256, 1)[0]             # fine regions: the first-order term dominates
    if not 0.47 <= ratio <= 0.53:
        return False, f"doubling theta_splits from 256 scales the translation band by {ratio:.3f}, expected about 0.5 (first order)"
    return True, f"bands within the formula (translation {top:.3e} m, torque {max(widths['torque']):.3e} N m, rotation {e_rot:.3e} rad); doubling fine regions scales translation by {ratio:.3f}; sizing target needs {s} theta and {l} lambda splits"


def check_outer_pruning(cfg, inst):
    """Run B's own LP pruning: an edge type that holds a true sample must never be pruned."""
    md, modes, task, tr = inst
    light = deep_merge(cfg, {"relaxation": {"validity": {"samples": 200}}})
    samples = {r["type"]: r["samples"] for r in validate_relaxation(light, md, modes, task, tr)}
    copy_tr = dict(tr, type_alive=tr["type_alive"].copy())
    pruned = prune_outer(cfg, md, modes, task, copy_tr)
    wrong = [int(ti) for ti in pruned if samples.get(int(ti), 0) > 0]
    if wrong:
        return False, f"{len(wrong)} edge types with true samples were pruned, e.g. {wrong[:5]}"
    return True, f"{len(pruned)} of {len(samples)} LP-alive edge types pruned, none holding a true sample; {int(copy_tr['edge_alive'].sum())} of {len(tr['edge_tail'])} edges kept"


def selftest_m1(cfg):
    """Milestone M1 (plan 5.4, 17): the outer models on the full mode set, for every relaxation the run matrix uses."""
    st = cfg["selftest"]
    base = deep_merge(cfg, {"trellis": {"faces": [0, 1, 2, 3], "regions": "all"},
                            "relaxation": {"type": "sdp", "order": 1, "eliminate_linear": True, "regions": {"theta_splits": 1, "lambda_splits": 1},
                                           "taylor_mode": "band", "cost_model": "tangent_planes"}})
    sdp = {k: deep_merge(base, {"relaxation": {"order": k}}) for k in (1, 2, 3)}
    full_form = deep_merge(sdp[1], {"relaxation": {"eliminate_linear": False}})
    regioned = deep_merge(base, {"relaxation": {"regions": {"theta_splits": int(st["theta_splits"])}}})
    taylor = deep_merge(regioned, {"relaxation": {"type": "taylor"}})
    mcc = deep_merge(taylor, {"relaxation": {"taylor_mode": "mccormick"}})
    soc = deep_merge(taylor, {"relaxation": {"cost_model": "soc"}})
    t0 = time.perf_counter()
    build_all = lambda c: {t: build_instance(deep_merge(c, {"task": t})) for t in c["tasks"]}
    plain, split = build_all(sdp[1]), build_all(regioned)
    print(f"built {len(plain)} tasks without and with {st['theta_splits']} theta regions in {time.perf_counter() - t0:.1f}s; "
          f"with regions: {json.dumps(summary(split[st['task']][3], split[st['task']][1], split[st['task']][2]))}", flush=True)
    for k in sdp:
        moment_basis(k)                                         # built once, before any fork
    i1, i2 = plain[st["task"]], split[st["task"]]
    pair = {t: split[t] for t in (st["task"], base["task"])}    # the McCormick and SOC variants: fewer tasks
    record = {}

    def tables():
        record["sizes"] = {
            "sdp_k1_full": relaxation_table(full_form, *i1, "SDP kappa 1, full form (for comparison)"),
            "sdp_k1": relaxation_table(sdp[1], *i1, "SDP kappa 1"),
            "sdp_k2": relaxation_table(sdp[2], *i1, "SDP kappa 2"),
            "sdp_k3": relaxation_table(sdp[3], *i1, "SDP kappa 3"),
            "sdp_k1_regions": relaxation_table(regioned, *i2, "SDP kappa 1 with theta regions (A-reg)"),
            "taylor": relaxation_table(taylor, *i2, "Taylor (B)"),
        }
        return True, "printed above"

    checks = [
        ("M1 moment basis: sizes, order, index tables, interval box", check_moment_basis),
        ("M1 sampler against the scalar true step", lambda: check_sampler(i1)),
        ("M1 region copies hold every true step", lambda: check_region_graph(i2)),
        ("M1 lifted rows match their polynomials, kappa 1", lambda: check_lift_identity(sdp[1], i1)),
        ("M1 lifted rows match their polynomials, kappa 2", lambda: check_lift_identity(sdp[2], i1)),
        ("M1 lifted rows match their polynomials, kappa 3 (3 edge types)", lambda: check_lift_identity(sdp[3], i1, 3)),
        ("M1 elimination leaves the model unchanged, kappa 1", lambda: check_elimination(sdp[1], i1)),
        ("M1 elimination leaves the model unchanged, kappa 2 (12 edge types)", lambda: check_elimination(sdp[2], i1, 12)),
        ("M1 validity 5.4, SDP kappa 1, full form", lambda: check_validity(full_form, plain, record, "sdp_k1_full")),
        ("M1 validity 5.4, SDP kappa 1", lambda: check_validity(sdp[1], plain, record, "sdp_k1")),
        ("M1 validity 5.4, SDP kappa 2", lambda: check_validity(sdp[2], plain, record, "sdp_k2")),
        ("M1 validity 5.4, SDP kappa 3", lambda: check_validity(sdp[3], plain, record, "sdp_k3")),
        ("M1 validity 5.4, SDP kappa 1 with theta regions (A-reg)", lambda: check_validity(regioned, split, record, "sdp_k1_regions")),
        ("M1 validity 5.4, Taylor bands (B)", lambda: check_validity(taylor, split, record, "taylor_band")),
        ("M1 validity 5.4, Taylor McCormick", lambda: check_validity(mcc, pair, record, "taylor_mccormick")),
        ("M1 validity 5.4, Taylor SOC costs", lambda: check_validity(soc, pair, record, "taylor_soc")),
        ("M1 negative controls, SDP kappa 1", lambda: check_negative_controls(sdp[1], i1)),
        ("M1 negative controls, SDP kappa 2", lambda: check_negative_controls(sdp[2], i1)),
        ("M1 negative controls, Taylor", lambda: check_negative_controls(taylor, i2)),
        ("M1 coupling and cost along the hand push, SDP kappa 1", lambda: check_path_lift(sdp[1], i1)),
        ("M1 coupling and cost along the hand push, SDP kappa 2", lambda: check_path_lift(sdp[2], i1)),
        ("M1 coupling and cost along the hand push, Taylor", lambda: check_path_lift(taylor, i2)),
        ("M1 Taylor bands and region sizing", lambda: check_bands(taylor, i2)),
        ("M1 Taylor feasibility pruning", lambda: check_outer_pruning(taylor, i2)),
        ("M1 sizes per group of edge types (plan 5.1.7)", tables),
    ]
    ok, results = run_checks(checks)
    out = ROOT / cfg["output"]["root"] / "selftest" / "m1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, results=results, record=record), indent=2))
    print(f"\n{'ALL M1 CHECKS PASSED' if ok else 'SOME M1 CHECKS FAILED'} in {time.perf_counter() - t0:.0f}s; details in {out}", flush=True)
    return ok


def cone_convention_value(cfg, solver):
    """min <C, X> subject to trace X = 1, X PSD, whose value is C's smallest eigenvalue: a wrong triangle order or scaling in svec_position gives another value. Returns (solver's value, exact value)."""
    C = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.5], [0.0, 1.5, 5.0]])
    iu, ju = np.triu_indices(3)
    pos = svec_position(iu, ju, 3, solver)
    c, trace = np.zeros(6), np.zeros(6)
    c[pos] = C[iu, ju] * np.where(iu == ju, 1.0, math.sqrt(2.0))
    trace[pos[iu == ju]] = 1.0
    A = sp.vstack([sp.csr_matrix(trace), -sp.identity(6, format="csr")], format="csc")
    prob = dict(solver=solver, A=A, b=np.r_[1.0, np.zeros(6)], c=c, nz=1, nl=0, sizes=[3])
    return float(c @ solve_conic(cfg, prob)["x"]), float(np.linalg.eigvalsh(C)[0])


def check_cone_conventions(cfg):
    """Each solver's PSD triangle convention, on a problem with a known answer."""
    solvers = ["clarabel", "scs"]
    results = pool_map(lambda s: cone_convention_value(cfg, s), solvers, cfg["build"])
    worst = max(abs(a - b) for a, b in results)
    return worst <= 1e-4, ", ".join(f"{s} {a:.6f}" for s, (a, _) in zip(solvers, results)) + f", exact {results[0][1]:.6f}"


def check_reference_assembly(cfg, inst, solver, record, key):
    """The assembled program holds the hand push: its lift is feasible and costs J_hand. Dropping one step must break conservation, so the check can fail. Stores J_hand in record[key]."""
    md, modes, task, tr = inst
    rels = edge_models(cfg, md, task, modes, tr)
    prob = assemble_conic(cfg, tr, rels, solver)
    steps = hand_push(cfg, md, task, modes)
    ids = path_edge_ids(tr, modes, steps)
    where = {int(e): i for i, e in enumerate(prob["alive"])}
    x = np.zeros(prob["c"].size)
    J_hand = 0.0
    for (kind, tail, head, v), e in zip(steps, ids):
        i = where[int(e)]
        x[prob["offs"][i]:prob["offs"][i + 1]] = lift_points(rels[int(tr["edge_type"][e])], md, task, kind, tail, head, v[None])[0]
        J_hand += poly_eval(edge_cost(cfg, tail, head), v)
    eq, ge, psd = conic_residuals(prob, x)
    J = float(prob["c"] @ x)
    i = where[int(ids[len(ids) // 2])]
    broken = x.copy()
    broken[prob["offs"][i]:prob["offs"][i + 1]] = 0.0
    eq_broken = conic_residuals(prob, broken)[0]
    record[key] = J_hand
    ok = eq <= 1e-9 and ge <= 1e-9 and psd <= 1e-9 and abs(J - J_hand) <= 1e-10 and eq_broken >= 0.5
    return ok, (f"{prob['c'].size} variables, {prob['A'].shape[0]} rows, {len(prob['sizes'])} PSD blocks: hand push residuals eq {eq:.1e}, ge {ge:.1e}, psd {psd:.1e}; "
                f"cost {J:.6f} vs J_hand {J_hand:.6f}; dropping a step breaks conservation by {eq_broken:.2f}")


def selftest_m2(cfg):
    """Milestone M2 (plan 10.4, 17): the reference solve on the small instance. J_rel stays under J_hand, the reduced and full forms agree, and the table shows how tight each relaxation is."""
    st, m2 = cfg["selftest"], cfg["selftest"]["m2"]
    t0 = time.perf_counter()
    small = deep_merge(cfg, {"trellis": {"N": st["N"], "faces": st["faces"], "regions": st["regions"]},
                             "relaxation": {"type": "sdp", "order": 1, "eliminate_linear": True, "regions": {"theta_splits": 1, "lambda_splits": 1},
                                            "taylor_mode": "band", "cost_model": "tangent_planes"}})
    regioned = deep_merge(small, {"relaxation": {"regions": {"theta_splits": int(st["theta_splits"])}}})
    reduced, full_form = "SDP order 1", "SDP order 1, full form"
    models = {reduced: small, full_form: deep_merge(small, {"relaxation": {"eliminate_linear": False}}),
              "Taylor bands": deep_merge(regioned, {"relaxation": {"type": "taylor"}}),
              "Taylor McCormick": deep_merge(regioned, {"relaxation": {"type": "taylor", "taylor_mode": "mccormick"}})}
    tasks = [str(t) for t in m2["tasks"]]
    plain = {t: build_instance(deep_merge(small, {"task": t})) for t in tasks}
    split = {t: build_instance(deep_merge(regioned, {"task": t})) for t in tasks}

    def instance(model, t):
        """The instance of one model and task; run B's edge types are pruned by its own rows, on a private copy."""
        mcfg = models[model]
        if mcfg["relaxation"]["type"] != "taylor":
            prebuild_bases(mcfg, *plain[t])
            return plain[t]
        md, modes, task, tr = split[t]
        own = dict(tr, type_alive=tr["type_alive"].copy())
        prune_outer(mcfg, md, modes, task, own)
        return md, modes, task, own

    jobs = [(models[model], instance(model, t), f"{t}: {model}") for t in tasks for model in models]
    short = deep_merge(small, {"task": m2["order2"]["task"], "trellis": {"N": int(m2["order2"]["N"])}, "reference": {"time_limit_s": float(m2["order2"]["time_limit_s"])}})
    short_two = deep_merge(short, {"relaxation": {"order": 2}})
    short_inst = build_instance(short)
    prebuild_bases(short, *short_inst)
    prebuild_bases(short_two, *short_inst)
    short_labels = [f"{short['task']}, N = {short['trellis']['N']}: SDP order {k}" for k in (1, 2)]
    print(f"built the M2 instances in {time.perf_counter() - t0:.1f}s; the solves print as they finish", flush=True)
    results, hand = {}, {}

    def run_jobs(batch):
        for r in pool_map(lambda i: reference_job(*batch[i]), range(len(batch)), cfg["build"]):
            results[r["label"]] = r

    def solve_small():
        run_jobs(jobs)
        errors = [r["label"] for r in results.values() if "error" in r]
        return not errors, f"{len(jobs)} solves" + (f"; errors on {errors}" if errors else "")

    def below_hand():
        tol = float(m2["tol_hand"])
        J_hand = hand[st["task"]]
        lines, ok = [], True
        for model in models:
            r = results[f"{st['task']}: {model}"]
            if r["solved"]:
                ok &= r["J"] <= J_hand + tol
            elif model != full_form:                # the full form has no interior, and may stop early: reported, not required
                ok = False
            lines.append(f"{model} {'not solved' if r['J'] is None else format(r['J'], '.6f')} ({r.get('status', 'error')})")
        return ok, f"J_hand {J_hand:.6f}; " + ", ".join(lines)

    def forms_agree():
        tol = float(m2["tol_forms"])
        lines, ok = [], True
        for t in tasks:
            a, b = results[f"{t}: {reduced}"], results[f"{t}: {full_form}"]
            if a["infeasible"] or b["infeasible"]:
                ok &= a["infeasible"] and b["infeasible"]
                lines.append(f"{t}: {a.get('status')} and {b.get('status')}")
                continue
            ok &= a["solved"]
            if a["J"] is None or b["J"] is None:
                ok = False
                lines.append(f"{t}: no value to compare")
                continue
            gap = abs(a["J"] - b["J"])
            if b["solved"]:
                ok &= gap <= tol * max(1.0, abs(a["J"]))
            lines.append(f"{t}: reduced {a['J']:.8f} in {a['seconds']:.1f} s, full {b['J']:.8f} in {b['seconds']:.1f} s ({b['status']}), difference {gap:.1e}")
        return ok, "; ".join(lines)

    def order_two():
        run_jobs([(short, short_inst, short_labels[0]), (short_two, short_inst, short_labels[1])])
        one, two = results[short_labels[0]], results[short_labels[1]]
        J_hand = hand["order2"]
        if not one["solved"] or two["J"] is None:
            return False, f"order 1 {one.get('status', one.get('error'))}, order 2 {two.get('status', two.get('error'))}"
        msg = f"J_hand {J_hand:.6f}; order 1 {one['J']:.6f}; order 2 {two['J']:.6f} ({two['solver']}, {two['status']}, {two['iters']} iterations, {two['seconds']:.0f} s)"
        if not two["solved"]:
            return True, msg + "; order 2 did not converge inside the time limit, so its value is approximate and not checked"
        slack = 1e-3                                # SCS's accuracy
        return one["J"] - slack <= two["J"] <= J_hand + slack, msg

    checks = [
        ("M2 PSD triangle conventions, on a known answer", lambda: check_cone_conventions(small)),
        ("M2 assembly holds the hand push, SDP order 1", lambda: check_reference_assembly(small, plain[st["task"]], "clarabel", hand, st["task"])),
        ("M2 assembly holds the hand push, SDP order 1, full form", lambda: check_reference_assembly(models[full_form], plain[st["task"]], "clarabel", {}, "full")),
        ("M2 assembly holds the hand push, Taylor", lambda: check_reference_assembly(models["Taylor bands"], split[st["task"]], "clarabel", {}, "taylor")),
        ("M2 assembly holds the hand push, Taylor McCormick", lambda: check_reference_assembly(models["Taylor McCormick"], split[st["task"]], "clarabel", {}, "mccormick")),
        ("M2 assembly holds the hand push, SDP order 2", lambda: check_reference_assembly(short_two, short_inst, "scs", hand, "order2")),
        ("M2 reference solves on the small instance", solve_small),
        ("M2 J_rel <= J_hand", below_hand),
        ("M2 the reduced and full forms give the same J_rel", forms_agree),
        ("M2 SDP order 2 on a shorter horizon", order_two),
    ]
    ok, checked = run_checks(checks)
    print(f"\nreference values, delta = {cfg['solver']['delta']}", flush=True)
    for r in results.values():
        print(reference_line(r), flush=True)
    out = ROOT / cfg["output"]["root"] / "selftest" / "m2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, results=checked, solves=list(results.values()), J_hand=hand), indent=2))
    print(f"\n{'ALL M2 CHECKS PASSED' if ok else 'SOME M2 CHECKS FAILED'} in {time.perf_counter() - t0:.0f}s; details in {out}", flush=True)
    return ok


def selftest(cfg):
    """The milestone checks in order; a milestone runs only after the one before it passes (plan 0)."""
    return selftest_m0(cfg) and selftest_m1(cfg) and selftest_m2(cfg)


def selftest_m0(cfg):
    """Milestone M0 checks on the tiny instance (plan 12.3) and on the full mode set."""
    st = cfg["selftest"]
    tiny_cfg = deep_merge(cfg, {"task": st["task"], "trellis": {"N": st["N"], "faces": st["faces"], "regions": st["regions"]}})
    full_cfg = deep_merge(cfg, {"task": st["task"], "trellis": {"faces": [0, 1, 2, 3], "regions": "all"}})
    t0 = time.perf_counter()
    tiny = build_instance(tiny_cfg)
    full = build_instance(full_cfg)
    for label, inst in (("tiny", tiny), ("full", full)):
        print(f"{label}: {json.dumps(summary(inst[3], inst[1], inst[2]))}", flush=True)
    print(f"built both instances in {time.perf_counter() - t0:.1f}s", flush=True)
    md, rng, record = full[0], np.random.default_rng(0), {}
    checks = [
        ("M0 geometry and limit surface", lambda: check_geometry(md, cfg)),
        ("M0 friction sign (force on the slider along the pusher's slide)", lambda: check_friction_sign(md)),
        ("M0 torque sign", lambda: check_torque_sign(md)),
        ("M0 true steps satisfy the constraint table", lambda: check_true_steps(md, full[1], rng)),
        ("M0 constraint table degrees and rows", lambda: check_table(md, full[2], full[1], full[3])),
        ("M0 mode graph (plan 4.4)", lambda: check_mode_graph(full[1])),
        ("M0 trellis counts", lambda: check_trellis_counts(full[3], full[1])),
        ("M0 flow and coupling signs", lambda: check_flow_signs(tiny_cfg, *tiny)),
        ("M0 pruning soundness, tiny", lambda: check_pruning(tiny_cfg, *tiny)),
        ("M0 pruning soundness, full", lambda: check_pruning(full_cfg, *full)),
        ("M0 hand-scripted push, tiny", lambda: check_hand_push(tiny_cfg, *tiny, record)),
        ("M0 hand-scripted push, full", lambda: check_hand_push(full_cfg, *full, record)),
    ]
    ok, results = run_checks(checks)
    out = ROOT / cfg["output"]["root"] / "selftest" / "m0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, results=results, record=record,
                                   tiny=summary(tiny[3], tiny[1], tiny[2]), full=summary(full[3], full[1], full[2])), indent=2))
    print(f"\n{'ALL M0 CHECKS PASSED' if ok else 'SOME M0 CHECKS FAILED'}; details in {out}", flush=True)
    return ok


# ===== MAIN =====

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="run config: a name in configs/runs/ or a path")
    ap.add_argument("--task", default=None, help="task name, overriding the config")
    ap.add_argument("--set", action="append", default=[], help="override one key, e.g. --set trellis.N=16")
    ap.add_argument("--selftest", action="store_true", help="run the milestone checks")
    ap.add_argument("--build", action="store_true", help="build, prune, validate and save the edge models of the run")
    ap.add_argument("--band_report", action="store_true", help="print the Taylor bands and the region sizing table of the run")
    ap.add_argument("--reference", action="store_true", help="solve the run's whole relaxation on the CPU for J_rel (small instances)")
    args = ap.parse_args()
    cfg = load_config(args.run, args.set)
    if args.task:
        cfg["task"] = args.task
    if args.selftest:
        sys.exit(0 if selftest(cfg) else 1)
    if args.band_report:
        band_report(cfg, *build_instance(cfg))
        return
    if args.reference:
        md, modes, task, tr = build_instance(cfg)
        build_relaxations(cfg, md, modes, task, tr)     # stops on a validity failure
        result = reference_job(cfg, (md, modes, task, tr), cfg["run_name"])
        path = ROOT / cfg["output"]["root"] / cfg["run_name"] / task["name"] / str(cfg["seed"]) / "reference.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(f"saved {path}")
        return
    if args.build:
        md, modes, task, tr = build_instance(cfg)
        rels = build_relaxations(cfg, md, modes, task, tr)
        relaxation_table(cfg, md, modes, task, tr, cfg["run_name"])
        path = ROOT / cfg["output"]["root"] / cfg["run_name"] / task["name"] / str(cfg["seed"]) / "problem.npz"
        save_problem(path, cfg, md, modes, task, tr, rels)
        print(json.dumps(summary(tr, modes, task), indent=2))
        print(f"saved {path}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
