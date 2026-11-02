"""Planar pushing on a time-layered trellis (plan sections 4-6): geometry and modes, the true per-edge constraints as one polynomial table, the true-model step, trellis assembly with pruning, and the milestone self-tests.

Math: docs/trajopt_detailed.tex (source of truth). Implementation: IMPLEMENTATION_PLAN.md.

    python push_model.py --selftest            # milestone checks implemented so far (M0)
    python push_model.py --run A --build       # build, prune and save one run's trellis
"""
import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

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
    """Geometry, limit surface and limits of plan 4.1 as plain floats and arrays."""
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
                p_box=np.asarray(m["workspace"]["p_box"], float), r_ws=float(m["workspace"]["r_ws"]))


def build_modes(cfg):
    """Contact modes (configured faces x friction types) first, then the configured non-contact regions (plan 4.3)."""
    tr = cfg["trellis"]
    faces = [int(j) for j in tr.get("faces", [0, 1, 2, 3])]
    use_all = tr.get("regions", "all") == "all"
    regions = list(REGION_NAMES) if use_all else list(tr["regions"])
    unknown = [r for r in regions if r not in REGION_NAMES] + [c for c in tr["contact_types"] if c not in CONTACT_TYPES]
    if unknown:
        raise ValueError(f"unknown regions or contact types: {unknown}")
    if use_all and int(tr.get("noncontact_regions", 8)) != 8:
        raise ValueError("trellis.noncontact_regions must be 8 when trellis.regions is 'all'")
    modes = [dict(name=f"c{j}_{ct}", contact=True, face=j, ctype=ct, region=None)
             for j in faces for ct in tr["contact_types"]]
    modes += [dict(name=r, contact=False, face=None, ctype=None, region=r) for r in regions]
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


def edge_box(md, tail):
    """Variable box of plan 4.5 item 9, before LP tightening; it holds every true point of the edge."""
    lo, hi = np.empty(NV), np.empty(NV)
    for off in (0, 6):
        lo[off + 0], hi[off + 0] = md["p_box"][0]
        lo[off + 1], hi[off + 1] = md["p_box"][1]
        lo[off + 2:off + 4], hi[off + 2:off + 4] = -1.0, 1.0
        lo[off + 4:off + 6], hi[off + 4:off + 6] = -md["r_ws"], md["r_ws"]
    if tail["contact"]:
        tb = md["tau_bound"][tail["face"]]
        lo[FN:], hi[FN:] = (0.0, -md["ft_max"], -tb), (md["fn_max"], md["ft_max"], tb)
    else:
        lo[FN:], hi[FN:] = 0.0, 0.0
    return lo, hi


def edge_cost(cfg, tail, head):
    """Quadratic edge cost of plan 4.6 as a polynomial over v (the same for all runs)."""
    w = cfg["costs"]
    diff = lambda a, b: lin({a: 1.0, b: -1.0})
    sq = lambda a, b: poly_mul(diff(a, b), diff(a, b))
    switch = head is not None and head["name"] != tail["name"]
    return poly_add(poly_scale(poly_add(sq(PX1, PX0), sq(PY1, PY0)), w["w_p"]),
                    poly_scale(poly_add(const(2.0), poly_mul(var(C0), var(C1), -2.0), poly_mul(var(S0), var(S1), -2.0)), w["w_theta"]),
                    poly_scale(poly_add(sq(RX1, RX0), sq(RY1, RY0)), w["w_r"]),
                    poly_scale(poly_add(poly_mul(var(FN), var(FN)), poly_mul(var(FT), var(FT))), w["w_f"]),
                    const(w["w_switch"] if switch else 0.0))


# ===== RELAXATIONS =====
# Milestone M1 (plan section 5): the SDP moment relaxation and the Taylor outer model, both
# built from the constraint table above.


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
    by_name = {m["name"]: m for m in modes}
    strip, stick, j = by_name["strip_2"], by_name["c2_stick"], 2
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


# ===== ASSEMBLY =====
# Vertices: start s (layer 0), then (k, mode) for k = 1..N-1, then goal t (layer N). Edge k joins
# layer k to layer k+1, and edges are listed layer by layer, i.e. in topological order. Flow
# convention (tex Def 2.2): sum over edges into v minus sum over edges out of v = beta_v, with
# beta_s = -1, beta_t = +1: the start supplies one unit and the goal absorbs it. In the coupling
# operator K an edge enters its head with sign + and its tail with sign -. The Python loops here
# run once at build time; the solver itself never loops over edges.

def build_trellis(cfg, task, modes):
    """Vertices, edges and edge types of plan 6.1, before pruning."""
    N, M = int(cfg["trellis"]["N"]), len(modes)
    if N < 2:
        raise ValueError("N >= 2: the trellis needs a start step and a goal step")
    names = [m["name"] for m in modes]
    if task["start_region"] not in names or modes[names.index(task["start_region"])]["contact"]:
        raise ValueError(f"start region {task['start_region']!r} is not a configured non-contact region")
    sm = names.index(task["start_region"])
    nv = 2 + (N - 1) * M
    s, t = 0, nv - 1
    vid = lambda k, m: 1 + (k - 1) * M + m
    vert_layer, vert_mode = np.zeros(nv, np.int64), np.full(nv, -1, np.int64)
    vert_layer[t] = N
    for k in range(1, N):
        vert_layer[vid(k, 0):vid(k, 0) + M] = k
        vert_mode[vid(k, 0):vid(k, 0) + M] = np.arange(M)
    allow = np.array([[transition_allowed(a, b) for b in modes] for a in modes])
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
    from scipy.optimize import linprog
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
    from scipy.optimize import linprog
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
    for ti in range(nt):
        lo, hi = edge_box(md, type_modes(tr, modes, ti)[0])
        box, st = tighten_box(type_rows(md, task, modes, tr, ti), lo, hi, margin)
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
    return md, modes, task, prune(cfg, md, task, modes, build_trellis(cfg, task, modes))


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


def save_problem(path, cfg, md, modes, task, tr):
    """Writes the graph part of problem.npz (plan 6.1); the relaxation data joins it in M1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(summary=summary(tr, modes, task), modes=[m["name"] for m in modes], var_names=list(VAR_NAMES),
                edge_kinds=list(EDGE_KINDS), task=dict(name=task["name"], start=task["start"].tolist(),
                goal=task["goal"].tolist(), pusher_start=task["pusher_start"].tolist(), start_region=task["start_region"]),
                model={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in md.items()})
    np.savez_compressed(path, meta=json.dumps(meta),
                        **{k: tr[k] for k in ("vert_layer", "vert_mode", "edge_tail", "edge_head", "edge_type", "edge_layer",
                                              "edge_lp_alive", "edge_alive", "type_kind", "type_tail", "type_head",
                                              "type_alive", "type_lo", "type_hi", "allow")},
                        type_status=tr["type_status"].astype("U32"), s=tr["s"], t=tr["t"], N=tr["N"], M=tr["M"])


# ===== SELFTEST =====
# Milestone checks (plan 17). M0: sign conventions, the mode graph, pruning soundness, and a
# hand-scripted true push that must be a valid start->goal path.

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
    for ti in pruned:
        lo, hi = edge_box(md, type_modes(tr, modes, ti)[0])
        miss = infeasibility_margin(type_rows(md, task, modes, tr, ti), lo, hi)
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


def selftest(cfg):
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
    ap.add_argument("--build", action="store_true", help="build, prune and save the trellis of the run")
    args = ap.parse_args()
    cfg = load_config(args.run, args.set)
    if args.task:
        cfg["task"] = args.task
    if args.selftest:
        sys.exit(0 if selftest(cfg) else 1)
    if args.build:
        md, modes, task, tr = build_instance(cfg)
        path = ROOT / cfg["output"]["root"] / cfg["run_name"] / task["name"] / str(cfg["seed"]) / "problem.npz"
        save_problem(path, cfg, md, modes, task, tr)
        print(json.dumps(summary(tr, modes, task), indent=2))
        print(f"saved {path}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
