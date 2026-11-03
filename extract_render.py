"""Extracts trajectories from a solve and renders them on the GPU (plan section 13), four ways: heaviest paths, randomized rounding, the certificate's k shortest paths, and greedy. The metrics table and the filmstrip come at M10.

    python extract_render.py --run ref_tiny --hand       # the scripted true push through the same steps, which checks this script
    python extract_render.py --run A                     # all four on the run's final_state.npz: a guess file per path, a video of each method's best
    python extract_render.py --run A --methods greedy,flow_decomp --videos 3
    python extract_render.py --run A --no_video          # guess files and the printed numbers only
    python extract_render.py --run ref_tiny --reference  # the same on Clarabel's solution from push_model.py --reference, which has no PDHG error in it

What it renders is a guess, not a true trajectory: first moments average all the flow through an edge, and nothing here enforces the true model (plan 13.3). Each guess prints how far it is from one, and the video's label gives the mode and the pusher's gap on every frame.
"""
import argparse
import heapq
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# MuJoCo draws on the GPU through EGL. Unlike scripts that fall back to osmesa, this one has no CPU
# fallback, so a missing EGL fails loudly. It has to be set before mujoco is imported.
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import push_model as pm

ROOT = Path(__file__).resolve().parent
# The extraction methods (plan 13.2): each one's name in titles, and what its weight means.
METHODS = dict(flow_decomp=("heaviest path", "flow"), random_round=("rounded path", "share of samples"), cert_kshortest=("certificate path", "value"), greedy=("greedy path", "smallest flow"))
SLIDER_HALF_HEIGHT = 0.02       # drawing only: the model is planar
FLOW_FLOOR = 1e-12
FLOW_LEFT = 1e-6                # path decomposition stops below this (plan 13.2)
# The true-model numbers (plan 13.4). Each group covers the rows of the constraint table whose tags start so.
ERROR_GROUPS = dict(translation=("trans_",), rotation=("rot_", "unit_norm"), torque=("torque",), friction=("fn_", "stick_", "slide_"), contact=("contact_", "head_region", "tail_region"),
                    free_motion=("static_", "zero_"), speed=("v_max",), ends=("start_", "goal_"))
MOVE_FLOOR = 1e-6               # m: a slider step below this counts as not moving
FORCE_FLOOR = 1e-6              # N
GAP_FLOOR = 1e-3                # m: a pusher further than this is not touching


# ===== PROBLEM =====
# The same steps as run_trajopt.py, without JAX: this script must not take GPU memory from a solve.

def run_dir(cfg):
    return ROOT / cfg["output"]["root"] / cfg["run_name"] / cfg["task"] / str(cfg["seed"])


def ensure_problem(cfg, run, task, sets):
    """Loads the run's problem.npz, first building it when it is missing or came from another config (plan 12.1). Returns it as a dict with s, t, N and M as plain ints, as the trellis helpers in push_model.py expect, plus the alive edges and each edge's place among them."""
    path = run_dir(cfg) / "problem.npz"
    fresh = False
    if path.exists():
        with np.load(path) as z:
            fresh = json.loads(str(z["meta"])).get("fingerprint") == pm.reference_fingerprint(cfg)
    if not fresh:
        cmd = [sys.executable, str(ROOT / "push_model.py"), "--build"] + (["--run", run] if run else []) + (["--task", task] if task else [])
        for s in sets:
            cmd += ["--set", s]
        print(f"building {path}", flush=True)
        subprocess.run(cmd, check=True)
    with np.load(path) as z:
        prob = {k: z[k] for k in z.files}
    if str(prob["rel_kind"]) != "sdp":
        raise NotImplementedError("extraction reads the SDP layout so far; run B's Taylor layout comes at M10")
    alive = np.flatnonzero(prob["edge_alive"])
    place = np.full(prob["edge_tail"].size, -1, np.int64)
    place[alive] = np.arange(alive.size)
    return dict(prob, s=int(prob["s"]), t=int(prob["t"]), N=int(prob["N"]), M=int(prob["M"]), alive=alive, place=place)


def load_reference(cfg, prob, path):
    """Clarabel's solution from push_model.py --reference, after checking that it belongs to this config and problem. It has no prices or edge bounds, so the certificate's paths are not available from it."""
    if not path.exists():
        raise SystemExit(f"no reference solution at {path}; run: python push_model.py --run {cfg['run_name']} --task {cfg['task']} --reference")
    with np.load(path) as z:
        solve = dict(m=z["m"], p=None, alive=z["alive"], lb=None, J=float(z["J"]), fingerprint=str(z["fingerprint"]))
    if solve["fingerprint"] != pm.reference_fingerprint(cfg) or not np.array_equal(solve["alive"], prob["alive"]):
        raise SystemExit(f"{path} came from another config; rerun: python push_model.py --run {cfg['run_name']} --task {cfg['task']} --reference")
    return solve


def load_solve(prob, path):
    """The final m and prices of a solve over the alive edges, and its edge bounds if it saved them, after checking that it belongs to this problem."""
    if not path.exists():
        raise SystemExit(f"no solve at {path}; run_trajopt.py saves it when a run ends by itself, so a run stopped with Ctrl-C leaves none")
    with np.load(path) as z:
        solve = dict(m=z["m"], p=z["p"], alive=z["alive"], lb=z["lb"] if "lb" in z.files else None)
    if not np.array_equal(solve["alive"], prob["alive"]):
        raise SystemExit(f"{path} came from another problem; rerun the solve")
    return solve


# ===== EXTRACTION =====
# Plan 13.2 and 13.3, on the numpy side since a path is one small search over a few thousand edges.

def heaviest_path(prob, y):
    """Method 1's top path (plan 13.2): the start-goal path whose smallest flow is largest, by a DP over the layers. It also gives a sensible path on an iterate that does not conserve flow exactly. y holds the alive edges' flows; returns (edge ids in layer order, the path's flow)."""
    tails, heads, layers, alive = prob["edge_tail"], prob["edge_head"], prob["edge_layer"], prob["alive"]
    flow = np.zeros(tails.size)
    flow[alive] = np.maximum(y, 0.0)
    best, came = np.full(prob["t"] + 1, -np.inf), np.full(prob["t"] + 1, -1)
    best[prob["s"]] = np.inf
    for k in range(prob["N"]):
        idx = alive[layers[alive] == k]
        value = np.minimum(best[tails[idx]], flow[idx])
        order = np.argsort(value, kind="stable")        # ascending, so the last write to a head is its largest
        best[heads[idx[order]]] = value[order]
        came[heads[idx[order]]] = idx[order]
    if came[prob["t"]] < 0:
        raise RuntimeError("no alive path reaches the goal")
    path, v = [], prob["t"]
    while v != prob["s"]:
        path.append(int(came[v]))
        v = int(tails[path[-1]])
    return path[::-1], float(best[prob["t"]])


def flow_paths(prob, y, k):
    """Method 1 (plan 13.2): repeatedly take the heaviest path and subtract its flow, until the flow left is negligible. The weights never rise, so the first k are the top k. Returns [(edges, weight)]."""
    left, paths = y.copy(), []
    for _ in range(k):
        edges, weight = heaviest_path(prob, left)
        if weight < FLOW_LEFT and paths:
            break
        paths.append((edges, weight))
        left[prob["place"][edges]] -= weight
    return paths


def rounded_paths(prob, y, m, samples, k, seed):
    """Method 2 (plan 13.2, tex Prop rounding has the flow's marginals): at each vertex take an outgoing edge with probability flow / outflow, for all samples at once per layer. Duplicates merge, and the k paths with the lowest relaxed cost stay. Every alive edge keeps a tiny weight, so a sample that reaches a vertex with no outflow still moves on. Returns [(edges, share of the samples)]."""
    rng = np.random.default_rng(seed)
    tails, heads, layers, alive = prob["edge_tail"], prob["edge_head"], prob["edge_layer"], prob["alive"]
    weight = np.zeros(tails.size)
    weight[alive] = np.maximum(y, 0.0) + FLOW_FLOOR
    at = np.full(samples, prob["s"])
    chosen = np.zeros((samples, prob["N"]), np.int64)
    for layer in range(prob["N"]):
        idx = alive[layers[alive] == layer]
        idx = idx[np.argsort(tails[idx], kind="stable")]
        total = np.cumsum(weight[idx])
        first, last = np.searchsorted(tails[idx], at, side="left"), np.searchsorted(tails[idx], at, side="right") - 1
        below = np.where(first > 0, total[np.maximum(first - 1, 0)], 0.0)
        pick = np.clip(np.searchsorted(total, below + rng.random(samples) * (total[last] - below)), first, last)
        chosen[:, layer] = idx[pick]
        at = heads[idx[pick]]
    unique, counts = np.unique(chosen, axis=0, return_counts=True)
    order = np.argsort(relaxed_costs(prob, alive, m)[prob["place"][unique]].sum(axis=1), kind="stable")[:k]
    return [([int(e) for e in unique[i]], float(counts[i]) / samples) for i in order]


def certificate_paths(prob, lb, k):
    """Method 3's paths (plan 13.2): the k cheapest start-goal paths by the certificate's edge bounds, from a DP over the layers that keeps the k best ways to reach each vertex. The first path's value is the free bound L_0 at these prices and multipliers. Returns [(edges, value)]."""
    tails, heads, layers, alive = prob["edge_tail"], prob["edge_head"], prob["edge_layer"], prob["alive"]
    best = {prob["s"]: [(0.0, -1, 0)]}                  # vertex: its best (value, incoming edge, rank in that edge's tail)
    for layer in range(prob["N"]):
        incoming = {}
        for e in alive[layers[alive] == layer]:
            for rank, (value, _, _) in enumerate(best.get(int(tails[e]), [])):
                incoming.setdefault(int(heads[e]), []).append((value + float(lb[prob["place"][e]]), int(e), rank))
        best.update({v: heapq.nsmallest(k, items) for v, items in incoming.items()})
    paths = []
    for value, e, rank in best[prob["t"]]:
        edges = []
        while e >= 0:
            edges.append(e)
            _, e, rank = best[int(tails[e])][rank]
        paths.append((edges[::-1], value))
    return paths


def greedy_path(prob, y):
    """Method 4 (plan 13.2): from the start, the outgoing edge with the most flow. Pruning keeps only edges on some start-goal path, so this always reaches the goal, though on an iterate that does not conserve flow it can walk through edges with no flow. Returns (edges, the smallest flow on them)."""
    tails, heads, alive = prob["edge_tail"], prob["edge_head"], prob["alive"]
    flow = np.maximum(y, 0.0)
    path, v = [], prob["s"]
    while v != prob["t"]:
        here = np.flatnonzero(tails[alive] == v)
        e = int(alive[here[np.argmax(flow[here])]])
        path.append(e)
        v = int(heads[e])
    return path, float(flow[prob["place"][path]].min())


def relaxed_costs(prob, edges, rows):
    """Each edge's relaxed cost per unit flow, c^T m / y, with the entries kept inside the edge type's box since a tiny flow makes the division noisy."""
    types = prob["edge_type"][edges]
    n = rows.shape[1]
    y = np.maximum(rows[:, :1], FLOW_FLOOR)
    return np.sum(prob["rel_c"][types, :n] * np.clip(rows / y, prob["rel_lo"][types, :n], prob["rel_hi"][types, :n]), axis=1)


def make_pricer(cfg, prob, md, modes, task, p):
    """A function from an edge id to arg min pi_e(lambda) as a row of m (plan 13.3, tex Initialization): the edge's own conic program at y = 1, with the final state prices in its cost. This runs Clarabel on the CPU, since a path needs only a few of these, and each is kept once solved."""
    import clarabel
    import scipy.sparse as sp
    alive = prob["alive"]
    verts = np.unique(np.r_[prob["edge_tail"][alive], prob["edge_head"][alive]])
    index = np.full(prob["t"] + 1, -1, np.int64)
    index[verts] = np.arange(verts.size)
    lam = p.copy()
    lam[:, 0] = 0.0                                     # flow prices cancel along a path
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = float(cfg["reference"]["clarabel"]["tol"])
    programs, priced = {}, {}

    def program(ti):
        rel = pm.relax_edge_type(cfg, md, task, modes, prob, ti)
        rows = pm.type_conic_rows(rel, "clarabel")
        flow = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, rel["n"]))
        A = sp.vstack([rows["zero"], flow, -rows["nonneg"], -rows["psd"]], format="csc")
        A.sort_indices()
        zeros = rows["zero"].shape[0] + 1
        b = np.zeros(A.shape[0])
        b[zeros - 1] = 1.0
        return rel, A, b, [clarabel.ZeroConeT(zeros), clarabel.NonnegativeConeT(rows["nonneg"].shape[0])] + [clarabel.PSDTriangleConeT(int(s)) for s in rows["sizes"]]

    def price(e):
        e = int(e)
        if e not in priced:
            ti = int(prob["edge_type"][e])
            if ti not in programs:
                programs[ti] = program(ti)
            rel, A, b, cones = programs[ti]
            cost = rel["c"] + rel["head_map"].T @ lam[index[prob["edge_head"][e]]] - rel["tail_map"].T @ lam[index[prob["edge_tail"][e]]]
            sol = clarabel.DefaultSolver(sp.csc_matrix((rel["n"], rel["n"])), cost, A, b, cones, settings).solve()
            if str(sol.status) not in ("Solved", "AlmostSolved"):
                print(f"  pricing edge {e} ended with status {sol.status}, so its point is unreliable", flush=True)
            priced[e] = np.zeros(int(prob["rel_n"].max()))
            priced[e][:rel["n"]] = np.asarray(sol.x)
        return priced[e]

    return price


def method_paths(cfg, prob, md, modes, task, solve, method):
    """One method's paths, best first, as [(edges, weight, rows of m)]. Rows are the solve's own, except on a certificate path, which has no flow to read and uses the priced minimizers (plan 13.3)."""
    ext, m = cfg["extraction"], solve["m"]
    k, y = int(ext["k_paths"]), m[:, 0]
    if method == "cert_kshortest":
        price = make_pricer(cfg, prob, md, modes, task, solve["p"])
        return [(edges, value, np.array([price(e) for e in edges])) for edges, value in certificate_paths(prob, solve["lb"], k)]
    if method == "flow_decomp":
        paths = flow_paths(prob, y, k)
    elif method == "random_round":
        paths = rounded_paths(prob, y, m, int(ext["samples"]), k, int(cfg["seed"]))
    else:
        paths = [greedy_path(prob, y)]
    return [(edges, weight, m[prob["place"][edges]]) for edges, weight in paths]


def check_path(prob, edges):
    """Stops on anything but a connected start-goal path of alive edges (plan 17, M10)."""
    tails, heads = prob["edge_tail"][edges], prob["edge_head"][edges]
    if tails[0] != prob["s"] or heads[-1] != prob["t"] or np.any(heads[:-1] != tails[1:]) or not np.all(prob["edge_alive"][edges]):
        raise RuntimeError(f"not a start-goal path of alive edges: {edges}")


def edge_points(prob, edges, rows):
    """Edge-layout points (len(edges), NV) from lifted rows: the first moments per unit flow, kept inside the edge type's box since a tiny flow makes the division noisy, then back through the saved elimination (plan 5.1.7). The first moments are entries 1 to nv, since monomials are ordered by degree."""
    out = np.zeros((len(edges), pm.NV))
    for i, e in enumerate(edges):
        ti = int(prob["edge_type"][e])
        nv = int(prob["rel_nv"][ti])
        w = rows[i, 1:nv + 1] / max(float(rows[i, 0]), FLOW_FLOOR)
        w = np.clip(w, prob["rel_lo"][ti, 1:nv + 1], prob["rel_hi"][ti, 1:nv + 1])
        out[i] = prob["rel_sub_const"][ti] + prob["rel_sub_coef"][ti, :, :nv] @ w
    return out


def knots_and_forces(v):
    """Knots (N + 1, 6), forces (N, 3) and the mismatch at each interior knot (N - 1) along a path. An interior knot is the average of what its two edges say (plan 13.3)."""
    x0, x1 = v[:, list(pm.KNOT0)], v[:, list(pm.KNOT1)]
    knots = np.vstack([x0[:1], 0.5 * (x1[:-1] + x0[1:]), x1[-1:]])
    return knots, v[:, pm.FN:], np.max(np.abs(x1[:-1] - x0[1:]), axis=1)


def step_modes(prob, modes, edges):
    """(tail mode, head mode or None) of each step."""
    return [pm.type_modes(prob, modes, int(prob["edge_type"][e])) for e in edges]


def step_costs(cfg, knots, forces, pairs):
    """The edge cost of each step at the extracted knots and forces (plan 4.6): what the rendered trajectory would cost if it were true."""
    return np.array([pm.poly_eval(pm.edge_cost(cfg, tail, head), pm.edge_vector(knots[k], knots[k + 1], *forces[k])) for k, (tail, head) in enumerate(pairs)])


def mode_sequence(pairs):
    """The tail modes along a path, with repeats counted."""
    return ", ".join(f"{name} x{len(list(run))}" for name, run in itertools.groupby(tail["name"] for tail, _ in pairs))


def pusher_gap(md, r):
    """Clearance between the pusher disk and the slider at pusher centers r (n, 2) in the body frame: the signed distance to the box minus the radius, so 0 is touching and negative is overlapping."""
    out = np.abs(r) - np.array([md["ax"], md["ay"]])
    return np.linalg.norm(np.maximum(out, 0.0), axis=1) + np.minimum(np.max(out, axis=1), 0.0) - md["rp"]


def true_model_numbers(md, task, prob, edges, pairs, knots, forces):
    """How far a guess is from the true model (plan 13.4), at the points the video draws: the largest error per group of rows in the constraint table, how many contact steps move the slider against the force on it or with no force at all, how many non-contact steps move it, how many steps move it with the pusher away from it, and the pusher's gap at each knot."""
    theta = np.arctan2(knots[:, 3], knots[:, 2])
    gap = pusher_gap(md, knots[:, 4:6])
    errors, contact_gap = {}, 0.0
    counts = dict(contact_steps=0, free_steps=0, moved_contact=0, moved_free=0, against=0, unforced=0, moved_away=0)
    for k, (e, (tail, head)) in enumerate(zip(edges, pairs)):
        kind = pm.EDGE_KINDS[int(prob["type_kind"][prob["edge_type"][e]])]
        point = pm.edge_vector(knots[k], knots[k + 1], *forces[k])
        for tag, value in pm.violations(pm.edge_constraints(md, task, kind, tail, head), point).items():
            group = next((name for name, starts in ERROR_GROUPS.items() if tag.startswith(starts)), "other")
            errors[group] = max(errors.get(group, 0.0), float(value))
        move = knots[k + 1, :2] - knots[k, :2]
        moved = bool(np.linalg.norm(move) > MOVE_FLOOR)
        counts["moved_away"] += int(moved and gap[k] > GAP_FLOOR)
        if not tail["contact"]:
            counts["free_steps"] += 1
            counts["moved_free"] += int(moved)
            continue
        counts["contact_steps"] += 1
        contact_gap = max(contact_gap, abs(float(gap[k])), abs(float(gap[k + 1])))
        n, t = md["normals"][tail["face"]], md["tangents"][tail["face"]]
        force = body_to_world(theta[k], -forces[k, 0] * n + forces[k, 1] * t)
        if moved:
            counts["moved_contact"] += 1
            counts["unforced"] += int(np.linalg.norm(force) <= FORCE_FLOOR)
            counts["against"] += int(np.linalg.norm(force) > FORCE_FLOOR and move @ force < 0.0)
    return dict(counts, errors=errors, contact_gap=contact_gap, gap=gap)


def hand_rows(cfg, prob, md, modes, task):
    """The scripted push (T1 only) as a flow of one on its edges and zero elsewhere. Only the entries extraction reads are filled. Returns (m over the alive edges, its edges, its true edge-layout points)."""
    steps = pm.hand_push(cfg, md, task, modes)
    edges = pm.path_edge_ids(prob, modes, steps)
    place = prob["place"]
    m = np.zeros((prob["alive"].size, int(prob["rel_n"].max())))
    for (_, _, _, v), e in zip(steps, edges):
        ti = int(prob["edge_type"][e])
        nv = int(prob["rel_nv"][ti])
        m[place[e], 0] = 1.0
        m[place[e], 1:nv + 1] = v[prob["rel_free"][ti, :nv]]
    return m, edges, np.array([s[3] for s in steps])


def check_hand(md, task, prob, path, weight, guess, edges, truth, J_hand, tol=1e-9):
    """Whether the steps above recover the scripted push: the independent truth is the true model's own points, not anything the relaxation stored. The true-model numbers must read zero on it, and must fire on a copy with its last contact step reversed and the pusher moved 1 cm off the face."""
    knots, forces, pairs, numbers = guess["knots"], guess["forces"], guess["pairs"], guess["numbers"]
    errors = dict(path=0.0 if path == edges else 1.0, flow=abs(weight - 1.0), points=float(np.max(np.abs(guess["v"] - truth))),
                  knots=float(np.max(np.abs(knots[:-1] - truth[:, list(pm.KNOT0)]))), cost=abs(float(guess["costs"].sum()) - J_hand), true_model=max(numbers["errors"].values()),
                  against=float(numbers["against"] + numbers["unforced"]), free_motion=float(numbers["moved_free"]), moved_away=float(numbers["moved_away"]), contact_gap=numbers["contact_gap"])
    k = max(i for i, (tail, _) in enumerate(pairs) if tail["contact"])
    bad = knots.copy()
    bad[k + 1, :2] = knots[k, :2] - (knots[k + 1, :2] - knots[k, :2])
    bad[k, 4:6] = knots[k, 4:6] + 0.01 * md["normals"][pairs[k][0]["face"]]
    flagged = true_model_numbers(md, task, prob, path, pairs, bad, forces)
    errors["control"] = 0.0 if flagged["against"] >= 1 and flagged["moved_away"] >= 1 and flagged["errors"]["translation"] > 1e-6 and abs(flagged["contact_gap"] - 0.01) < 1e-9 else 1.0
    ok = all(value <= tol for value in errors.values())
    print(f"{'PASS' if ok else 'FAIL'}: hand push through extraction: " + ", ".join(f"{k} error {value:.1e}" for k, value in errors.items()), flush=True)
    return ok


# ===== RENDERING =====
# Plan 13.5, in the way of scripts/render.py in residual-controller: mujoco.Renderer under EGL, frames
# through imageio. Nothing is simulated: the bodies are mocap bodies whose poses are set directly.

def scene_xml(md, rc):
    """A scene with no joints: ground (10 cm squares), slider with a dark bar on its +x side, pusher, and a see-through goal."""
    ax, ay, rp, hz = md["ax"], md["ay"], md["rp"], SLIDER_HALF_HEIGHT
    return f"""
<mujoco>
  <statistic center="0 0 0" extent="1"/>
  <visual>
    <global offwidth="{int(rc['width'])}" offheight="{int(rc['height'])}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0 0 0"/>
  </visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.93" rgb2="0.85 0.85 0.85" width="128" height="128"/>
    <material name="grid" texture="grid" texrepeat="5 5" texuniform="true"/>
  </asset>
  <worldbody>
    <geom type="plane" size="3 3 0.1" material="grid"/>
    <body name="slider" mocap="true" pos="0 0 {hz}">
      <geom type="box" size="{ax} {ay} {hz}" rgba="0.2 0.4 0.8 1"/>
      <geom type="box" size="0.012 {0.6 * ay} {hz + 0.0005}" pos="{ax - 0.012} 0 0" rgba="0.1 0.2 0.45 1"/>
    </body>
    <body name="pusher" mocap="true" pos="0 0 {hz}">
      <geom type="cylinder" size="{rp} {1.5 * hz}" rgba="0.85 0.25 0.2 1"/>
    </body>
    <body name="goal" mocap="true" pos="0 0 {hz}">
      <geom type="box" size="{ax} {ay} {hz}" rgba="0.2 0.75 0.35 0.3"/>
    </body>
  </worldbody>
</mujoco>"""


def yaw_quat(theta):
    return np.array([math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta)])


def body_to_world(theta, r):
    """R(theta) r for a body-frame vector."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * r[0] - s * r[1], s * r[0] + c * r[1]])


def view_of(md, rc, task, p, pusher):
    """Camera target and distance that keep the path, the pusher and the goal in frame."""
    pts = np.vstack([p, pusher, task["goal"][:2]])
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    half = 0.5 * (pts.max(axis=0) - pts.min(axis=0)) + math.hypot(md["ax"], md["ay"]) + 0.05
    aspect = float(rc["width"]) / float(rc["height"])
    return center, max(half[0] / aspect, half[1]) / math.tan(math.radians(22.5))    # MuJoCo's default fovy is 45 deg


def add_sphere(scene, mujoco, pos, radius, rgba):
    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scene.ngeom += 1


def add_arrow(scene, mujoco, start, end, width, rgba):
    if scene.ngeom < scene.maxgeom:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, float), np.asarray(end, float))
        scene.ngeom += 1


def make_labeler():
    """A function that writes text lines into a frame. The MuJoCo renderer draws no text, so this uses Pillow, with the font that ships with matplotlib."""
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), 20)

    def label(frame, lines):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        for i, text in enumerate(lines):
            draw.text((16, 12 + 26 * i), text, fill=(20, 20, 20), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
        return np.asarray(image)

    return label


def render_video(cfg, md, task, knots, forces, pairs, costs, title, path):
    """The video of plan 13.5: slider, goal, pusher, the trail of the slider's center, the contact force as an arrow, and text. One frame per render.fps, with a step lasting render.slow times dt."""
    import imageio.v2 as imageio
    import mujoco
    rc = cfg["render"]
    fps, N = int(rc["fps"]), len(pairs)
    per_step = max(1, round(fps * md["dt"] * float(rc["slow"])))
    theta = np.unwrap(np.arctan2(knots[:, 3], knots[:, 2]))
    pusher = np.array([knots[k, :2] + body_to_world(theta[k], knots[k, 4:6]) for k in range(N + 1)])
    center, distance = view_of(md, rc, task, knots[:, :2], pusher)
    model = mujoco.MjModel.from_xml_string(scene_xml(md, rc))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, int(rc["height"]), int(rc["width"]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, 90.0, float(rc["elevation"])
    cam.lookat[:] = (center[0], center[1], 0.0)
    slider_id, pusher_id, goal_id = (int(model.body(name).mocapid[0]) for name in ("slider", "pusher", "goal"))
    hz = SLIDER_HALF_HEIGHT
    data.mocap_pos[goal_id] = (task["goal"][0], task["goal"][1], hz)
    data.mocap_quat[goal_id] = yaw_quat(task["goal"][2])
    label = make_labeler()
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=None)

    def draw(k, a):
        """One frame at fraction a of step k; step N is the final knot."""
        b = min(k + 1, N)
        p = (1.0 - a) * knots[k, :2] + a * knots[b, :2]
        r = (1.0 - a) * knots[k, 4:6] + a * knots[b, 4:6]
        th = float((1.0 - a) * theta[k] + a * theta[b])
        data.mocap_pos[slider_id] = (p[0], p[1], hz)
        data.mocap_quat[slider_id] = yaw_quat(th)
        world_r = p + body_to_world(th, r)
        data.mocap_pos[pusher_id] = (world_r[0], world_r[1], hz)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        for q in knots[:k + 1, :2]:
            add_sphere(renderer.scene, mujoco, (q[0], q[1], 2.0 * hz + 0.002), 0.004, (0.1, 0.1, 0.1, 1.0))
        if k < N and pairs[k][0]["contact"]:
            j = pairs[k][0]["face"]
            n, t = md["normals"][j], md["tangents"][j]
            force = body_to_world(th, -forces[k, 0] * n + forces[k, 1] * t)
            if np.linalg.norm(force) > 1e-6:
                at = p + body_to_world(th, r - md["rp"] * n)
                tip = at + float(rc["force_scale"]) * force
                add_arrow(renderer.scene, mujoco, (at[0], at[1], 2.0 * hz + 0.004), (tip[0], tip[1], 2.0 * hz + 0.004), 0.006, (0.95, 0.6, 0.1, 1.0))
        where = f"step {k + 1}/{N}  {pairs[k][0]['name']}" if k < N else f"end, {N} steps"
        writer.append_data(label(renderer.render(), [title, f"{where}  gap {pusher_gap(md, r[None])[0] * 100:.1f} cm", f"cost so far {costs[:k + 1].sum():.5f}"]))

    for k in range(N):
        for i in range(per_step):
            draw(k, i / per_step)
    for _ in range(fps):
        draw(N, 0.0)
    writer.close()
    renderer.close()
    print(f"saved {path}", flush=True)


def report_guess(cfg, prob, md, modes, task, out, name, title, weight_name, edges, weight, rows, video):
    """One guess: its knots, forces and numbers, printed and saved as guess_<name>.npz (plan 13.3, 13.4), with its video if asked. Returns its pieces, which --hand checks."""
    check_path(prob, edges)
    v = edge_points(prob, edges, rows)
    knots, forces, mismatch = knots_and_forces(v)
    pairs = step_modes(prob, modes, edges)
    costs = step_costs(cfg, knots, forces, pairs)
    numbers = true_model_numbers(md, task, prob, edges, pairs, knots, forces)
    norm_error = float(np.max(np.abs(knots[:, 2] ** 2 + knots[:, 3] ** 2 - 1.0)))
    print(f"{name}: {weight_name} {weight:.6f}, relaxed cost {relaxed_costs(prob, edges, rows).sum():.6f}, cost at the extracted points {costs.sum():.6f}, largest knot mismatch {np.max(mismatch, initial=0.0):.2e}, largest |c^2 + s^2 - 1| {norm_error:.2e}", flush=True)
    print(f"  modes: {mode_sequence(pairs)}", flush=True)
    print("  true model, largest error: " + ", ".join(f"{group} {value:.1e}" for group, value in numbers["errors"].items()), flush=True)
    print(f"  slider moves on {numbers['moved_contact']} of {numbers['contact_steps']} contact steps ({numbers['against']} against the force, {numbers['unforced']} with no force) and {numbers['moved_free']} of {numbers['free_steps']} non-contact steps; "
          f"{numbers['moved_away']} steps move it with the pusher over {GAP_FLOOR * 1e3:.0f} mm away; largest gap on a contact step {numbers['contact_gap'] * 100:.2f} cm", flush=True)
    np.savez(out / f"guess_{name}.npz", modes=np.array([int(prob["type_tail"][prob["edge_type"][e]]) for e in edges]), edges=np.array(edges), knots=knots, forces=forces,
             dt=md["dt"], task=cfg["task"], weight=weight, knot_mismatch=mismatch, gap=numbers["gap"], true_model=json.dumps({k: value for k, value in numbers.items() if k != "gap"}))
    if video:
        render_video(cfg, md, task, knots, forces, pairs, costs, title, out / f"{name}.mp4")
    return dict(v=v, knots=knots, forces=forces, costs=costs, pairs=pairs, numbers=numbers)


# ===== MAIN =====

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="run config: a name in configs/runs/ or a path")
    ap.add_argument("--task", default=None, help="task name, overriding the config")
    ap.add_argument("--set", action="append", default=[], help="override one key, e.g. --set render.slow=1; keys the model depends on must match the solve")
    ap.add_argument("--hand", action="store_true", help="use the scripted true push (T1) instead of a solve, and check the extraction against it")
    ap.add_argument("--reference", action="store_true", help="use Clarabel's solution from push_model.py --reference instead of a PDHG solve: no solver error, so what remains is the relaxation's")
    ap.add_argument("--methods", default="all", help=f"all, or a comma-separated list of {', '.join(METHODS)}")
    ap.add_argument("--videos", type=int, default=1, help="videos per method, best paths first; every path gets a guess file")
    ap.add_argument("--no_video", action="store_true", help="extract and print only")
    args = ap.parse_args()
    if not args.run:
        ap.print_help()
        sys.exit(1)
    cfg = pm.load_config(args.run, args.set)
    if args.task:
        cfg["task"] = args.task
    prob = ensure_problem(cfg, args.run, args.task, args.set)
    md, modes, task = pm.build_model(cfg), pm.build_modes(cfg), pm.resolve_task(cfg)
    out = run_dir(cfg) / "extract"
    out.mkdir(parents=True, exist_ok=True)
    if args.hand:
        try:
            m, hand_edges, truth = hand_rows(cfg, prob, md, modes, task)
        except ValueError as exc:
            raise SystemExit(f"--hand needs a task the script covers: {exc}")
        edges, weight = heaviest_path(prob, m[:, 0])
        guess = report_guess(cfg, prob, md, modes, task, out, "hand", "hand push", "path flow", edges, weight, m[prob["place"][edges]], not args.no_video)
        J_hand = float(sum(pm.poly_eval(pm.edge_cost(cfg, tail, head), row) for row, (tail, head) in zip(truth, guess["pairs"])))
        sys.exit(0 if check_hand(md, task, prob, edges, weight, guess, hand_edges, truth, J_hand) else 1)
    solve = load_reference(cfg, prob, run_dir(cfg) / "reference_state.npz") if args.reference else load_solve(prob, run_dir(cfg) / "final_state.npz")
    source, prefix = ("Clarabel", "reference_") if args.reference else (cfg["run_name"], "")
    methods = list(cfg["extraction"]["methods"]) if args.methods == "all" else args.methods.split(",")
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; choose from {', '.join(METHODS)}")
    print(f"{cfg['run_name']} on {cfg['task']}" + (f": Clarabel's solution of the relaxation, J_rel {solve['J']:.6f}" if args.reference else ""), flush=True)
    for method in methods:
        if method == "cert_kshortest" and solve["lb"] is None:
            print(f"{method}: skipped, since this solution has no edge bounds; a solve from the current run_trajopt.py saves them", flush=True)
            continue
        label, weight_name = METHODS[method]
        for rank, (edges, weight, rows) in enumerate(method_paths(cfg, prob, md, modes, task, solve, method)):
            report_guess(cfg, prob, md, modes, task, out, f"{prefix}{method}_{rank}", f"{source}: {label} {rank}, {weight_name} {weight:.3f}", weight_name, edges, weight, rows, not args.no_video and rank < args.videos)


if __name__ == "__main__":
    main()
