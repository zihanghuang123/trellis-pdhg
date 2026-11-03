"""Extracts a trajectory from a solve and renders it on the GPU (plan section 13). So far: the top path of method 1 and its video; the other methods, the metrics table and the filmstrip come at M10.

    python extract_render.py --run ref_tiny --hand       # the scripted true push through the same steps, which checks this script
    python extract_render.py --run D                     # the heaviest path of the run's final_state.npz
    python extract_render.py --run D --no_video          # the same, without drawing

What it renders is a guess, not a true trajectory: first moments average all the flow through an edge, and nothing here enforces the true model (plan 13.3). The printed knot mismatch and rotation error say how far it is from one.
"""
import argparse
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
METHOD = "flow_decomp"
SLIDER_HALF_HEIGHT = 0.02       # drawing only: the model is planar
FLOW_FLOOR = 1e-12


# ===== PROBLEM =====
# The same steps as run_trajopt.py, without JAX: this script must not take GPU memory from a solve.

def run_dir(cfg):
    return ROOT / cfg["output"]["root"] / cfg["run_name"] / cfg["task"] / str(cfg["seed"])


def ensure_problem(cfg, run, task, sets):
    """Loads the run's problem.npz, first building it when it is missing or came from another config (plan 12.1). Returns it as a dict with s, t, N and M as plain ints, as the trellis helpers in push_model.py expect."""
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
    return dict(prob, s=int(prob["s"]), t=int(prob["t"]), N=int(prob["N"]), M=int(prob["M"]))


def load_solve(prob, path):
    """The final m of a solve over the alive edges, after checking that it belongs to this problem."""
    if not path.exists():
        raise SystemExit(f"no solve at {path}; run_trajopt.py saves it when a run ends by itself, so a run stopped with Ctrl-C leaves none")
    with np.load(path) as z:
        m, alive = z["m"], z["alive"]
    if not np.array_equal(alive, np.flatnonzero(prob["edge_alive"])):
        raise SystemExit(f"{path} came from another problem; rerun the solve")
    return m


# ===== EXTRACTION =====
# Plan 13.2 and 13.3, on the numpy side since a path is one small search.

def heaviest_path(prob, y):
    """Method 1's top path (plan 13.2): the start-goal path whose smallest flow is largest, by a DP over the layers. An iterate that does not conserve flow exactly still gives a connected path, which following the largest edge need not. y holds the alive edges' flows; returns (edge ids in layer order, the path's flow)."""
    tails, heads, layers, alive = prob["edge_tail"], prob["edge_head"], prob["edge_layer"], np.flatnonzero(prob["edge_alive"])
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


def hand_rows(cfg, prob, md, modes, task):
    """The scripted push (T1 only) as a flow of one on its edges and zero elsewhere. Only the entries extraction reads are filled. Returns (m over the alive edges, its edges, its true edge-layout points)."""
    steps = pm.hand_push(cfg, md, task, modes)
    edges = pm.path_edge_ids(prob, modes, steps)
    alive = np.flatnonzero(prob["edge_alive"])
    place = {int(e): i for i, e in enumerate(alive)}
    m = np.zeros((alive.size, int(prob["rel_n"].max())))
    for (_, _, _, v), e in zip(steps, edges):
        ti = int(prob["edge_type"][e])
        nv = int(prob["rel_nv"][ti])
        m[place[e], 0] = 1.0
        m[place[e], 1:nv + 1] = v[prob["rel_free"][ti, :nv]]
    return m, edges, np.array([s[3] for s in steps])


def check_hand(path, weight, v, knots, costs, edges, truth, J_hand, tol=1e-9):
    """Whether the steps above recover the scripted push: the independent truth is the true model's own points, not anything the relaxation stored."""
    errors = dict(path=0.0 if path == edges else 1.0, flow=abs(weight - 1.0), points=float(np.max(np.abs(v - truth))),
                  knots=float(np.max(np.abs(knots[:-1] - truth[:, list(pm.KNOT0)]))), cost=abs(float(costs.sum()) - J_hand))
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
        writer.append_data(label(renderer.render(), [title, where, f"cost so far {costs[:k + 1].sum():.5f}"]))

    for k in range(N):
        for i in range(per_step):
            draw(k, i / per_step)
    for _ in range(fps):
        draw(N, 0.0)
    writer.close()
    renderer.close()
    print(f"saved {path}", flush=True)


# ===== MAIN =====

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="run config: a name in configs/runs/ or a path")
    ap.add_argument("--task", default=None, help="task name, overriding the config")
    ap.add_argument("--set", action="append", default=[], help="override one key, e.g. --set render.slow=1; keys the model depends on must match the solve")
    ap.add_argument("--hand", action="store_true", help="use the scripted true push (T1) instead of a solve, and check the extraction against it")
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
    alive = np.flatnonzero(prob["edge_alive"])
    if args.hand:
        try:
            m, hand_edges, truth = hand_rows(cfg, prob, md, modes, task)
        except ValueError as exc:
            raise SystemExit(f"--hand needs a task the script covers: {exc}")
    else:
        m = load_solve(prob, run_dir(cfg) / "final_state.npz")
    edges, weight = heaviest_path(prob, m[:, 0])
    place = {int(e): i for i, e in enumerate(alive)}
    v = edge_points(prob, edges, m[[place[e] for e in edges]])
    knots, forces, mismatch = knots_and_forces(v)
    pairs = step_modes(prob, modes, edges)
    costs = step_costs(cfg, knots, forces, pairs)
    norm_error = float(np.max(np.abs(knots[:, 2] ** 2 + knots[:, 3] ** 2 - 1.0)))
    print(f"{cfg['run_name']} on {cfg['task']}: path flow {weight:.6f}, cost at the extracted points {costs.sum():.6f}, largest knot mismatch {np.max(mismatch, initial=0.0):.2e}, largest |c^2 + s^2 - 1| {norm_error:.2e}", flush=True)
    print(f"modes: {mode_sequence(pairs)}", flush=True)
    ok = True
    if args.hand:
        J_hand = float(sum(pm.poly_eval(pm.edge_cost(cfg, tail, head), row) for row, (tail, head) in zip(truth, pairs)))
        ok = check_hand(edges, weight, v, knots, costs, hand_edges, truth, J_hand)
    name = "hand" if args.hand else f"{METHOD}_0"
    out = run_dir(cfg) / "extract"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"guess_{name}.npz", modes=np.array([int(prob["type_tail"][prob["edge_type"][e]]) for e in edges]), edges=np.array(edges), knots=knots, forces=forces,
             dt=md["dt"], task=cfg["task"], path_flow=weight, knot_mismatch=mismatch)
    if not args.no_video:
        title = "hand push" if args.hand else f"{cfg['run_name']}: heaviest path, flow {weight:.3f}"
        render_video(cfg, md, task, knots, forces, pairs, costs, title, out / f"{name}.mp4")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
