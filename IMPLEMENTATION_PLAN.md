# Implementation Plan — Certified Trellis Trajectory Optimization, Planar Pushing Prototype

**For:** Claude Code, on a server with 8× A100 SXM 80 GB.
**Source of truth:** the proposal PDF *Certified Trajectory Optimization on Time-Layered Graphs* (Nov 2, 2026), Sections 2–8 and 12. Put it in the repo at `docs/trajopt_proposal.pdf`.
**Rule:** where this plan and the proposal disagree, the proposal wins, except for the explicit deviations listed in §19.
**Not in scope now:** the local solver (LEAP). Extraction writes guesses to disk in a format LEAP can read later (§13.4).

---

## 0. How to work with this plan

- **Few large scripts.** No package structure, no class hierarchies, no plugin systems. Each script may be 1,000–3,000 lines. Top-level functions, generous comments, and clear `# ===== SECTION =====` banners. The goal is that a human or an AI can copy a whole section, change it, and rerun.
- **Settings live in YAML.** No magic constants buried in code. Every knob in §16 is read from the config.
- **Build in milestones (§17).** Each milestone has acceptance checks, and each check is a small function inside the relevant script, run with `--selftest`. Do not start a milestone until the previous one passes.
- **Parallelize everything (§15).** Within a run, every per-edge and per-vertex operation is batched with `jax.vmap` or segment ops; Python loops over edges are forbidden. Across runs, all 8 GPUs work in parallel.
- **Count everything.** Every solve, cache hit, zero-test pass, eigendecomposition, derivative evaluation, tree attempt and fallback is counted and logged. The comparisons depend on these counts.

---

## 1. Goal and run matrix

**Goal.** For planar pushing, build the relaxation and PDHG solver from the proposal. Compute the certified lower bound $L$. Extract initial guesses four ways, render them, and compare four solver variants.

| Run | Segment tree | Edge relaxation | Key knobs | Proposal § |
|---|---|---|---|---|
| **A** (default) | exact (cyclic reduction) | SDP moment relaxation | order κ (default 1) | 3 (Opt. B), 4, 5 |
| **A-κ2 / A-κ3** | exact | SDP, κ = 2 / κ = 3 | κ = 3 only on small instances | 3, 4 |
| **A-reg** | exact | SDP, same θ-regions as B | `theta_splits: 8` | 3 (regions) |
| **B** | exact | Taylor model + certified error band | `theta_splits`, `lambda_splits` | 3 (Opt. A) |
| **C-a** | low-rank, option (a): compressed exact responses | SDP κ = 1 | rank r ∈ {4, 16, 64} | 4, 10 |
| **C-b** | low-rank, option (b): **Section 12** position–velocity reference + ≤ r learned corrections | SDP κ = 1 | rank r ∈ {4, 16, 64}; plus r = 16 with 4 probes per refresh | **12** |
| **D** | none (plain PDHG) | SDP κ = 1 | — | 4 |
| **A-noskip** | exact | SDP κ = 1, skipping off | verification only | 4 |

The table lists every configuration. **They are run in stages, starting with a lean first round.**

**Stage 1 (first round): 7 jobs on 8 GPUs, all at once, in timing mode (§15.2).**

| GPU | Job | Task |
|---|---|---|
| 0 | A (exact tree, SDP κ = 1) | T3 |
| 1 | A-κ2 | T3 |
| 2 | B (exact tree, Taylor) | T3 |
| 3 | C-a, r = 16 | T3 |
| 4 | C-b, r = 16 (Section 12) | T3 |
| 5 | D (no tree) | T3 |
| 6 | A-κ3, small instance (N = 8) | T2 |
| 7 | small-instance baselines: A (κ = 1), then A-κ2, run one after the other on the same N = 8 instance | T2 |

- GPU 7 would otherwise sit idle. Without these baselines, A-κ3 has nothing at the same size to compare against. Set `stage1.small_kappa_baselines: false` to drop them.
- One seed (0). The solver is deterministic. Randomized rounding draws its own samples inside extraction.
- Each job stops when the P_δ gap closes (§10.5): mismatch ≤ δ and $U_{\text{flow}} - L \le \eta$. The round cap and time cap are safety stops only.
- Before the solves, tune the step-size balance for each configuration (§7.3). These are short tuning jobs scored on counts, so they run in throughput mode.
- **For every job:** extraction and rendering for all 4 methods (§13), plus the metric tables.

**Stage 2:** the same 6 full-size configurations on T1, T2 and T4 (18 jobs, 3 waves).

**Stage 3, chosen after looking at stages 1–2:**
- the rank sweep r ∈ {4, 64} for whichever low-rank option looks useful;
- C-b-r16-p4 (probing);
- A-reg;
- the optional horizon sweep $N \in \{16, 32, 64, 128\}$ for A, C-a-r16, C-b-r16 and D on T3 (`sweep_N`). Trees exist to address the horizon, so this is where they should differ most.

**A-noskip** is not a matrix job. It is the one-time equality check of milestone M5.

### 1.1 Where each part of the proposal lives in this plan

| Proposal | Plan | Status in this phase |
|---|---|---|
| §1–2 problem, trellis, perspective, δ | §4, §6, §7.2 | implemented |
| §3 Option A (Taylor + band), Option B (Shor / moment), regions, offline sizing | §5.2, §5.1, §5.3 | implemented (bands first order for bilinear terms, §19.6) |
| §4 PDHG round, diagonal steps, restarts | §7 | implemented |
| §4 zero test, unchanged face | §9.1, §9.2 | implemented (face cache on for B) |
| §4 tree pass behind SuperMann / HPE | §11.1, §11.2, §11.6 | implemented |
| §4 warm starts | §12.1 (`--warm_start`) | hook only |
| §5 certificate: free each round, lazy exact, true-set tightening, signed error, ceiling | §10 | implemented |
| §6 four extraction methods and initial guesses | §13 | implemented (no local solve) |
| §7 screening, refinement, factoring | §12.1, §19.4 | screening coded and unit-tested but off (needs U); rest not implemented |
| §8 loop and stopping | §12.1, §10.5 | implemented; the solver stops on the P_δ gap with $U_{\text{flow}}$; the trajectory gap $U_{\text{traj}} - L$ comes with LEAP |
| §12.1–12.5 low-rank correction | §11.3, §11.5, §11.6, §14 | implemented as run C-b |

---

## 2. Repository layout

```
trellis_push/
  docs/trajopt_proposal.pdf
  configs/
    base.yaml                 # all defaults (§16)
    runs/A.yaml  A_k2.yaml  A_k3_small.yaml  A_small.yaml  A_k2_small.yaml  A_reg.yaml  B.yaml
         C_a_r4.yaml C_a_r16.yaml C_a_r64.yaml
         C_b_r4.yaml C_b_r16.yaml C_b_r64.yaml C_b_r16_p4.yaml
         D.yaml  A_noskip.yaml
    matrix.yaml               # stage1 (7 jobs + tuning), stage2, stage3
  push_model.py               # §4–§6: geometry, modes, graph, relaxations, data arrays, reference CPU solve
  run_trajopt.py              # §7–§12: PDHG, edge solvers, skipping, certificate, trees, safeguard, main loop
  extract_render.py           # §13: 4 extraction methods, metrics, mp4 + filmstrip rendering
  compare.py                  # §14: cross-run tables and plots
  launch.py                   # §15: runs the matrix on 8 GPUs, with several processes per GPU
  runs/<run_name>/<task>/<seed>/...   # outputs (§12.4)
  runs/tuned/<config>_<task>.yaml     # tuned step-size balance (§7.3)
```

Shared helpers (YAML loading, index maps) can be duplicated between scripts. A little copy-paste is preferable to a shared module.

---

## 3. Environment

- Python 3.11+, `jax[cuda12]` (recent stable), `numpy`, `scipy`, `pyyaml`, `matplotlib`, `ffmpeg` (system), `cvxpy`, `clarabel`, `scs`. Add `mosek` if a license exists; it is optional and used only for reference checks.
- `jax.config.update("jax_enable_x64", True)` in every script. Bounds must be computed in float64. The iteration dtype is a config flag (`solver.dtype`, default `float64`).
- Environment for each process: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=<1/procs_per_gpu - margin>`.
- If available, enable CUDA MPS on each GPU so several small processes can share it efficiently (§15.3).
- **Persistent compilation cache**, shared by all jobs on the server: `jax.config.update("jax_compilation_cache_dir", <shared dir>)` and `jax_persistent_cache_min_compile_time_secs = 0`. Jobs with the same shapes then reuse compiled code instead of each paying the slow first compile.
- Record the JAX, jaxlib, CUDA and driver versions and the XLA flags in `timing.json`.

---

## 4. Planar pushing model (`push_model.py`, section MODEL)

### 4.1 Objects
- **Slider:** a rigid box with half-sizes $(a_x, a_y)$, mass $m$, and table friction $\mu_t$. Faces $j \in \{0,1,2,3\}$ are $+x, +y, -x, -y$ in the body frame. Each face has an outward normal $n_j$, a tangent $t_j$ (counter-clockwise), a center $c_j$ and a half-length $\ell_j$.
- **Pusher:** a disk of radius $r_p$, contact friction $\mu_c$, and speed limit $v_{\max}$ per step (body frame, $\infty$-norm).
- **Limit surface (quasi-static, ellipsoidal):**
  - $f_{\max} = \mu_t m g$ and $\tau_{\max} = \mu_t m g \, \bar r$, where $\bar r$ is the mean distance to the centroid for uniform pressure. Compute $\bar r$ numerically on a grid at startup.
  - The body twist is $\xi = (v_x, v_y, \omega) = c_v \,\mathrm{diag}(1/f_{\max}^2,\ 1/f_{\max}^2,\ 1/\tau_{\max}^2)\, w$, with $w = (f_x, f_y, \tau)$ the body-frame wrench and $c_v$ a config constant that sets the speed scale.

### 4.2 Knot state and per-edge variables
- **Knot state, SDP runs:** $x = (p_x, p_y, c, s, r_x, r_y)$, so $D = 6$. Here $(c, s)$ encode the slider rotation, and $r$ is the pusher center in the slider body frame.
- **Knot state, Taylor runs (B):** $x = (p_x, p_y, \theta, r_x, r_y)$, so $D = 5$ (§5.2).
- **Edge variables, SDP:** $v = (x_k, x_{k+1}, f_n, f_t, \tau) \in \mathbb{R}^{15}$. Non-contact edges set $f_n = f_t = \tau = 0$ through linear equalities, so they share the same 15-variable layout. Keeping one layout makes batching uniform (§6.2).

### 4.3 Modes
- **Contact modes:** face $j \in \{0..3\}$ × type ∈ {stick, slide+, slide−}, which gives 12 modes.
- **Non-contact modes:** 8 convex regions around the box in the body frame: 4 face strips (in front of each face) and 4 corner regions. Each is a polytope, kept $\ge r_p$ away from the box and clipped to a body-frame workspace box $|r| \le R_{ws}$.
- **Total:** $M = 20$, plus regionalized copies (§5.3).

### 4.4 Allowed transitions (mode graph, the same at every layer)
- Within face $j$, any of {stick, slide+, slide−} may follow any other, including itself.
- Contact on face $j$ ↔ the face-$j$ strip region, in both directions.
- Non-contact region ↔ itself and its adjacent regions (they share a boundary).
- Start: a vertex with the given initial pose and pusher position, in a given non-contact region.
- Goal: a vertex with the given final slider pose. The pusher is free at the goal, and goal edges come from every mode.

### 4.5 Constraints per edge (true, nonconvex model)
Write every constraint over $v$ as a polynomial of degree ≤ 2. Linear terms stay linear. **Keep the list in a single table-driven function, so the SDP builder and the Taylor builder read the same source.**

1. **Rotation unit norm:** $c_k^2 + s_k^2 = 1$ and $c_{k+1}^2 + s_{k+1}^2 = 1$.
2. **Relative rotation (angular step):** $c_k s_{k+1} - s_k c_{k+1} = \Delta t\, \omega$, where $\omega = c_v \tau / \tau_{\max}^2$. Also require $c_k c_{k+1} + s_k s_{k+1} \ge \cos(\theta_{\text{step,max}})$, a bilinear inequality. *(This form is consistent with unit norms at both knots. Do **not** use the Euler update $c_{k+1} = c_k - \Delta t\,\omega s_k$, which contradicts the unit norm.)*
3. **Translation (contact edges):** $p_{k+1} - p_k = \Delta t\, R(c_k, s_k)\, (v_x, v_y)$, with $(v_x, v_y) = c_v (f_x, f_y)/f_{\max}^2$ and $(f_x, f_y) = f_n (-n_j) + f_t t_j$. Two bilinear equalities.
4. **Torque definition (contact edges):** $\tau = r_c \times f_{\text{body}}$, with contact point $r_c = r_k - r_p n_j$. This is bilinear in $r$ and $(f_n, f_t)$.
5. **Contact geometry (contact edges, both knots):** $n_j^\top r = n_j^\top c_j + r_p$ (on the face line, offset by the radius) and $|t_j^\top (r - c_j)| \le \ell_j$. Linear.
6. **Friction:**
   - always $f_n \ge 0$ and $f_n \le f_{n,\max}$;
   - stick: $|f_t| \le \mu_c f_n$ and $r_{k+1} = r_k$;
   - slide+: $f_t = +\mu_c f_n$ and $t_j^\top (r_{k+1} - r_k) \ge 0$;
   - slide−: $f_t = -\mu_c f_n$ and $t_j^\top (r_{k+1} - r_k) \le 0$.

   *Here $f_t$ is the tangential force the pusher applies **to the slider**. Friction on the pusher opposes the pusher's sliding relative to the slider, so by Newton's third law the force on the slider points **along** that sliding. Define this once in a comment and unit-test it on a hand case: the pusher slides in $+t_j$ while pushing, so the slider is dragged toward $+t_j$.*
7. **Non-contact edges:** the slider is static ($p_{k+1} = p_k$, $c_{k+1} = c_k$, $s_{k+1} = s_k$). Both $r_k$ and $r_{k+1}$ lie in the edge's region polytopes: the tail mode's region at $k$ and the head mode's region at $k+1$, i.e. **two-sided**. The pusher moves at most $v_{\max}$ on **every** edge, contact or not: $\|r_{k+1} - r_k\|_\infty \le v_{\max}$.
8. **Head-mode membership at $k+1$:** $x_{k+1}$ satisfies the head mode's state constraints (contact line or region). This is the two-sided convention used in the earlier tests.
9. **Bounds:** workspace box on $p$; $|c|, |s| \le 1$; $|r| \le R_{ws}$; force bounds. These bounds are also used for the interval moment bounds (§5.1.4).
10. **Start and goal edges:**
    - start: $x_0$ fixed;
    - goal: $p_N$ and $(c_N, s_N) = (\cos\theta_g, \sin\theta_g)$ fixed, pusher free.

### 4.6 Cost per edge (quadratic, the same for all runs)
$$
\ell_e = w_p\|p_{k+1}-p_k\|^2 + w_\theta\, 2\big(1 - c_k c_{k+1} - s_k s_{k+1}\big) + w_r \|r_{k+1}-r_k\|^2 + w_f (f_n^2 + f_t^2) + w_{sw}\,[m \ne m'].
$$
In the SDP runs, all terms are linear in the moments (exact). For run B, see §5.2.4.

### 4.7 Tasks (in `base.yaml`, selected by name)
- `T1_push_straight`: from $(0, 0, 0)$ to $(0.30, 0, 0)$.
- `T2_rotate_90`: from $(0, 0, 0)$ to $(0, 0, \pi/2)$.
- `T3_translate_rotate`: from $(0, 0, 0)$ to $(0.20, 0.15, \pi/2)$.
- `T4_rotate_180`: from $(0, 0, 0)$ to $(0.10, 0, \pi)$. This needs face switches.

The pusher starts in the strip behind face 2 ($-x$). Defaults: $N = 32$ steps (a power of 2 simplifies the tree; the code must still handle any $N$), $\Delta t = 0.1$. These tasks are *in the spirit of* Graesdal et al. (RSS 2024), not exact copies. Reproducing their benchmark is a later milestone.

---

## 5. Edge outer models (`push_model.py`, section RELAXATIONS)

The output of this section, for every **edge type**, is a set of homogeneous linear and conic constraints on the edge vector, plus a linear cost (§6). An edge type is (kind ∈ {start, mid, goal}, tail mode, head mode, regions).

### 5.1 SDP moment relaxation (runs A, A-κ, A-reg, C-a, C-b, D)

**5.1.1 Monomials and moments.**
- Use monomials in $v \in \mathbb{R}^{15}$ of degree ≤ $2\kappa$, ordered by degree, then lexicographically.
- Let $m$ be the moment vector, with $m_0$ the moment of the constant monomial.
- Build index maps `mono_index[exponent_tuple] -> int` and `mat_index[κ]`, which maps a monomial pair to the index of its product.

| κ | variables in $m$ | moment matrix size $s$ |
|---|---|---|
| 1 | 1 + 15 + 120 = 136 | 16 |
| 2 | C(19,4) = 3,876 | 136 |
| 3 | C(21,6) = 54,264 | 816 (small instances only) |

**5.1.2 Constraints in moments (slice $m_0 = 1$, then homogenized).**
- **Moment matrix:** $\mathcal M_\kappa(m) \succeq 0$, where entry $(\alpha, \beta)$ is $m_{\alpha+\beta}$ over monomials of degree ≤ κ.
- **Polynomial equalities $h(v) = 0$ of degree $d_h$:** for every monomial $u$ with $\deg u \le 2\kappa - d_h$, add the linear row $L_m(h\,u) = 0$. For κ = 1 that means $u = 1$ for quadratic $h$, and $u \in \{1, v_i\}$ for linear $h$.
- **Polynomial inequalities $g(v) \ge 0$:**
  - κ = 1: linear $g$ gives a linear row $L_m(g) \ge 0$; quadratic $g$ (the rotation-step bound) gives a linear row in second moments.
  - κ ≥ 2: add the localizing matrix $\mathcal M_{\kappa - \lceil \deg g/2 \rceil}(g\, m) \succeq 0$ **and** the linear row.
- **Optional RLT cuts** (`relaxation.rlt: true`, default false): pairwise products of linear inequalities as rows in second moments.

**5.1.3 Homogenization (perspective).** Replace every constant (the coefficient of monomial $1$) by $m_0$, and add $0 \le m_0 \le 1$. All rows and all PSD blocks become homogeneous in $m$. Then $m_0 = y_e$ is the edge flow. This is exactly the perspective of the outer set (proposal §2).

**5.1.4 Interval moment bounds (needed for the zero test and for certificate soundness).**
- From the variable box in §4.5.9, compute $[\underline{m}_\alpha, \overline{m}_\alpha]$ for every monomial by interval arithmetic.
- Add homogenized rows $\underline{m}_\alpha m_0 \le m_\alpha \le \overline{m}_\alpha m_0$.
- These rows are **valid for true points**, so the outer set stays an outer set, and they give the box that the dual bound in §9.1 needs.
- The IPM treats them as **variable bounds**, i.e. diagonally, not as general rows.

**5.1.5 Vertex coupling (knob `relaxation.couple_second_moments`, default true).**
- Coupled coordinates for each vertex: flow $m_0$, first moments of the knot state (6), and, if the knob is on, second moments of the knot state ($6\cdot7/2 = 21$). So $q = 28$ per vertex, or $q = 7$ with the knob off.
- An edge contributes its post-knot coordinates to the head vertex (sign $+$) and its pre-knot coordinates to the tail vertex (sign $-$). Flow appears with $+$ at the head and $-$ at the tail. The start supplies 1 and the goal absorbs 1.
- The start edge's pre-knot moments are fixed by rows in the edge itself: $m_{x_0} = x_0 m_0$, $m_{x_0 x_0^\top} = x_0 x_0^\top m_0$. The same holds for the fixed parts of the goal knot.

**5.1.6 Polyhedral block vs PSD blocks (decision 1a; see §8).** For each edge, collect:
- the polyhedral part $P_e$: equality rows, inequality rows, variable bounds, $0 \le m_0 \le 1$;
- the PSD part: a list of blocks $\{\mathcal B_{e,b}(m) \succeq 0\}$, each a linear map from $m$ to symmetric matrices.

Store the PSD maps as sparse `(row, col, moment_index, coefficient)` lists for each edge type. They are the same for all time steps.

### 5.2 Taylor outer model with certified band (run B)

**5.2.1 Variables.**
- Knot $x = (p_x, p_y, \theta, r_x, r_y)$; edge $v = (x_k, x_{k+1}, f_n, f_t, \tau)$, 13 variables.
- No moment lifting. The edge set is polyhedral in $v$; costs are handled in §5.2.4.

**5.2.2 Regions.**
- Each mode is copied for each θ-interval $[\theta_i - h, \theta_i + h]$ (`theta_splits`, default 8 over $[-\pi, \pi)$).
- Optionally the contact position along the face is split too (`lambda_splits`, default 1).
- A region transition is allowed between the same interval or adjacent intervals, provided the step bound allows it.

**5.2.3 Linearization with a certified band.** For each nonconvex term $\phi(v)$ in §4.5:
- Expand to first order at the region center $\bar v$ (the θ-center; the center of the variable box for other factors).
- Replace $\phi(v) = 0$ by $|\phi_{\text{lin}}(v)| \le e_\phi$. The band $e_\phi$ is computed **offline** by interval arithmetic over the region box, as the sum of the second-order remainder and the bilinear cross terms.
- Specific terms:
  - $R(\theta)$: $\cos$ and $\sin$ remainders are bounded by $h^2/2$.
  - Product terms such as $(R(\theta) - R(\bar\theta))\,v$: the band is $h \cdot v_{\max}$. **This is first order in $h$.** Proposal §3 says Option A's error scales like (region radius)²; that holds for pure configuration terms only, not for bilinear ones (§19, item 6). Report the bands so the scaling is measured.
  - Torque $r_c \times f$: the band is (contact-position range) × (force range) for the bilinear remainder.
  - Rotation update: the true model is $\sin(\theta_{k+1}-\theta_k) = \Delta t\,\omega$ (§4.5.2, the same for every run). B uses $|\theta_{k+1} - \theta_k - \Delta t\,\omega| \le e_{\text{rot}}$, with $e_{\text{rot}} = \max_{|u|\le\sin\theta_{\text{step,max}}}|\arcsin u - u|$ (about 3e-3 rad at 15°). The step bound $|\theta_{k+1}-\theta_k| \le \theta_{\text{step,max}}$ is linear in θ.
  - **Both relaxations must contain the same true set $X_e$.** Run the §5.4 validity check for B with samples generated from the sine-form true model.
- Option `taylor_mode: mccormick`: use McCormick envelopes for the bilinear terms instead of bands. This is tighter and still polyhedral.
- Write a function `band_report(region)` that prints every band width. **Unit-test validity:** sample true feasible points in the region and check that each one satisfies the outer constraints (§5.4).

**5.2.4 Costs for B (keeps edges polyhedral, so the face cache is exact).**
- Replace each quadratic cost term by a valid **underestimator**: the maximum of `cost_planes` tangent planes (default 16 per term), using epigraph variables.
- The proposal allows $\hat\ell_e \le \ell_e$. The perspective of a maximum of affine functions is polyhedral.
- Flag `cost_model: soc` keeps the exact perspective-quadratic form. That makes the edges second-order-cone edges, so the face cache must be disabled.

### 5.3 Regions as modes and offline sizing (proposal §3)
- `regions.theta_splits` and `regions.lambda_splits` apply to every run. They multiply the modes and filter transitions.
- Function `size_regions(target_band)` (Taylor): returns the smallest split counts whose bands are all ≤ `target_band`. Print a table. This is "computed offline".
- Run A-reg uses B's region counts with the SDP relaxation, so the two relaxations are compared on the same trellis.
- **θ-regions in the SDP model.** A θ-interval $[\theta_i - h, \theta_i + h]$ becomes linear constraints on $(c, s)$, all valid for unit-norm points:
  - two half-planes bounding the angular sector;
  - the chord cut $c\cos\theta_i + s\sin\theta_i \ge \cos h$.
  The interval moment bounds (§5.1.4) are then recomputed on the smaller box.

### 5.4 Validity checks (milestone M1)
- Sample $10^4$ true feasible transitions per edge type: random states and forces satisfying §4.5, with the next knot computed exactly by the true update.
- **SDP:** form the rank-1 moment vector of each sample. Check every row and every PSD block (smallest eigenvalue ≥ −1e-10).
- **Taylor:** check every band row.
- Any violation is a bug that would make $L$ invalid. **The run must stop.**

---

## 6. Trellis assembly and data layout (`push_model.py`, section ASSEMBLY)

### 6.1 Graph
- Vertices: start, then $(k, \text{mode})$ for $k = 1..N-1$, then goal. Edges are listed layer by layer, in topological order.
- Prune edges with an empty slice: one batched feasibility LP per edge type, using CVXPY or HiGHS on CPU at build time.
- Prune edges on no start→goal path (forward/backward reachability).
- Save `problem.npz`: index arrays, edge-type ids, and all per-type matrices.

### 6.2 Batching by edge type
- Group edge types into **shape buckets**: contact-mid, noncontact-mid, start, goal, and (for B) one bucket per region pattern. Each bucket has identical array shapes, padded within the bucket.
- Pad inequality rows as $0^\top m \ge -1$ (always slack, never active). **Do not pad equality rows.** Equality counts are equal within a bucket by construction.
- For each edge: `type_id`, `tail_vertex`, `head_vertex`, `tail_slots` and `head_slots` (indices of its coupled coordinates in the vertex price vector), `layer`.
- Store matrices **per type**. Edges index into them with `type_id`, so memory scales with the number of types (≈ transitions per layer), not with $E$.

### 6.3 Coupling operator $K$ without sparse matrices
- $Kx$: `jax.ops.segment_sum` of each edge's signed coupled coordinates into vertex slots.
- $K^\top p$: gather the vertex prices at each edge's slots, with signs.
- Row and column absolute sums for diagonal step sizes are computed once with the same gather/scatter.

---

## 7. PDHG core (`run_trajopt.py`, section PDHG)

### 7.1 Variables
- **Primal, per edge:**
  - the moment vector $m_e$ (the polyhedral block);
  - one copy matrix $S_{e,b}$ per PSD block (SDP runs);
  - run B: the Taylor vector $v_e$ plus epigraph variables.
- **Dual:**
  - vertex prices $p_v \in \mathbb{R}^q$;
  - edge-local duals $\Lambda_{e,b}$ for the coupling $S_{e,b} = \mathcal B_{e,b}(m_e)$;
  - κ ≥ 2 only (§8.3): edge-local duals for the dualized equality rows.

### 7.2 One round (Chambolle–Pock, $\theta = 1$ extrapolation)
1. **Edges, in parallel.**
   - Polyhedral block: $m_e^+ = \arg\min_{m \in P_e} \langle c_e + K_e^\top p + \mathcal B_e^*(\Lambda_e),\, m\rangle + \tfrac12\|m - m_e\|^2_{T_e^{-1}}$. This is a QP (§8.1), with exact skipping (§9).
   - PSD copies: $S_{e,b}^+ = \Pi_{\succeq 0}\big(S_{e,b} + T_S\,\Lambda_{e,b}\big)$, a batched eigendecomposition.
   - **Convention.** Use the Lagrangian term $\langle \Lambda, \mathcal B(m) - S\rangle$ throughout. The $m$-step then carries $+\mathcal B^*(\Lambda)$ and the $S$-step $-\langle\Lambda, S\rangle$. At optimum $-\Lambda \succeq 0$. Unit-test the signs.
2. **Duals, in parallel.**
   - Vertex prices: $p^+ = \mathrm{prox}_{\sigma g^*}\big(p + \Sigma(K(2m^+ - m) - b)\big)$. Flow rows are exact. State rows use a δ-ball, i.e. the shrink operator for the dual of $\{\|r_v\| \le \delta\}$. Use ℓ∞ per coordinate, `delta` in config, default 1e-4.
   - Edge duals: $\Lambda^+ = \Lambda + \Sigma_\Lambda\big(\mathcal B(2m^+ - m) - (2S^+ - S)\big)$.

### 7.3 Step sizes
- Pock–Chambolle diagonal preconditioning with α = 1, applied to the full operator (vertex coupling, PSD coupling, and dualized rows if any): $T_j = \theta_{pd}/\sum_i |K_{ij}|$ and $\Sigma_i = 1/(\theta_{pd} \sum_j |K_{ij}|)$.
- Columns with no coupling get $T_j = \theta_{pd}$.
- Multiply all $T$ and $\Sigma$ by 0.99. This makes the PDHG metric $P$ strictly positive definite, which the safeguard (§11.6) needs, since its norms are $P$-norms.
- **Step-size settings.**
  - `theta_pd`: the primal/dual balance.
  - `theta_psd`: the balance for the PSD coupling (copies $S$ and duals $\Lambda$). Default `same`, i.e. equal to `theta_pd`.
  - `primal_weight_update`: adaptive balance (below). Default `none`.
- **Tuning (`theta_pd: auto`): every configuration is tuned on its own, under identical rules.** Tuning once on run D and reusing the result everywhere would bias the comparison.
  - Use the same grid (`tune.grid`), the same budget (`tune.rounds` = 2,000) and the same task as the job.
  - The full configuration is used, trees included.
  - Score: the normalized KKT error at the end (§7.4).
  - Store the result in `runs/tuned/<config>_<task>.yaml`. Report the chosen θ and the tuning time in the tables, separately from the solve time.
  - If the PSD coupling residual (logged separately) lags the vertex residual by more than 10×, tune `theta_psd` on the same grid as a second pass, and report that.
- **`primal_weight_update: pdlp`** (optional; if switched on, it is on for every run).
  - At each restart, $\omega \leftarrow \exp\!\big(\tfrac12\log(\|\Delta y\|/\|\Delta x\|) + \tfrac12\log\omega\big)$, where $\Delta x, \Delta y$ are the primal and dual moves since the previous restart.
  - Then set $\theta_{pd} = 1/\omega$.
  - Skip the update when either move is below 1e-10.
  - The PSD balance follows the same ω unless `theta_psd` is set explicitly.
- **Use svec scaling** ($\sqrt2$ on off-diagonals) for all symmetric-matrix variables, so the inner products and the PDHG metric are consistent.

### 7.4 Restarts (proposal §4: "restarts help on sharp problems")
- PDLP-style restart to the average. Track the running average since the last restart.
- Compute a normalized KKT error (primal residual, dual residual, gap estimate) for the current iterate and for the average.
- Restart to the better candidate when its KKT error ≤ 0.2 × the error at the last restart. Also restart when it is ≤ 0.8 × that error and is larger than the previous candidate's.
- A fixed-frequency option is available as a fallback.
- Restarts move every edge's center. Skipping handles this automatically (§9.1 uses the current center).

### 7.5 Execution
- Write the round as one jitted function `round_fn(state) -> state, stats`.
- Run blocks of `rounds_per_scan` rounds (default 50) with `jax.lax.scan`. Synchronize with the host only between blocks, for logging, the certificate, the tree, extraction and checkpoints.
- `state` is a single pytree: primal, dual, stored duals, the face cache, counters, restart averages.
- Checkpoint `state` every `checkpoint_every` rounds so runs can resume.
- **Compile everything before the clock starts.**
  - Ahead-of-time compile every variant with `jax.jit(f).lower(...).compile()` before any timer starts: the scan block, every IPM bucket size, the tree attempt and its trials, the certificate passes, and the stopping test.
  - In `--mode timing`, any compilation after the clock starts is an error. Count compilation events, e.g. with `jax_log_compiles` and a logging handler, and abort if the count changes.

---

## 8. Edge block solvers (`run_trajopt.py`, section EDGE_SOLVERS)

### 8.1 Polyhedral block, κ = 1 and run B: batched exact-to-tolerance IPM
- Primal-dual Mehrotra predictor–corrector for $\min \tfrac12 m^\top H m + f^\top m$ subject to equality rows $A m = 0$, inequality rows $G m \ge h$ (only the $m_0 \le 1$ row has a nonzero right-hand side), and variable bounds.
- $H = \mathrm{diag}(T^{-1})$ is diagonal. Eliminate the bounds diagonally, then solve the reduced normal equations with dense Cholesky: size 136 for κ = 1, about 13–30 for B.
- Fixed iteration cap (`ipm_iters`, default 40), tolerance `ipm_tol` 1e-11. Use `vmap` over a bucket.
- Return the solution, the multipliers of every row, and the **active set** (rows with slack < 1e-9 and multiplier > 1e-9).
- At the end, clean the output: if every entry of the solution is < 1e-13 in magnitude, set it to exactly 0.

### 8.2 PSD projection
- `jnp.linalg.eigh` with `vmap` over all blocks of a bucket. Clip eigenvalues at 0 and reconstruct.
- **Gershgorin skip** (exact, optional, default on): if every Gershgorin disc of the argument lies in $(-\infty, 0]$, the projection is exactly 0, so skip the eigendecomposition. Count these.
- Keep the eigenvectors of the latest projection. The optional frozen-rank Jacobian (§11.1) uses them.
- **Benchmark batched `eigh` first** (milestone M0): 16×16 (κ = 1), 136×136 (κ = 2), 816×816 (κ = 3), at realistic batch sizes. Some GPU `eigh` paths loop over large matrices instead of batching them.
  - If 136×136 is slow, add `psd_proj: polar_ns`: $\Pi(X) = (X + |X|)/2$, with $|X| = X\,\mathrm{sign}(X)$ computed by batched Newton–Schulz iterations (matmuls only, FP64 tensor cores).
  - Accept its result only after checking the projection error, $\|X_+ X_-\|_F$ and $\lambda_{\min}(X_+)$, against `psd_proj_tol`. The PDHG then runs inexactly to tolerance (Rasch & Chambolle); log the errors.
  - The optional frozen-rank Jacobian always uses `eigh`.

### 8.3 κ ≥ 2: split the equalities too (deviation, §19)
- For κ ≥ 2 the polyhedral block has thousands of variables, which is too large for a dense IPM on each edge.
- So dualize the equality rows as well (edge-local duals). The primal step for $m_e$ then becomes a **box projection** (bounds plus $0 \le m_0 \le 1$), which is exact and trivial.
- Inequality rows are already handled by the localizing PSD blocks plus their linear copies. Dualize those linear rows too, with a nonnegative-cone dual projection.
- Flag `qp_block: exact_ipm | split_rows`. Default `exact_ipm` for κ = 1 and `split_rows` for κ ≥ 2.

---

## 9. Exact skipping (`run_trajopt.py`, section SKIPPING) — proposal §4

### 9.1 Zero test (exact, any convex block; default on in all runs)
- Polyhedral block step data: $g_e = c_e + K_e^\top p + \mathcal B_e^*(\Lambda_e) - T_e^{-1} m_e$. The step is zero **iff** $\pi_e(g_e) = g_{e,0} + \min_{m \in P_e \cap \{m_0 = 1\}} g_{e,\setminus0}^\top m \ge 0$. This is Theorem 1 of the edge-solve report.
- Test with stored multipliers $(\nu_e, \mu_e \ge 0)$ for the rows of $P_e$:
$$
\mathrm{LB}_e(g) = g_0 + b_{\text{eq}}^\top \nu + b_{\text{in}}^\top \mu + \sum_i \min(r_i \underline m_i,\ r_i \overline m_i),\qquad r = g_{\setminus0} - A_{\text{eq}}^\top \nu - A_{\text{in}}^\top \mu .
$$
  Here $b$ collects each row's $m_0$-coefficient (with sign), and the box comes from §5.1.4. If $\mathrm{LB}_e(g_e) \ge$ `lb_tol` (1e-12), set the step to exactly 0 and skip the solve.
- **Refresh:** if the test fails and the edge's center is 0, solve the *pricing LP* $\min_{P_e \cap \{m_0=1\}} g^\top m$ (batched IPM, linear objective, same machinery). Store its duals. If its value is ≥ 0, the step is 0. Otherwise run the step QP and store the QP's row multipliers as the new $(\nu, \mu)$.
- **Unit test:** at optimal duals, LB equals the pricing value (to 1e-10). Random dual-feasible multipliers always give LB ≤ value. "Zero step ⇔ π ≥ 0" holds on 1,000 random edges.

### 9.2 Face cache (exact for polyhedral blocks; default **on for B**, off for the SDP runs per the agreed default)
- After a solve with active set $\mathcal A$, store the KKT inverse for (edge type, $\mathcal A$).
- Next round, compute the candidate as one matrix–vector product. **Verify:** all inactive rows are feasible (≥ −1e-11) and all active inequality multipliers are ≥ −1e-11. On success there is no solve.
- Storage: for a first version, keep a per-edge padded inverse. Later, use a shared table keyed by (type, hash of $\mathcal A$).
- **Note:** with decision 1(a), the polyhedral block of the SDP runs is also polyhedral, so the cache is exact there too. Keep the flag `face_cache: true|false` per run. Turning it on for the SDP runs is a free extra experiment (A-cache).

### 9.3 Compaction, so skipping actually saves GPU time
- Each round, first run the zero test and the cache check on **all** edges. These are cheap, vectorized operations.
- Gather the indices that still need a solve with `jnp.nonzero(mask, size=B)`, where $B$ is the smallest bucket size from {64, 128, ..., E} that fits. Run the IPM on that buffer only, and scatter the results back.
- Pre-compile one IPM per bucket size; there are only a few.
- Flag `compaction: none` (compute everything under a mask; simplest, used for verification) or `bucketed` (the default).

### 9.4 Counters, per round
`n_zero`, `n_cache_hit`, `n_cache_miss`, `n_qp_solve`, `n_pricing_refresh`, `n_eigh`, `n_gershgorin_skip`.

---

## 10. Certificate (`run_trajopt.py`, section CERTIFICATE) — proposal §5

### 10.1 Free bound, every round (`free_every`, default 1, as proposal §4 step (d) and §5 specify)
- Compute the bound inside the `lax.scan` round and keep the running maximum $L_{\text{best}} = \max_{\text{rounds}} L$. Any prices give a valid bound, so the maximum is valid.
- State prices $\lambda$: the state rows of $p$, with flow rows set to zero.
- For each edge: $g^{\text{cert}}_e = c_e + K_e^\top(0, \lambda)$, with **no** center term and **no** $\Lambda$ term.
- The edge's bound covers its whole outer set, including the PSD blocks:
$$
\mathrm{LB}_e = g_0 + b_{\text{eq}}^\top\nu + b_{\text{in}}^\top\mu - \langle \hat\Lambda, \mathcal B_0\rangle + \sum_i \min(r_i\underline m_i, r_i \overline m_i),\quad r = g_{\setminus0} - A_{\text{eq}}^\top\nu - A_{\text{in}}^\top\mu - \mathcal B^*_{\setminus0}(\hat\Lambda),
$$
  with $\hat\Lambda = \Pi_{\succeq0}(-\Lambda_e)$ under the §7.2 convention. Reason: for $\Gamma\succeq0$ and $\mathcal B(m)\succeq0$, $\langle\Gamma,\mathcal B(m)\rangle\ge0$, so $g^\top m \ge (g - \mathcal B^*\Gamma)^\top m$. Every multiplier is projected onto its dual cone, so the bound is valid for **any** stored values. Unit test: it never exceeds Clarabel's $\pi_e$.
- Then $L = \mathrm{SP}[\mathrm{LB}_e] - \delta\sum_v \|\lambda_v\|_1$ (ℓ∞ tolerance, so the dual norm is ℓ1).
- **Shortest path:** a DP over layers with `jax.ops.segment_min` over incoming edges. That is O(E) work with $N$ sequential steps. A min-plus `associative_scan` gives $\log N$ depth, but is optional.

### 10.2 Exact bound and lazy evaluation (`exact_every`, default 200 rounds)
- Exact $\pi_e$: solve each edge's pricing problem over its full outer set, including PSD blocks, at $g^{\text{cert}}_e$. Use CPU Clarabel or SCS in a process pool (`cert_workers`, default 32), or a batched GPU first-order solver if implemented.
- **Soundness:** never use the solver's objective value. Plug its duals into the LB formula above, after projecting them onto their cones. Floating point gets a signed slack: add `-eps_cert * (1 + |terms|)` per edge, with `eps_cert` = 1e-12.
- **Lazy SP:** start from the free LB values. Repeat: find the shortest path; evaluate every not-yet-evaluated edge on it exactly; stop when the path consists only of evaluated edges. Log the number of evaluations.
- Log both the full exact $L$ (all edges; only every `full_exact_every` = 1,000 rounds, for diagnostics) and the lazy one.

### 10.3 Tighter bounds on the true sets (proposal §5, "tighter, periodically"; default on, every 500 rounds)
- For edges on the current lazy shortest path only, run a small branch-and-bound on θ: split the edge's θ range into ≤ `bnb_max_splits` (8) pieces.
- Bound each piece with the same outer model restricted to that piece, and take the minimum over pieces. The result is valid and tighter.
- Replace the edge's value in the SP and recompute $L$. Log the improvement.

### 10.4 Ceiling diagnostic
On small instances only: solve the full relaxation with CVXPY to get $J_{\text{rel}}$. Check $L \le J_{\text{rel}}$ at every logged round.

### 10.5 Stopping test on P_δ (the η stop)
- **P_δ** is the fractional relaxation of proposal §2: the declared outer sets, flow conserved, states conserved up to δ. Every $L$ of §10.1–10.3 satisfies $L \le J_{P_\delta} \le J^\star$.
- **$U_{\text{flow}}$** is the relaxed cost $\sum_e \hat\ell_e(m_e)$ of a primal candidate. It is an upper bound on $J_{P_\delta}$ when the candidate is feasible for $P_\delta$.
- **Candidates:** the current iterate and the restart average (§7.4). Check both and use the smallest $U_{\text{flow}}$ among those that pass.
- **Feasibility checks.** PDHG reaches these only in the limit, so each uses a signed numerical tolerance, logged at stop:
  1. Flow conservation: $|K_{\text{flow}}m - b_{\text{flow}}| \le$ `eps_flow` (1e-6) at every vertex. Flow is exact in the proposal; this is only numerical slack.
  2. State mismatch: $\|K_{\text{state}}m - b_{\text{state}}\|_\infty \le \delta$ at every vertex. This includes second-moment coordinates when they are coupled.
  3. Each edge's polyhedral rows hold to the IPM tolerance. For κ ≥ 2 (`split_rows`), also the dualized rows, to `eps_row` (1e-8).
  4. Each PSD block: $\lambda_{\min}(\mathcal B_{e,b}(m_e)) \ge -\,$`eps_psd`$\cdot(1 + \|\mathcal B_{e,b}(m_e)\|)$, with `eps_psd` = 1e-8. Use batched `eigvalsh` on the blocks.
- **Stop** when a candidate passes 1–4 and $U_{\text{flow}} - L_{\text{best}} \le \eta\cdot\max(1, |L_{\text{best}}|)$, with η = `stop.eta_rel` (1e-3). $L_{\text{best}}$ is the best certified bound so far (free, lazy, B&B).
- **Assert** $U_{\text{flow}} \ge L_{\text{best}} - 10^{-8}\max(1, |L_{\text{best}}|)$. A violation means a bug in the bound or in the checks; stop with an error.
- **When:** at every scan-block boundary (every 50 rounds), where the host already synchronizes.
- **What the tolerances affect.** They change only when the run stops. $L$ stays rigorous whatever they are.
- **Status** goes to `cert.json`: `converged_eta`, `round_cap`, `time_cap` or `stalled`. Only `converged_eta` counts as reaching η in comparisons; the others are reported with their final gap.
- **Not used for stopping:** the true-model cost of an extracted guess (§13.4). It is not a feasible trajectory. The trajectory gap $U_{\text{traj}} - L$ is computed only once LEAP exists.

---

## 11. Segment trees (`run_trajopt.py`, section TREES) — proposal §4 (tree pass) and §12

All tree types (exact, C-a, C-b) share §11.1 and §11.6. They differ only in the model of the price operator they solve.

### 11.1 Common pieces

**What the tree models (proposal §4).**
- Freeze active sets. Each edge update then responds linearly to its endpoint prices.
- The price operator $S = \sum_e K_e J_e K_e^\top$ is SPD and block tridiagonal by layer. Its block width is $w = M_{\text{layer}}\cdot q$, where $q$ is the number of coupled coordinates per vertex: 28 for SDP with second-moment coupling, 7 without, 6 for run B.
- The tree solves a model of $S$. The safeguard (§11.6) decides whether the result is used.

**Edge Jacobian $J_e$** (exact tree and C-a only; C-b never forms one).
- The edge update is the polyhedral-block prox QP of §8.1, with the PSD duals $\Lambda$ held fixed.
- **Default, `tree.jacobian: one_step`.** Freeze the active set of the last solve: active rows and active variable bounds become equalities.
  - This is exactly the response that a trial with changed prices realizes, since the trial keeps the centers and $\Lambda$ (see *Direction* below).
  - It is also what C-b measures from its trials, so all three tree types model **the same object**.
  - For run B, which has no PSD blocks, it is the only option.
- **Option, `tree.jacobian: frozen_rank`** (SDP runs). Also freeze the rank of each PSD copy:
  - let $V$ be the eigenvectors of $S^+_{e,b}$ with eigenvalue > `tree.eig_tol` (relative, 1e-8) and $V_\perp$ the rest;
  - add the equalities $V_\perp^\top\,\mathcal B_{e,b}(m)\,V_\perp = 0$.

  These describe the tangent space of the fixed-rank PSD set, so a rank-1 moment matrix can still rotate, i.e. its state can change. Freezing the whole face would hide the state response.

  This option models the response *after* $\Lambda$ has adapted. A prices-only trial does not realize it in one step, so it is an experiment, not the default.
- **Formula.** Stack the frozen equalities as $E_e$ and let $H_e = \mathrm{diag}(T_e^{-1})$. Then
$$J_e^{\text{full}} = H^{-1} - H^{-1}E_e^\top\,(E_e H^{-1}E_e^\top)^{+}\,E_e H^{-1}.$$
  Compute the pseudo-inverse by `eigh` with relative tolerance 1e-12, because frozen rows can be redundant.
- $S$ receives $K_e J_e^{\text{full}} K_e^\top$ on the edge's coupled coordinates ($2q-1$ of them; flow is shared).
- Batch over edges in buckets. **Count each $J_e$ as one derivative evaluation.**

**Exact assembly.**
- Scatter the edge blocks into the diagonal blocks $D_k$ and off-diagonal blocks $E_k$ ($w\times w$).
- Add `tree.reg` · (mean diagonal) · I, default 1e-8.

**Right-hand side** (proposal §12.4: "the uncompressed chosen model's residual on the right").
- Compute it from the **actual** edge outputs $m^+$ of the current ordinary round, never from a compressed or reference model:
$$r = \Sigma_p^{-1}\,(\hat p - p),\qquad \hat p = \mathrm{prox}_{\sigma g^*}\!\big(p + \Sigma_p(Km^+ - b)\big)\quad\text{(non-extrapolated dual update).}$$
- For flow rows, $r = Km^+ - b$. For state rows, $r$ includes the δ-ball shrink.
- $r = 0$ exactly at a PDHG fixed point, so every tree type returns $\Delta p = 0$ there. Tested in M7: "A zero residual must produce zero correction even after compression."

**Direction: trees propose price changes only, for all three tree types.**
- Current point: $z = (m, S, p, \Lambda)$. The tree solves $\tilde S\,\Delta p = r$ for the vertex prices. The direction is $d = (0, 0, \Delta p, 0)$.
- The edges' response to the new prices is obtained by the **actual edge updates** inside the check (§11.6): $T(z + \tau d)$ re-solves the edges, with exact skipping, at the same prox center $m$ and prices $p + \tau\Delta p$.
- Using the same form for all tree types keeps the comparison fair. It also means each trial evaluation and the ordinary round $T(z)$ share the same centers, so their difference is a clean measurement of the edges' response to a price change. C-b uses this measurement as data (§11.5).
- Optional flag `tree.primal_predict: true` (exact tree and C-a only): also move the center to $m^+ - J^{\text{full}}K^\top\Delta p$. Default false.

### 11.2 Exact tree (runs A, A-κ, A-reg, B): odd–even block cyclic reduction
- **Level loop.** All blocks of a level are processed in parallel with batched dense algebra.
  - Eliminate the odd layers:
    $D'_{2i} = D_{2i} - E_{2i-1}^\top D_{2i-1}^{-1}E_{2i-1} - E_{2i}D_{2i+1}^{-1}E_{2i}^\top$,
    $E'_{2i} = -E_{2i}D_{2i+1}^{-1}E_{2i+1}$,
    and the right-hand side the same way.
  - Recurse on the even layers.
  - Back-substitute the odd layers on the way down.
  - The Cholesky factors of $D_{\text{odd}}$ are computed once per level and reused in both passes.
- Pad $N$ to a power of two with identity blocks. Depth $2\lceil\log_2 N\rceil$.
- **Cost** $O(Nw^3)$:
  - A, $w = 560$: milliseconds;
  - B, $w = 160\cdot6 = 960$;
  - A-reg, $w = 160\cdot28 = 4480$: about $3\times10^{12}$ flops and about 10 GB per pass, so it gets its own GPU.
- **Unit test:** matches a dense solve of the assembled $S$ to relative error < 1e-10.

### 11.3 Compressed tree engine (shared by C-a and C-b) — proposal §12.3
**Node model.**
- A node covers a time interval $I$ with boundary layers $(a, b)$. Its boundary vector holds all coupled coordinates of layers $a$ and $b$; for C-b, also the reference's latent channels on those layers (§11.5).
- It stores
$$\tilde S_I = R_I + U_IU_I^\top,\qquad U_I\in\mathbb R^{n_\partial\times r_I},\ r_I\le r,$$
  where $R_I$ is a structured SPD part (§11.4, §11.5).
- **Leaves** are single layers of edges, $[k, k+1]$. **Merges** join $[a, m]$ and $[m, b]$.

**Merge (exact formula).**
1. On layers $(a, m, b)$: $R = R_L \oplus R_R$ (added on the shared layer $m$) and $V = [U_L\ U_R]$, embedded with zeros. $V$ has $k = r_L + r_R \le 2r$ columns.
2. Eliminate $m$. Write $\partial = \{a, b\}$ and set
$$G = R_{mm}^{-1}V_m,\quad C = I_k + V_m^\top G,\quad X = V_\partial - R_{\partial m}G .$$
   Then the Schur complement of the children's model is
$$(R + VV^\top)/m \;=\; R/m \;+\; X C^{-1} X^\top .$$
   $R/m$ stays structured (§11.4, §11.5). The correction $W = XC^{-1}X^\top$ is PSD with rank ≤ $k$.
   **Unit test:** check this identity against a dense computation.
3. **Truncate** to rank $r$:
   - thin QR, $X = Q_X R_X$;
   - `eigh` of the $k\times k$ matrix $R_X C^{-1}R_X^\top$;
   - keep the top $r$ eigenpairs, $U_{\text{parent}} = Q_X V_r \Theta_r^{1/2}$.
   The discarded part $\Delta = W - U_{\text{parent}}U_{\text{parent}}^\top$ is PSD. The parent's model, $R/m + U_{\text{parent}}U_{\text{parent}}^\top$, is still $\succeq R/m \succ 0$.
4. **Consistency (12.3: "factors and right-hand-side information must remain consistent").**
   - The parent's matrix equals the exact Schur complement of the children's models with the boundary block $A_{\partial\partial}$ replaced by $A_{\partial\partial} - \Delta$.
   - The right-hand-side condensation, $\rho_\partial \leftarrow \rho_\partial - A_{\partial m}A_{mm}^{-1}\rho_m$, and the downward recovery, $x_m = A_{mm}^{-1}(\rho_m - A_{m\partial}x_\partial)$, do not involve $A_{\partial\partial}$. They are therefore exact for that modified model.
   - Applied at every merge, the whole up–down pass is an **exact solve of one well-defined global SPD matrix** $\tilde S$. Hence:
     - (i) the map $r\mapsto\Delta p$ is linear, symmetric and positive definite;
     - (ii) a zero residual gives a zero correction;
     - (iii) the stored factors and the condensed right-hand side match the compressed model.
   - Truncating a matrix while condensing the right-hand side some other way (or averaging unrelated states) breaks all three. Do not do it.
5. **Per-node storage for the downward pass:** the structured factor of $R_{mm}$, plus $V_m$, $G$, $C^{-1}$ ($k\times k$), $V_\partial$ and the condensed $\rho$. $A_{mm}^{-1}$ is applied by Woodbury: $A_{mm}^{-1} = R_{mm}^{-1} - G C^{-1}G^\top$.
6. **Root:** boundary = the start and goal layers. The rank cap applies there too. Solve the small root system densely.

**Cost** (12.5): $O(n_\partial k^2 + k^3)$ per merge, i.e. $O(N[wr^2 + r^3])$ for the low-rank algebra, plus the structured $R$ work. Log the **critical path** (sum over levels of the slowest merge) separately from total work.

**Unit tests (M7):**
- no truncation gives the same result as the uncompressed model;
- the tree map, applied to unit vectors on a tiny instance, is symmetric and positive definite;
- zero right-hand side gives exactly zero;
- the merge identity in step 2.

### 11.4 Option (a), run C-a: compressed exact responses
- **Exact leaf block** $L_k = \sum_{e\in\text{layer }k}K_eJ_e^{\text{full}}K_e^\top$ on layers $(k, k+1)$. It is applied only through matvecs, which use the $J_e$ (counted as derivative evaluations).
- **Spectral compression** with a randomized range finder: $r + 8$ probe matvecs, then a Rayleigh–Ritz step, giving Ritz values $\theta_1\ge\dots\ge\theta_{r+8}$ and vectors $Q$.
  - Base $R_{\text{leaf}} = \beta_k I$ with $\beta_k = \theta_{r+1}$ (the level of the tail of the spectrum).
  - Correction $U_{\text{leaf}} = Q_r(\Theta_r - \beta_k I)^{1/2}$. This is PSD by construction.
  - *A block-diagonal base does not work here:* "exact minus its diagonal blocks" is indefinite (zero diagonal blocks), so it cannot equal $UU^\top$.
- **Structured part:** $R$ is a multiple of the identity on each layer, with no coupling across layers, so $R/m = R_{\partial\partial}$. All cross-layer coupling is carried by the capped corrections.
- **Count:** probe matvecs, $J_e$ evaluations, Rayleigh–Ritz time.

### 11.5 Option (b), run C-b: **Section 12 of the proposal, implemented faithfully**
Keep the subsection numbers (12.1–12.5) as code comments.

**12.1, what is predicted.**
- C-b predicts the *optimizer's response* of edge updates to endpoint prices, not the robot's feasible set.
- The outer models, contact constraints, objective and certificate are exactly those of §5 and §10. C-b changes only the tree direction. Add an `assert` that C-b code never writes problem data or $L$.

**12.2, simple reference motion: a position–velocity response, repeated across coordinates and mode pairs.**
- **Channels.** Each coupled coordinate $j$ at each vertex gets `channels_per_coord` = 2 channels:
  - a **value** channel: it is the vertex coordinate itself, receives the residual $r$, and returns $\Delta p$;
  - a latent **trend** channel: its right-hand side is 0 and it is never output.
- **Reference edge** (identical for every allowed transition $(m\to m')$ and every coordinate, unit-scaled). It is the response $J^{\text{unit}} = (\nabla^2 h)^{-1}$ ($4\times4$, SPD) of
$$h(a,v,a',v') = \tfrac12(a^2 + a'^2) + \tfrac{1}{2}\gamma_v^{-1}(v^2+v'^2) + \tfrac12\,\frac{w_c}{\Delta t^2}\Big(a'-a-\Delta t\,\tfrac{v+v'}{2}\Big)^2 + \tfrac12\,\frac{w_a}{\Delta t^2}(v'-v)^2,$$
  with $(a, v)$ the value and trend at the tail and $(a', v')$ at the head.
  - The $w_c$ term says the value changes by the trend times $\Delta t$.
  - The $w_a$ term says the trend changes slowly.
  - Together they mean a change of trend now changes values later (12.2: "changing velocity now also changes position later").
  - Knobs: `gamma_v`, `w_c`, `w_a`. They are dimensionless.
- **Per-coordinate scale.** $J_{\text{ref},j} = t_j\,J^{\text{unit}}$, where $t_j$ is the mean PDHG primal step $T$ of coordinate $j$ over edges, i.e. the response of an unconstrained prox. Hence
$$S^{\text{ref}} = R^{\text{small}}\otimes \mathrm{diag}(t),$$
  where $R^{\text{small}}$ is built from $J^{\text{unit}}$ on the (layer, mode, channel) index only. Schur complements preserve this Kronecker form, because each coordinate is $t_j$ times the same matrix. **Do all reference algebra on $R^{\text{small}}$** (per node $4M_{\text{layer}}\times4M_{\text{layer}}$, i.e. 80×80 for $M = 20$).
- **Mode graph.** Reference edges exist only on allowed, unpruned transitions. For a general (non-complete) mode graph this is the "$O(NM^3)$ reference setup" of 12.5. Applying $(R^{\text{small}})^{-1}\otimes\mathrm{diag}(1/t)$ to correction directions is the "$O(NM^2Dr)$" term. Time both.
- **Flow** is one of the $q$ coordinates, with its own $t_j$.
- **Zero-flow edges.** Because the reference is the same on zero-flow edges, C-b can propose changes through edges whose local derivative is zero (12.2). Log **activations**: edges with flow < 1e-9 before an accepted tree step and ≥ 1e-6 after it. Log the same per ordinary round for comparison. Activation is not guaranteed; the check decides.

**12.3, what each segment retains: corrections from saved edge inputs and outputs.**
- **Data source.** The trial evaluations of §11.6 and the ordinary round $T(z)$ share the same prox centers and $\Lambda$. They differ only in the prices: $p$ vs $p + \tau\Delta p$. So for every leaf (one layer of edges, boundary layers $k$ and $k+1$), each trial gives a **clean response pair**:
$$d^\partial = \tau\Delta p\big|_{\text{layers }k,k+1},\qquad y^\partial = -K_{\text{leaf}}\big(m^+(p+\tau\Delta p) - m^+(p)\big).$$
  Here $y^\partial \approx S_{\text{leaf}}\,d^\partial$, with $S_{\text{leaf}}$ the leaf's true linearized response.
- **Probing** (optional, `probe_solves_per_refresh`, counted): evaluate all edges at the current center with prices $p + \epsilon\,\xi$, where $\xi$ is random and boundary-restricted, with independent directions for each leaf. One probe is one round of edge work and gives one pair for every leaf at once. $\epsilon$ = `probe_scale` × ‖r‖ / max(t).
- **Ordinary-round differences** (`secant_sources` includes `rounds`; default off). Successive rounds change the prox centers as well as the prices, so these pairs mix two effects. They are available for comparison only.
- **Storage:** a ring buffer of the last `secant_history` pairs per leaf (default max(8, r)). Only vertex-sized vectors are stored, never edge data.
- **Fit** (positive block SR1 on top of the reference):
  1. Stack the pairs: $D = [d^\partial_i]$, $Y = [y^\partial_i]$.
  2. Orthogonalize $D$ by pivoted QR and drop near-dependent columns (tol 1e-8).
  3. Residual of the reference: $\hat Y = Y - S^{\text{ref,val}}_{\text{leaf}}D$, where $S^{\text{ref,val}}_{\text{leaf}}$ is the leaf's reference block with its latent channels eliminated locally.
  4. $C = \mathrm{sym}(D^\top\hat Y)$ with eigenpairs $(c_i, v_i)$. Keep only $c_i > 10^{-10}\max|c|$ (the curvature condition). This gives the *chosen positive correction model* of 12.3.
  5. $U_{\text{leaf}} = \hat Y\,V_+\,\mathrm{diag}(c_+^{-1/2})$, then truncate to rank $r$. $U$ is zero on latent rows.
  6. **Unit test:** when $D^\top\hat Y$ is symmetric positive definite and the number of pairs is ≤ r, the model reproduces the pairs: $(S^{\text{ref,val}} + UU^\top)D = Y$.
- **The rank cap limits the proposal model only.** The code comment must say that the true mechanics are not assumed to be low rank.
- **Merges** use the engine in §11.3, with the cap at every level and at the root. The structured part is the Kronecker reference: $R/m$ is computed on $R^{\text{small}}$.

**12.4, how it fits into one iteration.** Every `tree.every` rounds (default 10):
1. **Build or refresh the proposal model** from saved pairs. Optionally add probes. The very first attempt has no pairs, so it uses the pure reference ($U = 0$). **Count** every new edge solve, derivative evaluation (none in C-b) and probe.
2. **Pass upward, then downward** (§11.3), batched per level. Right-hand side: the uncompressed residual $r$ of §11.1. Output: $\Delta p$. Zero residual gives zero correction.
3. **Check the actual update** with the safeguard (§11.6). Trials evaluate the original edge updates with exact skipping. A failing check is damped or rejected according to the safeguard, with the ordinary PDHG step as the fallback. **Each trial counts as one round of edge work.** Every trial also yields new pairs, whether it is accepted or not.
- **Rules for the bound and the logs:**
  - The reference model's predicted cost is never used as $L$. $L$ is written only by §10 functions; enforce this with an `assert`.
  - Log exact-reuse hit rates (zero test, cache, Gershgorin) **separately** for trial evaluations and ordinary rounds. A tree step can cross several active regions, so ordinary-PDHG hit rates need not carry over.

**12.5, cost and evaluation.**
- **Time separately:**
  - reference setup;
  - internal tree construction (low-rank algebra);
  - application of reference factors to corrections;
  - pair acquisition (trials, probes);
  - aggregation over the $E$ edges (forming $y^\partial$);
  - the safeguard trials;
  - the tree's critical path.
- **Empirical quantities to report:** useful rank, reference quality, acceptance rate, and total time to the requested gap.
- **Reference quality, per attempt:**
  - the ratio $\|r(z + \tau d)\| / \|r(z)\|$ (the linear model predicts 0);
  - on small instances, with `--diag_compare_exact`: the cosine and the relative error between C-b's $\Delta p$ and the exact tree's $\Delta p$ at the same point. Diagnostic work is excluded from timings.
- **Compare rank choices by total checked progress per unit time, including failures and fallback work** (§14).

### 11.6 Safeguard (proposal §4: "SuperMann or a hybrid proximal extragradient test"), shared by all trees
**Setting.**
- One ordinary round is the operator $T$: preconditioned PDHG, a proximal-point step in the metric
$$P = \begin{bmatrix}T^{-1} & -K^\top\\ -K & \Sigma^{-1}\end{bmatrix}.$$
  Here $K$ includes the vertex coupling and the PSD coupling. $T$ is firmly nonexpansive in $\|\cdot\|_P$ when $P\succ0$; the 0.99 step-size factor in §7.3 ensures this.
- $R(z) = z - T(z)$. $T(z)$ is already computed by the ordinary round, so $R(z)$ and the fallback step are free.
- All norms and inner products are in $P$; they cost a few $K$ matvecs.

**Default: SuperMann (Themelis & Patrinos 2019), with the blind step disabled.** Parameters $c_1 = 0.99$, $q = 0.99$, $\sigma = 0.1$, relaxation $\lambda = 1$, backtracking $\beta = 0.5$, `max_trials` = 3.
- The K0 step (accept $z+d$ without evaluating it, when $\|R(z)\|\le c_0\eta$) is off by default ($c_0 = 0$), because 12.4 requires checking the actual update. `c0 > 0` enables it.
- **Per attempt, with $\tau = 1$:**
  1. $w = z + \tau d$. Evaluate $T(w)$ (one round of edge work, with skipping). Set $R_w = w - T(w)$.
  2. **K1** (accept the step): if $\|R(z)\|_P \le r_{\text{safe}}$ and $\|R_w\|_P \le c_1\|R(z)\|_P$, set $z^+ = w$ and $r_{\text{safe}} \leftarrow \|R_w\|_P + q^{j}$, where $j$ counts accepted K1 steps.
  3. **K2** (projection step): let $\rho = \|R_w\|_P^2 - \tau\langle R_w, d\rangle_P$. If $\rho \ge \sigma\|R_w\|_P\|R(z)\|_P$, set $z^+ = z - \lambda\,(\rho/\|R_w\|_P^2)\,R_w$. This projects onto a halfspace that contains every fixed point, since $T$ is firmly nonexpansive.
  4. Otherwise $\tau \leftarrow \beta\tau$ and repeat, up to `max_trials`.
  5. **Fallback:** $z^+ = T(z)$, the ordinary PDHG step. It is the K2 step with $\tau = 0$, so the scheme stays inside SuperMann's convergence framework.
- **Initialize** $r_{\text{safe}} = \|R(z_0)\|_P$.
- **Restarts** (§7.4) reset $r_{\text{safe}}$ and the trial counters.
- **Before coding, check the conventions against Algorithm 2 of the paper.**

**Option `safeguard.type: hpe`** (Solodov & Svaiter; Monteiro & Svaiter).
- **Trial:** $w = z + \tau d$, $\tilde z = T(w)$, $u = w - \tilde z$. Then $P u \in A(\tilde z)$ exactly, by the prox identity.
- Let $\lambda^* = \langle z - \tilde z, u\rangle_P / \|u\|_P^2$.
- **Accept** if $\lambda^* \ge \lambda_{\min}$ (0.1) and $\|\lambda^* u + \tilde z - z\|_P \le \sigma_{\text{hpe}}\|\tilde z - z\|_P$ ($\sigma_{\text{hpe}}$ = 0.9). Then set $z^+ = z - \lambda^* u$.
- If $\|u\|_P$ ≤ 1e-14·$\|z\|_P$, $\tilde z$ is a solution: accept $z^+ = \tilde z$.
- Otherwise damp $\tau$ as above, then fall back to $T(z)$. With $d = 0$ the test gives exactly $z^+ = T(z)$.
- The guarantee is the averaged $O(1/\sum\lambda)$ restricted saddle gap, for the $\lambda$-weighted average of the $\tilde z$.

**Counters, per tree type:** `tree_attempts`, `accepted_K1`, `accepted_K2` (or `accepted_hpe`), `accepted_at_tau{1,0.5,0.25}`, `fallback`, `trial_rounds`, `trial_edge_solves`, `probe_rounds`, `derivative_evals`.

---

## 12. Main loop and outputs (`run_trajopt.py`, section MAIN)

### 12.1 Flow (proposal §8, without the local solver)
1. **Load the config** and build or load `problem.npz` (§6). On build, run the validity checks (§5.4).
2. **Initialize** state: zeros (cold start), or `--warm_start <run_dir>` for later replanning experiments.
3. **Loop until the η stop (§10.5) fires.** Safety stops: `rounds_max` (50,000), `time_budget_s` (1,800 s), or `stalled` (neither $L_{\text{best}}$ nor the best $U_{\text{flow}}$ improves by more than 1e-6 relative over `stall_rounds` = 2,000 rounds). A safety stop marks the run as not converged. Each pass of the loop:
   - run one scan block of PDHG rounds (§7), with skipping (§9);
   - every `tree_every` rounds, if `tree.type != none`: a tree attempt (§11), then the safeguard;
   - every round (inside the scan): free certificate (§10.1); the running maximum is logged per block;
   - every scan block: the stopping test (§10.5);
   - every `exact_every`: lazy exact certificate (§10.2); every 500 rounds: true-set tightening (§10.3);
   - every `extract_every`: dump flows and first moments to `snapshots/round_XXXXX.npz` for extraction.
4. **Screening (§7 of the proposal)** needs $U$ from a feasible trajectory, so it is **off** until LEAP exists. Implement `screen_edges(U)` anyway, with a unit test using a synthetic $U$; it is used later.
5. **Mode refinement** (proposal §7) is **off** in this phase. Leave a `TODO` hook only.

### 12.2 Logging (`log.jsonl`, one line per scan block)
- round, wall time, time per phase (edges, eigh, duals, tree, certificate);
- primal and dual residuals, normalized KKT error, objective;
- $L_{\text{free}}$, and when available $L_{\text{lazy}}$, $L_{\text{full}}$, $L_{\text{bnb}}$;
- $U_{\text{flow}}$ of both candidates, their four feasibility violations, the gap $(U_{\text{flow}} - L_{\text{best}})/\max(1,|L_{\text{best}}|)$, and the stop status;
- all counters (§9.4, §11.6), lazy evaluation count, restart events.

### 12.3 Self-tests (`--selftest`)
All unit tests from §5.4, §9.1, §9.2, §11.2, §11.3, §11.6 and §17 run on a tiny instance ($N = 8$, 2 faces, 4 regions) on CPU or one GPU.

### 12.4 Output files, in `runs/<run>/<task>/<seed>/`
`config.resolved.yaml`, `problem.npz`, `log.jsonl`, `final_state.npz` (flows, moments, prices, stored duals), `cert.json`, `tree.json`, `timing.json`, `snapshots/`.

---

## 13. Extraction and rendering (`extract_render.py`) — proposal §6

### 13.1 Input
A run directory (final state, or a chosen snapshot), plus `--methods all|<list>`.

### 13.2 The four methods (all vectorized where possible)
1. **Flow path decomposition (heaviest).** Repeatedly take the max-bottleneck start→goal path (a DP over layers with max-min), subtract the bottleneck, and stop when the flow left is < 1e-6. Keep the top `k_paths` (8) by weight.
2. **Randomized rounding.** Sample `samples` (default 256) paths. At each vertex, choose the next edge with probability (edge flow) / (vertex outflow). Vectorize all samples as one batched categorical sampling per layer. Deduplicate and keep the best `k_paths` by relaxed cost.
3. **Certificate shortest path + k-shortest.** On the certificate edge values (free LB, or exact if available), compute the $k$ best paths with a k-best DP on the DAG (top-$k$ lists per vertex).
4. **Greedy.** From the start, follow the outgoing edge with the largest flow until the goal.

### 13.3 Initial guess along a path
- **Methods 1, 2, 4:** knot states from first moments $z_e/y_e$. At an interior knot, average the incoming edge's post-knot and the outgoing edge's pre-knot (log the difference as a consistency violation). Forces from $f_n/y, f_t/y$.
- **Method 3:** use each edge's pricing minimizer (first moments of the pricing solution), as in proposal §6 "Initialization".
- **Rotation:** keep the raw $(c, s)$ for metrics. For rendering use $\theta = \operatorname{atan2}(s, c)$.

### 13.4 Metrics, written per guess to `extract/metrics.csv`
- method, path id, mode sequence, number of switches, path flow weight (or probability);
- **relaxed cost** of the path (sum of edge costs at the relaxed values) and **true-model cost** (the true cost evaluated on the extracted states and forces);
- **uncertified gap:** (true-model cost − $L$)/|$L$| and (relaxed cost − $L$)/|$L$|, where $L$ is the best certified bound of the run;
- **true-model violations:** max $|c^2+s^2-1|$, translation residual, rotation-relation residual, torque residual, contact-line and extent violation, penetration depth, friction-cone violation, consistency at shared knots;
- **smoothness:** $\sum\|\Delta^2 p\|^2$ and $\sum (\Delta^2\theta)^2$.
- Also write a LEAP-ready guess file `extract/guess_<method>_<id>.npz` (mode ids per step, knot states $(N+1)\times D$, forces $N\times 2$, dt, task).

### 13.5 Rendering (no simulation)
- Per guess, an `mp4` (matplotlib animation + ffmpeg) showing:
  - the slider box at pose $(p, \theta)$, the goal box dashed, and a trail of the box center;
  - the pusher disk at $p + R(\theta) r$;
  - a contact force arrow at the contact point (world frame);
  - text: method, step, mode name, cost so far.
- Per run, `extract/filmstrip.png`: 4 rows (methods) × 8 columns (evenly spaced knots), plus a final column overlaying all poses.
- **Parallelize rendering** with a `multiprocessing.Pool` across CPU cores, one job per guess.

---

## 14. Cross-run comparison (`compare.py`)

**Tables** (CSV and Markdown), one row per run × task × seed:
- **primary:** the stop status, and the rounds, rounds of edge work and wall time to the η stop (§10.5);
- final $L$ (free, lazy, B&B), final $U_{\text{flow}}$ and gap, and $J_{\text{rel}}$ when known;
- the tuned θ and the tuning time, reported separately (§7.3);
- secondary: rounds and wall time to reach within 1% and 0.1% of a **common reference** $L^{\text{ref}}$ for the task: $J_{\text{rel}}$ where it is known, otherwise the best certified $L$ over **all** runs of that task. Never use a run's own final value, which would favor runs that stall early. Also report the time to reach $\|R\|_P \le$ 1e-6·$\|R(z_0)\|_P$;
- counts: QP solves, pricing refreshes, eigendecompositions, Gershgorin skips, zero tests, cache hits;
- tree: attempts, acceptance rate, fallback rate, trial rounds, derivative evaluations, probes, tree time and critical path;
- the best extraction metrics per method.

**Plots** (static PNG):
- $L$ vs rounds and $L$ vs wall time, for all runs of a task;
- the residual $\|R\|_P$ vs time;
- acceptance rate vs rank $r$ for C-a and C-b;
- **checked progress per unit time vs rank** (proposal §12.5): time to the $L^{\text{ref}}$ targets and to the residual target, as functions of $r$ for C-a and C-b. Count *all* work: failed trials, fallback rounds, probes, derivative evaluations. Report two companions: rounds of edge work to target (machine independent) and the per-attempt $\log$ decrease of $\|R\|_P$ per second. Do not merge them into one score;
- reference quality (C-b) and direction agreement with the exact tree (small instances, §11.5);
- activations through zero-flow edges per accepted tree step vs per ordinary round;
- the κ effect (A vs A-κ2 vs A-κ3): $L$ and time per round;
- regions: A-reg vs B.

---

## 15. Parallelization

### 15.1 Within one run (one GPU)
- All per-edge work is batched with `vmap` over shape buckets: QP block IPM, pricing LPs, eigendecompositions, the zero test, the cache check, $J_e$, extraction sampling.
- Vertex coupling uses `segment_sum` and gathers. No Python loops over edges or vertices.
- Rounds run in `lax.scan` blocks (§7.5).
- Bucketed compaction (§9.3) turns skipping into real GPU time savings.
- The tree runs level by level: all merges and Cholesky factors of a level form one batched call.
- The certificate's shortest path is $O(E)$ per layer step. Lazy exact pricing goes to a CPU worker pool that overlaps with GPU rounds: launch the pricing on a snapshot, and fold in the result when it is ready.

### 15.2 Across GPUs (`launch.py`)
- Expand `matrix.yaml` into concrete jobs (run × task × seed × knob values). Write one resolved YAML per job.
- **Two launch modes.**
  - `--mode timing`: one job per GPU, exclusive, 8 jobs at a time. **All wall-clock comparisons come from this mode.** Shared GPUs distort timings unequally.
  - `--mode throughput`: several jobs per GPU. Use it for count-based metrics, extraction and debugging.
  - Every table states which mode produced it.
- **Stages (§1).**
  - Tuning first: short, count-scored jobs in throughput mode. For stage 1 that is 9 configurations × 6 grid values.
  - Stage 1: 7 jobs, one per GPU, in timing mode. GPU 7 runs the two small baselines one after the other.
  - Stage 2: 18 jobs in 3 waves.
  - Stage 3: chosen later.
  - Each job ends at the η stop. The 30-minute time cap is a safety stop, not the expected runtime.
- **Seeds.** One seed by default. Seeds change only the randomized parts (rounding samples, probes). Add a second seed only to measure run-to-run timing variance.
- **Queue (throughput mode):** 8 GPUs × `procs_per_gpu` (default 4) worker slots. Each slot runs one job with `CUDA_VISIBLE_DEVICES=g` and a memory fraction of about 0.2. Planar pushing needs well under 20 GB per job, except κ = 3.
- κ = 3 jobs and the full-exact certificate diagnostics get a **whole GPU** (`exclusive: true`).
- After a job finishes, launch `extract_render.py` on it in a CPU pool, then `compare.py` at the end.
- Re-running `launch.py` skips jobs that have `DONE` markers.

### 15.3 GPU sharing
Enable CUDA MPS where possible. Otherwise keep `procs_per_gpu` ≤ 4.

### 15.4 Sharding one solve over several GPUs
Not needed for planar pushing. Leave a note: for Go2 and hands, shard the edge buckets over devices with `shard_map`, combine vertex sums with `psum`, and run the tree on one device.

### 15.5 Launch overhead and host syncs (measure first, in M0)
At this problem size, a round is likely dominated by fixed per-kernel launch cost (a few µs per kernel × hundreds of kernels per round) rather than arithmetic. These are estimates; measure before optimizing.
- **Measure.**
  - Time one scan block of rounds at E ≈ 10, 2,600 and 26,000 (the last by stacking copies of the trellis). Flat time means the round is overhead-bound.
  - Profile one block with Nsight Systems or the JAX profiler. Record kernels per round, host↔device synchronizations per round, and the time between kernels.
  - Write all three to `timing.json`.
- **Remove per-round host syncs.**
  - No Python control flow on device values inside the scan.
  - Fixed-count inner loops (IPM iterations) use `lax.fori_loop` with a static trip count, or are unrolled. No data-dependent `while_loop` inside a round.
  - Check in the profile whether `lax.scan`'s loop or `lax.cond` / `lax.switch` copy predicates to the host on your XLA version.
- **Compaction without a per-round sync.**
  - The host picks the bucket size once per scan block: the smallest bucket ≥ 1.5 × the largest solve count seen in the previous block.
  - Inside the block, `lax.cond(count ≤ B, bucket_solve, full_solve)` guarantees no needed solve is ever dropped. Count overflows.
  - If the profile shows the conditional costs a sync, compare against `compaction: none` (fixed shape, all edges under a mask). At E ≈ 2,600 the masked version may be faster. The skip counts (§9.4) are the skipping metric either way.
- **Fewer, larger kernels.**
  - The free certificate's shortest path over N layers can use a min-plus `associative_scan` (log N depth) instead of N sequential steps. Measure both.
  - Check whether your XLA version supports command buffers (CUDA graphs) for loop bodies, and whether they are on by default (`--xla_gpu_enable_command_buffer`). If they help, enable them for every run, so all runs get the same treatment.
- **Instance batching** (stage 2 and later, throughput option). `vmap` the whole round over several tasks that share a trellis, so each kernel launch does more work. Wall-clock comparisons still come from single-instance timing mode.

---

## 16. Configuration (`configs/base.yaml`, all defaults)

```yaml
run_name: A_exact_sdp_k1
seed: 0
task: T3_translate_rotate
model:
  slider: {half_x: 0.10, half_y: 0.075, mass: 0.1, mu_table: 0.5}
  pusher: {radius: 0.01, mu_contact: 0.5, v_max_step: 0.02}
  limit_surface: {c_v: 0.5}          # speed scale
  workspace: {p_box: [[-0.6, 0.6], [-0.6, 0.6]], r_ws: 0.4}
  force: {fn_max: 5.0}
  step: {theta_step_max_deg: 15}
trellis:
  N: 32
  dt: 0.1
  contact_types: [stick, slide_pos, slide_neg]
  noncontact_regions: 8
costs: {w_p: 1.0, w_theta: 0.5, w_r: 1.0, w_f: 0.01, w_switch: 0.05}
relaxation:
  type: sdp                 # sdp | taylor
  order: 1                  # kappa
  couple_second_moments: true
  rlt: false
  regions: {theta_splits: 1, lambda_splits: 1}
  taylor_mode: band         # band | mccormick   (B only)
  cost_model: tangent_planes  # tangent_planes | soc (B only)
  cost_planes: 16
solver:
  rounds_max: 50000         # safety stop only
  time_budget_s: 1800       # safety stop only
  stall_rounds: 2000        # safety stop: no L_best / U_flow progress > 1e-6 rel
  rounds_per_scan: 50
  theta_pd: auto            # per-config tuning (7.3): runs/tuned/<config>_<task>.yaml
  theta_psd: same           # same | <float>
  primal_weight_update: none   # none | pdlp
  tune: {grid: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], rounds: 2000, score: kkt}
  delta: 1.0e-4             # state tolerance of P_delta
  stop: {eta_rel: 1.0e-3, eps_flow: 1.0e-6, eps_psd: 1.0e-8, eps_row: 1.0e-8}   # 10.5
  compile: {aot: true, cache_dir: /shared/jax_cache, fail_on_late_compile: true}
  dtype: float64
  qp_block: auto            # exact_ipm for kappa=1, split_rows for kappa>=2
  ipm: {iters: 40, tol: 1.0e-11}
  restarts: {scheme: pdlp_kkt, beta_sufficient: 0.2, beta_necessary: 0.8}
  checkpoint_every: 2000
skipping:
  zero_test: true
  lb_tol: 1.0e-12
  face_cache: false         # true for B
  gershgorin_skip: true
  compaction: bucketed      # none | bucketed  (15.5: measure both)
tree:
  type: exact               # none | exact | lowrank_a | lowrank_b
  every: 10
  reg: 1.0e-8
  rank: 16
  jacobian: one_step        # one_step | frozen_rank (11.1)
  eig_tol: 1.0e-8           # frozen_rank only
  primal_predict: false     # exact / C-a only
  lowrank_a: {oversample: 8}
  reference: {model: pos_vel, channels_per_coord: 2, gamma_v: 1.0, w_c: 1.0, w_a: 1.0}   # C-b, 12.2
  secant_history: auto      # max(8, rank)
  secant_sources: [trials, probes]   # add 'rounds' to also use ordinary-round differences
  probe_solves_per_refresh: 0
  probe_scale: 1.0e-2
  safeguard:
    type: supermann         # supermann | hpe
    c0: 0.0                 # 0 disables blind K0 steps (12.4: check the actual update)
    c1: 0.99
    q: 0.99
    sigma: 0.1
    relax: 1.0
    beta: 0.5
    max_trials: 3
    hpe: {sigma: 0.9, lambda_min: 0.1}
certificate:
  free_every: 1             # rounds
  exact_every: 200
  full_exact_every: 1000
  lazy: true
  bnb: {enabled: true, every: 500, max_splits: 8}
  eps_cert: 1.0e-12
  cert_workers: 32
  reference_check_small: true
extraction:
  every: 1000
  methods: [flow_decomp, random_round, cert_kshortest, greedy]
  k_paths: 8
  samples: 256
render: {fps: 10, filmstrip_cols: 8}
output: {root: runs}
```

**Per-run overrides** (`configs/runs/*.yaml`):
- A: none.
- A-κ2: `relaxation.order: 2`.
- A-κ3: `relaxation.order: 3`, `trellis.N: 8`, `exclusive: true`. Moment matrix 816×816 per edge. If a round takes more than about 5 s after the `eigh` benchmark, also set `trellis.faces: [0, 2]` (the selftest's reduced mode set) and log that.
- A-small, A-κ2-small (stage-1 baselines for A-κ3): `trellis.N: 8`, task T2, `relaxation.order: 1` or `2`. Use the same mode set as A-κ3.
- A-reg: `relaxation.regions.theta_splits: 8`, `exclusive: true` (tree blocks of width 4,480).
- B: `relaxation.type: taylor`, `regions.theta_splits: 8`, `skipping.face_cache: true`.
- C-a-r{4,16,64}: `tree.type: lowrank_a`, `tree.rank: r`.
- C-b-r{4,16,64}: `tree.type: lowrank_b`, `tree.rank: r`.
- C-b-r16-p4: as C-b-r16, plus `tree.probe_solves_per_refresh: 4`.
- D: `tree.type: none`.
- A-noskip (M5 check only, not a matrix job): `skipping.zero_test: false`, `skipping.compaction: none`.

**Matrix** (`matrix.yaml`):
- `stage1`: the 7 jobs of §1, with their tuning jobs first (`small_kappa_baselines: true`);
- `stage2`: T1, T2 and T4 for the six full-size configurations;
- `stage3`: filled in after stages 1–2;
- `sweep_N`: off.

Seed 0 only.

---

## 17. Milestones and acceptance checks

| # | Milestone | Acceptance checks (all automated in `--selftest` or in a dedicated `--check_<name>` mode) |
|---|---|---|
| M0 | Model and graph | Sign conventions unit-tested. Mode graph matches §4.4. Pruning removes only infeasible or unreachable edges. A hand-scripted feasible push (forward-simulated with the true model) is a valid start→goal path; record its cost $J_{\text{hand}}$. Benchmarks recorded: batched `eigh` sizes (§8.2), per-round time vs E, kernels and syncs per round (§15.5). Ahead-of-time compile covers every variant; a timing-mode dry run reports zero late compiles. |
| M1 | Outer models | §5.4 validity: 0 violations over $10^4$ samples per edge type (SDP κ = 1, 2; Taylor). Band report printed. |
| M2 | Reference solve | CVXPY + Clarabel (or MOSEK) solves the full relaxation on the small instance and gives $J_{\text{rel}}$. $J_{\text{rel}} \le J_{\text{hand}}$. |
| M3 | PDHG, run D | On the small instance, the objective reaches within 1e-4 relative of $J_{\text{rel}}$. Every logged $L$ is ≤ $J_{\text{rel}}$ (+ eps). Restarts fire and reduce the KKT error. Stopping test (§10.5): when it fires, $L_{\text{best}} \le J_{\text{rel}} \le U_{\text{flow}}$ within tolerance and the gap is ≤ η. |
| M4 | Edge identities | §9.1 tests pass (LB equals the pricing value at optimal duals; random duals give a valid bound; "zero ⇔ π ≥ 0" on 1,000 edges). Face cache output matches the IPM to 1e-10. The free bound with projected PSD duals never exceeds Clarabel's $\pi_e$ on 1,000 random edges and prices. |
| M5 | Skipping | A vs A-noskip on the small instance: iterates agree to 1e-9 at every logged round (tolerance is the IPM's, not exact). Counts are logged. Bucketed compaction gives the same iterates as `compaction: none`. |
| M6 | Exact tree and safeguard | Cyclic reduction equals a dense solve (1e-10). The one-step Jacobian matches finite differences of the edge prox (same active set). The frozen-rank option is checked the same way at fixed rank. SuperMann: K2 steps never increase $\mathrm{dist}_P$ to a reference solution on the small instance (Fejér check). A fallback equals the plain PDHG step bit for bit. HPE with $d = 0$ reproduces PDHG. On a frozen-face linear test problem, one accepted tree step reduces the residual by ≥ 100×. |
| M7 | Compressed engine | Merge identity $(R+VV^\top)/m = R/m + XC^{-1}X^\top$ checked densely. Without truncation it equals the uncompressed solve. With truncation, the tree map (applied to unit vectors) is symmetric positive definite. Zero right-hand side gives exactly zero; at a PDHG fixed point $r = 0$ gives $\Delta p = 0$. The Kronecker reference Schur complement matches a dense computation. |
| M8 | C-a, C-b | Both converge (the safeguard holds). C-b: `assert` that $L$ is written only by §10. Trial pairs are clean (same centers; checked by recomputing one pair at fixed center). The positive SR1 fit reproduces stored pairs when the rank allows. The C-a base and correction are PSD. All Section 12 counters and timers are present, and the activation counter works on a hand case. |
| M9 | Certificate | Lazy $L$ equals full-exact $L$ (to 1e-9) whenever both are computed. B&B never lowers $L$. Signed slack is applied. |
| M10 | Extraction and rendering | All 4 methods return valid start→goal mode sequences. Metrics CSV complete. mp4s and the filmstrip render. Guess files load. |
| M11 | Stage 1 | Tuning jobs, then the 7 stage-1 jobs, run on 8 GPUs in timing mode. Every job ends with a recorded stop status. `compare.py` produces the tables and plots. Stages 2–3 reuse the same launcher. |

---

## 18. Known pitfalls (read before coding)

- **Sign conventions:** $K$ (tail −, head +), flow supply at start and goal, the $\Lambda$ sign in the PSD coupling ($\langle\Lambda,\mathcal B(m)-S\rangle$), friction direction (force on the slider points along the pusher's relative sliding), the secant sign ($Y\approx S D$). Write each convention once in a comment and unit-test it.
- **Homogenization:** every constant must be multiplied by $m_0$. Bugs here break validity silently; §5.4 catches them.
- **svec scaling:** inconsistent $\sqrt2$ factors corrupt the PDHG metric and the LB formula.
- **The interval moment box must contain the true points.** It is also what keeps the zero test and the free certificate sound.
- **Never trust solver objective values for $L$.** Always use duals in the LB formula, projected onto cones, with signed slack.
- **Recompilation:** keep shapes static. Use buckets, not dynamic shapes. Cache compiled functions by bucket size. Compile ahead of time and fail on late compiles in timing mode (§7.5).
- **Memory:** κ = 2 has about 3.9k moments per edge. Store per-type matrices, not per-edge copies. Edge vectors are only $E \times 3.9\text{k}$ floats.
- **The face cache must never return an unverified solution.** If verification fails, solve.
- **Euler rotation update vs unit norm:** use the relative-rotation form (§4.5.2).
- **The reference model must never enter $L$** (C-b).
- **Timing:** call `block_until_ready` before reading timers. Compile ahead of time (§7.5) so no compile lands inside the clock. Compare wall-clock only from `--mode timing`.
- **Small problem, latency-bound GPU.** Run A has only about $E \approx 2{,}600$ edges, so per-round time is dominated by kernel launch latency, not arithmetic. Measure it (§15.5). Report machine-independent counts next to every wall-clock number, and do not extrapolate wall-clock ratios to Go2 or hand scale.
- **$P$ must be positive definite** (0.99 step factor). Otherwise SuperMann and HPE norms are meaningless.

---

## 19. Deviations from the proposal (explicit)

0. **SDP edges are split** (approved default 1). The PSD blocks get edge-local copies and duals, so each edge's prox step runs over its polyhedral block with the PSD duals fixed, instead of over the whole outer set. It is PDHG on an equivalent saddle problem with the same optimum. The zero test applies to the polyhedral block; PSD copies skip `eigh` by the exact Gershgorin test.
1. **κ ≥ 2 polyhedral block:** equality and inequality rows are dualized (`split_rows`) instead of solved as an exact edge QP. The dense per-edge QP would have thousands of variables. Skipping then applies to the PSD blocks through the Gershgorin test, not to a QP solve.
2. **Run B costs:** tangent-plane underestimators of the quadratic costs ($\hat\ell_e \le \ell_e$, allowed by proposal §2) keep the edges polyhedral, so the face cache is exact. `cost_model: soc` restores the exact perspective costs, with the cache disabled.
3. **Safeguard:** SuperMann as in proposal §4, with its blind K0 step disabled by default ($c_0 = 0$) so every tree step is checked against actual edge updates, as 12.4 requires. HPE is the alternative, also from §4.
4. **Not implemented in this phase:** local solves and $U$ (LEAP), screening against $U$ (implemented and unit-tested, but off), mode refinement, factored transitions, warm-started replanning (only the hook).
5. **True-set tightening** (proposal §5) is implemented as θ-splitting branch-and-bound on lazily selected edges. Global per-edge solves are not attempted.
6. **Taylor band scaling.** Proposal §3 says Option A's error scales like (region radius)². For bilinear terms (rotation × force or velocity, contact position × force), the valid band is **first order** in the region size, times the range of the other factor. The plan uses the valid bands (or McCormick) and reports them. Validity comes first.
7. **Face-cache sharing.** The proposal shares one factorization across all time steps with the same mode pair and face. The first version stores one per edge, for simple batching. The number of distinct (type, face) pairs is logged, showing what sharing would save. A shared table is a later optimization with identical results.
8. **Newton-on-face** for conic edges (proposal §4) is not implemented. It would apply only to run B with `cost_model: soc`, where the cache is off.
9. **Option (a) base.** A block-diagonal base minus the exact response is indefinite, so option (a) uses a spectral base (tail level times identity) plus the top-r part (§11.4).
10. **Stopping rule.** The text of proposal §8 applies η to a real trajectory's U. This phase applies η to P_δ, with $U_{\text{flow}}$ from the current flow (§10.5), consistent with §9's round bracket. The trajectory gap $U_{\text{traj}} - L$ is reported once LEAP exists.

---

## Appendix A. Formula sheet

- **Zero step:** step $= 0 \iff g_0 + \min_{m\in P_e\cap\{m_0=1\}} g_{\setminus0}^\top m \ge 0$.
- **Dual bound with box** (any $\nu$; $\mu \ge 0$; $\Lambda \succeq 0$, where $\Lambda$ here means the projected $\Gamma = \Pi_{\succeq0}(-\Lambda_e)$ of §10.1):
$$
\pi_e(g) \ge g_0 + b_{\text{eq}}^\top\nu + b_{\text{in}}^\top\mu - \langle\Lambda,\mathcal B_0\rangle + \sum_i \min(r_i\underline m_i, r_i\overline m_i),\qquad r = g_{\setminus0} - A_{\text{eq}}^\top\nu - A_{\text{in}}^\top\mu - \mathcal B^*_{\setminus0}(\Lambda).
$$
- **Bound degradation** with fixed duals: $\mathrm{LB}(g') \ge \mathrm{LB}(g) - |g'_0 - g_0| - \sum_i |g'_i - g_i|\max(|\underline m_i|, |\overline m_i|)$.
- **Certificate:** $L(\lambda) = \mathrm{SP}[\mathrm{LB}_e(g^{\text{cert}}_e(\lambda))] - \delta\sum_v\|\lambda_v\|_1 \le J^\star$.
- **Cyclic reduction:** see §11.2.
- **Woodbury:** $A_{mm}^{-1} = R_{mm}^{-1} - GC^{-1}G^\top$, see §11.3.
- **Edge Jacobian (frozen active set):** $J^{\text{full}} = H^{-1} - H^{-1}E^\top(EH^{-1}E^\top)^+EH^{-1}$, see §11.1.
- **Low-rank merge:** $(R+VV^\top)/m = R/m + XC^{-1}X^\top$, $X = V_\partial - R_{\partial m}R_{mm}^{-1}V_m$, $C = I + V_m^\top R_{mm}^{-1}V_m$, see §11.3.
- **SuperMann K2 step:** $\rho = \|R_w\|_P^2 - \tau\langle R_w, d\rangle_P$, $z^+ = z - \lambda(\rho/\|R_w\|_P^2)R_w$, see §11.6.
- **Positive block-SR1 fit (C-b):** $UU^\top = \hat Y\,(\mathrm{sym}(D^\top\hat Y))^{+}_{\succ0}\,\hat Y^\top$ with $\hat Y = Y - S^{\text{ref,val}}D$, then truncate to rank $r$.
- **PDHG metric norm:** $\|z\|_P^2 = \langle x, T^{-1}x\rangle - 2\langle Kx, p\rangle + \langle p, \Sigma^{-1}p\rangle$.

## Appendix B. References for implementers

- Chambolle & Pock (2011), PDHG; Pock & Chambolle (2011), diagonal preconditioning.
- Applegate, Hinder, Lu & Lubin (2023), restarts and sharpness (PDLP restart criteria).
- Themelis & Patrinos (2019), SuperMann; Solodov & Svaiter (1999), HPE.
- Agrawal, Barratt, Boyd, Busseti & Moursi (2019), *Differentiating through a cone program* (background for differentiating conic edges).
- Lasserre (2001) and Waki et al. (2006), moment relaxations and correlative sparsity.
- Marcucci et al. (2024), *Shortest paths in graphs of convex sets* (perspective formulation).
- Graesdal et al. (2024), *Towards tight convex relaxations for contact-rich manipulation* (planar pushing context).
- Lynch, Mason, Goyal / Hogan & Rodriguez, quasi-static pushing and ellipsoidal limit surfaces.
- Hockney (1965), Buzbee–Golub–Nielson (1970), cyclic reduction.
