#!/usr/bin/env bash
# Creates the `trellis` conda environment for this repo: Python 3.11, JAX with CUDA 12, the CPU
# convex solvers used for reference checks (CVXPY, Clarabel, SCS), YAML, plotting, ffmpeg, and MuJoCo
# with imageio for the videos. No DiffMJX: this project does not simulate, and MuJoCo only draws frames.
#   bash setup.sh                # environment `trellis`
#   ENV=other bash setup.sh      # another name
set -euo pipefail
cd "$(dirname "$0")"
ENV=${ENV:-trellis}
if ! conda env list | awk '{print $1}' | grep -x "$ENV" >/dev/null; then
    conda create -y -n "$ENV" python=3.11
fi
ENVDIR="$(conda env list | awk -v e="$ENV" '$1 == e {print $NF}')"
PY="$ENVDIR/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "jax[cuda12]" numpy scipy pyyaml matplotlib cvxpy clarabel scs mujoco "imageio[ffmpeg]"
conda install -y -n "$ENV" -c conda-forge ffmpeg
"$PY" - <<'EOF'
import jax, numpy, scipy, yaml, matplotlib, cvxpy, clarabel, scs, mujoco, imageio
print("jax", jax.__version__, "| backend", jax.default_backend(), "| devices", jax.devices())
print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| cvxpy", cvxpy.__version__,
      "| clarabel", clarabel.__version__, "| scs", scs.__version__)
print("cvxpy solvers:", cvxpy.installed_solvers())
print("mujoco", mujoco.__version__, "| imageio", imageio.__version__)
EOF
"$ENVDIR/bin/ffmpeg" -version | head -1
echo "done. activate with: conda activate $ENV"
