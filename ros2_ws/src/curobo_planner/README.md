# curobo_planner

cuRobo GPU motion planning for the Kinova Gen3, in **simulation**: plan
collision-free trajectories to named targets in an obstacle scene, execute on
the fake-hardware arm, and watch it in Foxglove. A natural-language layer
("go to the bottle") resolves phrases to targets via Claude, with an offline
keyword fallback.

Kept in its own package (not `adl_primitives`) because cuRobo pulls a heavy GPU
stack — the primitive library stays lightweight.

## One-time install (Jetson AGX Orin, JetPack 6.x)

cuRobo needs a **CUDA-enabled PyTorch built for Jetson** — the usual
`pip install torch` gives a CPU wheel and cuRobo will crash at runtime. Do this
in a `tmux` (the build is ~20 min):

```bash
# 0. Verify (or install) a Jetson CUDA PyTorch — the #1 gotcha
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   if that prints False, install a Jetson wheel matching your CUDA (nvcc --version):
#   pip install --no-cache torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 1. Build cuRobo (PINNED to v0.7.8 — v0.8.0 is a rewrite with a changed API)
sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install
export TORCH_CUDA_ARCH_LIST="8.7+PTX"     # Orin = sm_87
export MAX_JOBS=4                          # cap parallel nvcc to avoid OOM
git clone https://github.com/NVlabs/curobo.git && cd curobo
git checkout tags/v0.7.8
pip install -U "packaging>=24.1"           # avoids a setuptools build crash
pip install -U "setuptools>=70,<80"        # new enough for the build, <80 keeps colcon happy
pip install -e . --no-build-isolation      # MUST use the Jetson torch already installed
pip install "warp-lang==1.5.1"             # v0.7.8 needs warp 1.5.x (1.14 removed wp.torch)

# 2. (optional) natural-language layer
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...        # or add to ~/.zshrc; offline fallback works without it
```

Then build this package:

```bash
cd ~/RAMMP-Kinova/ros2_ws
colcon build --packages-select curobo_planner --symlink-install
source install/setup.zsh   # or setup.bash
```

## Run the demo

One command brings up the fake arm, the Foxglove bridge, and the planner
(warmup takes a minute on first launch):

```bash
ros2 launch curobo_planner curobo_demo.launch.py
```

Open Foxglove (`ws://<jetson-ip>:8765`), add a 3D panel, and view the
`/curobo_planner/markers` MarkerArray — you'll see the table, cabinet, shelf,
and the labelled targets. Then drive it in another terminal:

```bash
ros2 run curobo_planner goto "go to the bottle"
ros2 run curobo_planner goto "grab the mug"         # BEHIND the arm
ros2 run curobo_planner goto "open the cabinet"     # -> cabinet_handle
ros2 run curobo_planner goto                        # interactive prompt
ros2 run curobo_planner goto --list                 # list known targets
```

The arm plans a collision-free path around the obstacles **and every prop in
the scene** (as bounding boxes) and executes it;
watch the frames move in Foxglove. A target's `ignore_objects` list exempts
the prop being reached for (you can't dodge the thing you're reaching for);
if the very next plan starts in collision with that prop, the planner retries
once with the previous ignore list merged in, so the arm can always leave.
Special commands: `home` (also collision-planned, via `plan_single_js`) and
`pose: x y z roll pitch yaw` (metres + degrees) for a raw goal.

## Editing the scene

`config/scene.yaml` is the **single source of truth** — the planner builds
cuRobo's collision world *and* the Foxglove markers from it, and **reloads it on
every command**, so you can edit a target's position and just re-send the
command. No node restart.

If a plan fails, the status message says so — tune the target `position`/`rpy_deg`
in `scene.yaml` and retry. That first-run tuning loop is expected: the shipped
poses are reasonable starting points, not guaranteed-reachable.

> Note: cuRobo's ee_link for the Gen3 is `tool_frame`, which sits **0.120 m
> beyond the wrist flange** — roughly the fingertip midpoint (verified from
> cuRobo v0.7.8's `kinova_gen3_7dof.urdf`). Target positions say where the
> fingertips go.

Free props (`free: true`) obey physics in MuJoCo, but the planner avoids them
at their **YAML pose** — there is no perception yet, so a knocked-over bottle
is still avoided where the YAML says it stands.

## How it fits together

```
"go to the bottle"
   │  goto CLI: Claude forced-choice over the scene's target enum
   ▼  (offline keyword fallback if no ANTHROPIC_API_KEY)
/curobo_planner/command (std_msgs/String: a target name)
   │  planner: reload scene -> update cuRobo world -> plan_single()
   ▼
/joint_trajectory_controller/joint_trajectory  ->  fake ros2_control arm
   │
   └── obstacles + targets + goal  ->  /curobo_planner/markers  ->  Foxglove
```

Real hardware: `curobo_demo.launch.py use_fake_hardware:=false` uses cuRobo's
bundled Gen3 collision model, but do a very slow, supervised first run and keep
the hardware E-stop in reach — cuRobo checks self/scene collision but the scene
must actually match reality.
