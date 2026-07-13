# curobo_planner

cuRobo GPU motion planning for the Kinova Gen3: collision-free trajectories
to named targets in the kitchen scene, executed on the MuJoCo-backed arm.
A natural-language layer resolves free text — typed or spoken — to targets,
and live perception keeps the goals and the collision world honest.

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
```

Then build this package:

```bash
cd ~/RAMMP-Kinova/ros2_ws
colcon build --packages-select curobo_planner --symlink-install
source install/setup.zsh   # or setup.bash
```

## Run

The planner starts as part of the one-command bringup
(`ros2 launch mujoco_sim mujoco_bringup.launch.py` — see the repo README).
Drive it from another terminal:

```bash
ros2 run curobo_planner goto                        # interactive prompt (fastest)
ros2 run curobo_planner goto "go to the bottle"     # one-shot
ros2 run curobo_planner goto --list                 # list known targets
```

The arm plans a collision-free path around the furniture **and every prop in
the scene** (as bounding boxes) and executes it. A target's `ignore_objects`
list exempts the prop being reached for — but only on the final few
centimetres: the planner transits to a standoff pose with the FULL world
first, and retreats back through it on departure. Special commands: `home`,
`check` (dry-plan every target — run it after scene edits), `stop`, and
`pose: x y z roll pitch yaw` (metres + degrees) for a raw goal.

To run the planner standalone instead: `ros2 run curobo_planner planner`
(the scene file defaults to the installed `config/scene.yaml`).

## Voice control ("computer")

`voice/computer.py` — native, fully OFFLINE, runs on the dev machine:

```powershell
pip install vosk sounddevice pyttsx3 roslibpy pyyaml
python voice\computer.py --host 192.168.1.11    # first run downloads a 40 MB model
```

It listens continuously with a grammar constrained to this project's
vocabulary (built from scene.yaml — fast, hard to mishear, no cloud),
publishes to the planner over the Jetson's rosbridge, and speaks the replies.
`--list-mics` / `--mic N` to pick a microphone. Restart it after adding
targets to scene.yaml so the new words enter the grammar.

- **"computer, go to my bottle"** — one breath, arm goes.
- **"computer"** alone arms a 6 s window, then say the command.
- **"computer, stop"** — the planner handles `stop` BEFORE its command lock:
  the arm holds position immediately, even mid-motion.
- Replies are shown and SPOKEN ("Going to the bottle.").

The planner itself resolves free text (same token matcher as the goto CLI),
so any text published to `~/command` works — the voice app needs no knowledge
of the scene.

## Live perception

With the detectors running (they're part of the bringup), fresh detections
override prop positions and targets follow their reach-for object
(`Target pills follows live pill_bottle ...` in the log). Two thresholds,
deliberately different: targets follow fine-grained (>1.5 cm, grasp
accuracy), collision boxes only move for real displacements
(`live_box_shift`, 4 cm) so measurement noise can't pinch a verified goal
into IK_FAIL. Stale detections (>10 s) fall back to YAML — a stale pose
beats a wrong one.

## Making it fast

Enter-to-motion latency has three parts; each has a lever:

1. **CLI startup dominates one-shot calls** (~1-2 s of Python + DDS discovery
   per `ros2 run`). Use the **interactive prompt** instead — start
   `ros2 run curobo_planner goto` once, then each typed command goes out in
   milliseconds.
2. **Planning time** is reported in every status line (`plan 0.36s`). The
   effective knob is
   `ros2 launch mujoco_sim mujoco_bringup.launch.py enable_finetune:=false`
   (~2x faster, still collision-free, slightly less smooth). Do NOT lower
   `finetune_attempts`
   to save time: the finetune loop exits on first success, so easy plans
   never pay for the extra attempts — fewer attempts only converts
   hard-goal successes into failures.
3. **Jetson clocks**: `sudo nvpmodel -m 0 && sudo jetson_clocks` (MAXN) is
   the single biggest lever on GPU planning time.

## Editing the scene

`config/scene.yaml` is the **single source of truth** — the planner builds
cuRobo's collision world from it and **reloads it on every command**, so you
can edit a target's position and just re-send the command. No node restart.
(MuJoCo geometry does need `build_scene` + a bringup restart after edits.)

If a plan fails, the status message says so — tune the target
`position`/`rpy_deg` in `scene.yaml`, verify with `check`, and retry.

> Note: cuRobo's ee_link for the Gen3 is `tool_frame`, which sits **0.120 m
> beyond the wrist flange** — roughly the fingertip midpoint (verified from
> cuRobo v0.7.8's `kinova_gen3_7dof.urdf`). Target positions say where the
> fingertips go.

## How it fits together

```
"computer, go to my bottle"        "go to the bottle"
   │  voice app (offline Vosk)        │  goto CLI
   └──────────────┬───────────────────┘
                  ▼
/curobo_planner/command (std_msgs/String — free text ok)
   │  planner: reload scene -> apply live detections -> update cuRobo world
   ▼            -> plan standoff + final segments -> collision-checked traj
/joint_trajectory_controller/joint_trajectory  ->  MuJoCo-backed ros2_control
```
