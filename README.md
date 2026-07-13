# RAMMP-Kinova

Software for automating **activities of daily living (ADLs)** with a **Kinova
Gen3** 7-DoF arm + Robotiq 2F-85 gripper, driven by voice and planned by AI.
Say *"computuh, get my pills"* — the arm plans a collision-free path through a
simulated kitchen, tracks the pill bottle with two cameras, and goes to where
it actually is.

This README is the **install / run guide** — for the design and roadmap, see
[`docs/architecture.md`](docs/architecture.md).

```
voice ("computuh, ...")  or  typed text (goto CLI)
   ▼
cuRobo GPU planner  — collision-free plans (~0.3 s) in the scene.yaml kitchen
   ▼                   live-updated by perception (two RGB-D cameras)
ros2_control  →  MuJoCo physics (sim today, the real arm later)
```

## Requirements

| | |
|---|---|
| Robot host | NVIDIA Jetson AGX Orin — Ubuntu 22.04 + **ROS 2 Humble** |
| GPU stack | cuRobo **v0.7.8** on Jetson CUDA PyTorch (install: [`curobo_planner`](ros2_ws/src/curobo_planner/README.md)) |
| Simulator | MuJoCo ≥ 3.1 + [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) + `mujoco_ros2_control` (install: [`mujoco_sim`](ros2_ws/src/mujoco_sim/README.md)) |
| Dev machine (optional) | Any Windows/macOS/Linux box for the mirror viewer + voice app — **no ROS needed** |
| Arm (later) | Kinova Gen3 at `192.168.1.10`; the Jetson is `192.168.1.11` |

## 1. Install (Jetson)

```bash
git clone https://github.com/chrisscox/RAMMP-Kinova.git && cd RAMMP-Kinova
bash scripts/setup_ros2_kortex.sh      # apt deps + kortex driver + our packages
```

The apt dependency list lives in [`requirements.txt`](requirements.txt)
(`sudo apt install -y $(cat requirements.txt)` if you prefer manual).
Then do the two one-time GPU installs, each documented in its package README:
**cuRobo** ([`curobo_planner`](ros2_ws/src/curobo_planner/README.md)) and
**MuJoCo** ([`mujoco_sim`](ros2_ws/src/mujoco_sim/README.md)).

Source the workspace in **every terminal** (or add to `~/.zshrc`):

```bash
source /opt/ros/humble/setup.bash && source ~/RAMMP-Kinova/ros2_ws/install/setup.bash
```

## 2. Generate the scene (once, and after scene edits)

```bash
ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie
```

## 3. Run — one command

```bash
ros2 launch mujoco_sim mujoco_bringup.launch.py
```

That's the whole stack: MuJoCo physics behind ros2_control, both perception
cameras + detectors, the cuRobo planner (~15 s warmup), and the rosbridge the
Windows tools ride. Then drive it from a second terminal:

```bash
ros2 run curobo_planner goto        # interactive: bottle / pills / home / check ...
```

Free text works — `"go to my bottle"`, `"grab the mug"` — typed or spoken.

## 4. Windows companions (optional, no ROS install)

```powershell
pip install mujoco roslibpy opencv-python vosk sounddevice pyttsx3 pyyaml
```

- **Mirror viewer** — a native MuJoCo window tracking the sim live, with
  perception's beliefs overlaid as labelled markers:
  `python -m mujoco_sim.mirror_viewer --host 192.168.1.11 --model scene_gen3.xml`
  (add `--camera /rammp_detector/debug_image` for a second window showing
  what the detector sees, detections painted on). See
  [`mujoco_sim`](ros2_ws/src/mujoco_sim/README.md) for the local scene-XML step.
- **Voice** — *"computuh, go to my bottle"*, fully offline recognition:
  `python voice\computuh.py --host 192.168.1.11` (in
  `ros2_ws/src/curobo_planner/voice/`).

Both have Desktop launchers if you've set them up before.

## Stopping the arm

- Say **"computuh, stop"** (or type `stop` into a goto prompt — a second one
  if yours is blocked awaiting a command's status) — handled before the
  planner's command lock; the arm holds position immediately, even mid-motion.
- **Hardware E-stop** stays authoritative on the real arm — the Gen3 has no
  mechanical brakes.

## Diagnostics

| Command | What it verifies |
|---|---|
| `check` (in goto) | Dry-plans every target + standoff — run after any scene.yaml edit |
| `ros2 run mujoco_sim check_traj` | Watches executed trajectories for real contacts (mm-level) |
| `ros2 run rammp_perception probe` | One-shot camera/detection health report with named causes |

## Repository layout

```
RAMMP-Kinova/
├── README.md                        # this guide
├── requirements.txt                 # apt dependencies (one per line)
├── docs/architecture.md             # design + roadmap
├── scripts/setup_ros2_kortex.sh     # one-shot setup + build (Linux only)
└── ros2_ws/src/
    ├── curobo_planner/              # GPU planning + NL targets + voice app
    ├── mujoco_sim/                  # physics bringup, scene builder, mirror viewer
    └── rammp_perception/            # cameras -> 3D object tracking -> planner
```

## License

MIT — see [LICENSE](LICENSE).
