# RAMMP-Kinova

Automating **activities of daily living (ADLs)** for individuals with motor
disabilities using a **Kinova Gen3** arm. Rather than hard-coding a script per
task, we expose a library of small, safe, reusable motion **primitives** and (later)
let a real-time multimodal AI decide which primitives to call, and in what order, to
accomplish a higher-level goal.

> **Status: Milestone 1** — bring up the arm under ROS 2 Humble and execute the most
> basic possible motion (a small, slow joint nudge + gripper open/close) from a single ROS 2 node.
> No AI, perception, or task planning yet.

---

## Hardware & network

| Thing | Value |
|---|---|
| Arm | Kinova Gen3, **7-DoF**, Robotiq **2F-85** gripper |
| Arm IP | `192.168.1.10` |
| Compute | NVIDIA Jetson AGX Orin 64 GB (**Ubuntu 22.04 + ROS 2 Humble**) |
| Jetson IP | `192.168.1.11` (SSH user `abra`) |
| Camera | Intel RealSense **D405** — *deferred, not used in Milestone 1* |

> 🔒 **No credentials live in this repo** (it is public). The Jetson SSH password is
> kept out of version control — see your team notes.

## Architecture: now vs. later

- **Now (Milestone 1):** Kinova's official **[`ros2_kortex`](https://github.com/Kinovarobotics/ros2_kortex)**
  driver (Humble). High-level control runs at ~40–50 Hz with built-in joint/Cartesian
  limits and protection zones — plenty for basic, safe motion.
- **Later:** when we need >50 Hz / contact-rich control (e.g. **NVIDIA cuRobo**, force
  control), we wrap a low-level 1 kHz engine (the in-house
  [`kinova-gen3-driver`](https://github.com/rammp-org/kinova-gen3-driver), once it grows
  basic move/gripper) in a ROS 2 node. See [`docs/architecture.md`](docs/architecture.md).

## Development workflow

Develop on your machine, **build and run on the Jetson** (ROS 2 Humble is Linux-only —
there is no Windows/macOS build of this stack).

```
edit locally  ->  git push  ->  (on Jetson) git pull  ->  colcon build  ->  test
```

VS Code **Remote-SSH** into `abra@192.168.1.11` is the smoothest loop.

---

## One-time setup (on the Jetson)

Prerequisites: Ubuntu 22.04 with ROS 2 Humble already installed (`/opt/ros/humble`).

```bash
git clone https://github.com/chrisscox/RAMMP-Kinova.git
cd RAMMP-Kinova
bash scripts/setup_ros2_kortex.sh   # clones ros2_kortex (humble), imports deps, rosdep, colcon build
```

This builds the upstream driver **and** our package into `ros2_ws/`.

To rebuild only our package after editing it:

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select adl_primitives --symlink-install
source install/setup.bash
```

---

## Bring up the arm

**Always source the overlay first:**
```bash
source /opt/ros/humble/setup.bash
source ~/RAMMP-Kinova/ros2_ws/install/setup.bash
```

Dry bring-up with **fake hardware** (no robot, validates the stack):
```bash
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.1.10 use_fake_hardware:=true gripper:=robotiq_2f_85
```

**Real arm:**
```bash
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.1.10 gripper:=robotiq_2f_85
```

Verify the interfaces are live (in another terminal, overlay sourced):
```bash
ros2 control list_controllers     # expect joint_trajectory_controller + robotiq_gripper_controller active
ros2 action list                  # expect /joint_trajectory_controller/follow_joint_trajectory + /robotiq_gripper_controller/gripper_cmd
ros2 topic echo /joint_states --once
```

> ⚠️ If any of those names differ from the defaults, **don't edit code** — update
> [`ros2_ws/src/adl_primitives/config/hello_arm.yaml`](ros2_ws/src/adl_primitives/config/hello_arm.yaml).
> Everything is parameterized.

---

## Run the Milestone-1 demo (`hello_arm`)

The demo: read current pose → open gripper → nudge one wrist joint a few degrees and
back (slow) → close → open.

**1) Dry run first (default — connects, prints the plan, moves NOTHING):**
```bash
ros2 launch adl_primitives hello_arm.launch.py
```

**2) Real motion (only after the safety checklist below):**
```bash
ros2 launch adl_primitives hello_arm.launch.py dry_run:=false
```

Useful overrides: `nudge_deg:=5.0 nudge_joint_index:=6 move_time_s:=6.0`

---

## Stopping the arm

| Method | What it does | When |
|---|---|---|
| **Hardware E-stop** (red button on power cable) | Cuts power — **authoritative** | Any real emergency |
| **Software e-stop service** | Cancels active goals + deactivates the motion controller | Stop an in-progress motion from code/CLI |
| **Ctrl-C** | Exits the program (best-effort soft-stop) | Normal shutdown |

Software e-stop from another terminal:
```bash
ros2 service call /hello_arm/estop std_srvs/srv/Trigger "{}"
```

> ⚠️ **The Gen3 has no mechanical brakes.** Cutting power makes it settle slowly (the
> actuators damp the fall). The software e-stop is a convenience, **not** a substitute
> for the hardware E-stop. **Ctrl-C may not halt a trajectory already in progress** — use
> the e-stop service or the hardware button.

### First real-run safety checklist
- [ ] Workspace clear of people and obstacles; arm has room for the nudge.
- [ ] A person is within reach of the **hardware E-stop**.
- [ ] You ran the **dry run** and the printed plan looks sane.
- [ ] `nudge_deg` is small (≤ ~10°) and `move_time_s` is generous (slow).

---

## What to confirm on the Jetson (and tune in `config/hello_arm.yaml`)
- Exact action names (`ros2 action list`) and controller names (`ros2 control list_controllers`).
- Exact joint names (`ros2 topic echo /joint_states --once`) — defaults are `joint_1..joint_7`.
- The clear-faults mechanism (`ros2 service list | grep -i fault`). It's **off by default**
  (`clear_faults_service: ""`); set it once confirmed.

## Repository layout
```
RAMMP-Kinova/
├── README.md
├── LICENSE                         # MIT
├── docs/architecture.md            # long-term vision + roadmap
├── scripts/setup_ros2_kortex.sh    # one-shot Jetson setup (Linux only)
└── ros2_ws/src/adl_primitives/     # our ROS 2 Humble (Python / rclpy) package
    ├── adl_primitives/             # kinova_primitives.py, hello_arm.py
    ├── config/hello_arm.yaml       # all tunables (action/joint names, safety limits)
    ├── launch/hello_arm.launch.py
    ├── setup.py  setup.cfg  package.xml
    └── resource/adl_primitives
```

## License
MIT — see [LICENSE](LICENSE).
