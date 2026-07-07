# RAMMP-Kinova

Software for automating **activities of daily living (ADLs)** with a **Kinova Gen3** arm:
a library of small, safe motion **primitives** that a real-time AI will later orchestrate.
This README is the **download / install / run guide** — for the design and roadmap, see
[`docs/architecture.md`](docs/architecture.md).

---

## Requirements

| | |
|---|---|
| OS / middleware | **Ubuntu 22.04 + ROS 2 Humble** (Linux only) |
| Build/run host | NVIDIA Jetson AGX Orin, or any Ubuntu 22.04 machine |
| Arm | Kinova Gen3 7-DoF + Robotiq 2F-85 gripper — or `use_fake_hardware:=true` (no robot) |
| Arm IP | `192.168.1.10` (default) |

Install ROS 2 Humble first if you haven't:
<https://docs.ros.org/en/humble/Installation.html>

---

## 1. Download

```bash
git clone https://github.com/chrisscox/RAMMP-Kinova.git
cd RAMMP-Kinova
```

## 2. Install dependencies + build

**One command** installs the build tools and this package's dependencies, pulls the
`ros2_kortex` driver, runs `rosdep`, and builds the whole workspace:

```bash
bash scripts/setup_ros2_kortex.sh
```

Prefer to install the apt dependencies yourself? They're listed (one per line, no comments)
in [`requirements.txt`](requirements.txt):

```bash
sudo apt update && sudo apt install -y $(cat requirements.txt)
```

To rebuild just this package after editing it (`--symlink-install` means edits to
*existing* `.py` files need no rebuild — new files, entry points, launch/config files do):

```bash
cd ros2_ws
colcon build --packages-select adl_primitives --symlink-install
```

## 3. Source the workspace

Run this in **every new terminal**:

```bash
source /opt/ros/humble/setup.bash
source ~/RAMMP-Kinova/ros2_ws/install/setup.bash
```

## 4. Bring up the arm

**Fake hardware** (no robot — validates the stack):

```bash
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.1.10 use_fake_hardware:=true gripper:=robotiq_2f_85
```

**Real arm:**

```bash
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.1.10 gripper:=robotiq_2f_85
```

Check the interfaces are live (new terminal, sourced):

```bash
ros2 control list_controllers   # joint_trajectory_controller + robotiq_gripper_controller = active
ros2 action list
ros2 topic echo /joint_states --once
```

## 5. Run the demo

`test_arm` reads the current pose, opens the gripper, nudges one wrist joint a few degrees
and back (slowly), then closes and reopens the gripper.

```bash
# Dry run (default): connects and prints the plan, moves NOTHING
ros2 launch adl_primitives test_arm.launch.py

# Real motion: only after the safety checklist below
ros2 launch adl_primitives test_arm.launch.py dry_run:=false
```

Handy overrides: `nudge_deg:=5.0 nudge_joint_index:=6 move_time_s:=6.0`

## 6. Jog the arm from a browser

`jog_ui` serves a joystick web panel on port 8080, plus gripper open/close and a
soft-stop. Every command goes through the same primitives as `test_arm`, and `dry_run`
defaults to **true**.

Drag the round pad to move the hand in the robot's base frame
(forward/back/left/right), drag the side strip for up/down. Click **ENABLE JOYSTICK**
first — the panel boots disarmed. The arm moves **only while you hold** — releasing,
closing the tab, or losing WiFi stops it (300 ms deadman). Speed is capped at
`max_linear_mps` (default 10 cm/s) times the on-screen slider.

> 💡 Click **Home pose** first (joystick disabled). The startup all-zeros "candle"
> pose is a kinematic singularity — from there only forward/back can move; from the
> bent-elbow Home pose every direction works.

Two backends:

- **Real arm** (default): uses the `twist_controller`; the panel switches controllers
  automatically and rotates commands into the Kinova tool frame via TF.
- **Simulation** (`sim:=true`): differential IK streamed through the trajectory
  controller — works on **fake hardware**, so you can drive the simulated arm and watch
  it move in Foxglove:

  ```bash
  # terminal A: fake bringup (+ foxglove_bridge in another terminal if you want viz)
  ros2 launch kortex_bringup gen3.launch.py \
    robot_ip:=192.168.1.10 use_fake_hardware:=true gripper:=robotiq_2f_85 launch_rviz:=false
  # terminal B: joystick against the sim
  ros2 launch adl_primitives jog_ui.launch.py sim:=true dry_run:=false
  ```

> ⚠️ **Automatic software e-stop.** `jog_ui.launch.py` also starts `estop`, a separate
> small process. If the `jog_ui` *process* is killed uncleanly (SIGKILL, out-of-memory)
> while the joystick is held, the Kinova base latches its last velocity command —
> `jog_ui`'s own safety code can't run if `jog_ui` is dead. The `estop` node listens to
> `jog_ui`'s heartbeat and, on loss, zeroes the twist and swaps the trajectory
> controller back in. Clean `jog_ui` exits disarm it quietly; only unclean death makes
> it fire. Run at most **one** live (non-dry-run) `jog_ui` at a time (live sessions
> share the heartbeat topic). The hardware E-stop remains authoritative — total
> power/PC loss is still its job.

The soft-stop cancels goals, zeroes the velocity command, and deactivates **both** motion
controllers; **Resume** reactivates the trajectory controller (you re-enter Joystick mode
manually).

Upgrading an existing checkout? The jog UI adds a dependency and new files, so once:

```bash
sudo apt install python3-flask
cd ros2_ws && colcon build --packages-select adl_primitives --symlink-install
```

```bash
# Dry run (default): the UI works, clicks are logged, the arm does NOT move
ros2 launch adl_primitives jog_ui.launch.py

# Real motion: only after the safety checklist below
ros2 launch adl_primitives jog_ui.launch.py dry_run:=false
```

Then open `http://<jetson-ip>:8080` (e.g. `http://192.168.1.11:8080`) from any machine on
the LAN. The red **SOFT-STOP** button cancels goals and deactivates the motion controller;
**Resume** reactivates it.

> ⚠️ The page has no authentication — anyone who can reach the port can command the arm.
> Keep it on the robot LAN, or set `ui_host: "127.0.0.1"` in `config/jog_ui.yaml`.

---

## Stopping the arm

- **Hardware E-stop** (red button on the power cable) — **authoritative**; use it for any real emergency.
- **Software e-stop** (cancels goals + deactivates the motion controller), from another terminal:
  ```bash
  ros2 service call /test_arm/estop std_srvs/srv/Trigger "{}"   # /jog_ui/estop for the jog panel
  ```
- **Ctrl-C** — exits the program; may not halt a trajectory already in progress.

> ⚠️ The Gen3 has **no mechanical brakes** — cutting power lets it settle slowly. The
> software e-stop is a convenience, **not** a replacement for the hardware E-stop.

### Before your first real run
- [ ] Workspace clear; the arm has room for the nudge.
- [ ] A person is within reach of the hardware E-stop.
- [ ] You ran the dry run and the printed plan looks right.
- [ ] `nudge_deg` is small (≤ ~10°) and `move_time_s` is generous (slow).

---

## Configuration

Every interface name and safety limit is a ROS parameter in
[`ros2_ws/src/adl_primitives/config/test_arm.yaml`](ros2_ws/src/adl_primitives/config/test_arm.yaml)
(and `config/jog_ui.yaml` for the jog panel — keep the interface names in sync).
If `ros2 action list` or `ros2 control list_controllers` show different names on your robot,
**edit the YAML — don't touch the code.** Two values worth confirming on the real arm:

- `gripper_close_position` — the 2F-85's closed-joint value.
- `clear_faults_service` — off by default; set it once you find it via `ros2 service list | grep -i fault`.

## Developing on another machine

Edit on your workstation, **build and run on the Jetson over SSH** (ROS 2 Humble is
Linux-only). VS Code **Remote-SSH** into `abra@192.168.1.11` is the smoothest loop.
No credentials are stored in this repo.

## Repository layout

```
RAMMP-Kinova/
├── README.md                        # this guide
├── requirements.txt                 # apt dependencies (one per line)
├── LICENSE                          # MIT
├── docs/architecture.md             # design + roadmap
├── scripts/setup_ros2_kortex.sh     # one-shot setup + build (Linux only)
└── ros2_ws/src/adl_primitives/      # our ROS 2 Humble (Python / rclpy) package
    ├── adl_primitives/              # kinova_primitives.py, test_arm.py, jog_ui.py
    ├── config/                      # test_arm.yaml, jog_ui.yaml (names, limits)
    └── launch/                      # test_arm.launch.py, jog_ui.launch.py
```

## License

MIT — see [LICENSE](LICENSE).
