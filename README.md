# RAMMP-Kinova
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

To rebuild just this package after editing it (`--symlink-install` means `.py` edits need
no rebuild):

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

`hello_arm` reads the current pose, opens the gripper, nudges one wrist joint a few degrees
and back (slowly), then closes and reopens the gripper.

```bash
# Dry run (default): connects and prints the plan, moves NOTHING
ros2 launch adl_primitives hello_arm.launch.py

# Real motion: only after the safety checklist below
ros2 launch adl_primitives hello_arm.launch.py dry_run:=false
```

Handy overrides: `nudge_deg:=5.0 nudge_joint_index:=6 move_time_s:=6.0`

---

## Stopping the arm

- **Hardware E-stop** (red button on the power cable) — **authoritative**; use it for any real emergency.
- **Software e-stop** (cancels goals + deactivates the motion controller), from another terminal:
  ```bash
  ros2 service call /hello_arm/estop std_srvs/srv/Trigger "{}"
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
[`ros2_ws/src/adl_primitives/config/hello_arm.yaml`](ros2_ws/src/adl_primitives/config/hello_arm.yaml).
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
    ├── adl_primitives/              # kinova_primitives.py, hello_arm.py
    ├── config/hello_arm.yaml        # all tunables (names, limits)
    └── launch/hello_arm.launch.py
```

## License

MIT — see [LICENSE](LICENSE).
