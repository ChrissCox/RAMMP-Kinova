# adl_primitives

Milestone-1 ROS 2 (Humble) **Python / rclpy** package: basic Kinova Gen3 motion
primitives plus a `hello_arm` demo node. Built on top of
[`ros2_kortex`](https://github.com/Kinovarobotics/ros2_kortex).

> Python (not C++) on purpose: at the ros2_kortex high-level tier (~40 Hz) there's no
> latency benefit to C++, and this layer will call cuRobo / Gemini / perception, which
> are all Python. C++ is reserved for the future 1 kHz low-level control node.

## What it provides
- `KinovaPrimitives` (rclpy node) exposing reusable primitives:
  - `move_to_joint_positions(target, time_s)` via `control_msgs/action/FollowJointTrajectory`.
  - `open_gripper()` / `close_gripper()` / `command_gripper(pos, effort)` via
    `control_msgs/action/GripperCommand`.
  - `soft_stop()` — cancels active goals and (optionally) deactivates the motion controller.
- `~/estop` service (`std_srvs/srv/Trigger`) — software e-stop.
- `hello_arm` console entry point — the Milestone-1 demo sequence.

## Safety
- **`dry_run` defaults to `True`** — nothing moves until you pass `dry_run:=false`.
- The software e-stop is **not** a substitute for the **hardware E-stop**. The Gen3 has
  no mechanical brakes.

## Run
See the repository [README](../../../README.md) for full bring-up and run instructions.

```bash
ros2 launch adl_primitives hello_arm.launch.py            # dry run
ros2 launch adl_primitives hello_arm.launch.py dry_run:=false   # real motion
ros2 service call /hello_arm/estop std_srvs/srv/Trigger "{}"    # software e-stop
```

All names and limits are parameters — see [`config/hello_arm.yaml`](config/hello_arm.yaml).
