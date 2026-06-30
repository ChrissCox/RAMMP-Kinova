# adl_primitives

Milestone-1 ROS 2 (Humble) C++ package: basic Kinova Gen3 motion primitives plus a
`hello_arm` demo node. Built on top of [`ros2_kortex`](https://github.com/Kinovarobotics/ros2_kortex).

## What it provides
- `KinovaPrimitives` — a node exposing reusable primitives:
  - `moveToJointPositions(target, time_s)` — slow absolute joint move via
    `control_msgs/action/FollowJointTrajectory`.
  - `openGripper()` / `closeGripper()` / `commandGripper(pos, effort)` via
    `control_msgs/action/GripperCommand`.
  - `softStop()` — cancels active goals and (optionally) deactivates the motion controller.
- `~/estop` service (`std_srvs/srv/Trigger`) — software e-stop.
- `hello_arm` executable — the Milestone-1 demo sequence.

## Safety
- **`dry_run` defaults to `true`** — nothing moves until you pass `dry_run:=false`.
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
