# mujoco_sim

MuJoCo physics as the **ros2_control backend** for the RAMMP stack, plus the
one-command bringup for everything. The mock hardware is replaced by real
contact physics; the planner drives the same topics
(`/joint_trajectory_controller/joint_trajectory`, `/joint_states`, ...) it
will use on the real arm.

Built on `ros-controls/mujoco_ros2_control` (official, apt-installable on
Humble arm64) and the DeepMind MuJoCo Menagerie models (Gen3 joints/actuators
are named `joint_1..joint_7` — identical to ros2_kortex, so everything maps
by name).

## One-time install (Jetson)

```bash
sudo apt install ros-humble-mujoco-ros2-control ros-humble-rosbridge-server xvfb
pip install mujoco                                   # aarch64 wheels exist
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
```

Build this package:

```bash
cd ~/RAMMP-Kinova/ros2_ws
colcon build --packages-select mujoco_sim --symlink-install && source install/setup.zsh
```

## Generate the scene (once, and after editing scene.yaml)

Composes Menagerie Gen3 + Robotiq 2F-85 (per the documented attach recipe) and
dresses the world from `curobo_planner/config/scene.yaml` (`scenery.py`): a
KITCHEN — the arm stands on a pedestal on a stone-topped center island (floor
at z=-0.75; scene.yaml z's are base_link-frame, island top at -0.07) with
props spread 360° around it and real kitchen mass along the wall. Furniture is
styled by name; props are composed multi-geom bodies. Styled geometry stays
inside the YAML envelopes, so the sim still physically matches what cuRobo
avoids. Two RGB-D cameras are declared here: the fixed `scene_cam` and the
eye-in-hand `d405` on the wrist.

```bash
ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie
# -> writes ~/.ros/mujoco_sim/scene_gen3.xml, validates it compiles, then
#    steps physics briefly: free props must NOT drift from their YAML poses
#    (a drift warning means a prop isn't resting on its furniture).
```

## Run — the whole stack, one command

```bash
ros2 launch mujoco_sim mujoco_bringup.launch.py
```

Starts: robot_state_publisher, the MuJoCo controller manager (wrapped in
`xvfb-run` so the camera renderer has a GL display on a headless Jetson),
the trajectory + gripper controllers, rosbridge (for the Windows tools),
both perception detectors, and the cuRobo planner. Wait for
`cuRobo planner ready`, then in another terminal:

```bash
ros2 run curobo_planner goto
```

## Watching it (Windows dev machine, no ROS)

```powershell
pip install mujoco roslibpy opencv-python
git clone https://github.com/google-deepmind/mujoco_menagerie.git C:\path\to\mujoco_menagerie
# run from this package's directory so `python -m mujoco_sim.*` imports:
cd C:\RAMMP-Kinova\ros2_ws\src\mujoco_sim
# regenerate the scene locally so mesh paths resolve:
python -m mujoco_sim.build_scene --menagerie C:\path\to\mujoco_menagerie `
  --scene ..\curobo_planner\config\scene.yaml --out .\scene_gen3.xml
python -m mujoco_sim.mirror_viewer --host 192.168.1.11 --model .\scene_gen3.xml
```

The mirror is a native MuJoCo window tracking `/joint_states` live. The ARM
is pinned kinematically (truth); the PROPS run local physics so the arm can
visibly push them (an approximation — Backspace resets local props).
Perception's current beliefs are overlaid as small labelled cyan spheres.
Add `--camera /rammp_detector/debug_image` (or `/d405_detector/debug_image`
for the wrist camera) to open a second window showing the annotated camera
frame — accepted detections tinted, rejects dimmed red.

## Checking trajectories

```bash
ros2 run mujoco_sim check_traj
```

Watches every published trajectory: waypoint-level collision checks against
the scene, closest-approach reporting, a live execution-contact monitor, and
trajectory recording to `~/.ros/mujoco_sim/traj_log`.

## Notes / gotchas

- MuJoCo box `size` is **half**-extents; cuRobo cuboid `dims` are full extents.
  `build_scene` converts — keep authoring full dims in `scene.yaml`.
- MJCF camera `resolution` defaults to **1×1** — build_scene sets 640×480 on
  the published cameras (the field failure looked like "perception finds
  nothing"; the probe names it).
- The gripper actuator is renamed to `robotiq_85_left_knuckle_joint` with a
  0–0.8 rad ctrlrange so the stock `robotiq_gripper_controller` drives it by name.
- The kortex display xacro path defaults to `kortex_description/robots/gen3.xacro`;
  if your ros2_kortex layout differs, pass `description_file:=<path>`.
- Do not run this and the kortex fake bringup at the same time — both spawn a
  `/controller_manager`.
