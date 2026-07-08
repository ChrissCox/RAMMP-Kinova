# mujoco_sim

MuJoCo physics as the **ros2_control backend** for the RAMMP stack. The mock
hardware is replaced by real contact physics; everything above the controllers —
`jog_ui`, `curobo_planner`, primitives — keeps working unchanged, on the same
topics (`/joint_trajectory_controller/joint_trajectory`, `/joint_states`, ...).

Built on `ros-controls/mujoco_ros2_control` (official, apt-installable on
Humble arm64) and the DeepMind MuJoCo Menagerie models (Gen3 joints/actuators
are named `joint_1..joint_7` — identical to ros2_kortex, so everything maps by
name).

## One-time install (Jetson)

```bash
sudo apt install ros-humble-mujoco-ros2-control ros-humble-mujoco-ros2-control-demos \
                 ros-humble-rosbridge-server
pip install mujoco                                   # aarch64 wheels exist
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie

# sanity check the simulator itself:
ros2 launch mujoco_ros2_control_demos 01_basic_robot.launch.py headless:=true
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
props spread 360° around it and real kitchen mass along the wall (counter,
microwave, upper cabinet, fridge, stool, trash bin). Furniture is styled by
name; props are composed multi-geom bodies (bottle with neck+cap, mug with
handle, bowl, apple, plate, snack box). Styled geometry stays inside the YAML
envelopes, so the sim still physically matches what cuRobo avoids:

```bash
ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie
# -> writes ~/.ros/mujoco_sim/scene_gen3.xml, validates it compiles, then
#    steps physics briefly: free props must NOT drift from their YAML poses
#    (a drift warning means a prop isn't resting on its furniture).
```

> First-run note: this uses the mjSpec composition API and is the most
> version-sensitive step. It validates itself (compiles the model + checks all
> actuator/joint names ros2_control needs) and fails loudly — if it errors,
> paste the message.

## Run

```bash
# Terminal A — MuJoCo-backed bringup (RSP + physics + controllers + foxglove):
ros2 launch mujoco_sim mujoco_bringup.launch.py            # add mirror:=true for the Windows viewer

# Terminal B — cuRobo planner against the SAME scene:
ros2 run curobo_planner planner --ros-args \
  -p scene_file:=$(ros2 pkg prefix curobo_planner)/share/curobo_planner/config/scene.yaml \
  -p use_sim_time:=true

# Terminal C — drive it:
ros2 run curobo_planner goto "go to the bottle"
```

`jog_ui` also works against this sim: `ros2 launch adl_primitives jog_ui.launch.py
sim:=true dry_run:=false` (the sim backend streams through the JTC, which now
moves real physics). Add `use_sim_time` where relevant — MuJoCo publishes `/clock`.

## Watching it

- **Foxglove (always works):** the 3D panel shows the arm via TF (now driven by
  MuJoCo physics) plus `/curobo_planner/markers` for obstacles/targets.
- **Native MuJoCo window on Windows (pretty):** launch with `mirror:=true`, then
  on the dev machine (no ROS needed):

  ```powershell
  pip install mujoco roslibpy
  git clone https://github.com/google-deepmind/mujoco_menagerie.git
  # regenerate the scene locally so mesh paths resolve:
  python -m mujoco_sim.build_scene --menagerie .\mujoco_menagerie --scene scene.yaml --out scene_gen3.xml
  python -m mujoco_sim.mirror_viewer --host 192.168.1.11 --model scene_gen3.xml
  ```

  It's a passive puppet of `/joint_states` — smooth, interactive camera, and it
  cannot affect the real simulation. (ROS 2 Humble itself doesn't support
  Windows 11, which is why the mirror uses rosbridge instead of native DDS.)

## Notes / gotchas

- MuJoCo box `size` is **half**-extents; cuRobo cuboid `dims` are full extents.
  `build_scene` converts — keep authoring full dims in `scene.yaml`.
- The gripper actuator is renamed to `robotiq_85_left_knuckle_joint` with a
  0–0.8 rad ctrlrange so the stock `robotiq_gripper_controller` drives it by name.
- The kortex display xacro path defaults to `kortex_description/robots/gen3.xacro`;
  if your ros2_kortex layout differs, pass `description_file:=<path>`.
- Do not run this and the kortex fake bringup at the same time — both spawn a
  `/controller_manager`.
