# RAMMP-Kinova — agent instructions

Assistive robotics: a Kinova Gen3 7-DoF + Robotiq 2F-85 performs daily-living
tasks for people with motor disabilities, driven by voice, planned by cuRobo,
simulated in MuJoCo (real arm later). Read `docs/architecture.md` (design +
phases) and `docs/operational-stack.md` (runtime order) before large changes.

## Environment (Jetson AGX Orin)

- Ubuntu 22.04, ROS 2 Humble, **zsh** — source `install/setup.zsh`, never
  `.bash`. `ROS_LOCALHOST_ONLY=1` is set in `~/.zshrc`. User `abra`;
  arm 192.168.1.10, this Jetson 192.168.1.11.
- cuRobo **v0.7.8 pinned** (v0.8 is an API rewrite — do not upgrade) on a
  Jetson CUDA torch. MuJoCo via pip; menagerie at `~/mujoco_menagerie`;
  `mujoco_ros2_control` from apt.
- `ANTHROPIC_API_KEY` in `~/.zshrc` (repo is PUBLIC — never commit keys).
- Windows dev machine runs the mirror viewer + voice app over rosbridge
  (`C:\RAMMP-Kinova` checkout + Desktop launchers) — keep their interfaces
  (topics `/rammp/task`, `/rammp/task_status`, `/rammp/say`,
  `/curobo_planner/status`, rosbridge :9090) stable or tell the user to
  update that side.

## Run / build / test

```zsh
# the whole stack, one command (sim, controllers, rosbridge, 2 detectors,
# planner, brain):
ros2 launch mujoco_sim mujoco_bringup.launch.py     # brain_model:=... enable_finetune:=...
ros2 run curobo_planner goto                        # talk to it (--direct = skip the brain)

# after EVERY scene.yaml edit that touches geometry (the planner warns at
# startup if you forget — believe it):
ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie   # then restart bringup

# rebuild (only needed for NEW files/entry points/launch/config — .py edits
# are symlinked):
cd ~/RAMMP-Kinova/ros2_ws && colcon build --packages-select curobo_planner mujoco_sim rammp_perception --symlink-install && source install/setup.zsh
```

Verification ladder (use it after any planner/perception/scene change):
1. `python3 tools/ik_check.py` — DLS IK ground truth for every target (+
   ad-hoc poses: `name,x,y,z,r,p,yaw` args). 30/30 clean is the bar.
2. `python3 tools/perception_test.py` — renders scene_cam offline, runs the
   real detector code, compares vs sim truth. Bar: every prop ≤ 40 mm
   (typically ≤ 10 mm).
3. `check` in goto — dry-plans every target + standoff through cuRobo.
4. `ros2 run mujoco_sim check_traj` — live mm-level contact monitor during
   execution.
5. `ros2 run rammp_perception probe` — one-shot camera health with named
   failure causes (also `--ros-args -p rgb_topic:=/d405/color ...`).

## Invariants — hard-won, do not relearn these

cuRobo v0.7.8:
- `plan_single_js` is UNUSABLE on this Jetson (internal graph fallback calls
  torch.svd → missing cusolver → DT_EXCEPTION even at attempts=1). Home is
  planned in POSE space via cached FK.
- `finetune_attempts` stays 5: the loop exits on first success; fewer
  attempts converts hard-goal successes into failures.
- `update_world`: zero cuboids silently keeps the previous world; more than
  `collision_cache_obb` raises. Cylinders/spheres in a WorldConfig are
  SILENTLY DROPPED — props go in as cuboids only.
- Result buffers are padded — NEVER read `optimized_plan[-1]` (stale tail
  garbage caused violent 360° jerks).
- `retract_config` must be seeded to OUR home family (the bundled default is
  the opposite elbow family → winding).

Motion/safety:
- `velocity_scale` MUST stay 1.0 (1.4 torque-saturated the motors and cut
  corners off collision-checked paths). Faster motion needs actuator gains,
  not faster setpoints.
- The standoff→final→retreat segmentation is load-bearing: the reach-for
  prop is exempt ONLY on the final few cm, never in transit.
- STOP words (stop/halt/freeze/cancel/estop) are handled before the
  planner's command lock and bypass the brain's API entirely. Never put an
  API call or lock on the stop path.
- world_padding 0.02 + boosted arm spheres + gripper shell are audit-tuned;
  a grasp needs ~10 cm free span around the object (padded neighbor boxes
  pinch goals into IK_FAIL — that's honest, move the prop or the target).
- Tool-down poses above tool_frame z≈0.35 are kinematically unreachable.

Sim (MuJoCo):
- base_link↔shoulder_link meshes interpenetrate → contact exclude in
  build_scene (without it, ~105 Nm of phantom friction brakes joint_1).
- MJCF camera `resolution` defaults to 1×1 — published cameras must set it.
- TRANSPARENT geometry skips the depth buffer in the live camera pass →
  detector depth reads the background BEHIND it. All perception-tracked
  props must be opaque. (Offline `mujoco.Renderer` depth ignores alpha, so
  offline tests will NOT catch this class of bug.)
- Free props need condim 6 + rolling friction (default condim 3 = perpetual
  wobble, spheres roll off tables).
- The gripper attach quat and tool_spin_deg=90 encode the physical gripper's
  90° mounting twist — goals are authored for the physical fingers.

Perception honesty rules (each one bought with a field failure):
- No camera_info yet → NO detection tick (a guessed focal length shrank
  every position ~25% and locked out the real values).
- Eye-in-hand: color+depth must share a stamp; TF is looked up AT that
  stamp; unservable ticks are skipped.
- Depth from the mask interior (2 px erosion — AA halos poison the median),
  low-percentile anchor + size-hint push to object CENTER.
- Workspace box + max_jump 0.25 gates; unseen props keep last-known pose.
- Planner live thresholds: targets follow >1.5 cm, collision boxes move only
  >4 cm (`live_box_shift`) — noise must not reshape a verified world.
- Debug images are JPEG (`~/debug_image/compressed`) — raw frames fragment
  over rosbridge and roslibpy cannot reassemble them.

The brain (`curobo_planner/brain.py`):
- `/rammp/task_status` is a STRICT terminal protocol: '...'-prefix = interim,
  first plain message ends the task for every listener. Speech goes on
  `/rammp/say`, never as a plain status mid-task.
- Never publish a planner command after `_abort` — a fresh command
  legitimately clears the planner's stop-hold.
- Every tool result carries a fresh world snapshot; keep it that way — the
  model must never reason about a stale world. Honest failure text beats
  fake success (the kinova-gemini anti-pattern).

rclpy: logger severity is cached per source LINE — keep `.error()` and
`.info()` on separate lines or the node dies mid-flight.

## Conventions (the user: Chris)

- Commit AND push freely without asking (standing instruction). Repo is
  public: no keys, no credentials, ever.
- No "milestone" framing in docs/code. Plain names (`estop`, `brain`,
  `goto`). FEWER launch files — extend the one bringup, don't add more.
- Every failure message must NAME the cause and, when possible, the fix
  command. If tests fail, say so with output — never soften results.
- Debug artifacts from Chris (screenshots etc.) land in docs/ untracked —
  leave them uncommitted unless asked.

## Current state (2026-07-14, evening)

Grasp-and-lift CONFIRMED on a fresh `build_scene`: grab the bottle → 30 mm
neck pinch (gripper-verified), 12 cm lift, release sets it down and escapes
up, home — all clean in check_traj, full voice→brain→planner path. What it
took, and must not regress:
- `tool_tip_offset` (planner param, 0.021 m): cuRobo's tool frame is NOT
  the fingertips — measured 21 mm short of the pad centers. Every
  fingertip goal is pulled back along tool z; authored targets now MEAN
  pad centers. Home/escape (FK poses, `spin=False`) are exempt.
- Detector partial-visibility gates: border-clipped masks and abrupt blob
  shrink (the ARM occluding the object mid-grasp biased the bottle 2 cm)
  skip the tick; detections carry their camera in `Detection3D.id`.
- The brain aborts mid-API-call now (streamed response, abort polled per
  event): stop → 'Task aborted.' in <0.1 s, was 40 s.
- Stop is sacred client-side too (voice/computer.py rewrite + 29 offline
  cases in tools/voice_gate_test.py; computuh.py fork deleted).
- Launch hygiene: kill by `pkill -9 -f 'ros[-]args'` ([-] avoids
  self-match) — killing by node names missed the planner, and TWO stacks
  ran at once publishing conflicting detections from two worlds.

KNOWN OPEN: `check` fails cabinet_handle / shelf_edge / pills dry-plans
(IK_FAIL at goals equal to their historical values — likely stale since a
world edit; retune with pose probes). Mug grasp untested since the tip
calibration (its old rim-clip may be fixed by it). In flight: AnyGrasp
license (docs/grasp-proposer-memo.md — HGGD recommended; off critical
path). Next: place-on/handover tools, NanoOWL backend, real-arm bridge
(velocity_scale stays 1.0).
