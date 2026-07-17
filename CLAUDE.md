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

## Current state (2026-07-15)

Bottle AND mug grasp-and-lift CONFIRMED (gripper-verified 30 mm pinch /
60 mm body grip, 12 cm lift, release, home). The mug took a day of forensic
sim work — the lesson is an invariant now:
- The gripper VERDICT chain can lie without the grasp failing: a close
  onto a wide body CREEPS (~5 mrad/s squeezing the 64 mm mug), the
  controller's stall detector (default < 1 mrad/s) never fires, the
  GripperCommand action never returns, and the planner's timeout read a
  firmly HELD mug as 'closed on air' four times running. Fixed at both
  ends: controllers.yaml treats slow creep as stall=success
  (`allow_stalling` + `stall_velocity_threshold 0.02`), and the planner
  falls back to the live knuckle angle from /joint_states when the action
  times out. Never diagnose a MISSED from the verdict alone — knuckle_watch
  (/joint_states through the close) and an offline MuJoCo close-replay are
  the instruments that cracked it.
- check_traj CANNOT see the close (it replays trajectories; the gripper
  action isn't one) and its prop poses are SPAWN poses, not live sim truth.
- Grasp synthesis grips near the CoM now (0.9*half-height, capped 39 mm =
  palm clearance) and every synthesized pose is logged.
- The brain defaults to claude-haiku-4-5, extended thinking OFF: 0.87 s
  decision latency warm, ~1.6 s cold (was 2-15 s on sonnet+thinking).
  `brain_model:=claude-sonnet-4-6 brain_thinking:=true` for hard tasks.
  haiku does NOT support adaptive thinking (API 400) — never pair them.
- Still true, must not regress: tool_tip_offset 0.021 m (pad centers);
  detector occlusion gates; streamed brain abort (<0.1 s); stop sacred
  client-side (tools/voice_gate_test.py); launch ONLY via
  tools/launch_stack.zsh (two stacks = two worlds).

AnyGrasp RUNS NATIVELY on the Jetson (2026-07-15, likely first-on-record):
license is PERMANENT and bound to this Jetson's feature ID; MinkowskiEngine
0.5.4 built from the maintainer's cuda-12-1 fork (the documented
shared_ptr_base.h fix applied via an include-shadowed COPY at ~/me_patch —
system headers untouched, no sudo); SDK deps live in ~/anygrasp_venv
(--system-site-packages; graspnetAPI installed --no-deps, its numpy==1.20.3
pin is toxic); demo.py passes on the example data (~19 s incl. model load).
Layout on the Jetson: ~/anygrasp_sdk (dev branch), ~/AnyGrasp (license +
checkpoints), ~/anygrasp_venv. The proposer node RUNS IN THE BRINGUP
(rammp_perception grasp_proposer): loads once (~8 s), on-demand via
/grasp_proposer/request (object name or '' → JSON proposals in base frame
on .../proposals; errors are named, incl. the camera's actual footprint).
Every grasp now SCANS FIRST (use_anygrasp, scan_views params): the arm
orbits the object (_vantage_pose aims the camera via _VIEW_IN_TOOL
[0.20,0.61,0.77] + _CAM_IN_TOOL [0,-0.058,-0.24], both MEASURED
2026-07-15), pools proposals from every reachable view, gates them
(score>=0.05 / width / not-from-below / on-object / island margin /
goal_feasible) with PER-GATE rejection counts logged, and executes the
best survivor; the tool-down synthesizer is the loud last resort, and
too-wide objects (plate) get scan+proposals instead of instant refusal.
Scan-view tracking failures are SOFT (skip the view, keep the pool) —
views pass ~30 mm from the shelf post and controller lag clips it
sometimes; the start-in-collision escape ladder recovers (field-proven:
a failed view left the arm ON the apple and the grasp still completed).
THE BRAIN HAS EYES (2026-07-16, kinova-gemini pattern — see that repo's
ag_kinova_final.py): look tool → INSTANT scene-camera capture (zero arm
motion — wrist-vantage looks were slow, contorted, and worse; killed
same day) → the proposer freezes image+cloud+pose together and ZOOMS the
JPEG on the named object (the detector knows where it is; the brain only
boxes the PART — whole-frame boxing wandered) → part_box → proposals on
the frozen frame; if the scene view yields nothing (it is at 1.6 m, out
of the net's 0.4-0.7 m training range), the box's 3D volume rides to a
WRIST-camera retry at the standoff the grasp already flies to — semantic
choice from the scene view, metric proposals from the in-range view,
zero added motion. dense_grasp=True; part-box pools accept score>=0.015
(brain-vouched + four safety layers); orbit scan is opt-in
(scan_views:=N, default 0). 'grab the mug' ~18-25 s task time. Mug
handle is SOLID (scenery.py); scene_cam is 1280x960 (D405-native
parity); proposer respawns (a whole-kitchen cloud once OOM-killed it —
scene clouds are workspace-cropped now). GRASPED names its path and the
brain is prompt-bound to never embellish. A PEEK hop (one short tool-down move, offset by the camera's measured
view offset) lands the wrist camera on the object for the part retry —
the D405 physically cannot see straight down from a standoff. Brain
default is claude-sonnet-4-6 now (reasoning+vision > raw speed; haiku
asked questions as text, embellished, and boxed sloppily); prompt-bound:
no question marks ever, garbled ASR input ('google', 'the bug') ends in
a task_complete statement, done-when-released. Voice open-vocab text is
ACCEPTED ONLY when it agrees with the grammar hearing (>=60% content-
word overlap — free-model hallucinations once turned 'go bowl' into
'google'). The planner's latched 'ready' is filtered from the brain's
verdict stream (it once terminated 'go home' instantly). Look images
carry a 250/500/750 grid in box coordinates. HONEST STATE: the part
pipeline is mechanically complete (scene box -> 3D region -> peek ->
wrist proposals in 0.6 s) but sim proposals still score ~0.01-0.03 =
below even the relaxed floor — THE wall is AnyGrasp's real-sensor
training vs clean sim clouds, and fine-tuning is vetoed (keep it
general). Expect part grasps to come ALIVE on the real D405 (real-data
scores run 0.9+, per the 2026-07-16 research); in sim the geometric
fallback carries. Brain: ask_user REMOVED. Feature-ID reboot-drift
(#164) still unprobed.

GRASPGEN-X REPLACED ANYGRASP (2026-07-17, authored, NOT yet
Jetson-validated): the proposer is now a torch-free ZMQ client to
GraspGenX's own shipped server (NVlabs/GraspGenX, Apache-2.0, native
robotiq_2f_85; trained on synthetic clouds — the sim score collapse that
crippled AnyGrasp in MuJoCo should not apply, MEASURE IT). Bringup
starts the server from ~/graspgen_venv + ~/GraspGenX
(tools/install_graspgen.zsh installs; --no-deps playbook, needs git-lfs;
ROS python needs pyzmq msgpack msgpack-numpy). Conventions: +Z approach,
+X closing, pose at gripper base link, fingertips +0.136 m along +Z
(their config.json — do not resurrect 0.1303); width is computed by the
proposer from cloud extent (GraspGen emits poses+scores only); scores
are discriminator confidences [0,1] — planner floors are params
(proposal_min_score 0.5 / _part 0.35, sim-UNCALIBRATED, tune on the
Jetson). AnyGrasp venv/license remain untouched on the Jetson; git
revert restores it. First Jetson session: run the install script, then
calibrate floors + pad-center depth (fingertip vs pad-center bias
~2 cm class — measure like the bottle z-window was measured).

KNOWN OPEN: `check` fails cabinet_handle / shelf_edge / pills dry-plans
(IK_FAIL at goals equal to their historical values — likely stale since a
world edit; retune with pose probes). Next: place-on/handover tools,
NanoOWL backend, real-arm bridge (velocity_scale stays 1.0).
