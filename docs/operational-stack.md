# Operational stack — what runs, and in what order

Two kinds of flow: **continuous loops** that keep the system's beliefs true,
and the **command pipeline** that turns an utterance into motion. Everything
below exists and runs today (sim); the marked seams are where phase-2/3
components drop in without reordering anything.

## Continuous loops (always on, independent rates)

| Loop | Rate | What it maintains |
|---|---|---|
| MuJoCo physics + ros2_control | 500 Hz | ground truth + `/joint_states`, `/clock`, TF |
| Camera rendering | ~5 Hz | scene_cam + d405 RGB-D topics |
| Detector × 2 (scene_cam, d405) | 2 Hz duty cycle | `/perception/objects` (base-frame positions), debug images |
| Planner's live cache | on arrival | freshest position per object (10 s staleness → YAML fallback) |

Rules the loops obey (each earned by a field failure):
- No camera_info → no detection tick (never guess intrinsics).
- Eye-in-hand: color+depth must share a stamp; TF is looked up **at that
  stamp**; unservable ticks are skipped, not guessed.
- Depth from the mask interior only; positions gated by workspace box and
  max-jump; unseen objects keep their last-known pose. A stale pose beats a
  wrong one.
- Targets follow detections >1.5 cm; collision boxes move only >4 cm.

## The brain (system modulation)

Tasks land on `/rammp/task` first, where the **brain** (Claude, tool-use)
sees the live world — object positions from perception, geometry, what the
gripper holds — and picks from a tool hierarchy: `reach`, `grasp`,
`release`, `home`, `say`, `ask_user`, `task_complete`, and **`move_tool`,
where it creates its own endpoints** (base-frame x/y/z + an orientation
family). Every tool returns the planner's real verdict plus a fresh world
snapshot — the model never reasons about a stale world (the honesty gap in
the kinova-gemini reference this design is modeled on). The semantic/metric
split holds: the brain owns intent and coarse geometry; cuRobo may refuse
any endpoint, and the refusal text is treated as ground truth. STOP words
bypass the brain entirely (forwarded to the planner instantly, mid-task
included), and without an API key the brain is a verbatim passthrough —
the pipeline below, unchanged.

## Command pipeline (per planner command)

```
"computer, grab my water"          (voice / goto CLI / the brain's tools)
 1. STOP check           — stop words handled BEFORE the command lock
 2. intent               — RELEASE? GRASP? else reach/home/check/pose
 3. scene reload         — scene.yaml re-read, live detections applied
 4. object resolution    — free text -> object (name match, or target's
                           reach-for prop: "pills" -> pill_bottle)
 5. grasp synthesis      — LIVE position + geometry -> fingertip pose:
                           top-down family, grip depth from object height,
                           pad clearance floor, width vs 85 mm stroke,
                           yaw candidates (boxes: across the minor axis)
                           [SEAM: AnyGrasp/learned proposer replaces this]
 6. reachability filter  — candidates tried in order; cuRobo plans the
                           standoff with the FULL world; first success wins
 7. approach             — standoff (full world) -> descent (only the
                           grasped object exempt)
 8. close + VERIFY       — GripperCommand; the achieved position is the
                           verdict: full close = MISSED (announced, gripper
                           reopens, arm retreats); early stall = holding
 9. lift + hold state    — 12 cm lift; the object becomes part of the arm
                           (exempt from the world while held); "release"
                           opens and clears the hold
10. report               — every outcome named and spoken, incl. the
                           gripper gap vs the expected object width
```

Reach-only commands ("go to the bottle") run the same pipeline minus 5/8/9.
While holding, all planning keeps the held object exempt; a second grasp is
refused until "release".

## Failure ladder (what happens when steps fail)

- Start in collision → diagnose touching props → escape up → back out along
  the last trajectory → reject with names, never guess.
- No reachable grasp angle → honest "no reachable approach (tried N angles)".
- Object wider than the gripper → honest refusal, no attempt.
- Gripper closed on air → reopen, retreat, report MISSED + hint to check
  the camera windows.
- Arm settles far from the commanded endpoint → TRACKING FAILURE named in
  the log (physical contact or saturation).

## Phase seams

- **Grasp proposal (step 5)** is a pure function: (position, geometry) →
  ranked fingertip poses. AnyGrasp or any point-cloud proposer replaces it
  behind the same interface — everything from step 6 down is unchanged.
  (AnyGrasp SDK is closed-source, licensed per machine, x86-centric —
  research memo pending on Jetson feasibility; graspnet-baseline /
  lightweight depth nets are the fallback path.)
- **Detection (loop)**: NanoOWL replaces the color backend for real-world
  objects; same 3D path and gates.
- **Real arm**: mujoco_sim's bringup swaps for ros2_kortex; controllers,
  topics, and this entire pipeline are unchanged (velocity_scale stays 1.0).
- **Phase 4**: a Claude tool-use loop calls these same steps as typed,
  honest actions (grasp/release/reach/look), replanning per result.
