# Architecture & Roadmap

## Goal
Automate activities of daily living (ADLs) for individuals with motor disabilities with a
Kinova Gen3 arm. The thesis: instead of a brittle, hand-written script per task, build a
library of small, safe, composable **actions** and let an AI orchestrate them to
accomplish open-ended goals — paired with whatever input modality (voice, switch,
eye-gaze...) the user can operate.

## Layered design

```
                 ┌─────────────────────────────────────────┐
   user input ▶  │  voice ("computuh, ...") / typed text    │   more modalities later
                 └───────────────────┬─────────────────────┘
                 ┌───────────────────▼─────────────────────┐
                 │  AI orchestrator (Claude tool-use)       │   phase 4: decides WHICH
                 │  plan-first, typed results, closed-loop  │   action, in WHAT order
                 └───────────────────┬─────────────────────┘
                 ┌───────────────────▼─────────────────────┐
                 │  Action library (approach/grasp/place/   │   phase 3 ("modulation")
                 │  retreat/inspect — honest, typed results)│
                 └───────────────────┬─────────────────────┘
                 ┌───────────────────▼─────────────────────┐
                 │  cuRobo GPU planner + live perception    │   TODAY: working
                 │  collision-free segments, ~0.3 s plans   │
                 └───────────────────┬─────────────────────┘
                 ┌───────────────────▼─────────────────────┐
                 │  ros2_control                            │
                 │  • now: MuJoCo physics (mujoco_sim)      │
                 │  • later: ros2_kortex on the real Gen3   │
                 └───────────────────┬─────────────────────┘
                              ┌──────▼──────┐
                              │ Kinova Gen3 │  + RealSense D405 on the wrist
                              └─────────────┘
```

## What exists today

- **Simulation**: MuJoCo physics behind ros2_control (`mujoco_sim`) — same
  controllers and topics as the real kortex bringup, so the stack transfers.
  A full kitchen scene with free-physics props, generated from one
  `scene.yaml` that also feeds the planner (one source of truth).
- **Planning**: cuRobo v0.7.8 (`curobo_planner`) — collision-free plans in
  ~0.3 s, standoff/final/retreat segmentation around grasp targets,
  escape/back-out recovery, `check` self-test. Field-validated: 22/22
  collision-free tours.
- **Perception** (`rammp_perception`): two RGB-D cameras (fixed scene_cam +
  eye-in-hand D405), chromaticity detection → depth → 3D with honesty gates;
  the planner's world and targets follow live detections.
- **Voice**: offline Vosk app ("computuh, ..."), grammar built from the
  scene, sub-second speech-to-motion, spoken replies.
- **Diagnostics**: `check` (dry-plan all targets), `check_traj` (mm-level
  contact monitoring of executed trajectories), perception `probe`.

## Language split (deliberate)
- **Python (`rclpy`)** for orchestration, perception, planning (**cuRobo is
  PyTorch**), and the action library. At this tier there is no latency benefit
  to C++, and models stay in-process instead of crossing a language boundary.
- **C++ (`rclcpp`)** is reserved for a future **1 kHz low-level control node**
  (contact-rich manipulation), where Python's GIL and GC jitter disqualify it.
  ROS 2 is language-agnostic, so the layers interoperate over topics/actions.

## Phased plan

**Phase 2 — perception for the real world:** NanoOWL (TensorRT OWL-ViT,
~40-60 ms/frame on Orin) replaces the color backend: open-vocabulary text
prompts instead of declared colors. Same 3D path, gates, and planner wiring.

**Phase 3 — the action library ("modulation"):** small, safe actions —
approach / grasp / place / retreat / inspect — each returning **typed,
honest results** (success/failure + why + world state). The anti-pattern to
avoid, verified in the kinova-gemini reference: actions that return success
strings without checking reality.

**Phase 4 — AI orchestration:** a Claude tool-use loop over the action
library: plan-first, closed-loop (fresh perception after every
state-changing action), one retry per subgoal, semantic choices delegated to
the model and metric geometry kept in code (MOKA-style split).

**Phase 5 — multi-input:** route the input-device survey findings
(docs/input-devices-survey.md) into the same command path the voice app
uses — switches, head arrays, eye-gaze all publish the same text commands.

**Later — the real arm:** swap mujoco_sim's bringup for ros2_kortex on the
Gen3 (same controllers/topics; `velocity_scale` stays 1.0). The 1 kHz
low-level backend ([`rammp-org/kinova-gen3-driver`](https://github.com/rammp-org/kinova-gen3-driver))
and PREEMPT-RT belong to the contact-rich phase after that.

## Reference repositories
- `Kinovarobotics/ros2_kortex` — official ROS 2 Humble driver (the real-arm backend).
- `google-deepmind/mujoco_menagerie` — the Gen3 + 2F-85 models the sim composes.
- `rammp-org/kinova-gen3-driver` — in-house low-level 1 kHz C++ engine (future).
- `jakmilller/kinova-gemini` — AI-orchestrated manipulation reference (studied;
  its orchestration pattern informs phase 4, its honesty gaps inform phase 3).
