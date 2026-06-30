# Architecture & Roadmap

## Goal
Automate activities of daily living (ADLs) for individuals with motor disabilities with a
Kinova Gen3 arm. The thesis: instead of a brittle, hand-written script per task, build a
library of small, safe, composable **primitives** and let a real-time multimodal AI
orchestrate them to accomplish open-ended goals — the way a person coordinates their
senses and limbs to carry out a task.

## Layered design (target)

```
                 ┌─────────────────────────────────────────┐
                 │  AI orchestrator (real-time, multimodal) │   e.g. Gemini Live,
                 │  decides WHICH primitive, in WHAT order  │   function-calling
                 └───────────────────┬─────────────────────┘
                                     │ calls named primitives
                 ┌───────────────────▼─────────────────────┐
   THIS REPO ▶   │  Primitive library (C++ / ROS 2)         │   move_to_pose, grasp,
                 │  small, safe, reusable skills            │   open/close, approach...
                 └───────────────────┬─────────────────────┘
                                     │ commands
                 ┌───────────────────▼─────────────────────┐
                 │  Control backend                         │
                 │  • now: ros2_kortex (high-level ~40-50Hz)│
                 │  • later: 1 kHz low-level + cuRobo       │
                 └───────────────────┬─────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │ Kinova Gen3 │  + RealSense D405 (perception, later)
                              └─────────────┘
```

## Phased plan

**Milestone 1 (current):** ROS 2 Humble C++ workspace; bring up the arm via `ros2_kortex`;
one node executes a basic, slow, visible motion + gripper actuation, with a software
e-stop. This establishes the primitive layer's first two skills (move-to-joint-pose,
open/close gripper) and the dev→Jetson loop.

**Next:** grow the primitive library (Cartesian move, approach/retreat, grasp), add a
clean C++ API/action interface, wire MoveIt 2 for planning.

**Later — performance:** the high-level `ros2_kortex` path tops out around 40–50 Hz, which
is fine for coarse motion but not for reactive / contact-rich manipulation. For that we
move to a **1 kHz low-level** backend (the in-house
[`rammp-org/kinova-gen3-driver`](https://github.com/rammp-org/kinova-gen3-driver) — currently
gravity-comp + Cartesian impedance only) wrapped in a ROS 2 node, and adopt **NVIDIA
cuRobo** for fast GPU motion generation on the Jetson.

**Later — perception:** integrate the RealSense **D405** for vision-guided grasping.
Reference architecture: [`jakmilller/kinova-gemini`](https://github.com/jakmilller/kinova-gemini)
(C++ `kortex_controller` for execution + Python Gemini brain doing function-calling, with
SAM 2 segmentation + AnyGrasp 6-DoF pose estimation). Note it uses a D435i, not a D405.

**Vision (long-term):** fuse multiple real-time inputs simultaneously (vision, force /
proprioception, …) so tasks are carried out robustly under uncertainty.

## Reference repositories
- `Kinovarobotics/ros2_kortex` — official ROS 2 Humble driver (our current backend).
- `rammp-org/kinova-gen3-driver` — in-house low-level 1 kHz C++ engine (future backend).
- `rammp-org/kinova-quest-teleop` — Meta Quest 3 teleop (Python) reference.
- `jakmilller/kinova-gemini` — AI-orchestrated manipulation reference architecture.
