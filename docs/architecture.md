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
   THIS REPO ▶   │  Primitive library (Python / rclpy)      │   move_to_pose, grasp,
                 │  small, safe, reusable skills            │   open/close, approach...
                 └───────────────────┬─────────────────────┘
                                     │ commands
                 ┌───────────────────▼─────────────────────┐
                 │  Control backend                         │
                 │  • now: ros2_kortex (high-level ~40 Hz)  │
                 │  • later: 1 kHz low-level + cuRobo        │
                 └───────────────────┬─────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │ Kinova Gen3 │  + RealSense D405 (perception, later)
                              └─────────────┘
```

## Language split (deliberate)
- **Python (`rclpy`)** for the orchestration (Gemini), perception (SAM 2 / AnyGrasp /
  RealSense), planning (**cuRobo is PyTorch**), and the **primitive library**. At the
  ros2_kortex high-level tier (~40 Hz) there is no latency benefit to C++, and keeping
  this layer in Python means cuRobo / models are direct in-process calls, not a
  cross-language ROS boundary.
- **C++ (`rclcpp`)** is reserved for the future **1 kHz low-level control node**, where
  Python's GIL + garbage-collection jitter make it unsuitable for a hard-real-time 1 ms
  loop. ROS 2 is language-agnostic, so the two layers interoperate over topics/actions.

## Phased plan

**Milestone 1 (current):** ROS 2 Humble workspace; bring up the arm via `ros2_kortex`;
one rclpy node executes a basic, slow, visible motion + gripper actuation, with a software
e-stop. Establishes the primitive layer's first skills (move-to-joint-pose, open/close
gripper) and the dev→Jetson loop.

**Next:** grow the primitive library (Cartesian move, approach/retreat, grasp) behind a
backend-agnostic interface so the control backend is swappable, then wire MoveIt 2 / cuRobo.

**Later — performance:** the high-level `ros2_kortex` path runs ~40 Hz, fine for coarse
motion but not for reactive / contact-rich manipulation. For that we move to a **1 kHz
low-level** backend (the in-house
[`rammp-org/kinova-gen3-driver`](https://github.com/rammp-org/kinova-gen3-driver) — currently
gravity-comp + Cartesian impedance only) wrapped as a `ros2_control` hardware interface, and
adopt **NVIDIA cuRobo** for fast GPU motion generation on the Jetson. The 1 kHz control loop
must live in a dedicated PREEMPT-RT thread — never routed through ROS 2 middleware.

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
```
