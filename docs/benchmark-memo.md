# RAMMP Benchmark Memo — rigorous evaluation for the arm system

Date: 2026-07-17. Sources: deep-research sweep (105 agents, adversarially
verified claims) + a dedicated rulebook study; every load-bearing fact below
is primary-sourced; sections marked INFERENCE are design synthesis, not found
fact. Companion to docs/grasp-proposer-memo.md.

## Why two layers

GraspNet-1Billion (Fang, Wang, Gou, Lu — CVPR 2020) benchmarks *grasp
proposals*, not robots: its evaluator is analytic (force-closure + scene
collision), with no arm in the loop. A defensible RAMMP evaluation therefore
needs two layers: the **proposer** scored on the community benchmark, and the
**system** (voice→brain→planner→grasp→verify) scored by a physical-protocol
suite adapted from the rulebooks below.

## Layer 1 — proposer, offline, zero protocol invention

Score our proposer (AnyGrasp and/or fallback) on GraspNet-1B's 90 test scenes
via graspnetAPI. Verified mechanics (graspnet.net/evaluation.html, graspnetAPI
docs + source):

- Dump predictions per scene/view as (N,17) `.npy`: `[score, width, height,
  depth, R(9), t(3), object_id]` → `dump_folder/scene_XXXX/<camera>/NNNN.npy`;
  run `GraspNetEval.eval_seen/eval_similar/eval_novel/eval_all`.
- Protocol applied by the evaluator: pose-NMS 3 cm / 30°, top-10 per object,
  top-50 overall; online analytic scoring (object association by
  points-inside-gripper; collision ⇒ negative; force-closure per µ).
- **AP = mean Precision@k (k=1..50) averaged over µ = 0.2..1.2, 6 bins.**
  NOTE: the arXiv paper says µ→1.0; the live benchmark + shipping code say
  1.2 — the code is canonical; cite both and note the discrepancy.
- Splits: scenes 100–129 seen / 130–159 similar / 160–189 novel; report per
  camera (RealSense/Kinect).
- Context anchors (verified in the model-landscape research, see
  grasp-proposer-memo): baseline 27.6 → GSNet 67.1 ≈ AnyGrasp 66.1 →
  EconomicGrasp 68.2 → FineGrasp 71.7+ (RealSense, seen, +CD).

What this buys: a community-comparable generality number that is independent
of our 8 objects — and it evaluates the *proposer*, on their fixed
D435/Kinect scenes, NOT our D405/MuJoCo stack. Open question (from the
research): re-hosting the analytic evaluator on OUR meshes/scenes is
possible in principle (it needs models + scene reconstructions) and would
give a domain-matched twin of this layer.

## Layer 2 — the system benchmark in sim (build now)

INFERENCE, grounded in the cited rulebooks. Design:

### Layouts (GRASPA pattern — arXiv:2002.05017, hsp-iit/GRASPA-benchmark)
- 2–3 FROZEN layouts of increasing difficulty (GRASPA uses 5/7/11 objects on
  printed A2 boards with ArUco frames; poses shipped as XML). Ours: authored
  scene.yaml layouts, poses published in the repo. Run **isolation** and
  **clutter** modes separately.
- GRASPA's honesty mechanism, adopted: gates are prerequisites, not scores.
  Map to our verification ladder — reachability S0 ≈ `ik_check.py`,
  calibration S1 ≈ `perception_test.py`, graspability S2 = payload/aperture
  check. A failed gate reports **N/A, never 0**.

### Success + stability definitions (GRASPA verbatim — cheap and citable)
- Grasp success (binary/trial): **lift 0.15 m, hold 5 s** (align our 12 cm
  lift to 0.15 m for comparability).
- Stability (optional score): after lift, a fixed 5-waypoint in-hand
  trajectory (±45° about approach, 30° toward table); score = waypoints
  retained.
- Task stages get PARTIAL CREDIT (reach/grasp/lift/place/release) — the YCB
  protocol template (Calli et al., RAM 2015) explicitly prescribes avoiding
  binary scoring; N-SCORE (arXiv:2603.13616) shows partial credit cuts trial
  burden up to ~70%.

### Trials, seeds, randomization (LIBERO/COLOSSEUM/robomimic norms)
- Pin seeded initial-state sets per task (object-pose jitter inside the
  workspace annulus). **≥25 episodes per task per condition** (COLOSSEUM's
  25/task/perturbation), 50 where cheap (Meta-World+/robomimic convention:
  50 rollouts; MW+ adds 10 seeds + IQM + 95% CI for learned policies).
- Record `success_once` vs `success_at_end` separately (ManiSkill3
  distinction — a release can drop the object after a "successful" grasp).

### Perturbation matrix (COLOSSEUM taxonomy — arXiv:2402.08191)
Their 14 axes; the MuJoCo-expressible subset for us: object pose, distractor
count, light color, camera pose, object color/size, mass/friction. Even 4–5
axes × 25 episodes = a defensible robustness table. COLOSSEUM findings to
cite: −30–50% success per factor, sim-real R² = 0.614.

### Reporting (per task)
successes/trials as RAW COUNTS + success rate with **Wilson 95% CI** (never
Wald — Brown/Cai/DasGupta 2001), planning time, execution time, end-effector
path length (Morgan's robot-BBT reporting list, RA-L 2020), and NAMED failure
causes (the YCB "to submit" field; also this repo's own law). CI reality
check (NVIDIA eval note): at 90% observed success, 70 trials ⇒ ±7.7 pp;
±2 pp costs ~1,030 trials. A 10-pp policy gap can need up to 500 trials/side
(STEP, RSS 2025) — for A/B comparisons use paired initial conditions +
sequential stopping (TRI LBM template: 200 rollouts/task in sim, 50 on
hardware, blind randomized A/B, Bonferroni-corrected sequential tests).

### Optional community anchor
robosuite ships **Kinova3 + Robotiq85 in MuJoCo** (the only major sim
benchmark with our exact hardware pair) — running its Lift/PickPlace at the
standard 500-step horizon yields one directly comparable external number.

## Layer 3 — real arm + assistive framing (later)

- Print a GRASPA-style A2 board with marker frame for our object set (or use
  GRASPA's actual layouts — object overlap is substantial) — kills the
  placement-reproducibility problem.
- Trial discipline: GRASPA T=5/object is the floor; headline rates need
  20–50 trials + Wilson CIs. Adopt Morgan/Robothon's credibility device:
  **N consecutive executions, uncut video** (Robothon: 5 consecutive, board
  re-randomized per trial).
- Assistive write-up framing (field norms, small-n): 16-task ADL battery
  with dependent/assisted/independent classification + per-task time
  (RESNA 2003 WMRA protocol); headline-metric format = Maheu et al. 2011
  (n=14): 72% task-time reduction, 41% projected caregiver-time reduction.
  Instruments: QUEST 2.0, PIADS, TEMPA, NASA-TLX; participant norms run
  7–31 in this literature. No ARAT-derived robot protocol exists — the
  clinical scale that crossed over is Box-and-Blocks (robot-BBT, Morgan
  et al.: 16 blocks, zero-score conditions, min 5 consecutive runs).
- Bridge sim→real EXPLICITLY: run the same frozen layouts on both, report a
  sim-real correlation (SRCC — Kadian et al. 2020: naive 0.18 → 0.844 tuned;
  COLOSSEUM R² = 0.614) instead of asserting transfer.

## Ecosystem notes (verified)

- AnyGrasp is first-party to GraspNet-1B (same four authors; project page on
  graspnet.net) — Layer-1 numbers are directly comparable to its lineage.
  Its own system protocol (T-RO 2023): bin-clearing 300+ unseen objects,
  93.3% success, human-parity claim (n=2 humans), picks/hour timing — the
  verified template Layer 2's tiering follows.
- SuctionNet-1B transplanted the online-analytic evaluation to suction
  (S = S_seal × S_wrench, same splits/NMS pattern) — evidence the protocol
  generalizes across end-effectors.
- R2SGrasp (arXiv:2410.06521) trains the detector purely in sim and repairs
  REAL depth toward sim at inference — citable support for our
  sim-now/real-later methodology.
- ARMBench (Amazon, ICRA 2023) is a perception dataset (segmentation mAP,
  Recall@k ID, defect recall@FPR), not a physical protocol — pattern source
  for perception-layer metrics only.
- NIST task boards fix the statistics, not the trial count: report mean/SD/
  95% CI and justify n via the binomial reliability bound
  F(m−1; n, P_S) ≥ CL. Falco et al. hand protocols: min 32 cycles.

## Build order (proposal)

1. `tools/benchmark.py` — Layer-2 runner: frozen layouts + seeded jitter,
   N episodes/task, stage partial credit, Wilson CIs, JSON+markdown report.
   (The probes built 07-15/16 are its embryo.)
2. Layer-1 run: download GraspNet-1B test split, dump proposer predictions,
   `eval_all` — one number per split/camera.
3. Perturbation matrix once (1) is stable.
4. Layer 3 when the real arm arrives.
