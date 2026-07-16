# Research Adoption Memo — what the GraspNet lineage offers RAMMP

Date: 2026-07-16. Synthesis of five primary-source research sweeps
(execution layer; language/part-conditioned grasping; the GraspNet-1B
descendant map; failure recovery; outcome verification), plus a second
wave the same day: LLM planning & the VLA landscape (nine fact-checked
reports — see that section below). Facts are cited; INFERENCE marked.
Companions: grasp-proposer-memo.md (model landscape), benchmark-memo.md
(evaluation).

## The headline validations

- **Our architecture is published, twice**: LAN-Grasp (arXiv 2310.05239 —
  LLM names the part → open-vocab grounding → classical planner) and
  AffordGrasp (arXiv 2503.00778 — GPT-4o task→object→part reasoning →
  pixel MASK → AnyGrasp on the part). The field converged on exactly our
  decomposition; our one weak joint (VLM boxes) is the field's documented
  weakness, with a known fix.
- **Open-loop execution costs ~30 points**: Grasp-MPC (arXiv 2509.06201)
  measured 43% open-loop vs 73% closed-loop in clutter; GG-CNN
  (arXiv 1804.05172) hit 88% on objects MOVED mid-grasp. Our
  "nudged object → closed on air" class is the known price of open-loop.
- **OK-Robot (arXiv 2401.12202, MIT code) published the integration
  recipe**: project AnyGrasp proposals into the image, keep those inside
  the language-selected mask (filter, don't crop), rank by
  `S − θ⁴/10` (orientation penalty tuned to what your calibration
  tolerates), approach in shrinking staged waypoints
  (0.2→0.08→0.04 m→contact). 58.5% across 10 real homes; its successor
  DynaMem kept the same pattern.
- **COME-robot (arXiv 2404.10220) is our exact platform published**:
  Kinova Gen3 7-DoF + Robotiq gripper + wrist RGB-D + GPT-4V brain. Its
  recovery vocabulary — {re-observe, retry-with-next-candidate,
  reposition, replan, report} — recovered **70.8% of failed grasps**
  (17/24); 75% end-to-end vs 47.5% for open-loop Code-as-Policies.
- **Retry is the cheapest success multiplier, IF the retry differs**:
  Robot Utility Models +15.6 pp with avg 1.31 tries (verified retry);
  FAR's controlled baseline: naive retry +12.1 pp, perturbed retry
  +16.4 pp. The trap is published too: an unchanged world re-ranks the
  SAME failing grasp forever — Dex-Net 4.0's failure-memory mask took
  adversarial objects 63→80%, and non-Markov retry policies beat Markov
  by +107% MPPH (arXiv 2007.10420). OK-Robot's authors name their lack
  of retry as limitation #1 (58.5% overall, errors multiply per stage).

## Unified implementation queue (value/effort, cheapest first)

### Tier 1 — hours each, sim-testable now
1. **`approach_steering` toward top-down** — the licensed SDK we run has
   it (USAGE.md; we already pass `region_steering` but leave
   `approach_steering: None`). Biasing proposals toward the net's own
   viewpoint sweet spot (its front/side-view failures are documented,
   issue #133) attacks the sim score-suppression at zero cost.
2. **Vision lift-verification** — after lift, the EXISTING detector says
   whether the object left its table pose (QT-Opt's image-difference
   pattern, arXiv 1806.10293). Kills aperture-verdict false positives;
   also verifies release placement.
3. **OK-Robot ranking in _anygrasp_select** — orientation penalty
   (adapt `S − θ⁴/10`: penalize approaches our calibration and
   reachability envelope handle worst) + prefer proposal-in-mask
   membership over the current hard 3D-region crop.
4. **Failure-class enum in planner verdicts** — adopt AHA's taxonomy
   (arXiv 2410.00371): no_grasp (closed fully) / slip (width changed in
   transit) / mis_position (object moved post-attempt) / unreachable
   (IK_FAIL) / wrong_object. Each class names its own recovery (CoPAL's
   error-routing insight, arXiv 2310.07263); our "failure messages must
   name the cause" law, formalized so the brain selects recovery by
   class instead of parsing prose.

### Tier 2 — days each
5. **Bounded verified retry, never identical** — on a MISS: retreat
   (exists) → one fresh detector tick → if the object MOVED, re-plan at
   the new pose; if the world is UNCHANGED (<1.5 cm — our threshold),
   never re-send the same goal: perturb z/yaw or take the next-ranked
   proposal (Dex-Net 4.0 failure memory; non-Markov policies +107% MPPH;
   FAR: perturbed beats naive retry +4.3 pp). Bound ≤2 retries (RUM used
   10 unattended; ReplanVLM capped 5; RUM succeeded within 1.31 tries on
   average). Expose COME-robot's vocabulary as brain-visible outcomes;
   retries are interim '...' messages under the strict terminal protocol.
   Note: no literature exists for a scripted top→side fallback ladder —
   next-ranked-candidate IS the published analog.
6. **Claude points + NanoSAM mask** — the brain emits point(s) on the
   part (+ box as SAM prompt fallback); NanoSAM/EfficientViT-SAM-L0
   (TensorRT) refines to a pixel mask in **~10–20 ms on this exact SoC**
   (Frontiers 10.3389/frobt.2025.1693988); proposals filtered by mask
   membership. Field consensus: masks beat boxes, VLM points beat VLM
   boxes (MOKA/RoboPoint/Molmo/FreeGrasp line). Bonus: NanoOWL
   (9.8 ms/frame) is our planned phase-2 detector backend anyway — one
   TensorRT install serves both. Two-stage object→part conditional
   queries (LERF-TOGO's ablation; NanoOWL tree mode does it natively).
7. **Late re-target** — re-read the tracked pose at standoff + early
   descent; moved >1.5 cm → re-plan the short final segment (~1–2 Hz
   replans, no servo layer). Published floor: 5 Hz re-detect + re-plan
   grasped conveyor objects (Columbia dynamic grasping, arXiv
   2103.10562) — their one trick we lack: SEED each re-plan with the
   previous solution so successive trajectories stay similar. D405
   eye-in-hand precedent at 20 Hz exists (VFAS-Grasp, arXiv 2310.18459)
   and even it goes open-loop past the fingertips; GG-CNN froze updates
   at fingertips−70 mm. Every published wrist-camera system opens the
   loop for the final segment — our standoff→final segmentation is the
   field-converged design, keep it.
8. **In-transit hold monitoring** — poll gripper aperture during
   lift/carry; width creep toward full-close = slip → set down and
   re-grasp instead of continuing (Levine 2016 closure check;
   width-vs-expected-diameter monitoring, arXiv 2401.09772; the 2F-85
   re-grasp feature does this in firmware when rFR≥1). Works in sim
   today via the knuckle joint we already read.

### Tier 3 — 1–2 weeks
9. **Guarded descent via Gen3 joint torques** — stop-on-contact and
   contact-triggered gentle placement using per-joint torque deltas
   against a descent-start baseline (~0.2 s sample), trip at
   ~1–1.5 N·m on joints 2/4, cancel the trajectory goal (controller
   holds position on cancel), then release+retreat for placement.
   Numbers: joint ripple floor ±0.2–0.3 N·m (measured, joint 7,
   PMC11644453); 3–5 N tool contact × 0.4–0.7 m levers ≈ 1.5–3 N·m on
   joints 2/4 = ~10× floor — so ~2–5 N minimum detectable contact,
   sub-newton "gentle" is NOT achievable this way. CRITICAL: never
   `tool_external_wrench_*` — ~8% error static, worse moving (Kinova's
   own words, kortex #145/#87), and FROZEN under low-level servoing,
   which is what ros2_kortex uses — per-joint effort on /joint_states
   (code-verified in ros2_kortex) is the ONLY live force signal.
   Buildable in MuJoCo NOW: `jointactuatorfrc` sensors + noise 0.1–0.2
   N·m emulate the real floor; one monitor node then works unchanged in
   sim and real. Drift-robust upgrade path: OROCOS KDL
   ChainExternalWrenchEstimator (momentum observer, born from a Gen3
   user, kortex #52). Zero torque offsets at candle pose first (Kinova
   procedure). The monitor is an ADDITIONAL stop trigger, never a gate —
   matches our stop-is-sacred invariant. Precedent for the payoff: PR2
   contact-reactive grasping 66/68 vs 60/68 open-loop (Hsiao 2010).
10. **grasp_tracking probe (2 hours)** — the SDK's tracking module ships
   aarch64 cp310 binaries on the dev branch; our license folder likely
   validates it (machine-bound, not module-bound — INFERENCE). API:
   `create_tracker(cfgs)` → per-frame
   `tracker.update(points, colors, grasp_ids)`; 7 Hz on an RTX 2060 →
   Orin-plausible. If it loads: perturbation-robust grasping and the
   handover path (their handover paper: 78% over 31 objects). Its
   dynamic value assumed servo-based execution — pair with (6), not
   with a new servo layer.

### Real-arm phase
11. **gOBJ EMULATION, not gOBJ** — the 2F-85 firmware has object/drop
    detection (gOBJ: 0x02 = stopped-by-object on close, 0x02→0x03 =
    dropped), BUT it is NOT readable through the Gen3 interconnect:
    Kortex `GripperCyclic.MotorFeedback` carries only position /
    velocity / current_motor / voltage / temperature — no flag.
    Reproduce the semantics: stalled short of target + elevated
    current_motor = object; reached full close = miss; current collapse
    or position creep while holding = drop. Vendor caveats apply:
    command close well PAST expected width (detection needs the stall),
    thin objects can grasp without detection. (PickNik's
    ros2_robotiq_gripper parses real gOBJ over direct RS-485 if we ever
    bypass the interconnect.) Sim keeps the aperture check.
12. **Joint-torque payload delta (weigh the object)** — published on our
    EXACT platform (Gen3 + 2F-85, Kružliak arXiv 2404.07344): record
    empty-gripper reference torque τ₀ at a fixed measuring pose (joint
    axis ⊥ gravity), read τ after lift at the same pose, static —
    **±6 g accuracy up to 1 kg**. Kills the pinched-but-empty false
    verdict class outright, detects in-transit drops on re-check, and
    doubles as a fill-level estimator (full water bottle ~500 g). Same
    raw-joint-torque rule as (9): never tool_external_wrench.
13. **Glassware: TransCG/DFNet depth-completion front-end** — same lab,
    5.2 MB / 17 ms / 1.6 GB VRAM, RGB+broken-depth→fixed depth ahead of
    an untouched AnyGrasp (CC BY-NC-SA; FDCT is the MIT-licensed twin at
    ~70 FPS). ASGrasp is ruled OUT by hardware: the D405 has no IR
    projector (passive color stereo, 7–50 cm), and D400s documentedly
    fail on transparent objects — depth completion is a when, not if,
    for an assistive kitchen.

### Dataset positioning (when training re-enters scope)
- **GraspClutter6D** (RA-L 2025, ~220 GB HF): 1,000 real shelf/bin/table
  scenes, 62.6% occlusion, meshes + 6D poses (⇒ replayable in MuJoCo via
  build_scene — INFERENCE); proven transfer (Contact-GraspNet 77.5→93.4%
  real success when retrained on it). The fine-tune source for the
  real-clutter phase. License: HF says CC-BY-SA-4.0 — verify.
- **GraspGen** (NVIDIA 2025, CC-BY 4.0): 57M grasps, permissive — the
  fallback lineage if AnyGrasp licensing ever bites (no 2F-85 config;
  140→85 retarget nontrivial).
- **Grasp-Anything family**: the language-labeled corpus (10M+
  instructions) if we ever train a language-conditioned filter; 6-DoF
  labels are physics-unverified — weak supervision only.

## Settled non-adoptions
LERF-TOGO/GraspSplats (minutes/scene on a 4090, unlicensed, unmaintained
— but we keep their object→part conditional-query and semantic-rerank
ideas); GraspMolmo (open weights, TaskGrasp SOTA, but 7B VLM + M2T2 and
61% end-to-end doesn't beat fixing our one weak joint — benchmark-only
candidate); language-conditioned diffusion grasp nets (gated data, no
weights, no Jetson evidence); GAP-RL and full velocity-servo closed loop
(displace cuRobo's load-bearing standoff/final/retreat segmentation —
defer until Tiers 1–3 plateau; the field agrees: every published
wrist-camera servo system opens the loop at the end anyway); learned
push-to-singulate (VPG: BSD-2 but GPU-weeks, real evidence is for PACKED
clutter — 83.3% vs 43.5% — not our singulated scenes; the published
lightweight version is Dex-Net 4.0's CONDITIONAL nudge: a scripted
3–5 cm push triggered only when every grasp on a target is
IK_FAIL/pinched, then detector re-acquire — part of their 63→80% gain;
keep as a parked primitive); camera-pointed-at-gripper success
classifiers (best published: GraspCheckNet precision 0.678 — worse than
our detector re-perception) and FoundationPose in-hand tracking
(documented to lose tracking exactly at the failure moment, V-HOP) —
scene re-perception via the existing detector beats both; tactile slip
detection (no hardware); Grasp-MPC (arXiv 2509.06201) goes on the WATCH
list — built on cuRobo's own MPPI solver with a Robotiq gripper, 60 Hz,
but code unreleased and needs a 6-day-4090 value function;
SuctionNet/AnyDexGrasp (no
suction tool / no dex hand; AnyDexGrasp's ~100-real-attempts per-effector
adapter recipe is the future-proofing pattern, and it corroborates the
0.9+ real-data score calibration).

## LLM planning & the VLA landscape (2026-07-16 second wave)

Papers: PlanGenLLMs (ACL 2025, 2025.acl-long.958 / arXiv 2502.11221);
"VLA Models for Robotics: A Review Towards Real-World Applications"
(arXiv 2510.07077, IEEE Access — co-authored by NVIDIA GEAR's co-lead);
SayCan (2204.01691); Code as Policies (2209.07753); Voyager
(2305.16291); RT-2 (2307.15818); OpenVLA (2406.09246); π0/openpi
(2410.24164); plus a dedicated VLA-on-Orin feasibility sweep. Every
report adversarially fact-checked (5 load-bearing claims each) + a
completeness critic; corrections applied below.

### The architecture verdict — validated from three independent directions
- **The planning survey**: its recommended configuration is "LLM +
  classical planner, explicit closed loop, fresh state per step,
  external verification of terminal claims" — the RAMMP brain verbatim.
  The measured cliffs it warns about are ones we architected out: plan
  quality collapses ≥~20 steps (ours: 3–8 tool calls); even o1 correctly
  refuses unsolvable problems only 27% of the time (54% confabulates a
  plan) — which is why honest failure needs a REGRESSION TEST, not just
  a design law. Executability is ~100% for us BY CONSTRUCTION (typed
  tools make inadmissible actions unrepresentable) — the survey treats
  achieving that as a research problem.
- **The VLA review**: defines our stack OUT of the VLA race (Def. I.1
  excludes skill-selection architectures), then its own safety section
  prescribes exactly us: "hybrid architectures that combine the
  generalization capabilities of learned policies with the reliability
  of model-based controllers" for human-proximate work. The E2E frontier
  (π0.5, GR00T N1, RT-H) is converging back to a two-system split that
  mirrors our brain/planner decomposition. The word "Jetson" does not
  appear in the survey; "failures are typically treated as terminal
  events" in VLAs — strictly worse than our named-cause verdicts.
  Assistive robotics coverage: near zero. Nothing deployable we are
  behind on.
- **RT-2's ablations**: web-scale co-training is the generalization
  engine (unseen 62% vs 32%; from-scratch collapses to 9%) but "the
  robot does not acquire any ability to perform new motions" — semantics
  transfer, motor skill doesn't. RAMMP spends the identical web
  knowledge at the PLANNING layer (1–3 Hz cloud latency lands on the
  decision cadence, harmless) instead of the ACTION layer (where it's
  fatal). Also corroborates the no-fine-tuning stance.
- SayCan lineage: the field's own verdict — the learned per-skill value
  function lost on economics (68k teleop demos over 11 months for 17
  objects) and closed vocabulary; the surviving pattern is
  tool-calling + verifier feedback + honest failure (Inner Monologue →
  COME-robot → us). Under forced failures: SayCan 30.8% vs Inner
  Monologue 60.4% — recovery beats pre-scoring. RAMMP is
  post-Inner-Monologue, not pre-SayCan.
- Code as Policies lineage: pure LLM-code-on-the-motion-path did NOT
  become the deployment standard; typed tool APIs + verification did.
  CaP's surviving unique value is compositional spatial arithmetic —
  see B4.

### Brain-layer adoption queue (B-items; sequence, don't union — four
new tools at once would bloat the prompt and confuse tool selection)
- **B1. Honest-refusal + perturbation eval battery** (~1 day). 10–15
  scripted IMPOSSIBLE voice tasks in sim (absent object, unreachable
  pose, nonsense) asserting honest terminal failure with named cause and
  zero confabulated success — the o1 54%-confabulation number says this
  is where frontier models break, and our honest-failure law has no
  regression coverage. Plus the Inner-Monologue adversarial protocol:
  nudge the object in MuJoCo between standoff and final, assert
  recovery-or-honest-failure (this is also the missing test for Tier 2
  item 7). Complements tools/voice_gate_test.py.
- **B2. `dry_plan(targets[])` brain tool** (0.5–1 day). SayCan's one
  surviving idea: pre-commitment feasibility across ALTERNATIVES. Wrap
  the existing `check` dry-plan path as a planner service + brain tool
  returning per-target feasible/IK_FAIL/collision-named verdicts in one
  call. Directly attacks the KNOWN OPEN cabinet_handle/shelf_edge/pills
  IK_FAILs (they become brain-visible data instead of mid-task
  surprises). Opt-in for the brain, '...' interim status while planning.
  Do NOT add a learned value function — abandoned by the field, and
  per-skill training data is the specialization we veto.
- **B3. Escalating recovery ladder on the failure-class enum** (~1 day).
  Three independent sources triangulate (VLA review's LoHoVLA pattern;
  Inner-Monologue lineage; Tier 1 item 4 + Tier 2 item 5 above): on
  repeated same-class failure, escalate one level — retry grasp →
  re-scan/re-box → re-plan task → report. Highest-confidence item of the
  wave; mostly prompt+plumbing in brain.py.
- **B4. `compute()` sandboxed geometry tool** (1–2 days). The safe
  middle ground of Code as Policies: the brain writes a short NumPy
  snippet over NAMED world-snapshot poses only (AST whitelist —
  arithmetic/indexing/np calls; no imports, no I/O; time+memory cap);
  output is poses/offsets that feed the EXISTING collision-checked tools.
  Generated code never touches an actuator or topic — STOP path and
  standoff segmentation untouched. Unlocks relational placement ("left
  of the plate", "between the mugs") without a new tool per behavior,
  and is a prerequisite for the place-on/handover tools in the Next
  list. Echo computed values back with the fresh snapshot (CodeAct
  lesson); clamp outputs to the workspace box (hallucinated math becomes
  a named refusal). Skip hierarchical code-gen — 2022 Codex needed it,
  a 2026 frontier model doesn't at this scope.
- **B5. Standing instrumentation bundle** (hours, zero risk). Per-task
  #LLM-calls / tokens / decision-latency / $ logging (the survey's
  efficiency criterion — makes haiku↔sonnet choices measured, not
  anecdotal; nobody has computed our $/task); `tegrastats` co-tenancy
  audit of the running stack (prerequisite #1 for ANY future VLA
  experiment); LeRobot-format episode recording as a side effect of
  normal operation once the real D405 arrives (an eventual fine-tune
  dataset falls out for free instead of a teleop campaign).
- **Deferred: Voyager-style recipe library** (was: adopt, 2–4 days;
  critic overturned to DEFER). Voyager's own ablation says the skill
  library's value is compounding/long-horizon ("plateaus in later
  stages" without it); our tasks are 3–8 calls and the projected 3–8 s
  saving is the weakest win of the wave, against retrieval-near-miss
  risk and prompt surface. The design is recorded (JSON recipes,
  symbolic references never coordinates, store only on plain-status
  success, retrieved as SUGGESTIONS under fresh snapshots): re-open when
  multi-object tasks ("make me a snack") exist. Physical-arm precedent
  exists (BOSS on a Kinova Jaco: only method with nonzero success on
  length-4 tasks; ViReSkill UR5 75% vs 30%) — the trigger is task
  length, not doubt about the pattern.

### VLA go/no-go: SKIP this hardware generation (measured, not vibes)
Orin-measured inference (model-alone, before our stack's contention):
OpenVLA 7B INT4 = 374.7 ms ≈ 2.67 Hz at 4.0 GB (QAIL, arXiv 2412.01034
Table 10); GR00T N1.7 = 2.9 Hz PyTorch / 4.6 Hz TRT-DiT (TensorRT on
Orin cannot compile the LLM backbone — NVIDIA's own deployment README);
LiteVLA-Edge 256M = 6.64 Hz; π0/openpi has NO working Orin path (issue
#386 unresolved; official edge = Jetson Thor, ~94 ms per 10-step
trajectory). Blocking facts: (1) memory — only INT4-OpenVLA/SmolVLA-class
coexist with cuRobo+AnyGrasp+detectors on unified LPDDR5 that already
OOM-killed one node; (2) zero published VLA fine-tune on a Kinova Gen3
— OXE's Kinova prior is 1,085 Jaco-2 episodes + 192 displacement-only
episodes; (3) data — 50–300 real teleop demos/task, no off-the-shelf
Gen3 teleop rig (TidyBot++ phone-WebXR is the cheapest port), and
sim-only training measures **0–5.4% real success** without heavy domain
randomization (arXiv 2603.22876, 10k real trials) — our MuJoCo episodes
can never be the whole diet; (4) safety — a VLA emits raw action chunks
executed open-loop 0.5–0.8 s, bypassing cuRobo collision checking, the
standoff segmentation, and the verified world entirely. Corrections
applied per fact-check: π0 weights ARE open (Apache-2.0 + Gemma terms,
openpi, since Feb 2025) — the barrier is the Orin path and data, not
the license; DSRL's 20%→~100% @10K is a LIBERO SIM figure (real: 20%→90%
in <50 episodes).
- **The field validated our topology**: LiLo-VLA (classical planner
  transports, local policy takes the contact-rich final segment) is our
  standoff→final split with a VLA in the last cm; CBF-filter papers run
  learned policies through classical geometric vetoes. A runtime "VLA
  chunk → cuRobo feasibility veto" gate appears UNPUBLISHED — if we ever
  do it, that's novelty, not an import.
- **Cheapest credible taste** (2–4 days, zero robot risk): tegrastats
  audit → SmolVLA-450M inference benchmark NEXT TO the live stack (the
  number nobody has published) → LeRobot episodes from the MuJoCo
  bringup → ~$10 rented-4090 fine-tune → offline/sim-rollout eval only.
  Expected outcome per the transfer evidence: works in sim, wouldn't
  transfer — but it prices the pipeline and produces the co-tenancy
  numbers.
- **Re-evaluation triggers**: real arm + a Gen3 teleop path; OR a
  sub-1B VLA publishing ≥10 Hz measured on AGX Orin; OR Thor-class
  hardware in the budget (π0 = 19 Hz on Thor); OR an openpi-class model
  at ≤25 demos/task with a chunk interface a collision checker can gate.

### Gaps the critic named (open work)
- **Claude's own pointing/boxing accuracy is verified nowhere** — the
  part-grasp pipeline and the NanoSAM upgrade (Tier 2 item 6) rest on
  "VLM points beat boxes" measured on OTHER models. A half-day in-sim
  battery (known ground truth vs brain-emitted points on look images)
  closes it. Do this BEFORE building the NanoSAM stage.
- Gemini Robotics-ER 1.5/1.6 (pointing, grasp prediction, 3D boxes,
  tool calling — the one commercial product doing the brain's exact
  job) was not primary-sourced; worth a dedicated look as
  competitor/benchmark for the Claude brain.
- Brain availability under network loss is unaddressed (capability, not
  safety — STOP is client-side); no local-VLM-fallback feasibility fact
  exists yet.

## Where the lineage is heading (context)
Grasp detection is treated as solved infrastructure by its authors: the
SDK gets maintenance (steering, aarch64) while the lab's research energy
moved to dexterous hands, human-data engines (RH20T, AirExo-2, DEXOP)
and imitation policies; H-S. Fang → MIT→UMD faculty; Lu commercializes
via Noematrix. INFERENCE: no AnyGrasp v2 or open weights coming;
language-conditioned grasping from this line will arrive as
VLM + steering composition — the pattern we already run. The ecosystem's
universally cited pain is the closed license — our backend-agnostic
proposer seam stays the insurance policy.

## Sources
See the five research transcripts (2026-07-16) — every entry above
carries its arXiv/repo citation inline; load-bearing items were
primary-source fetched: anygrasp_sdk USAGE.md/demo.py (steering + tracker
API), OK-Robot HTML full text (filter recipe, S − θ⁴/10, staged
approach, failure taxonomy), COME-robot HTML (platform + recovery
stats), Frontiers Orin benchmark (NanoOWL/NanoSAM latencies), kortex
issues #145/#87/#52/#222 (wrench accuracy + low-level freeze + momentum
observer), ros2_kortex hardware_interface.cpp (effort on /joint_states,
code-verified), Kružliak 2404.07344 (±6 g payload on Gen3+2F-85),
Robotiq 2F-85 manual + Kortex MotorFeedback.md (gOBJ vs interconnect),
GG-CNN IJRR full text (rates, 70 mm freeze), VFAS-Grasp (D405 20 Hz
precedent), Columbia dynamic grasping README (5 Hz + plan seeding),
Dex-Net 4.0 Science Robotics PDF (failure memory 63→80%), RUM/FAR/
ReplanVLM/AHA/Inner-Monologue/Levine/QT-Opt/Calandra papers, TransCG
repo (DFNet numbers), GraspClutter6D HF card, MuJoCo sensor XML
reference (jointactuatorfrc + noise).

Second wave (LLM planning / VLA, all adversarially fact-checked):
PlanGenLLMs ACL PDF + arXiv v1/v3 (taxonomy, o1 27%-refusal via
Valmeekam 2409.13373, TravelPlanner 0.6%), VLA review full HTML
(Def. I.1, safety-section quote, Jetson absence grepped), SayCan +
Inner Monologue ar5iv full texts (30.8→60.4 under disturbance, 68k-demo
cost), CaP ar5iv + CodeAct 2402.01030 + neuro-symbolic CaP 2510.21302,
Voyager ar5iv + BOSS/LRLL/ViReSkill, RT-2 ar5iv + OXE + FAST, OpenVLA
ar5iv + OFT tables + openvla repo (license split: MIT code, Llama
Community weights), openpi README + issue #386 + Jetson AI Lab Thor
tutorial, QAIL 2412.01034 Table 10 (Orin INT4 374.7 ms), Isaac-GR00T
deployment README (Orin 342.8/216.5 ms), LiteVLA-Edge 2603.03380,
sim-to-real controlled study 2603.22876 (0–5.4% clean-sim transfer),
TFDS catalog (jaco_play, uiuc_d3field).
