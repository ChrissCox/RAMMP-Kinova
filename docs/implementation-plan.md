# Implementation plan — research-adoption upgrades

Date: 2026-07-17. Working plan derived from docs/research-adoption-memo.md,
with every item grounded against the current code (seams verified at main
@ dd2b47c).

**2026-07-17 amendment — GraspGen-X replaced AnyGrasp** (user decision,
authored same day, Jetson validation pending). Consequences for this plan:
A1 approach_steering is CLOSED (the AnyGrasp SDK hook no longer exists;
GraspGen's `graspmoe` mode natively provides discriminator-scored
top-down candidates); A2's region-membership half is INHERENT now
(GraspGen takes the segmented crop as its input cloud — only the θ⁴
orientation penalty remains of A2); Phase 1's probe (a) is replaced by
the GraspGen bring-up session (install script, smoke test, score-floor
calibration for `proposal_min_score`/`proposal_min_score_part`); all
verdict/retry/eval items are backend-independent and unchanged.

Item keys:

- A1 approach_steering · A2 OK-Robot ranking + region membership ·
  A3 points+NanoSAM (cut line) · A4 brain pointing-accuracy battery
- B1 vision lift-verify · B2 failure-class tags + bounded never-identical
  retry + brain recovery ladder · B3 in-transit slip monitor ·
  B4 late re-target at standoff
- C1 dry_plan brain tool · C2 compute() sandboxed geometry tool ·
  C3 eval (tools/eval.py — do-cases + refuse-cases, THE improvement
  gauge) · C4 usage instrumentation
- D1 guarded descent groundwork (cut line) · D2 grasp_tracking probe
  (opportunistic)

**Headline properties:** exactly ONE colcon rebuild in the whole plan
(Phase 1's `set_prop_pose` service in mujoco_sim — it unlocks eval
jitter, physical sabotage cases, and the future perturbation matrix);
everything else is a symlinked .py edit or a tools/ script (C2's
evaluator stays INLINE in brain.py; A4's crop helper lands as an EDIT to
`rammp_perception/geometry.py`). Every phase ends with a bringup restart
via `tools/launch_stack.zsh` only. Exactly ONE brain prompt revision, in
Phase 5.

---

## 1. Phased sequence

### Phase 1 — Instruments before behavior (C3 eval + C4 + Jetson probe batch)
**Why now:** nothing in Phases 2–5 can be judged without a baseline.
C3 (eval) is THE improvement gauge and the honesty tripwire in one tool;
it must baseline the stack before any prompt or verdict text changes.
C4 is hours, zero protocol surface, and prices every later phase
(haiku/sonnet choice becomes measured). The probe batch retires the
three unknowns that redesign later phases.

**Items:**
- C3 `tools/eval.py` — one runner, one report, two case families:
  **do-cases** (grasp each of the 8 props under seeded pose jitter;
  scored success_once/success_at_end, stage partial credit
  reach/grasp/lift/release/home, task time, named failure cause; success
  rates reported as raw counts + Wilson 95% CI) and **refuse-cases**
  (absent object, unreachable pose, garbled ASR, mid-grasp sabotage via
  set_prop_pose; scored: terminal status names the cause, zero
  confabulated success, no question marks). Modes: `--quick` (~5 seeds ×
  8 props + all refuse-cases, one sitting — the per-phase gate) and full
  (25 seeds/prop — the milestone number). JSON + markdown report per
  run, kept under docs/ untracked.
- `set_prop_pose` service in mujoco_sim (THE one rebuild): teleport a
  named prop in the running sim — unlocks eval jitter, physical
  sabotage, and the future perturbation matrix.
- C4 usage/latency/$ logging + `tools/tegrastats_audit.py`. Price table
  hardcoded with dated comment; unknown model → cost `n/a`.
- Jetson probe batch (~1 h): (a) `~/anygrasp_sdk` USAGE.md —
  `approach_steering` vector/frame/thresh semantics (gates A1),
  (b) `inspect.signature(MotionGen.plan_single)` + `MotionGenPlanConfig`
  fields — seed param present? (expected NO → B4 seeding closes N/A),
  (c) `ros2 topic echo /joint_states --once` — effort populated? (D1
  go/no-go, filed for the cut line).

**Files:** new `tools/eval.py`, `tools/tegrastats_audit.py`;
`mujoco_sim` (set_prop_pose service — new interface, rebuild);
`brain.py` (`_loop` stream wrap, `_run_task` counters — different
functions than Phase 5 touches).

**Rebuild/restart:** ONE colcon rebuild (mujoco_sim) + restart.
**Windows vs Jetson:** author 100% on Windows now (eval runner with
GotoClient reuse from `nl_command.py`, service code, C4 code). Jetson:
rebuild, probe batch, one task run for the `[usage]` line, quick-eval
baseline + one FULL eval baseline run (~1 day).
**Gate:** `[usage]` line with nonzero tokens+cost on a real task; FULL
eval baseline recorded (every refuse-case PASS or a NAMED baseline
failure; do-case success rates + CIs written down — later phases' bar is
"no new refuse-failures, do-case success within/above baseline CI");
three probe answers written down; `voice_gate_test.py` pass.

### Phase 2 — Verdict structure + the first false-positive killer (B2-tags + B1)
**Why now:** B2's tag constants are a declared dependency of B1, B3, and
B4's abort branch — they must exist first. B1 attacks the field-proven
worst verdict class (the mug "held read as closed-on-air" lesson) and is
the highest value-per-line item in the wave. Both live in `_handle_grasp`
— one edit wave.

**Items:** B2 tags-only (`[no_grasp]/[slip]/[mis_position]/[unreachable]/
[wrong_object]` prepended to FAILURE statuses only — NEVER to
`GRASPED`/`Released`/`Looking at`, which brain.py prefix-matches); B1
lift-verify decision table (fresh-tick TIMESTAMP keyed, never dict
freshness — stale pose is a non-signal per the "unseen props keep
last-known pose" invariant; z disambiguates re-detected-in-hand); new
shared instrument `tools/grasp_fault_test.py` cases 1–2 (phantom
detection injection on the detector topic).

**Safety note:** tags land three phases before the brain ladder — safe,
because "[no_grasp] Grasp ... MISSED" still contains MISSED, which the
current prompt already handles.

**Files:** `planner_node.py` (`_handle_grasp` ~1095–1157, verdict
strings, new `_lift_verify` helper, `_last_grasp_fail` field skeleton),
new `tools/grasp_fault_test.py`.
**Rebuild/restart:** none / restart.
**Windows vs Jetson:** author 100% now (~1 day). Jetson: fault-injection
session ~2 h.
**Gate:** grasp_fault_test cases 1–2 pass (phantom → `[no_grasp]`;
caged-but-on-table → B1 MISSED overriding the gripper); bottle AND mug
grasp-lift-release-home regression clean; voice_gate_test pass;
quick eval: no new refuse-failures, do-cases within baseline CI.

### Phase 3 — Close the loop in transit (B3 + B4-refresh) + A4 in parallel
**Why now:** B3 and B4's standoff refresh are the two remaining
`_handle_grasp`/wait-loop edits — batching them finishes the grasp-flow
churn before the ranking work touches `_anygrasp_select`. A4 is
independent, cheap, and its numbers set the A3 cut-line trigger — the
earlier it runs, the sooner that decision is data.

**Items:** B3 (flag-set in `_joint_state_cb` under `_state_lock`, action
in `_wait_until_stationary`/`_handle_command` — never in the callback;
gate strictly on `_held`; never publish a planner command on the slip
path); B4 part 1 only (standoff-arrival goal refresh with the explicit
table: ≤1.5 cm ignore / 1.5–4 cm retarget goal only / >4 cm abort
`[mis_position]`; final ≤6 cm unconditionally open-loop;
`descent_retarget` monitor stays default-OFF below the cut; seeding
closed N/A per Phase 1 probe); A4 (`tools/brain_look_test.py`, 1280×960
render parity, segmentation-mode ground truth, crop math refactored into
`geometry.py` as an EDIT shared with the proposer).

**Files:** `planner_node.py` (`_joint_state_cb`,
`_wait_until_stationary`, `_handle_grasp` post-standoff block),
`rammp_perception/geometry.py`, `grasp_proposer.py` (import the shared
crop), new `tools/brain_look_test.py`, grasp_fault_test cases 3–4.
**Rebuild/restart:** none / restart (proposer picks up geometry.py).
**Windows vs Jetson:** all authorable now (~1.5 days). Jetson: case 3
(forced gripper open mid-carry → `[slip]` before terminal), case 4
(±2.5 cm phantom shift between standoff and descent → "late re-target"
log + clean grasp), `check_traj` live during one retargeted grasp, A4
run (offline, no bringup, ~1 h).
**Gate:** grasp_fault_test 3–4 pass; check_traj clean;
`perception_test.py` still ≤40 mm (crop refactor didn't disturb detector
geometry); goto `check` regression; A4 numbers recorded (bar: hit-rate
≥0.8, median IoU ≥0.3, N≥10 per part) — this DECIDES the A3 trigger, it
does not block Phase 4.

### Phase 4 — Grasp quality: ranking, membership, never-identical retry, steering (A2 + B2-retry + A1)
**Why now:** these three co-edit `_anygrasp_select` and the proposer's
`_request_cb` — one wave each avoids double churn. Order within the
phase: A2 (soft penalty) before A1 (generation bias) so orientation
preference isn't double-counted; retry's signature-exclusion rides the
same `_anygrasp_select` edit.

**Items:** A2 (θ⁴ penalty with `anygrasp_orient_k` as a declared param,
sim default ~0.01, floor stays on RAW score; proposer-side region
membership BEFORE the top-10 truncation, written mask-or-box generic —
this is A3's future insertion point; `rejected_off_region` in the reply
JSON); B2-retry (never-identical by grasp SIGNATURE not index,
geometric-path z/yaw perturb with palm-cap respected, bound 2, then
`[no_grasp] ... two distinct grasps failed`; every perturb re-runs
`_goal_feasible`); A1 (per-request `approach_base` field, frame
conversion `cam_R.T @ approach_base` per source per the Phase 1
USAGE.md answer; applied ONLY to whole-object requests with a WIDE
thresh — never to `part_box`/`region_base` requests, which exist to find
horizontal handle approaches).

**Files:** `planner_node.py` (`_anygrasp_select`, `_handle_grasp` retry
head, `_request_proposals` emitters), `grasp_proposer.py`
(`_request_cb`), new `tools/anygrasp_select_test.py` (pure-function
refactor, canned JSON, runs on Windows).
**Rebuild/restart:** none (all params via `declare_parameter`) / restart.
**Windows vs Jetson:** author + unit tests now (~1.5 days). Jetson: k
tuning, per-gate rejection counters before/after, retry forcing,
USAGE.md-informed steering trial (~1 day).
**Gate:** anygrasp_select_test all-pass on Windows; live: rejection
counters show `from_below`/`ik` drop without the pool EVER emptying
(geometric fallback must remain reachable); mug part-path survivor still
surfaces within the 12-probe cap; forced double-failure logs two
DISTINCT signatures then the bounded refusal; quick eval clean — and
this is the phase where do-case success SHOULD move up vs baseline
(that's the point of A2/A1); if it doesn't, the constants get retuned
before the phase closes.

### Phase 5 — THE one brain revision (B2-ladder + C1 + C2), gated by the full battery
**Why now:** last, because every backend it references now exists and is
independently verified: tags (P2), planner auto-vary retry (P4),
dry-plan grammar (testable via `goto --direct "dry: ..."` before any
prompt change), compute evaluator (Windows unit-tested before wiring).
This is the ONLY edit to SYSTEM/TOOLS in the whole plan.

**Items:** ONE prompt/TOOLS revision containing: (a) the
class-conditioned ladder replacing the two existing retry lines
(`[no_grasp]/[slip]` → retry once → look+re-box → honest report;
`[unreachable]` → alternative or report, no blind retry;
`[mis_position]` → look first; max 2/object); (b) `dry_plan` tool + one
prompt line (command-grammar `dry:` path in `_run_check(candidates,
start_from_current)`, interim `'...dry_plan k/n'`, ONE terminal line,
`_stop_requested` polled between candidates); (c) `compute` tool + one
prompt line (AST whitelist, subprocess + `RLIMIT_AS` sandbox — NOT
`signal.SIGALRM`, which is dead in the worker thread; `_world_dict()`
refactor so prompt text and compute bindings share one snapshot builder;
outputs clamped to the workspace box AND the clamp named).

**Memo-warning reconciliation:** the revision is one TEXT change, but
tool ACTIVATION is sequenced via brain params (`enable_dry_plan`,
`enable_compute`, each gating both the TOOLS entry and its prompt line):
run C3 with neither → dry_plan only → both, re-running the battery at
each step. One revision, sequenced exposure, and a C3 regression bisects
any confusion to a single tool. If compute's activation step regresses
tool selection, `enable_compute` stays false (evaluator kept, tool
parked — becomes a cut-line item with trigger "place-on/handover work
begins").

**Files:** `brain.py` (SYSTEM, TOOLS, `_execute`, `_world_dict`),
`planner_node.py` (`_run_check` refactor + `dry:` grammar in
`_handle_command`), new `tools/compute_eval_test.py`, C3 case additions
(dry_plan-vs-check equivalence, relational-placement e2e, compute
failure-mode table).
**Rebuild/restart:** none / restart.
**Windows vs Jetson:** evaluator + unit tests + refactors + prompt text
fully authorable now (~2 days; compute_eval_test RUNS on Windows).
Jetson: dry-plan timing measurement, `check` byte-equivalence, staged C3
runs, one relational-placement task (~1 day).
**Gate:** compute_eval_test all-pass (Windows); `goto check` verdicts
byte-equivalent pre/post refactor AND matching `--direct "dry:"` per
target; full eval at each activation step (refuse-cases all-pass —
nonsense-ASR still zero `?`, no confabulated success); voice_gate_test;
`[usage]` per-task cost recorded for the sonnet brain; close the plan
with a FULL eval run vs the Phase 1 baseline — the before/after number. Natural follow-on (not in
scope): use dry_plan to retune the KNOWN OPEN
cabinet_handle/shelf_edge/pills targets.

---

## 2. Merge-churn groupings
- **G1 planner grasp flow** (`_handle_grasp`/`_joint_state_cb`/wait
  loops): B2-tags+B1 (P2), then B3+B4-refresh (P3) — each function
  edited once per phase, retry head lands with the P4 select wave.
  `_anygrasp_select` is touched exactly ONCE (P4: penalty + membership
  interface + signature exclusion together).
- **G2 brain**: ONE prompt/TOOLS revision (P5) covering ladder +
  dry_plan + compute (+ A3's point-emission text explicitly EXCLUDED —
  it rides a future revision when A3 un-parks). C4's P1 edit touches
  only `_loop`'s stream wrapper — disjoint functions, no churn.
- **G3 proposer `_request_cb`**: A2-membership + A1-steering together in
  P4; A3's mask filter later rebases on the mask-or-box membership
  interface built there.
- **G4 shared instruments**: `tools/grasp_fault_test.py` accretes cases
  across P2–P4 (one file, one injection mechanism); `geometry.py` crop
  helper shared by proposer + A4 (edit, not new file).
- **G5 rebuild consolidation**: ONE rebuild in the plan (Phase 1,
  mujoco_sim set_prop_pose). D1 is the only remaining rebuild-forcing
  item — if it un-parks, it gets its own colcon + build_scene window.

## 3. Cut line — NOT yet, with triggers
- **A3 points+NanoSAM**: trigger = A4 battery meets bar AND (real D405
  arrives OR wrist-retry part proposals exceed ~0.1 in practice). The
  sim score-suppression wall means the payoff is real-sensor; install
  NanoSAM in the same TensorRT session as the planned NanoOWL backend
  (one install, two consumers). Its prompt change joins the NEXT brain
  revision.
- **D1 guarded descent** (beyond the P1 effort probe): trigger =
  `/joint_states.effort` confirmed populated AND Phases 1–5 landed
  (memo: Tier 3 after Tier 1/2). A negative probe answer redesigns it
  entirely — that answer is already bought in P1.
- ~~C3 Tier B physics nudge~~ — CLOSED into Phase 1: `set_prop_pose`
  IS the physics-nudge mechanism; sabotage cases are physical from day
  one, no spoofing tier needed.
- **B4 descent_retarget monitor + plan seeding**: seeding = CLOSED N/A
  the day the P1 probe confirms `plan_single` has no seed param
  (expected); the in-descent monitor's trigger = field evidence of
  misses with fresh ticks showing movement AFTER the standoff refresh.
- **D2 grasp_tracking probe**: scripts authorable any time; trigger =
  any free 2 h Jetson slot after P2, or mandatory before handover work.
  Zero risk, future-facing value — opportunistic, not scheduled.
- **Voyager recipe library**: stays deferred per the memo; trigger =
  multi-object tasks exist.

## 4. Total effort
Windows authoring: ~6.5–8 days (starts immediately, nothing
Windows-blocked; Phase 1 grew by the eval runner + service). Jetson
testing: ~4.5–5.5 days in phase-batched sessions (arm/stack contention
is the constraint — every phase's Jetson work is one sitting).
**Total ~11–13 working days, ~2.5 calendar weeks.**
Cut-line items excluded (A3 ~2–3 d + install; D1 ~3–4 d; Tier B ~1 d;
D2 ~2 h).

## 5. Top 3 risks and mitigations
1. **Terminal-protocol regression.** Five phases edit status strings
   that brain.py prefix-matches and goto's `'...'` protocol depends on;
   one wrong prepend and tasks stop terminating. Mitigation: tags on
   failure statuses ONLY, success prefixes frozen by rule;
   voice_gate_test + a C3 no-new-failures run is a gate of EVERY phase;
   grasp_fault_test asserts the success-prefix strings byte-unchanged.
2. **Sim/real score-scale poisoning + emptied pools.** Every orientation
   constant (A2's k, A1's thresh, floors) is tuned against sim scores
   0.01–0.03; real-D405 scores run 0.9+, and A1+A2 together can
   double-count orientation and empty the pool onto the fallback
   silently. Mitigation: every constant is a declared ROS param with a
   sim default and a written real-arm retune checklist; A2-soft before
   A1-wide, steering never on part requests; per-gate rejection counters
   logged and compared as a gate; geometric fallback path untouched.
3. **Branch interaction in the closed grasp loop.** B1+B3+B4+retry
   compound inside one function: duplicate slip statuses, retarget
   consuming the retry budget, monitors firing during the close,
   callback/command-thread races, anything touching the stop path.
   Mitigation: hard rules — callbacks set flags only, actions run in the
   command thread; everything gates on `_held`; stop check FIRST in
   every poll loop; a retarget is by definition not a retry; first-writer
   -wins via `_held` clear; the interactions each have a named
   grasp_fault_test case, phases land small with the bottle+mug
   regression between every one, and check_traj rides one full
   retargeted+retried grasp live.
