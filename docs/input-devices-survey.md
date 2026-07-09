# Input Devices for Wheelchair Users & Assistive Arms — Cited Census

**Purpose.** Stage 1 of RAMMP input-device selection: enumerate every control interface
used by power-wheelchair users to drive the chair and to operate assistive devices —
especially wheelchair-mounted robotic arms — across the full ability spectrum (full hand
function down to eye/brain-only). This document is the raw material for a later ranking
pass; it does **not** yet pick a device.

**Method & confidence.** Compiled from four deep-research passes (fan-out web search →
source fetch → per-claim 3-vote adversarial verification → synthesis). ~40 claims survived
verification; every quantitative figure below is attributed to its primary source. A
recurring caveat runs through the whole field and is flagged per row: **most performance
numbers come from able-bodied participants**, and the one head-to-head study with SCI
participants shows able-bodied results do *not* transfer cleanly. Where evidence involves
actual people with motor disabilities (PWMD), it is marked **[PWMD]**.

---

## TL;DR for RAMMP

1. **The proportional joystick dominates real-world use** — 9 of 11 long-term Kinova JACO
   owners drive the *arm* with one, 8 of 11 drive the *chair* with one, even with very
   limited hand function. But this is partly selection bias: **4 of 9 SCI participants in a
   controlled study could not use a joystick at all.**
2. **The core problem is dimensionality mismatch.** A 1–3 DoF input driving a 6–7 DoF arm
   forces mode switching, which eats **~17.4% of task time** on the stock Kinova joystick —
   and only **2 of 11** long-term owners are satisfied with their control setup.
3. **The remedy the field endorses is minimal input + autonomy, not richer steering.** The
   canonical clinician survey concluded exactly this in 2000; the strongest template is a
   C5-SCI user driving a Kinova arm to **100% task success with just a 1-D speed signal + a
   binary switch** under shared autonomy.
4. **RAMMP's architecture matches the commercial precedent.** Kinova's own intended control
   path is pass-through from *whatever* wheelchair input the user has (joystick, head array,
   sip-and-puff) — the same input-agnostic idea as RAMMP's `/api/twist` channel.
5. **The AI layer must be interface-aware.** The same intent-inference algorithm produces
   significantly different error rates per input device; autonomy that doesn't wait for a
   noisy signal (sip-and-puff, gaze) to settle will fight capable users and get abandoned.
6. **Users prefer agency.** Across studies, people accept *full* autonomy less than manual
   or shared control — they trade speed for control. Design autonomy as power steering, not
   a chauffeur.

---

## Summary table

Bandwidth = usable control signal. "DoF" = continuous degrees of freedom; "cmds" = discrete
command vocabulary. Min ability = minimum residual function to operate at all.

| Modality | Min residual ability | Bandwidth | Maturity | Arm-control evidence |
|---|---|---|---|---|
| **Proportional joystick** (std/compact/chin/foot) | Some limb/chin/foot control; ~C5 for standard | 2–3 DoF continuous, proportional | Commercial (8 g–800 g force range) | **[PWMD]** dominant real-world arm input (9/11 JACO owners) |
| **Switched / non-proportional drive** | Any repeatable gross movement | 4–8 discrete directions, latched/momentary | Commercial | via mode-scanning, high burden |
| **Head array** (proximity switch) | Head control + neck support | ~1–2 D discrete | Commercial (ASL, Stealth) | **[PWMD]** 2/11 JACO owners; Argall studies |
| **Head-tracking / IMU** (AMiCUS, Munevo) | Head orientation control | 2–3 DoF continuous proportional | Commercial (drive) / lab (arm) | **[PWMD]** AMiCUS: 6 tetraplegics, 100% pick-place on UR5 |
| **Sip-and-puff** | Breath control | ~1-D, 2–4 discrete cmds | Commercial | **[PWMD]** worst of the wheelchair-input tier; 9 modes for a JACO |
| **Tongue** (Tongue Drive System) | Intact tongue motor control (spared by high SCI) | up to ~6 discrete cmds | Lab (never fully commercialized) | **[PWMD]** n=1 C1–C2 controlled all 14 JACO signals |
| **Intraoral joystick** (IntegraMouse+, Jouse) | Lip/tongue control | 2-D continuous + clicks | Commercial | 5-DoF exoskeleton (lab) |
| **Eye-gaze** (Tobii PCEye class) | Intact eye movement (works locked-in) | discrete dwell-select | Commercial tracker; MyEccPupil drives JACO | 6-DoF arm ADLs (able-bodied); thin PWMD base |
| **Voice / speech** | Intelligible speech (excludes anarthric) | discrete task-level cmds | Commercial ASR + emerging LLM | LLM-era: VoicePilot on Obi feeder; discrete only |
| **sEMG** (NeuroNode class) | Any voluntary muscle (single motor unit for ALS) | discrete triggers / low-D proportional | Commercial | ⚠️ evidence not verified this survey (gap) |
| **Body-machine interface** (shoulder IMU) | Residual shoulder motion (C5–C6) | 2-D continuous proportional | Lab prototype | **[PWMD]** n=1 C5 → 100% on Kinova MICO (shared auton.) |
| **EEG BCI — non-invasive** | Brain only | 2–6 discrete cmds, ~60–95% acc | Lab-bound | reach+grasp only by chaining low-D controls (able-bodied) |
| **BCI — invasive intracortical** | Brain only (motor cortex intact) | up to 10 DoF | Research (N=1–2, surgical) | **[PWMD]** BrainGate drink task; 10-DoF; sensory feedback |

---

## Taxonomy by modality

### 1. Proportional joysticks and variants
**How / min ability.** Speed and direction scale continuously with deflection. Standard
joystick is the most intuitive control for SCI at ~C5 and below; variants (compact, chin,
foot, finger) trade force/throw to reach weaker users. Commercial operating forces span
**mo-vis Micro ~8 g / 3.3 mm throw → Permobil Compact 220 g / 28 mm → Heavy Duty 800 g /
46 mm** (Permobil hub; mo-vis specs).
**Arm control. [PWMD]** In a RESNA 2023 study of 11 long-term Kinova JACO/JACO2 owners
(avg 3.2 yr ownership; ALS, MD, CP; very limited upper-limb function), **9/11 controlled
the arm with a proportional joystick**; for driving, 8/11 used a standard hand joystick,
the rest a foot joystick, chin joystick, and head array. The stock JACO interface is a
3-axis joystick partitioning the arm into translation/wrist/finger modes cycled by buttons
with LED feedback. In a 2011 study of 31 wheelchair users, most completed ADL tasks with
the JACO joystick on the first attempt (learnable), though authors were Kinova-affiliated.
**Caveat.** Dominance is partly prescription selection — see §Minimum-ability gating.

### 2. Switched / non-proportional drive controls
Binary on/off, typically 4–8 discrete directions (one switch per direction), momentary
(active while held) or latched (until re-triggered / timed). Indicated when the user lacks
the strength, ROM, or coordination for proportional control. Speed is preset, not graded.
For arm control this forces scanning through a mode list — the high-burden path.

### 3. Head-based interfaces
**Head array (proximity/switch).** Pads in the headrest sense head position; discrete,
non-proportional (~1–2 D). Commercial (ASL, Stealth Products). **[PWMD]** 2/11 JACO owners
used head input (one an ATOM electronic head array). Needs head control **and** neck
support — one SCI participant in the Argall study couldn't use it for lack of neck support.
**Head-tracking / IMU (proportional).** A head-worn IMU/AHRS gives continuous proportional
control from head pitch/roll/yaw. Commercial for driving (Munevo Drive on smart-glasses).
**[PWMD] AMiCUS (Sensors 2019)** gave 6 tetraplegic users (C0–C4) real-time proportional
control of a UR5 arm + Robotiq gripper from head motion alone — **100% completion on a
pick-and-place task**, 2–3 DoF simultaneous with a nod-gesture switching among 4 control
groups. Directly relevant: same class of arm+gripper as RAMMP.

### 4. Sip-and-puff
Positive "puffs" / negative "sips" through a straw; extremely low-dimensionality (~1-D,
expanded to ~2–4 discrete commands via hard/soft or long/short thresholds). Cost-effective,
moves with the user, but limits speech while driving, causes respiratory fatigue, needs
hygiene upkeep. **Arm control:** commercial sip-and-puff runs a JACO by auto-scrolling **9
discrete modes** with an exhale to latch — characterized in the literature as "tedious, if
not impossible" and the paradigm case for pairing with autonomy. In the Argall head-to-head
it was the **worst** interface: not just slowest but *noisiest* — the largest gap between
first response and settled response (more vacillation).

### 5. Tongue interfaces
**Tongue Drive System (Georgia Tech / Ghovanloo).** A magnetic tongue tracker; minimum
ability is intact tongue motor control, which hypoglossal innervation typically **spares
after high cervical SCI**. **[PWMD]** In Sci Transl Med 2013, 11 people with tetraplegia
(C6 or above) drove a wheelchair and used a computer **up to 3× faster than sip-and-puff at
equal accuracy**, and TDS kept improving with practice while SnP (their daily habit)
plateaued. **[PWMD] Arm control:** a person with **C1–C2** tetraplegia used a wireless
intraoral tongue interface to command **all 14 control signals of a JACO/Kinova arm** in 3D
(JNER 2017) — first such demo, n=1, ~50% pick-up success (capability, not reliability).
Maturity: TDS was never fully commercialized. **Intraoral joysticks** (IntegraMouse Plus,
Jouse) are commercial, lip/tongue-operated 2-D continuous + click; one lab intraoral device
gave full continuous control of a 5-DoF exoskeleton with no arm function required.

### 6. Eye-gaze
**How / min ability.** Screen-based gaze tracker; needs only intact eye movement — works
for locked-in / late ALS. Selection is dwell-based (discrete), so the **Midas-touch
problem** (unintended gaze triggering) is the core reliability issue, mitigated by
~200–700 ms dwell windows. Core constraint: mapping **2-D gaze to 3-D robot motion**,
usually solved with an on-screen GUI of Cartesian/task buttons (which also lets bedridden
users operate without seeing the robot's view).
**Arm control.** A JNER 2021 study drove a wheelchair-mounted 6-DoF arm (xArm 6 on a
Permobil) for ADLs via a Tobii PCEye5 and dwell-selected GUI — **100% task success** — but
on **10 able-bodied** participants, zero PWMD. A 2024 scoping review found only **6 of 39**
gaze-arm studies included disabled participants. Maturity: trackers are commercial (Tobii
Dynavox class); **MyEccPupil (HomeBrace)** is a commercial, insurance-reimbursable gaze
controller that explicitly drives JACO-class arms. The **HARMONIC dataset** (24 able-bodied
users, gaze+joystick+EMG on a 6-DoF eating task) is the reference resource for gaze intent
inference under shared autonomy — methodological, not PWMD-validated.

### 7. Voice / speech
**Why discrete, not continuous.** The AT literature (Simpson & Levine 2002; Peixoto 2013;
ASSETS 2022, the last with 12 PWMD) is consistent: voice's low bandwidth and
activation→command→wait latency preclude the frequent small adjustments continuous driving
needs — but PWMD users judge voice **best** for discrete task-level commands that name exact
parameters ("set water to 39°"). Min ability: intelligible speech (excludes anarthric
ALS/locked-in). This maps cleanly onto AI-planning/shared-autonomy manipulation.
**LLM/VLM era (2023–2026).** **VoicePilot (UIST 2024)** put an LLM speech interface on the
**Obi** commercial feeding arm, evaluated with 11 older adults — the closest analogue to a
voice-commanded assistive arm on a real product (older adults, not explicitly PWMD).
**VoxPoser (CoRL 2023)** and **Code as Policies (ICRA 2023)** are the general
natural-language-to-manipulation exemplars (zero-shot trajectory synthesis / LLM-generated
policy code) — lab prototypes, no disability evaluation. A 2024 edge system (GPT-4 + VOSK +
RealSense → discrete ROS arm actions) hit 81.8% command pass rate but on a 4-DoF hobby arm,
no disabled participants; **a Kinova Gen3 PWMD study is named as future work** — i.e., the
exact thing RAMMP could be.

### 8. sEMG (surface electromyography)
Face/neck/shoulder/residual-muscle EMG for discrete triggers or low-D proportional control;
can work down to single-motor-unit signals for ALS/locked-in (Control Bionics NeuroNode
class). **⚠️ Evidence gap:** despite targeted searching, **no claims about NeuroNode specs,
FDA status, cost, or sEMG arm-teleoperation with PWMD survived verification** in this
survey. Treat as a known unknown — see §Gaps.

### 9. Body-machine interfaces (shoulder-motion IMU)
Vest-mounted IMUs map residual shoulder kinematics (PCA) to continuous proportional
commands — **body**-machine, not brain. **[PWMD]** For driving, C5–C6 tetraplegic users
were ~2× slower than their own joystick at first but reached joystick-equivalent smoothness
after five sessions (IEEE TNSRE 2015, n=3). **The strongest RAMMP template:** the same vest
let a **C5-SCI** user control a **Kinova MICO under shared autonomy** with just a **1-D
continuous speed signal + a 1-D binary segment switch** → **100% success** on a pick-and-pour
task, comparable to able-bodied controls (ICORR 2015, n=1). This is the existence proof that
minimal input + robot planning defeats the bandwidth problem.

### 10. BCI — non-invasive EEG
Brain-only; P300 / SSVEP / motor-imagery. Very low bandwidth: motor-imagery arm control
achieved only by **chaining** sequential low-D controls (2-D reach then 1-D grasp), on 13
able-bodied subjects with notable dropout (Sci Rep 2016). Wheelchair EEG BCIs discriminate
just **2–6 commands at ~60–95% accuracy**, mostly able-bodied labs. Stays lab-bound: gel
setup, low information-transfer rate, poor match to continuous arm control.

### 11. BCI — invasive intracortical
The **only** modality here with actual severe-tetraplegia participants doing functional arm
control. **[PWMD]** BrainGate2 (Nature 2012): two people with tetraplegia used a 96-channel
motor-cortex array to reach-and-grasp; participant S3 drank coffee from a bottle (4/6
attempts), array usable **>5 years** post-implant. **10-DoF** anthropomorphic arm control by
one tetraplegic participant (J Neural Eng 2015). Bidirectional BCI with tactile feedback via
cortical microstimulation **halved** clinical task times 20.9→10.2 s in a C5/C6 participant
(Science 2021). All are N=1–2 surgical research demonstrations — not products; neurosurgery,
clinic-only. (Neuralink PRIME / Synchron Stentrode current status did not survive
verification and is not covered here.)

---

## Cross-cutting findings

**The DoF-mismatch / mode-switch tax.** 1–3 DoF inputs can't cover a 6–7 DoF end-effector,
so users mode-switch between control subsets (a 1-D input needs to cycle **six** modes for a
6-DoF arm, seven with the gripper). Measured cost: **17.4 ± 0.8% of task time** spent
switching, not moving (Herlant, HRI 2016, n=6 able-bodied — likely a *floor* for impaired
users). Automating it works: **LAMS (HRI 2025)**, an LLM auto-switching modes on a **Kinova
Gen3** (same arm family as RAMMP) via a 2-DoF joystick, cut manual switches **50–71%**.
Herlant also found auto-switching **preferred** even when it didn't improve raw speed.

**Shared autonomy must be interface-aware.** The same intent-inference algorithm yields
significantly different goal-prediction error rates per interface (sip-and-puff worst).
Autonomy that acts before a noisy/delayed signal settles takes control from capable users →
disagreement, dissatisfaction, non-adoption (Argall, ICORR 2019 + THRI 2019).

**The acceptance paradox.** End users accept **fully automatic** manipulator control *less*
than manual or shared control — repeatedly shown to prefer retaining control despite slower
performance. Autonomy should feel like assistance, not takeover.

**Minimum-ability gating is real and selective.** In the Argall study, **4 of 9 SCI
participants could not use the joystick at all**, 1 could not use the head array. Every SCI
participant who *could* use a joystick was already a daily joystick user — so "joysticks
dominate" is contaminated by the fact that joystick-incapable candidates often never get an
arm prescribed. RAMMP's value is largest exactly for the users the joystick excludes.

**The unmet-need baseline (Fehr 2000, 200 clinicians).** 9–10% of trained power-wheelchair
patients find the chair extremely hard/impossible for ADLs; **40%** struggle specifically
with steering; **85%** of clinicians see patients who can't drive at all; nearly half of
non-drivers were judged to need **automated navigation**. The paper's own conclusion: the
field needs "not more innovation in steering interfaces, but … supervised autonomous"
control — RAMMP's thesis, stated in 2000.

---

## Implications for RAMMP

- **Don't pick one device — keep the input-agnostic channel.** Kinova ships exactly this
  (wheelchair-input pass-through via the Jaco Blue box). RAMMP's `/api/twist` (lease + seq +
  deadman) already is this abstraction; adding an input = writing an adapter, not new safety.
- **Match the input tier to residual ability, not to a single "best" device.** Realistic
  spread for RAMMP's population: proportional joystick / head-IMU (AMiCUS-style) for users
  with limb/head control → tongue or intraoral joystick for high-C SCI with spared cranial
  function → eye-gaze or voice for the lowest-mobility → invasive BCI is out of scope
  (surgical, N=1–2).
- **Bet on discrete task-level intent + autonomy, which every low-bandwidth modality
  supports.** Voice, gaze-dwell, tongue, sip-and-puff, single-switch all express "do X" or
  "go here / stop" well and continuous 6-DoF teleop badly. RAMMP's AI-planning layer is the
  thing that turns their weakness into a non-issue — and the ICORR 2015 result (1-D speed +
  binary switch → 100% success on a Kinova arm) is the proof it works.
- **Make the autonomy interface-aware and interruptible.** Track *which* input is live and
  how noisy it is; let the human veto/override; never feel like a chauffeur.
- **Nearest concrete next step:** a **voice + LLM task-command** path on the Gen3 is
  low-hanging — the 2024 edge-robotics paper explicitly names "Kinova Gen3 + PWMD study" as
  its future work, and VoicePilot shows the pattern on a real assistive arm.

---

## Evidence gaps (what four passes did *not* verify)

These were searched but did not produce verified claims — real holes, not settled absences:

- **sEMG entirely:** Control Bionics NeuroNode specs, FDA status, price, single-motor-unit
  signal requirements, and any PWMD sEMG arm-teleoperation. Highest-value gap to fill.
- **Canadian JACO long-term clinical studies** (Routhier / Archambault / Maheu / Beaudoin,
  CIRRIS/Laval): usage patterns, PIADS/QUEST satisfaction, **abandonment rates**, ADL
  success, day-to-day input choice. Directly decision-relevant.
- **FRIEND** (Bremen wheelchair-arm shared control) input scheme; **Obi** feeder's native
  switch-based selection (Obi appears here only under the VoicePilot LLM overlay).
- **Quantitative bandwidth (bits/min) and unit cost** for nearly every modality — the field
  reports throughput comparatively ("3× faster") and rarely prices devices.
- **Neuralink PRIME / Synchron Stentrode** current demonstrated device control and clinical
  status (2025–2026).
- **Population validity:** eye-gaze, voice, and EEG evidence is overwhelmingly able-bodied;
  the PWMD-validated arm evidence concentrates in tongue (n=1 + n=11), body-machine
  interface (n=1 + n=3), AMiCUS head control (n=6), and invasive BCI (n=1–2).

---

## Primary sources

- Fehr, Langbein & Skaar (2000), *J Rehabil Res Dev* — clinician unmet-need survey. PMID 10917267
- Chung et al. (2023), RESNA — 11 long-term JACO owners, input census & satisfaction
- Herlant, Holladay & Srinivasa (2016), HRI — 17.4% mode-switch time; auto-switch preference. PMC6053067
- Nejati Javaremi & Argall (2020), IROS / (2019) ICORR — wheelchair-input hierarchy, MICO teleop. arXiv:2008.00109
- Tao, Yang, Ding & Erickson (2025), HRI — **LAMS** LLM mode-switching on Kinova Gen3. arXiv:2501.08558
- Rudigkeit & Gebhard (2019), *Sensors* — **AMiCUS** head-motion arm control (6 tetraplegics). PMC6630260
- Thorp et al. (2015/16), IEEE TNSRE — shoulder-IMU body-machine wheelchair control. PMC4742425
- Jain et al. (2015), ICORR — body-machine + shared autonomy on Kinova MICO (C5, n=1). PMC4737957
- Sunny et al. (2021), *JNER* — eye-gaze control of wheelchair-mounted 6-DoF arm. PMC8684692
- Newman/Aronson/Admoni et al., *IJRR* — **HARMONIC** gaze intent dataset. arXiv:1807.11154
- Hochberg et al. (2012), *Nature* — BrainGate2 reach-and-grasp/drink. PMC3640850
- Wodlinger et al. (2015), *J Neural Eng* — 10-DoF intracortical arm control. PMID 25514320
- Flesher et al. (2021), *Science* — bidirectional sensory-feedback BCI grasp. doi:10.1126/science.abd0380
- Meng et al. (2016), *Sci Rep* — non-invasive EEG chained arm control. PMC5155290
- Kim/Ghovanloo et al. (2013), *Sci Transl Med* — **Tongue Drive** vs sip-and-puff (11 tetraplegia). PMID 24285485
- Tongue-controlled JACO (2017), *JNER* — n=1 C1–C2, 14 arm signals. doi:10.1186/s12984-017-0330-2
- Padmanabha et al. (2024), UIST — **VoicePilot** LLM on Obi feeder. arXiv:2404.04066
- Huang et al. (2023), CoRL — **VoxPoser**. arXiv:2307.05973 · Liang et al. (2023), ICRA — **Code as Policies**. arXiv:2209.07753
- Simpson & Levine (2002) PMID 12236450; Peixoto (2013); ASSETS 2022 (arXiv:2207.04344) — voice-control AT literature
- Permobil hub / mo-vis — commercial joystick force/throw specs
- Kinova JACO user guide & Jaco Blue Universal Interface — wheelchair pass-through control

*Compiled for RAMMP device selection. Uncommitted working draft — verify decision-critical
figures against the cited primary source before relying on them.*
