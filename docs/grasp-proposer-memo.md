# Decision Memo: 6-DoF Grasp Pose Generation for RAMMP (Kinova Gen3 + 2F-85 + D405 on Jetson AGX Orin)

Date: 2026-07-02 (research verified against sources as of the July 2026 state of the repos)

---

## 1. Can AnyGrasp run on our Orin?

**Conditionally — as of this month, and unproven.** Until July 2026 the answer was flatly no: the maintainer stated on 2026-04-10 that "the current SDK does not support aarch64 machines" (verbatim, [issue #137](https://github.com/graspnet/anygrasp_sdk/issues/137), confirmed via GitHub API), and the old x86_64-only `license_checker` died with `Exec format error` on ARM ([issue #141](https://github.com/graspnet/anygrasp_sdk/issues/141)). Two things changed:

- **2026-07-04**: new license tool replaced `lib_cxx.so`/`license_checker`; feature ID now comes from the gsnet `.so` itself (`get_feature_id()`), Python 3.14 + CUDA 13 added ([CHANGELOG](https://github.com/graspnet/anygrasp_sdk), commit e1a1b31).
- **2026-07-13**: dev branch shipped aarch64 binaries — verified via GitHub contents API: `grasp_detection/gsnet_versions/aarch64/` contains gsnet `.so` for CPython 3.6–3.14, **including cp310, which matches JetPack 6's default Python 3.10**. README news: "We are testing the aarch64 version SDK. Try it out if you have interest." Explicitly experimental; **zero user success reports on any Jetson yet**.

**License path and lead time:** (1) generate feature ID with the dev-branch aarch64 `.so`; (2) submit the Google Form ([forms.gle/XVV3Eip8njTYJEBo6](https://forms.gle/XVV3Eip8njTYJEBo6) — email, affiliation, advisor, non-commercial + non-distribution agreement, feature ID); (3) "we usually reply in 5 workdays" per README — budget **1–2 weeks** given documented email delays (issues #19, #101, #139, #160; check spam). Weights arrive via links in the license email, not the repo. Non-commercial terms are fine for RAMMP research. Two caveats: the license is **machine-locked** and feature-ID drift is documented under Docker/WSL and even across reboots ([#164](https://github.com/graspnet/anygrasp_sdk/issues/164), filed against the new tool, unanswered); and license expiry is documented nowhere.

**The real technical gate is not the `.so` — it's MinkowskiEngine.** AnyGrasp requires the maintainer's fork (chenxi-wang/MinkowskiEngine, `cuda-12-1` branch for JetPack 6's CUDA 12.x, `--blas=openblas`). Adversarial search found **no documented successful MinkowskiEngine build on any Jetson, ever** — only failed attempts (NVIDIA/MinkowskiEngine #544 OOM on Nano/NX; forum thread 158475 on TX2). The Orin's RAM removes the OOM issue, and nothing in ME is x86-specific, but a build against NVIDIA's aarch64 PyTorch wheels would be a first-on-record; budget for the CUDA-12 patch set (ME issues #543, #601, #621). Expected latency if it works: ~0.3–0.6 s/inference extrapolated from the T-RO paper's ~100–200 ms on 2080 Ti-class hardware (low confidence, no Orin benchmark exists) — adequate for grasp-then-execute ADL tasks. Mandatory: voxel-downsample the D405 cloud (a raw ~1M-point cloud triggered >30 GB allocation, issue #29).

**Verdict: apply for the license this week (it's free and parallel to everything else), but do not put AnyGrasp on the critical path.**

### Addendum 2026-07-15 — license in hand; binary forensics

License + weights are now IN HAND. New facts from direct analysis of `gsnet.cpython-310-aarch64-linux-gnu.so` (dev branch, 1,602,144 bytes):

- **No native torch linkage.** `DT_NEEDED` is only libdl/libm/libc; torch and MinkowskiEngine are imported as *Python modules* at runtime. No C++ ABI coupling → the Jetson's cuRobo-pinned CUDA torch should serve as-is. glibc floor 2.29 (22.04 ships 2.35).
- **License internals:** Python-level checker doing public-key signature verification over a bundle (`licenseCfg.json` + `.public_key` + `.signature` + `.lic`); error strings include `feature id mismatch: local=%s license=%s` and `expired:` — licenses carry an expiry, term undocumented. Fails loud: "license failed to pass, AnyGrasp will be disabled!".
- **Maintainer is testing aarch64 actively** — chenxi-wang on [#141](https://github.com/graspnet/anygrasp_sdk/issues/141), 2026-07-15: "I'm testing the aarch64 version. See the dev branch." Report Orin results there; support is likely.
- **Free first test (no torch/ME needed):** the `.so` only needs libc — on the Jetson, `python3 -c "from gsnet import get_feature_id; print(get_feature_id())"` and compare against the license's `licenseCfg.json`; then reboot twice and re-run (probes the [#164](https://github.com/graspnet/anygrasp_sdk/issues/164) feature-ID reboot-drift risk before any build investment).
- **Ordered plan:** (1) feature-ID check + reboot drift probe; (2) timebox the ME build ~1 day (`cuda-12-1` branch, `TORCH_CUDA_ARCH_LIST=8.7`, `--blas=openblas`, `MAX_JOBS=4`); (3) pointnet2 + demo.py, measure Orin latency; (4) only then wire behind the step-5 seam as a *plugin with the geometric synthesizer as automatic fallback* (a license drift must degrade, never halt). Output convention: AnyGrasp +X = approach, translation = grasp CENTER not tip — convert into pad-center convention incl. `tool_tip_offset` 0.021 and `tool_spin_deg` 90.
- **Remote-service fallback is poorly matched here:** the dev machine is Windows; the `.so` is linux-gnu only, and WSL2/Docker are exactly where feature-ID drift is documented ([#87](https://github.com/graspnet/anygrasp_sdk/issues/87), #164). HGGD remains the license-clean fallback.
- **Unverified:** which machine's feature ID the issued license is bound to — resolve before anything else.

### Addendum 2026-07-16 — the open-family research (post-AnyGrasp pivot)

Field motivation: AnyGrasp proposals on our sim clouds score 0.01–0.15 (domain
gap: it trains on private real-sensor data) and the gates reject them — the
robot falls back or gives up. Research verdict across the GraspNet family:

- **AnyGrasp is no longer ahead of the open field.** GraspNet-1B RealSense AP
  (seen, +CD rows): graspnet-baseline 47.5 → **AnyGrasp 66.1 ≤ GSNet 67.1**
  → **EconomicGrasp (ECCV24, MIT, weights released, PyTorch 2.5/CUDA 12)
  68.2** → FineGrasp 71.7+ → RNGNet (2024) 75.2 (openness unverified —
  check if it ever becomes the fine-tune target). Nuances verified against
  primary sources: the AnyGrasp paper itself never published an AP table
  (third-party tabulated, FineGrasp Table I / RNGNet Table 1); apples-to-
  apples, GSNet+CD actually edges AnyGrasp on Seen. Benchmark AP = fraction
  of top-50 proposals that are force-closure-valid, averaged over friction
  0.2-1.2 (graspnetAPI/graspnet_eval.py).
- **graspnet-baseline practicals** (if ever needed): its pointnet2 is
  file-identical to anygrasp_sdk's (already compiled on our Orin); only the
  small knn op needs building (upstream THC-free since 2025-02); weights
  `checkpoint-rs.tar` on Google Drive; output is the same graspnetAPI
  GraspGroup → drop-in behind grasp_proposer. **License: SJTU non-commercial,
  NO redistribution, fine-tuned derivatives owned by SJTU** — poor fit for a
  public repo. EconomicGrasp's MIT is the clean iterate-and-publish path.
- **Fine-tuning on sim is proven** (R2SGrasp, arXiv 2410.06521: sim-trained,
  beats real-trained) and cheap (EconomicGrasp full train: 8.3 h on one
  RTX 3090). Our inference domain IS sim → domain-matched training kills the
  score-suppression problem. Gap: the analytic force-closure annotator was
  never released — write it (~200 lines antipodal sampling + friction sweep,
  label format documented in graspnetAPI docs; rhett-chen/grasp-auto-annotation
  is a deprecated reference implementation).
- **PLAN:** (1) force-closure annotator over our 8 exact meshes → a
  deterministic per-object GRASP LIBRARY backend (zero learning, no domain
  gap, fixes give-ups for known objects); (2) same annotator + MuJoCo scene
  randomizer → GraspNet-format dataset → fine-tune EconomicGrasp (clutter /
  pose-noise robustness, the real-world path); (3) AnyGrasp stays as A/B
  reference. The grasp_proposer node is backend-agnostic (GraspGroup in/out).

---

## 2. Pragmatic recommendation: HGGD now, GG-CNN as fallback

**Primary: HGGD** ([github.com/THU-VCLab/HGGD](https://github.com/THU-VCLab/HGGD), RA-L 2023) — the only choice that clears license, sensor, and Jetson bars simultaneously:

- **MIT license** — clean for any future path (vs. graspnet-baseline's SJTU noncommercial-only license, Contact-GraspNet's NVIDIA non-commercial license, AnyGrasp's gated binary).
- **Single-view RGBD in, 6-DoF grasps in clutter out** — matches the D405 feed directly, no segmentation stage required.
- **The only modern 6-DoF net with measured Jetson numbers**: 418 ms on Xavier NX / 649 ms on TX2 ([E3GNet paper benchmarks, arXiv:2410.22980](https://arxiv.org/html/2410.22980v2)); extrapolated ~100–200 ms (5–10 Hz) on AGX Orin in plain PyTorch. In the same benchmark, **GSNet (MinkowskiEngine-based, AnyGrasp's cousin) failed to run on Jetson at all**.
- **No MinkowskiEngine.** Stack is PyTorch ≥1.10 + pytorch3d (aarch64 source build — annoying but known-good).
- Caveats: no ROS 2 wrapper exists (true of every modern 6-DoF net — see section 3); set gripper width to the 2F-85's 85 mm; trained on GraspNet-1Billion at 0.4–0.7 m standoff while the D405 is optimal at 7–50 cm, so plan a pre-grasp scan pose and validate the domain gap early on real hardware.

**Fallback: GG-CNN** ([github.com/dougsm/ggcnn](https://github.com/dougsm/ggcnn), BSD-3-Clause) — 62,420 parameters, 19 ms full pipeline on desktop GPU, trivially TensorRT-able, and it has Kinova heritage ([ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping), ROS1 Mico-era). It is 4-DoF top-down only — not a full replacement, but it covers many tabletop ADL picks at near-zero integration risk and its closed-loop wrist-camera design philosophy matches the D405-on-wrist setup.

**Watch list:** E3GNet (same THU group, 127 ms on Xavier NX, 94% real-robot success on UR5e + **Robotiq 2F-85** + RealSense — ideal on paper but **no code released**; grab immediately if it drops); RNGNet (HGGD successor with closed-loop grasping and handover — assistive-relevant, but **no license file**; worth emailing THU); NVIDIA GraspGen (20 Hz reported, but NVIDIA research license, no documented Jetson support, needs a SAM2-class segmenter upstream); AnyGrasp aarch64 (re-check in ~1–2 months once early-adopter reports land).

---

## 3. Integration pattern to build TODAY (proposer-swappable)

Build a thin **grasp-proposal ROS 2 node behind a stable interface**, following the MoveIt Deep Grasps pattern of "grasp generator as a server, planner as client" ([tutorial](https://moveit.picknik.ai/humble/doc/examples/moveit_deep_grasps/moveit_deep_grasps_tutorial.html)). No maintained Humble wrapper exists for any modern 6-DoF net (all published wrappers are ROS1 or stale; the one Contact-GraspNet ROS 2 wrapper targets Jazzy and is WIP), so this node is required regardless of proposer choice — which makes it the right seam.

**Contract:** aligned RGBD + CameraInfo in → ranked grasp array out (PoseArray or a custom msg mirroring `moveit_msgs/Grasp`'s pre_grasp_approach / post_grasp_retreat parameterization). Everything downstream already exists in the stack:

1. **Normalize frames inside the node** — this is the classic integration bug, verified in detail: graspnetAPI/AnyGrasp uses **+X = approach, +Y = jaw-closing**, translation = grasp center with tip = `translation + depth·R[:,0]` (translation is NOT the tip — [USAGE.md](https://github.com/graspnet/anygrasp_sdk/blob/main/grasp_detection/USAGE.md)); Contact-GraspNet uses **+Z = approach, +X = baseline**, translation at the gripper base ([arXiv:2103.14127](https://arxiv.org/abs/2103.14127)). Each proposer plugin owns its per-detector rotation AND translation offset into a single canonical convention (recommend Kinova tool-Z-approach at the 2F-85 fingertip plane). All detectors output camera-frame; apply hand-eye transform at the node boundary.
2. **Two-cloud discipline**: propose on the target-object cloud (mask from user/LLM selection — kinova-gemini's segment-before-proposal or OK-Robot's mask-filter-after-proposal both work), collision-check against the full scene cloud. Note the MoveIt demo does NOT actually do scene-cloud collision checking (verified against its source — it uses hand-spawned primitives), so RAMMP must supply its own scene representation via cuRobo's world model.
3. **cuRobo already has the execution layer**: `MotionGen.plan_grasp` (added v0.7.5, 2024-11-22, [releases](https://github.com/NVlabs/curobo/releases)) takes the top-k grasp set, does reachability/collision arbitration across candidates, and plans approach-to-pre-grasp offset, constrained linear grasp approach, and constrained lift, with gripper commands interleaved by the caller. **Action item: verify the local cuRobo version ≥0.7.5** — and note v0.8.0 (Apr 2026) restructured the API, so pin deliberately. Cheap win from kinova-gemini: on IK failure, retry with a 180° flip about the approach axis.
4. **Grasp verification** via `control_msgs/GripperCommand` on ros2_kortex's `/robotiq_gripper_controller/gripper_cmd`: success = `stalled && !reached_goal` at a gap clearly larger than fully closed (maps to Robotiq gOBJ 0x02 = contact while closing vs 0x03 = no object/dropped); then lift ~10 cm and re-check the gap.

**Sequence:** wire this node with HGGD against the simulated D405 now; the proposer is a plugin, so AnyGrasp (if the aarch64 build pans out), E3GNet, or RNGNet slot in later without touching the planner side.

**Reference implementation to crib:** [jakmilller/kinova-gemini](https://github.com/jakmilller/kinova-gemini) (updated Jun 2026) — Gen3 7-DoF + Robotiq 2F + RealSense + Humble, documented SAM2→AnyGrasp→IK-with-flip-retry→pre-grasp→close→lift pipeline; nearly our exact stack.

---

## 4. Top 3 risks

1. **D405 close-range domain gap.** Every candidate net (HGGD, RNGNet, graspnet-baseline, AnyGrasp) trained on GraspNet-1Billion captured with D435/Kinect at ~0.4–0.7 m; the D405 is optimized for 7–50 cm. Wrist-camera viewpoints will be out-of-distribution. Mitigation: dedicated pre-grasp scan pose (see GraspView, [arXiv:2511.04199](https://arxiv.org/abs/2511.04199), for the wrist-camera next-best-view pattern); validate on real hardware in week one of D405 availability; budget for fine-tuning with near-range crops. This risk applies regardless of proposer choice.

2. **If AnyGrasp is pursued: unproven aarch64 chain with a machine-locked license.** Three stacked unknowns: MinkowskiEngine fork has never been built on Jetson (no success on record anywhere); the aarch64 `.so` is one week old with no user reports; and feature-ID drift across reboots is reported against the new license tool (#164, unanswered) — a re-lock on a deployed assistive robot would be an outage with a multi-day re-licensing loop. Treat as "announced, unproven"; timebox any build attempt.

3. **No ROS 2 wrapper exists for anything, and upgrade paths have license holes.** The thin Humble node is unavoidable custom work on the critical path — schedule it now, in sim, not after hardware arrives. Downstream: RNGNet (best assistive-feature fit) has no license file (all-rights-reserved by default), E3GNet has no code, GraspGen has no Jetson support and a research-only code license. If HGGD underperforms at close range, the fallback ladder is GG-CNN (planar-only) → license outreach (THU for RNGNet) → AnyGrasp aarch64 — none is a drop-in 6-DoF replacement today. Mitigation: send the RNGNet license email and the AnyGrasp license form this week; both are free options with long lead times.

**Bottom line:** Build the swappable grasp-proposal node around HGGD in sim now; file the AnyGrasp license application in parallel but keep it off the critical path; keep GG-CNN warm as the planar fallback; re-evaluate AnyGrasp-on-Orin and the watch-list nets in 6–8 weeks.
