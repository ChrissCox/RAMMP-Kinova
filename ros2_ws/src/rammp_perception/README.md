# rammp_perception

Continuous scene perception: the arm **sees** its surroundings and the
planner's collision world **follows reality** instead of trusting scene.yaml
forever. Knock the bottle across the island, say "computer, go to the
bottle" — the arm goes to where it actually is.

```
scene_cam (fixed, over the kitchen)     d405 (eye-in-hand, on the wrist)
   │  RGB-D from mujoco_ros2_control       │
   ▼                                       ▼
detector node (one instance per camera; both start with the bringup)
   │  /perception/objects        vision_msgs/Detection3DArray, base frame
   │  ~/debug_image              camera frame with detections painted on
   ▼
curobo_planner — fresh detections override prop poses; targets follow
                 their reach-for object
```

Both detector instances start automatically with
`ros2 launch mujoco_sim mujoco_bringup.launch.py`. Run one manually only for
debugging — the d405 instance is the default parameters plus:

```bash
ros2 run rammp_perception detector --ros-args -r __node:=d405_detector \
  -p rgb_topic:=/d405/color -p depth_topic:=/d405/depth \
  -p info_topic:=/d405/camera_info -p camera_attached_frame:=bracelet_link
```

## Backend

**`color`** — segments the sim props by CHROMATICITY (color ratios, so
lighting/highlights don't break matching; absolute-RGB lost the mug to its
own specular). The scene palette is tuned so every free prop is chromatically
distinct (minimum pairwise distance 0.125 vs the 0.08 match tolerance). Zero
ML dependencies — it exercises the full camera→world pipeline exactly the way
a neural backend will. Offline validation vs sim ground truth: **≤10 mm**
horizontal error on all eight props.

**NanoOWL** (phase 2, Jetson production): NVIDIA's TensorRT OWL-ViT
(~40-60 ms/frame on AGX Orin, runtime-changeable text prompts), duty-cycled
at 2-5 Hz so cuRobo keeps the GPU. It slots in as another backend — the 3D
path, gates, and planner wiring don't change.

## 3D localization (why the positions are honest)

- Depth anchors at the prop's robust nearest surface (5th percentile), and
  only depths within the prop's own physical span count — background bleed
  at mask edges is excluded no matter how much of the mask it poisons.
- The point is pushed half the prop's known size along the view ray:
  surface → center.
- **Eye-in-hand:** color + depth must carry the same timestamp, and the
  camera pose comes from TF **at that timestamp** — "latest TF" on a moving
  wrist smeared positions by centimetres (field: phantom "bottle moved
  84 mm"). Ticks TF can't serve are skipped, not guessed.

## Honesty gates (why it doesn't lie)

- Detections outside the island workspace box are dropped (the white arm
  and beige wall otherwise masquerade as props).
- A detection farther than `max_jump` (0.25 m) from a label's last-known
  position is rejected — MISS, not mismatch. Unseen props keep their
  last-known pose: a stale pose beats a wrong one.
- Duplicate-color ambiguity resolves by position continuity: each label
  keeps the candidate nearest its last-known position.

## Diagnostics

```bash
ros2 run rammp_perception probe                    # scene_cam health report
ros2 run rammp_perception probe --ros-args -p rgb_topic:=/d405/color \
  -p depth_topic:=/d405/depth -p info_topic:=/d405/camera_info
```

One frame in, everything out: encodings, depth statistics with named
failure modes (1×1 frames, millimetre depth, z-buffers, far-plane), raw
blobs, and each candidate's 3D position with the exact gate verdict. The
`~/debug_image` topics show the same continuously — view from Windows via
the mirror viewer's `--camera` flag.

## Cameras

`scene_cam` is defined in `mujoco_sim/scenery.py` (pos `[-0.75, 0, 1.45]`,
fovy 58°, looking down at the island); the detector's `camera_position` /
`camera_xyaxes` parameters MUST match it. The `d405` rides `bracelet_link`
(defined in `build_scene.py`; the node's mount parameters MUST match).
Intrinsics come from `camera_info` at runtime. The same node runs against a
real RealSense later — same topic types, extrinsics become a calibration
instead of a copy.
