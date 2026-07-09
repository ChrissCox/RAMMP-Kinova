# rammp_perception

Continuous scene perception: the arm **sees** its surroundings and the
planner's collision world **follows reality** instead of trusting scene.yaml
forever. Knock the bottle across the island, say "computuh, go to the
bottle" — the arm goes to where it actually is, dodging things where they
actually are.

```
scene_cam (RGB-D, in the MuJoCo scene, published by mujoco_ros2_control)
   │
   ▼
detector node  — backend detects objects, depth turns masks into 3D
   │  /perception/objects   (vision_msgs/Detection3DArray, base frame)
   │  /perception/markers   (Foxglove: cyan spheres + labels)
   ▼
curobo_planner — fresh detections OVERRIDE prop poses everywhere
                 (collision world, touch diagnosis, target following)
```

## Run (Jetson, after the bringup)

```bash
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge python3-scipy
ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie   # adds scene_cam
# restart the bringup, then:
ros2 run rammp_perception detector
```

The planner picks detections up automatically (`live_objects` param,
default true; `live_staleness` 10 s — stale detections fall back to YAML,
because a stale pose beats a wrong one).

## Backends

- **`color`** (default) — segments the sim props by CHROMATICITY (color
  ratios, so lighting/highlights don't break matching; absolute-RGB lost the
  mug to its own specular). Tracks the chromatic props (bottle, mug,
  snack box, apple); white/cream props (plate, bowl) are indistinguishable
  from the white arm by color and stay on YAML truth. Zero ML dependencies —
  this backend exists to validate the full camera→world pipeline, which it
  does: offline test vs sim ground truth shows ≤27 mm error on all tracked
  props and 8.7 mm re-acquisition after a 12 cm knock.
- **`owlvit`** (`backend:=owlvit`) — real open-vocabulary detection
  (HuggingFace OWL-ViT; `pip install transformers`). Slow without a GPU
  engine; the stepping stone to NanoOWL.
- **NanoOWL** (phase 2, Jetson production): NVIDIA's TensorRT OWL-ViT
  (~40-60 ms/frame on AGX Orin, runtime-changeable prompts, Apache-2.0),
  duty-cycled at 2-5 Hz so cuRobo keeps the GPU. Runs as its own ROS node
  (`jetson-containers run $(autotag nanoowl)` + ROS2-NanoOWL); wiring its
  Detection2DArray into this node's 3D path is the next step.

## Honesty gates (why it doesn't lie)

- Detections outside the island workspace box are dropped (the white arm
  and beige wall otherwise masquerade as props).
- A detection farther than `max_jump` (0.25 m) from a label's last-known
  position is rejected — MISS, not mismatch.
- Duplicate-color ambiguity (mug and apple are both red) resolves by
  position continuity.

## Camera

`scene_cam` is defined in `mujoco_sim/scenery.py` (pos `[-0.75, 0, 1.45]`,
fovy 58°, looking down at the island). The detector node's
`camera_position` / `camera_xyaxes` parameters MUST match it — intrinsics
come from `camera_info` at runtime. The same node structure runs against a
real camera later (D405 publishes the same topic types; extrinsics become a
calibration instead of a copy).
