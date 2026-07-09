"""cuRobo motion-planning node for the Kinova Gen3 (simulation).

Pipeline:
  /curobo_planner/command (std_msgs/String)
      -> resolve to a Cartesian goal pose (named target from the scene,
         "home", or "pose: x y z roll pitch yaw")
      -> cuRobo MotionGen.plan_single() against the scene's collision world
      -> publish trajectory_msgs/JointTrajectory to the joint_trajectory_controller
         (the streaming topic the fake ros2_control JTC already accepts)
  Obstacles + targets + the active goal are published as a MarkerArray for Foxglove.

Uses cuRobo's bundled `kinova_gen3.yml` (joint_1..7, ee_link=tool_frame, Robotiq
2F-85 collision spheres) — no robot config generation needed. Pin cuRobo v0.7.8.

The collision world = scene obstacles + props (as bounding boxes), rebuilt per
command, MINUS the props the target lists in `ignore_objects` (you reach FOR
the bottle, you can't also dodge it). `home` is planned through cuRobo too —
every motion this node emits is collision-checked.

Conventions that bite (all verified against v0.7.8 source; handled here):
  * ee_link is `tool_frame` = 0.120 m BEYOND the wrist flange, roughly the
    fingertip midpoint. Scene targets say where the FINGERTIPS go.
  * cuRobo Pose quaternion is [w, x, y, z]; ROS/scene is [x, y, z, w].
  * cuRobo joint order follows its config, not the controller's — remap by NAME.
  * Cylinder/Sphere entries in a WorldConfig are SILENTLY IGNORED by the
    collision checkers (only cuboid + mesh load) — props go in as cuboids.
  * update_world() with ZERO cuboids silently keeps the previous world (early
    return before the disable line); more cuboids than collision_cache raises.
  * warmup() once (the first plan is otherwise many seconds); its default
    warmup_js_trajopt=True also pre-captures the plan_single_js graphs.
  * The executed trajectory is uniform at interpolation_dt; time_from_start = (k+1)*dt.
"""

import math
import re
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

from curobo_planner.scene import load_scene, resolve_phrase, scene_markers

STOP_WORDS = {'stop', 'halt', 'freeze', 'cancel', 'estop'}


class CuroboPlanner(Node):

    def __init__(self):
        super().__init__('curobo_planner')
        self._cb = ReentrantCallbackGroup()

        self.scene_file = self.declare_parameter('scene_file', '').value
        self.robot_config = self.declare_parameter('robot_config', 'kinova_gen3.yml').value
        self.joint_names = list(self.declare_parameter(
            'controller_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7'],
        ).value)
        self.joint_states_topic = self.declare_parameter(
            'joint_states_topic', '/joint_states'
        ).value
        self.jtc_topic = self.declare_parameter(
            'jtc_topic', '/joint_trajectory_controller/joint_trajectory'
        ).value
        # 0.02: dense waypoints make the JTC's spline replay visibly smoother
        # on big base turns (25 Hz targets read as jerky at 500 Hz control).
        self.interpolation_dt = self.declare_parameter('interpolation_dt', 0.02).value
        self.max_attempts = self.declare_parameter('max_attempts', 8).value
        self.execute = self.declare_parameter('execute', True).value
        # Latency knobs. The honest fast switch is enable_finetune=false
        # (~2x faster, still collision-free, slightly less smooth).
        # finetune_attempts stays at cuRobo's default 5: the loop exits on
        # the FIRST success (easy plans never pay for the extra attempts),
        # and when ALL attempts fail v0.7.8 DISCARDS the already-valid
        # trajectory — lowering it converts hard-goal successes to failures.
        self.finetune_attempts = self.declare_parameter('finetune_attempts', 5).value
        self.enable_finetune = self.declare_parameter('enable_finetune', True).value
        # Graph (PRM) warmup needs torch.svd, which on some Jetson torch wheels
        # requires a newer cuSOLVER than JetPack ships. Trajopt does not need
        # it, so the graph planner is opt-in.
        self.enable_graph = self.declare_parameter('enable_graph', False).value
        # Collision-cache headroom so live-editing scene.yaml can ADD obstacles
        # (cuRobo's update_world is only safe up to this many boxes).
        self.collision_cache_obb = self.declare_parameter('collision_cache_obb', 40).value
        # Physical safety margin: every collision box is inflated by this on
        # each side. cuRobo's hard feasibility allows ~0 mm clearance, the
        # JTC tracks with mm-cm error, and the boosted sphere model still
        # under-covers the elbow by up to 15 mm — 0.02 keeps real transit
        # clearance positive through all of that (0.01 left the elbow ~5 mm
        # of legal contact arcing over the cabinet). Goals live closer than
        # the padding to their objects; `check` validates them after edits.
        self.world_padding = self.declare_parameter('world_padding', 0.02).value
        # Stock kinova_gen3.yml fingertip pads carry a single r=0.01 sphere each
        # — thinner than the real 2F-85 fingertip, so "collision-free" paths
        # clip props in MuJoCo. Inflate to this radius (0.01 restores stock).
        self.pad_sphere_radius = self.declare_parameter('pad_sphere_radius', 0.02).value
        # Full gripper collision shell (see _gripper_shell): the stock model
        # leaves the knuckle housing bare and the fingers nearly so.
        self.gripper_shell = self.declare_parameter('gripper_shell', True).value
        # Live perception: rammp_perception publishes detected prop positions
        # (base frame) on /perception/objects; fresh detections OVERRIDE the
        # YAML poses of matching props in everything downstream (collision
        # world, touch diagnosis, markers), and targets follow their
        # reach-for object. Stale detections fall back to YAML — a stale
        # pose beats a wrong one.
        self.live_objects = self.declare_parameter('live_objects', True).value
        self.live_staleness = self.declare_parameter('live_staleness', 10.0).value
        # Boosted ARM spheres (see _ARM_SPHERES): mesh-vs-sphere audit showed
        # real geometry poking up to 64 mm (base), 47 mm (shoulder), 36 mm
        # (upper arm) outside the stock model — invisible centimetres that
        # were "smacking into things". The boosted set is audit-tuned to
        # p95 protrusion ~0 and verified self-collision-free (positive
        # margins at home + every scene target under cuRobo's ignore rules).
        self.arm_sphere_boost = self.declare_parameter('arm_sphere_boost', True).value
        # The PHYSICAL gripper is mounted 90° twisted about the tool axis
        # relative to cuRobo's URDF gripper. Goal orientations are authored
        # for the physical fingers ("straddle the bottle"), so every goal is
        # spun by this angle about its own tool z before it reaches cuRobo.
        self.tool_spin_deg = self.declare_parameter('tool_spin_deg', 90.0).value
        # Distance (m) at which the trajopt collision cost starts pushing away
        # (soft standoff). cuRobo's default 0.025 lets transit graze; 0.03
        # buys margin for the coarse gripper sphere model.
        self.collision_activation_distance = self.declare_parameter(
            'collision_activation_distance', 0.03).value
        # Joint-speed scaling for retiming. KEEP AT 1.0: 1.4 demanded speeds
        # the sim's motors cannot track — they torque-saturate and CUT
        # CORNERS off the collision-checked path (field: a waypoint-clean
        # trajectory ended wedged in the shelf). Faster motion needs actuator
        # gains raised to match, not just faster setpoints.
        self.velocity_scale = self.declare_parameter('velocity_scale', 1.0).value
        # Kinova "Home" (bent-elbow) — well-conditioned; also the safe default start.
        self.home_pose = list(self.declare_parameter(
            'home_pose_rad', [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
        ).value)

        if not self.scene_file:
            raise RuntimeError('scene_file parameter is required')

        self._state_lock = threading.Lock()
        self._q_now = None            # latest arm joints (controller order), or None
        self._v_max = None            # latest max |joint velocity|, or None
        self._traj_end = None         # node-clock Time the last published traj ends
        self._plan_lock = threading.Lock()
        self._scene = load_scene(self.scene_file)
        self._goal_name = None        # last goal, for marker highlighting
        self._last_ignore = set()     # props ignored by the last EXECUTED plan
        self._last_standoff = None    # (xyz, wxyz) to retreat through, or None
        self._home_pose_fk = None     # cached FK of home_pose (cuRobo frame)
        self._last_traj = None        # last executed JointTrajectory (for back-out)
        self._stop_requested = False  # voice/CLI "stop": abort segment waits

        self.create_subscription(
            JointState, self.joint_states_topic, self._joint_state_cb, 10,
            callback_group=self._cb)
        self._live = {}          # label -> ([x, y, z], monotonic stamp)
        if self.live_objects:
            try:
                from vision_msgs.msg import Detection3DArray
                self.create_subscription(
                    Detection3DArray, '/perception/objects',
                    self._perception_cb, 5, callback_group=self._cb)
            except ImportError:
                self.get_logger().warning(
                    'vision_msgs unavailable — live perception disabled '
                    '(sudo apt install ros-humble-vision-msgs)')
                self.live_objects = False
        self._jtc_pub = self.create_publisher(JointTrajectory, self.jtc_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        # Latched so a client that connects just after a terminal status still
        # receives it (fast error replies were racing DDS discovery).
        self._status_pub = self.create_publisher(
            String, '~/status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(1.0, self._publish_markers, callback_group=self._cb)

        # Heavy GPU init BEFORE the command subscription exists: a command sent
        # during the (possibly minutes-long) warmup must not silently queue and
        # then move the arm long after the client gave up.
        self._init_curobo()
        self.create_subscription(
            String, '~/command', self._command_cb, 10, callback_group=self._cb)
        self._status_pub.publish(String(data='ready'))
        self.get_logger().info(
            "cuRobo planner ready. Send targets to '%s/command': %s | home | "
            "'pose: x y z r p yaw' | check (dry-plan all targets)"
            % (self.get_fully_qualified_name(), ', '.join(self._scene.target_names)))

    # --------------------------------------------------------------- cuRobo setup
    def _init_curobo(self):
        import torch  # noqa: F401  (import here so the module loads without CUDA)
        try:
            # Newer warp-lang (>=~1.6) requires the torch interop submodule to
            # be imported explicitly; cuRobo v0.7.8 assumes implicit `wp.torch`
            # and crashes in its mesh collision checker otherwise.
            import warp.torch  # noqa: F401
        except ImportError:
            pass
        try:
            # Route linalg through MAGMA where available: this wheel's cuSOLVER
            # path needs cusolverDnXsyevBatched, newer than JetPack's library.
            torch.backends.cuda.preferred_linalg_library('magma')
        except Exception:
            pass
        from curobo.types.base import TensorDeviceType
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        self._torch = torch
        self._tensor_args = TensorDeviceType()
        self.get_logger().info('Loading cuRobo MotionGen (%s)...' % self.robot_config)
        kw = {}
        if float(self.velocity_scale) != 1.0:
            kw['velocity_scale'] = float(self.velocity_scale)
        cfg = MotionGenConfig.load_from_robot_config(
            self._load_robot_config(),
            self._world_dict(self._scene),
            tensor_args=self._tensor_args,
            interpolation_dt=self.interpolation_dt,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={'obb': int(self.collision_cache_obb), 'mesh': 10},
            collision_activation_distance=float(self.collision_activation_distance),
            **kw)
        self._motion_gen = MotionGen(cfg)
        # cuRobo consumes the start state in ITS cspace order, not by name —
        # plan_single does not reorder. Build start states in this order.
        self._curobo_joint_names = list(self._motion_gen.kinematics.joint_names)
        if set(self._curobo_joint_names) != set(self.joint_names):
            raise RuntimeError(
                'cuRobo joints %s != controller joints %s'
                % (self._curobo_joint_names, self.joint_names))
        self.get_logger().info('Warming up cuRobo (first-run kernel compile)...')
        self._motion_gen.warmup(enable_graph=bool(self.enable_graph))
        self.get_logger().info('cuRobo warmup complete.')

    def _load_robot_config(self):
        """The bundled robot config with the fingertip pads inflated.

        Returns the robot_cfg dict (the canonical cuRobo pattern for
        modifying a bundled config before MotionGenConfig.load_from_robot_config).
        The spheres may live inline or in a referenced spheres/*.yml — handle
        both, then bump the two inner_finger_pad spheres up to
        pad_sphere_radius so the collision model matches a real fingertip.
        """
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        cfg = load_yaml(join_path(get_robot_configs_path(), self.robot_config))['robot_cfg']
        kin = cfg['kinematics']
        spheres = kin.get('collision_spheres')
        if isinstance(spheres, str):
            loaded = load_yaml(join_path(get_robot_configs_path(), spheres))
            spheres = loaded.get('collision_spheres', loaded)
            kin['collision_spheres'] = spheres
        r = float(self.pad_sphere_radius)
        for link in ('left_inner_finger_pad', 'right_inner_finger_pad'):
            for s in spheres.get(link, []) if isinstance(spheres, dict) else []:
                s['radius'] = max(float(s['radius']), r)
        if self.gripper_shell and isinstance(spheres, dict):
            base = spheres.setdefault('robotiq_arg2f_base_link', [])
            base.extend(self._gripper_shell())
        if self.arm_sphere_boost and isinstance(spheres, dict):
            for link, sph in self._ARM_SPHERES.items():
                spheres[link] = [{'center': list(c), 'radius': rr}
                                 for c, rr in sph]
        # Bias cuRobo toward OUR home family. The bundled retract_config
        # ([0,-0.8,0,1.5,0,0.4,0]) lives in the opposite ELBOW FAMILY from
        # the Kinova Home this stack uses (joint_3 = pi): IK seeded there
        # keeps picking flipped configurations, and every plan that crosses
        # families is a huge, winding, wrist-twisting reconfiguration.
        try:
            cspace = kin['cspace']
            by_name = dict(zip(self.joint_names, self.home_pose))
            cspace['retract_config'] = [
                float(by_name[n]) for n in cspace['joint_names']]
        except Exception as exc:
            self.get_logger().warning('retract_config bias failed: %s' % exc)
        return cfg

    # Audit-tuned replacement arm spheres (mesh-vs-sphere protrusion p95 ~0,
    # worst 15 mm vs stock's 64 mm) with verified positive self-collision
    # margins at home and all scene targets. Bracelet mains grow only +5 mm
    # and the forearm tip stays stock — fatter versions of those two put the
    # MUG config into forearm/bracelet self-collision (-2.5 mm, caught by
    # the offline gate before shipping).
    _ARM_SPHERES = {
        # base sphere bottom must stay ABOVE the pedestal box top (-0.03,
        # scene.yaml) or the robot permanently collides with its own mount
        # and every start state is invalid (field regression).
        'base_link': [([0, 0, 0.065], 0.075), ([0, 0, 0.125], 0.065)],
        'shoulder_link': [([0, 0, -0.04], 0.07), ([0, 0, -0.10], 0.072),
                          ([0, 0, -0.16], 0.062)],
        'half_arm_1_link': [([0, 0, 0], 0.062), ([0, -0.06, 0], 0.062),
                            ([0, -0.12, 0], 0.062), ([0, -0.17, 0], 0.06)],
        'half_arm_2_link': [([0, 0, 0], 0.058), ([0, 0, -0.07], 0.056),
                            ([0, 0, -0.15], 0.056), ([0, 0, -0.21], 0.056)],
        'forearm_link': [([0, 0, 0], 0.06), ([0, -0.06, 0], 0.058),
                         ([0, -0.12, 0], 0.058), ([0, -0.17, 0], 0.055)],
        'spherical_wrist_1_link': [([0, 0, 0], 0.06), ([0, 0, -0.085], 0.06)],
        'spherical_wrist_2_link': [([0, 0, 0], 0.055), ([0, -0.085, 0], 0.055)],
        'bracelet_link': [([0, 0, -0.045], 0.045), ([0, -0.05, -0.045], 0.045),
                          ([0.045, 0, -0.05], 0.036)],
    }

    @staticmethod
    def _gripper_shell():
        """Sphere shell covering the whole open 2F-85, on the gripper BASE link.

        Stock coverage is one r=0.04 sphere on the base, one thin sphere per
        outer finger and per pad — knuckle housing and finger bodies are
        essentially invisible to cuRobo, which is exactly where the gripper
        was clipping the scene. All spheres go on robotiq_arg2f_base_link:
        its frame is the flange (z toward the fingertips), the gripper is
        rigid in this model (all joints fixed), and the v0.7.8
        self_collision_ignore already exempts base-vs-every-gripper-link and
        base-vs-bracelet, so the shell cannot create phantom self-collisions.
        Rings (not finger-aligned pairs) so coverage holds regardless of the
        gripper's mounting twist; the grasp gap between the fingertips stays
        OPEN so a future grasp goal isn't self-blocked.
        """
        shell = [([0.0, 0.0, 0.055], 0.042),      # knuckle housing core
                 ([0.04, 0.0, 0.06], 0.032), ([-0.04, 0.0, 0.06], 0.032),
                 ([0.0, 0.04, 0.06], 0.032), ([0.0, -0.04, 0.06], 0.032),
                 ([0.0, 0.0, 0.078], 0.035),      # palm / finger roots
                 ([0.045, 0.0, 0.10], 0.026), ([-0.045, 0.0, 0.10], 0.026),
                 ([0.0, 0.045, 0.10], 0.026), ([0.0, -0.045, 0.10], 0.026),
                 ([0.045, 0.0, 0.13], 0.022), ([-0.045, 0.0, 0.13], 0.022),
                 ([0.0, 0.045, 0.13], 0.022), ([0.0, -0.045, 0.13], 0.022),
                 # distal cap: the very fingertips ("barely clipping the tips")
                 ([0.045, 0.0, 0.145], 0.018), ([-0.045, 0.0, 0.145], 0.018),
                 ([0.0, 0.045, 0.145], 0.018), ([0.0, -0.045, 0.145], 0.018)]
        return [{'center': list(c), 'radius': r} for c, r in shell]

    def _world_dict(self, scene, ignore=frozenset()):
        """cuRobo world: obstacles + props (bounding boxes), minus `ignore`.

        Props MUST go in as cuboids: at v0.7.8 Cylinder/Sphere entries in a
        WorldConfig are silently dropped by both collision checkers (only the
        'cuboid' and 'mesh' lists load, and MotionGen never converts).
        """
        from curobo.geom.types import WorldConfig
        from curobo_planner.scene import euler_deg_to_quat
        # 2x: padding is per SIDE. The pedestal is excluded — its box is
        # already margin-sized to stay under the robot's own base sphere,
        # and padding it upward would invalidate every start state.
        pad = 2.0 * float(self.world_padding)
        cuboids = {}
        for o in scene.obstacles:
            x, y, z, w = euler_deg_to_quat(o.rpy_deg)
            p = 0.0 if o.name == 'pedestal' else pad
            cuboids[o.name] = {
                'dims': [d + p for d in o.dims],
                'pose': [o.position[0], o.position[1], o.position[2], w, x, y, z],
            }
        for o in scene.objects:
            if o.name in ignore:
                continue
            x, y, z, w = euler_deg_to_quat(o.rpy_deg)
            # 'obj_' prefix so a prop can share a name with an obstacle
            # (cuRobo fills cache slots positionally; names gate nothing).
            cuboids['obj_' + o.name] = {
                'dims': [d + pad for d in o.bounding_dims()],
                'pose': [o.position[0], o.position[1], o.position[2], w, x, y, z],
            }
        return WorldConfig.from_dict({'cuboid': cuboids})

    def _update_world(self, ignore=frozenset()):
        """Swap the collision world (v0.7.8-safe: guard the silent cases).

        An EMPTY world would silently leave the previous obstacles active
        (load_collision_model returns before its disable line), and MORE
        cuboids than the collision cache raises from inside cuRobo — refuse
        both up front so the world is never half-true.
        """
        world = self._world_dict(self._scene, ignore)
        n = len(world.cuboid)
        if n < 1:
            raise RuntimeError('collision world is empty; refusing to plan')
        if n > int(self.collision_cache_obb):
            raise RuntimeError(
                '%d collision boxes > collision_cache_obb=%d; raise the cache param'
                % (n, self.collision_cache_obb))
        self._motion_gen.update_world(world)
        note = (' (ignoring: %s)' % ', '.join(sorted(ignore))) if ignore else ''
        self.get_logger().info('Collision world: %d boxes%s' % (n, note))

    # ------------------------------------------------------------------ callbacks
    def _joint_state_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = [msg.position[idx[n]] for n in self.joint_names]
        except (KeyError, IndexError):
            return  # not all arm joints present in this message
        v = None
        if msg.velocity is not None and len(msg.velocity) == len(msg.name):
            v = max(abs(msg.velocity[idx[n]]) for n in self.joint_names)
        with self._state_lock:
            self._q_now = q
            self._v_max = v

    def _wait_until_stationary(self, timeout_s=30.0, thresh=0.02,
                               settle_samples=4):
        """Block until the arm is genuinely at rest; False if it never is.

        A plan starts from _q_now; if the arm is still executing the previous
        trajectory, that state is stale by the time the new one publishes —
        the controller jerks onto it (jitter) and the catch-up sweep is
        UNPLANNED, free to pass through obstacles. Chained goto commands hit
        this constantly.

        Two traps (found in adversarial review): the success status publishes
        the moment execution STARTS, when velocity is still ramping from rest
        — so a velocity check alone passes during the first ~200 ms of
        motion. Hold off until the published trajectory's scheduled END
        (node clock: correct under use_sim_time), then require several
        consecutive calm samples. And on timeout, FAIL CLOSED — the caller
        rejects the command; never plan from a moving arm.
        """
        # Fast path: the arm has been at rest since well before this command
        # (no trajectory in flight for 0.3 s+) — no ramp-up window to fear.
        # Still take TWO calm samples one period apart (a single instantaneous
        # read can be a settle-oscillation zero-crossing under clock skew,
        # e.g. a planner missing use_sim_time against a slow sim), and never
        # shortcut when velocity is unreported (v is None).
        with self._state_lock:
            v = self._v_max
            traj_end = self._traj_end
        long_idle = (traj_end is None
                     or self.get_clock().now() > traj_end + Duration(seconds=0.3))
        if long_idle and v is not None and v < thresh:
            time.sleep(0.05)
            with self._state_lock:
                v2 = self._v_max
            if v2 is not None and v2 < thresh:
                return True

        deadline = time.monotonic() + timeout_s
        calm = 0
        warned = False
        while time.monotonic() < deadline:
            if self._stop_requested:
                return False   # voice/CLI stop: abort the segment wait
            with self._state_lock:
                v = self._v_max
                traj_end = self._traj_end
            executing = (traj_end is not None
                         and self.get_clock().now() < traj_end)
            if not executing and (v is None or v < thresh):
                calm += 1
                if calm >= settle_samples:
                    return True
            else:
                calm = 0
                if not warned:
                    self.get_logger().info(
                        'Previous motion still running — waiting before '
                        'planning...')
                    warned = True
            time.sleep(0.05)
        return False

    def _perception_cb(self, msg):
        now = time.monotonic()
        for det in msg.detections:
            if not det.results:
                continue
            label = det.results[0].hypothesis.class_id
            p = det.bbox.center.position
            self._live[label] = ([float(p.x), float(p.y), float(p.z)], now)

    def _apply_live(self, scene):
        """Overwrite fresh-detected props' XY in the loaded scene (Z stays
        YAML: centroid depth biases the vertical, and props slide rather
        than levitate). Returns {name: (dx, dy)} so targets can follow."""
        deltas = {}
        if not self.live_objects:
            return deltas
        now = time.monotonic()
        for o in scene.objects:
            lv = self._live.get(o.name)
            if lv and now - lv[1] < float(self.live_staleness):
                dx = float(lv[0][0]) - o.position[0]
                dy = float(lv[0][1]) - o.position[1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    deltas[o.name] = (dx, dy)
                o.position[0] += dx
                o.position[1] += dy
        return deltas

    def _publish_markers(self):
        arr = scene_markers(self._scene, goal_target_name=self._goal_name,
                            stamp=self.get_clock().now().to_msg())
        self._marker_pub.publish(arr)

    def _status(self, text, error=False):
        # rclpy caches log severity per source LINE, so a single line that picks
        # .error vs .info raises "Logger severity cannot be changed between
        # calls" the first time the other branch fires. Keep them on two lines.
        if error:
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _command_cb(self, msg):
        raw = msg.data.strip()
        if not raw:
            return
        # STOP is handled BEFORE the plan lock so a voice "computuh, stop"
        # lands even while a multi-segment command is executing: hold the arm
        # where it is and abort any segment waits in flight.
        if set(re.findall(r'[a-z]+', raw.lower())) & STOP_WORDS:
            self._stop_requested = True
            self._hold_in_place()
            self._status('STOPPED — holding position. Send a new command '
                         'when ready.')
            return
        if not self._plan_lock.acquire(blocking=False):
            self._status('Busy planning; ignoring "%s".' % raw)
            return
        try:
            self._handle_command(raw)
        except Exception as exc:  # never let a bad command kill the node
            self._status('Command "%s" failed: %s' % (raw, exc), error=True)
        finally:
            self._plan_lock.release()

    def _handle_command(self, raw):
        # Reload the scene each command so live YAML edits take effect, and keep
        # the collision world in sync with what Foxglove shows. The world is
        # (re)built AFTER resolving the command: targets may ignore props.
        self._scene = load_scene(self.scene_file)
        live_deltas = self._apply_live(self._scene)
        if live_deltas:
            self.get_logger().info(
                'Live perception: %s'
                % ', '.join('%s moved %.0f/%.0f mm' % (n, dx * 1000, dy * 1000)
                            for n, (dx, dy) in live_deltas.items()))
        self._stop_requested = False
        if not self._settle_or_hold():
            self._status('Arm would not stop moving (even after hold-in-'
                         'place); command "%s" REJECTED.' % raw, error=True)
            return

        low = raw.lower()
        # NL fallback: anything that isn't a known command resolves through
        # the same token matcher the goto CLI uses — so raw voice text
        # ("go to my bottle") published straight to ~/command just works.
        if (low not in ('home', 'check') and not low.startswith('pose:')
                and self._scene.target(raw) is None):
            resolved = resolve_phrase(raw, self._scene)
            if resolved:
                self.get_logger().info('[nl] "%s" -> %s' % (raw, resolved))
                raw, low = resolved, resolved.lower()
        if low == 'home':
            self._goal_name = None
            if not self._retreat_if_needed():
                return
            self._update_world()
            # Pose-space ONLY. The js path's internal graph fallback engages
            # even within a SINGLE attempt when js-trajopt fails (field log:
            # DT_EXCEPTION with attempts=1) — it cannot be made safe on this
            # Jetson's torch. The pose plan, IK-seeded by retract_config =
            # our home pose, lands in the home family and ran 100% clean
            # across a full field session.
            pos, quat = self._home_fk()
            if pos is not None:
                self._plan_to_pose(pos, quat, label='home', spin=False)
            else:
                self._plan_to_joints(self.home_pose, label='home')
            return
        if low == 'check':
            self._run_check()
            return
        if low.startswith('pose:'):
            nums = low[len('pose:'):].replace(',', ' ').split()
            if len(nums) != 6:
                self._status('pose: needs "x y z roll pitch yaw" (deg).', error=True)
                return
            x, y, z = (float(v) for v in nums[:3])
            quat = _euler_deg_to_wxyz(nums[3:])
            self._goal_name = None
            if not self._retreat_if_needed():
                return
            self._update_world()
            self._plan_to_pose([x, y, z], quat, label='pose(%s)' % ' '.join(nums[:3]))
            return

        target = self._scene.target(raw)
        if target is None:
            self._status(
                'Unknown target "%s". Known: %s' % (raw, ', '.join(self._scene.target_names)),
                error=True)
            return
        self._goal_name = target.name
        ignore = frozenset(target.ignore_objects)
        if not self._retreat_if_needed():
            return
        xyz = list(target.position)
        qx, qy, qz, qw = target.quat_xyzw()
        wxyz = [qw, qx, qy, qz]
        # TASK ADAPTATION: a target follows its reach-for prop's live
        # position — knock the bottle across the island and "go to the
        # bottle" goes to where it actually is.
        for nm in target.ignore_objects:
            if nm in live_deltas:
                dx, dy = live_deltas[nm]
                xyz = [xyz[0] + dx, xyz[1] + dy, xyz[2]]
                self.get_logger().info(
                    'Target %s follows live %s (%+.0f/%+.0f mm)'
                    % (target.name, nm, dx * 1000, dy * 1000))
                break
        if ignore:
            # Two segments: transit to a standoff with the FULL world (the
            # reach-for prop is dodged like everything else), then the final
            # few cm with only that prop exempt. Whole-plan exemption let the
            # wrist sweep straight through the bottle mid-path (check_traj:
            # 39 mm deep) — the exemption must never apply to transit.
            sxyz, swxyz = self._standoff_for(target, xyz, wxyz)
            # auto pull-back standoffs derive from the already-shifted xyz;
            # only EXPLICIT standoff positions need the live shift themselves
            if target.standoff_position:
                for nm in target.ignore_objects:
                    if nm in live_deltas:
                        dx, dy = live_deltas[nm]
                        sxyz = [sxyz[0] + dx, sxyz[1] + dy, sxyz[2]]
                        break
            self._update_world()
            if not self._plan_to_pose(sxyz, swxyz,
                                      label=target.name + ' (standoff)',
                                      publish_status=False):
                return
            if not self._wait_motion_done('approach'):
                return
            self._update_world(ignore)
            if self._plan_to_pose(xyz, wxyz, label=target.name, ignore=ignore):
                self._last_standoff = (sxyz, swxyz)
        else:
            self._update_world()
            self._plan_to_pose(xyz, wxyz, label=target.name)

    @staticmethod
    def _standoff_for(target, xyz, wxyz):
        """The pre-approach pose: explicit from the YAML when the straight
        pull-back exits the reachable workspace (IK-verified per target),
        else `standoff` metres back along the tool axis."""
        if target.standoff_position:
            if target.standoff_rpy_deg:
                swxyz = _euler_deg_to_wxyz(target.standoff_rpy_deg)
            else:
                swxyz = list(wxyz)
            return list(target.standoff_position), swxyz
        axis = _tool_axis(wxyz)
        return ([xyz[k] - axis[k] * float(target.standoff) for k in range(3)],
                list(wxyz))

    def _run_check(self):
        """Dry-plan every target (and its standoff) from the home pose and
        report each verdict in one status — the instant validator for scene
        edits. Nothing is executed and no planner state is disturbed."""
        from curobo.types.math import Pose
        from curobo.types.robot import JointState as CuJointState
        by_name = dict(zip(self.joint_names, self.home_pose))
        qc = [by_name[n] for n in self._curobo_joint_names]
        start = CuJointState.from_position(
            self._torch.tensor([qc], device=self._tensor_args.device,
                               dtype=self._tensor_args.dtype),
            joint_names=list(self._curobo_joint_names))
        lines = []
        for t in self._scene.targets:
            ignore = frozenset(t.ignore_objects)
            xyz = list(t.position)
            qx, qy, qz, qw = t.quat_xyzw()
            wxyz = [qw, qx, qy, qz]
            segs = [(t.name, xyz, wxyz, ignore)]
            if ignore:
                sxyz, swxyz = self._standoff_for(t, xyz, wxyz)
                segs.insert(0, (t.name + '/standoff', sxyz, swxyz, frozenset()))
            for label, pos, quat, ign in segs:
                self._update_world(ign)
                if self.tool_spin_deg:
                    quat = _spin_about_tool(quat, float(self.tool_spin_deg))
                goal = Pose(
                    position=self._torch.tensor(
                        [pos], device=self._tensor_args.device,
                        dtype=self._tensor_args.dtype),
                    quaternion=self._torch.tensor(
                        [quat], device=self._tensor_args.device,
                        dtype=self._tensor_args.dtype))
                res = self._motion_gen.plan_single(start, goal,
                                                   self._plan_config())
                ok = self._plan_ok(res)
                lines.append('%s %s' % (label,
                                        'OK' if ok
                                        else 'FAIL(%s)' % self._result_status(res)))
                self.get_logger().info('check: %s' % lines[-1])
        self._update_world()
        bad = sum(1 for s in lines if 'FAIL' in s)
        self._status('check (%d segments, %d failing): %s'
                     % (len(lines), bad, ' | '.join(lines)), error=bad > 0)

    def _retreat_if_needed(self):
        """Back out through the last prop target's standoff before planning on.

        The arm parked centimetres from (or touching) the prop it reached
        for; a fresh plan with the full world would start in/near collision,
        and a plan that ignores the prop may sweep through it in transit.
        Retreating along the recorded approach first keeps the exemption
        confined to the same few cm it was granted for. Returns False only
        when the command must be aborted (error already published).
        """
        if not self._last_ignore or self._last_standoff is None:
            self._last_standoff = None
            return True
        sxyz, swxyz = self._last_standoff
        ignore = frozenset(self._last_ignore)
        self._last_standoff = None
        self._update_world(ignore)
        ok = self._plan_to_pose(sxyz, swxyz, label='retreat', ignore=ignore,
                                publish_status=False, publish_errors=False)
        if ok:
            if not self._wait_motion_done('retreat'):
                return False
            self._last_ignore = set()
        else:
            self.get_logger().warning(
                'Retreat plan failed; continuing — the start-state backstop '
                'will cover a stuck start.')
        return True

    def _escape_up(self, ignore):
        """Short lift straight up from the CURRENT pose with the touching
        props exempt — the bounded form of 'leave through the object you
        reached for'. Exemption applies to ~10 cm of motion, never transit.
        Returns True once the arm has moved and settled."""
        with self._state_lock:
            q = self._q_now
        if q is None:
            return False
        try:
            by_name = dict(zip(self.joint_names, q))
            qc = [by_name[n] for n in self._curobo_joint_names]
            t = self._torch.tensor([qc], device=self._tensor_args.device,
                                   dtype=self._tensor_args.dtype)
            st = self._motion_gen.kinematics.get_state(t)
            pos = [float(v) for v in st.ee_position[0].tolist()]
            quat = [float(v) for v in st.ee_quaternion[0].tolist()]
        except Exception as exc:
            self.get_logger().error('escape FK failed: %s' % exc)
            return False
        pos[2] += 0.10
        self._update_world(ignore)
        # check_start=False: the start IS the problem — trajopt's collision
        # cost pushes the arm out of shallow (padded-ghost) penetration.
        ok = self._plan_to_pose(pos, quat, label='escape', ignore=ignore,
                                spin=False, publish_status=False,
                                publish_errors=False, allow_escape=False,
                                check_start=False)
        return ok and self._wait_motion_done('escape')

    def _back_out(self, points=12, speed=0.5):
        """Deep-contact last resort: replay the tail of the LAST executed
        trajectory in REVERSE — retracing a path that was collision-legal
        inbound — to back the arm out of whatever it ended up against."""
        msg = self._last_traj
        if msg is None or not msg.points or not self.execute:
            return False
        # Retrace from the waypoint NEAREST the arm's actual position: a
        # mid-path wedge means the tail was never reached, and pulling
        # toward it drags the arm further into whatever it hit.
        with self._state_lock:
            q_ref = self._q_now
        idx = len(msg.points) - 1
        if q_ref is not None and msg.joint_names == list(self.joint_names):
            dists = [max(abs(a - b) for a, b in zip(p.positions, q_ref))
                     for p in msg.points]
            idx = dists.index(min(dists))
        lo = max(0, idx - int(points))
        tail = list(msg.points[lo:idx + 1])[::-1]
        out = JointTrajectory()
        out.joint_names = list(msg.joint_names)
        # START FROM WHERE THE ARM ACTUALLY IS: if execution was deflected,
        # the recorded waypoints are far from the real pose and replaying
        # them directly would command another unplanned jump (field bug).
        with self._state_lock:
            q_now = self._q_now
        prev = None
        t = 0.0
        if q_now is not None and msg.joint_names == list(self.joint_names):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q_now]
            t = 0.15
            p.time_from_start = Duration(seconds=t).to_msg()
            out.points.append(p)
            prev = list(p.positions)
        for src in tail:
            p = JointTrajectoryPoint()
            p.positions = list(src.positions)
            if prev is not None:
                d = max(abs(a - b) for a, b in zip(prev, p.positions))
                t += max(0.08, d / float(speed))
            else:
                t = 0.3
            p.time_from_start = Duration(seconds=t).to_msg()
            out.points.append(p)
            prev = list(p.positions)
        self.get_logger().warning(
            'Backing out along the last trajectory (%d points).' % len(tail))
        self._jtc_pub.publish(out)
        end = self.get_clock().now() + Duration(seconds=t + 0.3)
        with self._state_lock:
            self._traj_end = end
        return self._wait_until_stationary(timeout_s=20.0)

    def _hold_in_place(self):
        """Command the JTC to hold the CURRENT joint positions.

        A physically blocked trajectory leaves the controller pushing toward
        an unreachable setpoint forever: perpetual contact force, velocity
        that never settles, every future command rejected by the stationary
        gate — a deadlock only recoverable by re-targeting the controller at
        where the arm actually IS. No new motion is commanded.
        """
        with self._state_lock:
            q = self._q_now
        if q is None or not self.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start = Duration(seconds=0.3).to_msg()
        msg.points = [pt]
        self._jtc_pub.publish(msg)
        end = self.get_clock().now() + Duration(seconds=0.55)
        with self._state_lock:
            self._traj_end = end

    def _settle_or_hold(self, timeout_s=30.0):
        """Wait for rest; if the arm won't settle (wedged, controller still
        fighting a contact), hold-in-place and give it one more chance."""
        if self._wait_until_stationary(timeout_s=timeout_s):
            return True
        self.get_logger().warning(
            'Arm will not settle — commanding hold-in-place (a blocked '
            'trajectory leaves the controller fighting a contact).')
        self._hold_in_place()
        return self._wait_until_stationary(timeout_s=6.0)

    def _wait_motion_done(self, what):
        if not self.execute:
            return True
        if not self._settle_or_hold(timeout_s=45.0):
            self._status('Arm did not finish the %s motion; command aborted.'
                         % what, error=True)
            return False
        # Arrival check: a settled arm FAR from the commanded endpoint means
        # the controller lost the trajectory (physical contact or actuator
        # saturation) — name it now, not three failed commands later.
        if self._last_traj is not None and self._last_traj.points:
            goal = self._last_traj.points[-1].positions
            with self._state_lock:
                q = self._q_now
            if q is not None and len(goal) == len(q):
                err = max(abs(a - b) for a, b in zip(goal, q))
                if err > 0.08:
                    self.get_logger().error(
                        'TRACKING FAILURE: arm settled %.2f rad from the '
                        'commanded endpoint — physical contact or controller '
                        'saturation (velocity_scale too high?).' % err)
        return True

    def _home_fk(self):
        """The home joint pose's tool pose in cuRobo's frame (cached)."""
        if self._home_pose_fk is None:
            try:
                by_name = dict(zip(self.joint_names, self.home_pose))
                qc = [by_name[n] for n in self._curobo_joint_names]
                t = self._torch.tensor([qc], device=self._tensor_args.device,
                                       dtype=self._tensor_args.dtype)
                st = self._motion_gen.kinematics.get_state(t)
                self._home_pose_fk = (
                    [float(v) for v in st.ee_position[0].tolist()],
                    [float(v) for v in st.ee_quaternion[0].tolist()])  # wxyz
            except Exception as exc:
                self.get_logger().error('home FK failed: %s' % exc)
                self._home_pose_fk = (None, None)
        return self._home_pose_fk

    # -------------------------------------------------------------------- planning
    def _start_state(self):
        from curobo.types.robot import JointState as CuJointState
        with self._state_lock:
            q = self._q_now
        if q is None:
            return None
        # Reorder from controller order into cuRobo's cspace order (see init).
        by_name = dict(zip(self.joint_names, q))
        q_curobo = [by_name[n] for n in self._curobo_joint_names]
        t = self._torch.tensor([q_curobo], device=self._tensor_args.device,
                               dtype=self._tensor_args.dtype)
        return CuJointState.from_position(t, joint_names=list(self._curobo_joint_names))

    def _plan_to_pose(self, xyz, wxyz, label, ignore=frozenset(), spin=True,
                      publish_status=True, publish_errors=True,
                      allow_escape=True, check_start=True):
        from curobo.types.math import Pose
        start = self._start_state()
        if start is None:
            self._status('No joint_states yet; is the arm bringup running?', error=True)
            return False
        if spin and self.tool_spin_deg:
            wxyz = _spin_about_tool(wxyz, float(self.tool_spin_deg))
        goal = Pose(
            position=self._torch.tensor([xyz], device=self._tensor_args.device,
                                        dtype=self._tensor_args.dtype),
            quaternion=self._torch.tensor([wxyz], device=self._tensor_args.device,
                                          dtype=self._tensor_args.dtype),
        )

        def plan_fn(cfg):
            # Fetch the start FRESH per attempt: an escape motion may have
            # moved the arm between the first try and the re-plan.
            s = self._start_state()
            return None if s is None else self._motion_gen.plan_single(s, goal, cfg)

        return self._run_plan(plan_fn, label, ignore,
                              publish_status=publish_status,
                              publish_errors=publish_errors,
                              allow_escape=allow_escape,
                              check_start=check_start)

    def _plan_to_joints(self, positions, label, ignore=frozenset(),
                        publish_status=True, publish_errors=True,
                        attempts=None):
        """Collision-checked plan to a joint configuration (used for 'home')."""
        from curobo.types.robot import JointState as CuJointState
        start = self._start_state()
        if start is None:
            self._status('No joint_states yet; is the arm bringup running?', error=True)
            return False
        by_name = dict(zip(self.joint_names, positions))
        q = [by_name[n] for n in self._curobo_joint_names]  # cspace order, by name
        goal = CuJointState.from_position(
            self._torch.tensor([q], device=self._tensor_args.device,
                               dtype=self._tensor_args.dtype),
            joint_names=list(self._curobo_joint_names))

        def plan_fn(cfg):
            s = self._start_state()
            return None if s is None else self._motion_gen.plan_single_js(s, goal, cfg)

        return self._run_plan(plan_fn, label, ignore,
                              publish_status=publish_status,
                              publish_errors=publish_errors,
                              attempts=attempts)

    def _plan_config(self, attempts=None, check_start=True):
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        # enable_graph_attempt=None: cuRobo silently ENABLES the graph (PRM)
        # planner after 3 failed attempts — and the graph planner needs
        # torch.svd, which this Jetson torch wheel cannot run (missing
        # cusolverDnXsyevBatched -> TorchScript crash, DT_EXCEPTION). Same
        # reason graph warmup is off. Never let it auto-engage.
        kw = {}
        if not self.enable_graph:
            kw['enable_graph_attempt'] = None
        if not check_start:
            # Escape motions only: the start IS in (shallow, usually padded-
            # ghost) collision; trajopt's collision cost pushes it out.
            kw['check_start_validity'] = False
        return MotionGenPlanConfig(
            max_attempts=int(attempts or self.max_attempts),
            enable_graph=bool(self.enable_graph),
            enable_finetune_trajopt=bool(self.enable_finetune),
            finetune_attempts=int(self.finetune_attempts), **kw)

    def _find_touching(self):
        """(prop names, furniture names, worst-hit detail) at the current pose.

        cuRobo says INVALID_START_STATE_WORLD_COLLISION without naming the
        obstacle. Recompute it: FK the current joints into the robot's own
        collision spheres (get_robot_as_spheres, verified v0.7.8 API) and
        test them against the scene boxes. Props can be ignored away for one
        departure plan; furniture cannot. The detail string pins down the
        exact sphere and depth so a surprising verdict can be audited.
        """
        from curobo_planner.scene import euler_deg_to_quat
        with self._state_lock:
            q = self._q_now
        if q is None:
            return set(), set(), ''
        by_name = dict(zip(self.joint_names, q))
        qc = [by_name[n] for n in self._curobo_joint_names]
        t = self._torch.tensor([qc], device=self._tensor_args.device,
                               dtype=self._tensor_args.dtype)
        spheres = self._motion_gen.kinematics.get_robot_as_spheres(t)[0]
        pts = [(list(s.pose[:3]), float(s.radius)) for s in spheres
               if float(s.radius) > 0.0]
        worst = ['', 0.0]

        def hits_box(name, center, rpy_deg, dims):
            x, y, z, w = euler_deg_to_quat(rpy_deg)
            half = [d / 2.0 for d in dims]
            hit = False
            for p, r in pts:
                dx = [p[k] - center[k] for k in range(3)]
                # rotate into the box frame: v = R^T d  (conjugate quat)
                cx, cy, cz = -x, -y, -z
                uvx = cy * dx[2] - cz * dx[1]
                uvy = cz * dx[0] - cx * dx[2]
                uvz = cx * dx[1] - cy * dx[0]
                v = [dx[0] + 2 * (w * uvx + cy * uvz - cz * uvy),
                     dx[1] + 2 * (w * uvy + cz * uvx - cx * uvz),
                     dx[2] + 2 * (w * uvz + cx * uvy - cy * uvx)]
                d2 = sum(max(abs(v[k]) - half[k], 0.0) ** 2 for k in range(3))
                pen = (r + 0.008) - math.sqrt(d2)   # buffer 0.005 + slack
                if pen > 0:
                    hit = True
                    if pen > worst[1]:
                        worst[0] = ('sphere r=%.3f at [%.3f %.3f %.3f] into '
                                    '%s by %.0f mm'
                                    % (r, p[0], p[1], p[2], name, pen * 1000))
                        worst[1] = pen
            return hit

        props = {o.name for o in self._scene.objects
                 if hits_box(o.name, o.position, o.rpy_deg, o.bounding_dims())}
        furniture = {o.name for o in self._scene.obstacles
                     if hits_box(o.name, o.position, o.rpy_deg, o.dims)}
        return props, furniture, worst[0]

    def _run_plan(self, plan_fn, label, ignore, publish_status=True,
                  publish_errors=True, allow_escape=True, attempts=None,
                  check_start=True):
        """Plan + execute one segment; returns True on success.

        Success status is published only for the FINAL segment of a command
        (publish_status) — the goto CLI treats the first status as the
        command's terminal reply. Errors publish unless the caller has its
        own fallback (publish_errors=False).
        """
        # Log-only (NOT a status publish) so the ~/status topic carries only the
        # terminal result — the goto CLI waits for that.
        self.get_logger().info('Planning to %s...' % label)
        t0 = time.monotonic()
        used = set(ignore)   # the ignore set the EXECUTED plan actually ran with
        result = plan_fn(self._plan_config(attempts, check_start))
        status = self._result_status(result)
        # Enum may stringify as its NAME or its value ("Invalid Start State:
        # World Collision") — normalize before matching.
        if not self._plan_ok(result) and 'INVALID_START' in status.upper().replace(' ', '_'):
            # Backstop: the arm is parked against something. The old backstop
            # exempted the touched props for the WHOLE next plan — with 1 cm
            # world padding that fired constantly, so transit plans ran with
            # objects removed ("practically ignoring the scene"). The escape
            # is BOUNDED instead: lift 10 cm from the current pose with only
            # the touching props exempt, then re-plan with the full world.
            touching, furniture, detail = self._find_touching()
            escaped = False
            if allow_escape:
                escape_ignore = frozenset(set(ignore) | self._last_ignore | touching)
                self.get_logger().warning(
                    'Start in collision (touching: %s%s) — escaping upward, '
                    'then replanning with the full world.'
                    % (', '.join(sorted(touching | furniture)) or 'padded ghost only',
                       (' | ' + detail) if detail else ''))
                escaped = self._escape_up(escape_ignore)
                if not escaped:
                    # Deep contact defeats even a start-check-free escape
                    # plan. Model-free last resort: retrace the tail of the
                    # last executed trajectory — the path was legal inbound.
                    escaped = self._back_out()
            if escaped:
                self._update_world(frozenset(ignore))
                result = plan_fn(self._plan_config(attempts))
                status = self._result_status(result)
            elif furniture:
                text = ('Arm start state rejected: in collision with %s '
                        '[%s] — jog it clear (joystick) or edit scene.yaml.'
                        % (', '.join(sorted(furniture)), detail))
                if publish_errors:
                    self._status(text, error=True)
                else:
                    self.get_logger().error(text)
                return False
        if not self._plan_ok(result):
            s = status.upper().replace(' ', '_')
            if 'IK' in s:
                hint = ('no collision-free joint solution AT the goal — move the '
                        'target away from obstacles or relax rpy_deg in scene.yaml. '
                        'Remember: the target is the FINGERTIP midpoint; the flange '
                        'sits 12 cm behind it along the tool axis.')
            elif 'INVALID_START' in s:
                props, furniture, detail = self._find_touching()
                touching = ', '.join(sorted(props | furniture)) or 'nothing detected'
                hint = ("the arm's CURRENT pose collides with the world "
                        '(touching: %s%s) — send the target whose '
                        'ignore_objects covers it, or jog clear.'
                        % (touching, (' | ' + detail) if detail else ''))
            elif 'FINETUNE' in s:
                hint = ('a collision-free path WAS found but retiming it failed '
                        '(goal near a kinematic limit) — raise finetune_attempts '
                        'or move the target slightly.')
            elif 'TRAJOPT' in s:
                hint = ('the goal is reachable but no collision-free PATH was found — '
                        'clear the approach corridor or try an intermediate target.')
            else:
                hint = 'Try tuning the target pose in scene.yaml.'
            text = 'Plan to %s FAILED (%s). %s' % (label, status, hint)
            if publish_errors:
                self._status(text, error=True)
            else:
                self.get_logger().error(text)
            return False
        traj = result.get_interpolated_plan()
        traj = traj.get_ordered_joint_state(self.joint_names)  # remap to controller order
        dt = float(result.interpolation_dt)
        # NOTE: publish ONLY the trimmed interpolated plan. An earlier
        # attempt appended optimized_plan's last point as an "exact
        # endpoint", but cuRobo's result buffers are PADDED beyond the
        # actual solution (why interpolated_plan needs trim_trajectory) and
        # the untrimmed tail can be stale garbage from a previous plan —
        # field-correlated with two violent-motion incidents. The
        # settles-slightly-short case is handled by the bounded escape.
        self._publish_trajectory(traj, dt)
        self._last_ignore = used
        n = traj.position.shape[0]
        try:
            tp = traj.position
            travel = float((tp.max(dim=0).values - tp.min(dim=0).values).max())
        except Exception:
            travel = float('nan')
        text = ('Planned to %s: %d points, %.1fs (plan %.2fs, travel %.1f rad)%s.'
                % (label, n, n * dt, time.monotonic() - t0, travel,
                   '' if self.execute else ' (execute=false)'))
        if publish_status:
            self._status(text)
        else:
            # Interim segment ('...' prefix): the goto CLI prints it and
            # keeps waiting for the terminal status — a multi-segment
            # command's reply can be a minute away while the arm moves.
            self._status('... ' + text)
        return True

    @staticmethod
    def _plan_ok(result):
        return result is not None and bool(result.success.item())

    @staticmethod
    def _result_status(result):
        if result is None:
            return 'no result'
        return str(getattr(result, 'status', None) or 'unknown')

    def _publish_trajectory(self, traj, dt):
        if not self.execute:
            return
        pos = traj.position.cpu().numpy()
        vel = traj.velocity.cpu().numpy() if traj.velocity is not None else None
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        for k in range(pos.shape[0]):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in pos[k]]
            if vel is not None:
                p.velocities = [float(v) for v in vel[k]]
            p.time_from_start = Duration(seconds=(k + 1) * dt).to_msg()
            msg.points.append(p)
        end_t = pos.shape[0] * dt
        self._jtc_pub.publish(msg)
        self._last_traj = msg   # for _back_out()
        # Scheduled end of this motion (+ a settle margin), in node-clock
        # time so use_sim_time slowdowns are handled: the stationary gate
        # must not trust velocity readings before this.
        end = self.get_clock().now() + Duration(seconds=end_t + 0.25)
        with self._state_lock:
            self._traj_end = end


def _euler_deg_to_wxyz(rpy_deg_strs):
    from curobo_planner.scene import euler_deg_to_quat
    x, y, z, w = euler_deg_to_quat([float(v) for v in rpy_deg_strs])
    return [w, x, y, z]


def _spin_about_tool(wxyz, deg):
    """q ⊗ Rz(deg): spin a goal about its own tool (z) axis."""
    half = math.radians(deg) / 2.0
    sw, sz = math.cos(half), math.sin(half)
    w1, x1, y1, z1 = wxyz
    return [w1 * sw - z1 * sz,
            x1 * sw + y1 * sz,
            y1 * sw - x1 * sz,
            z1 * sw + w1 * sz]


def _tool_axis(wxyz):
    """The tool (z) axis of a wxyz orientation: R @ [0,0,1]."""
    w, x, y, z = wxyz
    return [2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)]


def main(args=None):
    rclpy.init(args=args)
    node = CuroboPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
