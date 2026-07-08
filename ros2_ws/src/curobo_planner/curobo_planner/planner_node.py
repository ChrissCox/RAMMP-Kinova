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

from curobo_planner.scene import load_scene, scene_markers


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
        self.interpolation_dt = self.declare_parameter('interpolation_dt', 0.04).value
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
        # Stock kinova_gen3.yml fingertip pads carry a single r=0.01 sphere each
        # — thinner than the real 2F-85 fingertip, so "collision-free" paths
        # clip props in MuJoCo. Inflate to this radius (0.01 restores stock).
        self.pad_sphere_radius = self.declare_parameter('pad_sphere_radius', 0.02).value
        # Full gripper collision shell (see _gripper_shell): the stock model
        # leaves the knuckle housing bare and the fingers nearly so.
        self.gripper_shell = self.declare_parameter('gripper_shell', True).value
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

        self.create_subscription(
            JointState, self.joint_states_topic, self._joint_state_cb, 10,
            callback_group=self._cb)
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
            "cuRobo planner ready. Send targets to '%s/command': %s | home | 'pose: x y z r p yaw'"
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
        cfg = MotionGenConfig.load_from_robot_config(
            self._load_robot_config(),
            self._world_dict(self._scene),
            tensor_args=self._tensor_args,
            interpolation_dt=self.interpolation_dt,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={'obb': int(self.collision_cache_obb), 'mesh': 10},
            collision_activation_distance=float(self.collision_activation_distance),
        )
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
        return cfg

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
        cuboids = {}
        for o in scene.obstacles:
            x, y, z, w = euler_deg_to_quat(o.rpy_deg)
            cuboids[o.name] = {
                'dims': list(o.dims),
                'pose': [o.position[0], o.position[1], o.position[2], w, x, y, z],
            }
        for o in scene.objects:
            if o.name in ignore:
                continue
            x, y, z, w = euler_deg_to_quat(o.rpy_deg)
            # 'obj_' prefix so a prop can share a name with an obstacle
            # (cuRobo fills cache slots positionally; names gate nothing).
            cuboids['obj_' + o.name] = {
                'dims': o.bounding_dims(),
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
        if not self._wait_until_stationary():
            self._status('Arm would not stop moving; command "%s" REJECTED — '
                         'resend once it settles.' % raw, error=True)
            return

        low = raw.lower()
        if low == 'home':
            self._goal_name = None
            self._update_world()
            self._plan_to_joints(self.home_pose, label='home')
            return
        if low.startswith('pose:'):
            nums = low[len('pose:'):].replace(',', ' ').split()
            if len(nums) != 6:
                self._status('pose: needs "x y z roll pitch yaw" (deg).', error=True)
                return
            x, y, z = (float(v) for v in nums[:3])
            quat = _euler_deg_to_wxyz(nums[3:])
            self._goal_name = None
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
        x, y, z = target.position
        qx, qy, qz, qw = target.quat_xyzw()
        self._update_world(ignore)
        self._plan_to_pose([x, y, z], [qw, qx, qy, qz], label=target.name,
                           ignore=ignore)

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

    def _plan_to_pose(self, xyz, wxyz, label, ignore=frozenset()):
        from curobo.types.math import Pose
        start = self._start_state()
        if start is None:
            self._status('No joint_states yet; is the arm bringup running?', error=True)
            return
        if self.tool_spin_deg:
            wxyz = _spin_about_tool(wxyz, float(self.tool_spin_deg))
        goal = Pose(
            position=self._torch.tensor([xyz], device=self._tensor_args.device,
                                        dtype=self._tensor_args.dtype),
            quaternion=self._torch.tensor([wxyz], device=self._tensor_args.device,
                                          dtype=self._tensor_args.dtype),
        )
        self._run_plan(lambda cfg: self._motion_gen.plan_single(start, goal, cfg),
                       label, ignore)

    def _plan_to_joints(self, positions, label, ignore=frozenset()):
        """Collision-checked plan to a joint configuration (used for 'home')."""
        from curobo.types.robot import JointState as CuJointState
        start = self._start_state()
        if start is None:
            self._status('No joint_states yet; is the arm bringup running?', error=True)
            return
        by_name = dict(zip(self.joint_names, positions))
        q = [by_name[n] for n in self._curobo_joint_names]  # cspace order, by name
        goal = CuJointState.from_position(
            self._torch.tensor([q], device=self._tensor_args.device,
                               dtype=self._tensor_args.dtype),
            joint_names=list(self._curobo_joint_names))
        self._run_plan(lambda cfg: self._motion_gen.plan_single_js(start, goal, cfg),
                       label, ignore)

    def _plan_config(self):
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        # enable_graph_attempt=None: cuRobo silently ENABLES the graph (PRM)
        # planner after 3 failed attempts — and the graph planner needs
        # torch.svd, which this Jetson torch wheel cannot run (missing
        # cusolverDnXsyevBatched -> TorchScript crash, DT_EXCEPTION). Same
        # reason graph warmup is off. Never let it auto-engage.
        kw = {}
        if not self.enable_graph:
            kw['enable_graph_attempt'] = None
        return MotionGenPlanConfig(
            max_attempts=int(self.max_attempts),
            enable_graph=bool(self.enable_graph),
            enable_finetune_trajopt=bool(self.enable_finetune),
            finetune_attempts=int(self.finetune_attempts), **kw)

    def _find_touching(self):
        """(prop names, furniture names) the robot currently penetrates.

        cuRobo says INVALID_START_STATE_WORLD_COLLISION without naming the
        obstacle. Recompute it: FK the current joints into the robot's own
        collision spheres (get_robot_as_spheres, verified v0.7.8 API) and
        test them against the scene boxes. Props can be ignored away for one
        departure plan; furniture cannot.
        """
        from curobo_planner.scene import euler_deg_to_quat
        with self._state_lock:
            q = self._q_now
        if q is None:
            return set(), set()
        by_name = dict(zip(self.joint_names, q))
        qc = [by_name[n] for n in self._curobo_joint_names]
        t = self._torch.tensor([qc], device=self._tensor_args.device,
                               dtype=self._tensor_args.dtype)
        spheres = self._motion_gen.kinematics.get_robot_as_spheres(t)[0]
        pts = [(list(s.pose[:3]), float(s.radius)) for s in spheres
               if float(s.radius) > 0.0]

        def hits_box(center, rpy_deg, dims):
            x, y, z, w = euler_deg_to_quat(rpy_deg)
            half = [d / 2.0 for d in dims]
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
                if d2 < (r + 0.008) ** 2:   # sphere_buffer 0.005 + slack
                    return True
            return False

        props = {o.name for o in self._scene.objects
                 if hits_box(o.position, o.rpy_deg, o.bounding_dims())}
        furniture = {o.name for o in self._scene.obstacles
                     if hits_box(o.position, o.rpy_deg, o.dims)}
        return props, furniture

    def _run_plan(self, plan_fn, label, ignore):
        # Log-only (NOT a status publish) so the ~/status topic carries only the
        # terminal result — the goto CLI waits for that.
        self.get_logger().info('Planning to %s...' % label)
        t0 = time.monotonic()
        used = set(ignore)   # the ignore set the EXECUTED plan actually ran with
        result = plan_fn(self._plan_config())
        status = self._result_status(result)
        # Enum may stringify as its NAME or its value ("Invalid Start State:
        # World Collision") — normalize before matching.
        if not self._plan_ok(result) and 'INVALID_START' in status.upper().replace(' ', '_'):
            # The arm is parked against something — usually the prop the LAST
            # plan was allowed to ignore (hovering at the bottle), sometimes a
            # prop it merely settled against. Work out exactly WHICH props the
            # current pose penetrates and retry once ignoring those too;
            # leaving through the object you just reached for is fine.
            touching, furniture = self._find_touching()
            extra = (self._last_ignore | touching) - set(ignore)
            if extra:
                merged = frozenset(set(ignore) | extra)
                self.get_logger().warning(
                    'Start state in collision; retrying, also ignoring: %s'
                    % ', '.join(sorted(extra)))
                self._update_world(merged)
                used = set(merged)
                result = plan_fn(self._plan_config())
                status = self._result_status(result)
            elif furniture:
                self._status('Arm is in collision with %s — jog it clear '
                             '(joystick) or edit scene.yaml.'
                             % ', '.join(sorted(furniture)), error=True)
                return
        if not self._plan_ok(result):
            s = status.upper().replace(' ', '_')
            if 'IK' in s:
                hint = ('no collision-free joint solution AT the goal — move the '
                        'target away from obstacles or relax rpy_deg in scene.yaml. '
                        'Remember: the target is the FINGERTIP midpoint; the flange '
                        'sits 12 cm behind it along the tool axis.')
            elif 'INVALID_START' in s:
                hint = ("the arm's CURRENT pose collides with the world — send the "
                        'target whose ignore_objects covers the prop it is touching.')
            elif 'FINETUNE' in s:
                hint = ('a collision-free path WAS found but retiming it failed '
                        '(goal near a kinematic limit) — raise finetune_attempts '
                        'or move the target slightly.')
            elif 'TRAJOPT' in s:
                hint = ('the goal is reachable but no collision-free PATH was found — '
                        'clear the approach corridor or try an intermediate target.')
            else:
                hint = 'Try tuning the target pose in scene.yaml.'
            self._status('Plan to %s FAILED (%s). %s' % (label, status, hint),
                         error=True)
            return
        traj = result.get_interpolated_plan()
        traj = traj.get_ordered_joint_state(self.joint_names)  # remap to controller order
        dt = float(result.interpolation_dt)
        self._publish_trajectory(traj, dt)
        self._last_ignore = used
        n = traj.position.shape[0]
        self._status('Planned to %s: %d points, %.1fs (plan %.2fs)%s.'
                     % (label, n, n * dt, time.monotonic() - t0,
                        '' if self.execute else ' (execute=false)'))

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
        self._jtc_pub.publish(msg)
        # Scheduled end of this motion (+ a settle margin), in node-clock
        # time so use_sim_time slowdowns are handled: the stationary gate
        # must not trust velocity readings before this.
        end = self.get_clock().now() + Duration(
            seconds=pos.shape[0] * dt + 0.25)
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
