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

Conventions that bite (handled here):
  * cuRobo Pose quaternion is [w, x, y, z]; ROS/scene is [x, y, z, w].
  * cuRobo joint order follows its config, not the controller's — remap by NAME.
  * CollisionCheckerType.MESH handles cuboids AND meshes; PRIMITIVE ignores meshes.
  * warmup() once (the first plan is otherwise many seconds).
  * The executed trajectory is uniform at interpolation_dt; time_from_start = (k+1)*dt.
"""

import math
import threading

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
        # Collision-cache headroom so live-editing scene.yaml can ADD obstacles
        # (cuRobo's update_world is only safe up to this many boxes).
        self.collision_cache_obb = self.declare_parameter('collision_cache_obb', 40).value
        # Kinova "Home" (bent-elbow) — well-conditioned; also the safe default start.
        self.home_pose = list(self.declare_parameter(
            'home_pose_rad', [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
        ).value)
        self.home_time_s = self.declare_parameter('home_time_s', 6.0).value

        if not self.scene_file:
            raise RuntimeError('scene_file parameter is required')

        self._state_lock = threading.Lock()
        self._q_now = None            # latest arm joints (controller order), or None
        self._plan_lock = threading.Lock()
        self._scene = load_scene(self.scene_file)
        self._goal_name = None        # last goal, for marker highlighting

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
        from curobo.types.base import TensorDeviceType
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        self._torch = torch
        self._tensor_args = TensorDeviceType()
        self.get_logger().info('Loading cuRobo MotionGen (%s)...' % self.robot_config)
        cfg = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            self._world_dict(self._scene),
            tensor_args=self._tensor_args,
            interpolation_dt=self.interpolation_dt,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={'obb': int(self.collision_cache_obb), 'mesh': 10},
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
        self._motion_gen.warmup()
        self.get_logger().info('cuRobo warmup complete.')

    def _world_dict(self, scene):
        """cuRobo world from the scene's obstacle boxes (cuboid dict form)."""
        from curobo.geom.types import WorldConfig
        from curobo.geom.sdf.world import CollisionCheckerType  # noqa: F401
        from curobo_planner.scene import euler_deg_to_quat
        cuboids = {}
        for o in scene.obstacles:
            x, y, z, w = euler_deg_to_quat(o.rpy_deg)
            cuboids[o.name] = {
                'dims': list(o.dims),
                'pose': [o.position[0], o.position[1], o.position[2], w, x, y, z],
            }
        return WorldConfig.from_dict({'cuboid': cuboids})

    # ------------------------------------------------------------------ callbacks
    def _joint_state_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = [msg.position[idx[n]] for n in self.joint_names]
        except (KeyError, IndexError):
            return  # not all arm joints present in this message
        with self._state_lock:
            self._q_now = q

    def _publish_markers(self):
        arr = scene_markers(self._scene, goal_target_name=self._goal_name,
                            stamp=self.get_clock().now().to_msg())
        self._marker_pub.publish(arr)

    def _status(self, text, error=False):
        (self.get_logger().error if error else self.get_logger().info)(text)
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
        # the collision world in sync with what Foxglove shows.
        self._scene = load_scene(self.scene_file)
        self._motion_gen.update_world(self._world_dict(self._scene))

        low = raw.lower()
        if low == 'home':
            self._goal_name = None
            self._execute_joint_goal(self.home_pose, self.home_time_s)
            self._status('Moved to home pose.')
            return
        if low.startswith('pose:'):
            nums = low[len('pose:'):].replace(',', ' ').split()
            if len(nums) != 6:
                self._status('pose: needs "x y z roll pitch yaw" (deg).', error=True)
                return
            x, y, z = (float(v) for v in nums[:3])
            quat = _euler_deg_to_wxyz(nums[3:])
            self._goal_name = None
            self._plan_to_pose([x, y, z], quat, label='pose(%s)' % ' '.join(nums[:3]))
            return

        target = self._scene.target(raw)
        if target is None:
            self._status(
                'Unknown target "%s". Known: %s' % (raw, ', '.join(self._scene.target_names)),
                error=True)
            return
        self._goal_name = target.name
        x, y, z = target.position
        qx, qy, qz, qw = target.quat_xyzw()
        self._plan_to_pose([x, y, z], [qw, qx, qy, qz], label=target.name)

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

    def _plan_to_pose(self, xyz, wxyz, label):
        from curobo.types.math import Pose
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        start = self._start_state()
        if start is None:
            self._status('No joint_states yet; is the arm bringup running?', error=True)
            return
        goal = Pose(
            position=self._torch.tensor([xyz], device=self._tensor_args.device,
                                        dtype=self._tensor_args.dtype),
            quaternion=self._torch.tensor([wxyz], device=self._tensor_args.device,
                                          dtype=self._tensor_args.dtype),
        )
        # Log-only (NOT a status publish) so the ~/status topic carries only the
        # terminal result — the goto CLI waits for that.
        self.get_logger().info('Planning to %s...' % label)
        result = self._motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(max_attempts=int(self.max_attempts)))
        if result is None or not bool(result.success.item()):
            status = getattr(result, 'status', 'unknown') if result is not None else 'no result'
            self._status('Plan to %s FAILED (%s). Try tuning the target pose in scene.yaml.'
                         % (label, status), error=True)
            return
        traj = result.get_interpolated_plan()
        traj = traj.get_ordered_joint_state(self.joint_names)  # remap to controller order
        dt = float(result.interpolation_dt)
        self._publish_trajectory(traj, dt)
        n = traj.position.shape[0]
        self._status('Planned to %s: %d points, %.1fs%s.'
                     % (label, n, n * dt, '' if self.execute else ' (execute=false)'))

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

    def _execute_joint_goal(self, positions, time_s):
        """Single-point trajectory (used for 'home'); bypasses cuRobo."""
        if not self.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in positions]
        pt.time_from_start = Duration(seconds=time_s).to_msg()
        msg.points = [pt]
        self._jtc_pub.publish(msg)


def _euler_deg_to_wxyz(rpy_deg_strs):
    from curobo_planner.scene import euler_deg_to_quat
    x, y, z, w = euler_deg_to_quat([float(v) for v in rpy_deg_strs])
    return [w, x, y, z]


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
