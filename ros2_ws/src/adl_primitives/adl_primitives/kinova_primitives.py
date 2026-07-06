"""Basic Kinova Gen3 motion primitives for the RAMMP ADL project.

Commands the arm through ros2_kortex's ros2_control interfaces:
  * joints  -> control_msgs/action/FollowJointTrajectory
  * gripper -> control_msgs/action/GripperCommand
and provides a software e-stop (``~/estop``, std_srvs/srv/Trigger).

SAFETY: ``dry_run`` defaults to True (nothing moves). The software e-stop is a
convenience, NOT a substitute for the hardware E-stop -- the Gen3 has no brakes.
"""

import math
import threading
import time
import xml.etree.ElementTree as ET

import numpy
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node

from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _quat_rotate(q, v):
    """Rotate 3-vector ``v`` by quaternion ``q`` = (x, y, z, w)."""
    x, y, z, w = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


class KinovaPrimitives(Node):
    """A small library of basic, safe Kinova Gen3 motion primitives."""

    def __init__(self, node_name='test_arm'):
        super().__init__(node_name)

        # -- Parameters (verify names/limits on the Jetson; see config/test_arm.yaml) --
        self.joint_names = self.declare_parameter(
            'joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7'],
        ).value
        self.jtc_action_name = self.declare_parameter(
            'joint_trajectory_action', '/joint_trajectory_controller/follow_joint_trajectory'
        ).value
        self.gripper_action_name = self.declare_parameter(
            'gripper_action', '/robotiq_gripper_controller/gripper_cmd'
        ).value
        self.joint_states_topic = self.declare_parameter(
            'joint_states_topic', '/joint_states'
        ).value
        self.switch_controller_service = self.declare_parameter(
            'switch_controller_service', '/controller_manager/switch_controller'
        ).value
        self.controller_to_deactivate = self.declare_parameter(
            'controller_to_deactivate', 'joint_trajectory_controller'
        ).value
        self.clear_faults_service = self.declare_parameter('clear_faults_service', '').value

        self.dry_run = self.declare_parameter('dry_run', True).value
        self.estop_deactivate_controller = self.declare_parameter(
            'estop_deactivate_controller', True
        ).value
        self.nudge_joint_index = self.declare_parameter('nudge_joint_index', 6).value
        self.nudge_deg = self.declare_parameter('nudge_deg', 8.0).value
        self.max_nudge_deg = self.declare_parameter('max_nudge_deg', 20.0).value
        self.move_time_s = self.declare_parameter('move_time_s', 5.0).value
        self.gripper_open_position = self.declare_parameter('gripper_open_position', 0.0).value
        self.gripper_close_position = self.declare_parameter('gripper_close_position', 0.7).value
        self.gripper_max_effort = self.declare_parameter('gripper_max_effort', 20.0).value
        self.server_wait_timeout_s = self.declare_parameter('server_wait_timeout_s', 10.0).value
        self.joint_state_max_age_s = self.declare_parameter('joint_state_max_age_s', 1.0).value

        # -- Cartesian jog (velocity) via ros2_kortex's twist_controller --
        self.twist_topic = self.declare_parameter(
            'twist_topic', '/twist_controller/commands'
        ).value
        self.twist_controller_name = self.declare_parameter(
            'twist_controller_name', 'twist_controller'
        ).value
        self.max_linear_mps = self.declare_parameter('max_linear_mps', 0.10).value
        self.max_angular_rps = self.declare_parameter('max_angular_rps', 0.3).value
        self.twist_timeout_s = self.declare_parameter('twist_timeout_s', 0.3).value
        self.list_controllers_service = self.declare_parameter(
            'list_controllers_service', '/controller_manager/list_controllers'
        ).value
        # ros2_kortex hardcodes CARTESIAN_REFERENCE_FRAME_TOOL for twists, so
        # base-frame commands must be rotated into the tool frame via TF.
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.tool_frame = self.declare_parameter('tool_frame', 'end_effector_link').value
        self.tf_max_age_s = self.declare_parameter('tf_max_age_s', 0.5).value
        # 'kortex' streams to the real twist_controller (real hardware only).
        # 'sim_jtc' integrates the twist via differential IK and streams small
        # position steps through the trajectory controller — for FAKE-hardware
        # testing (e.g. watching the arm in Foxglove); it works on real
        # hardware too, so speeds stay clamped either way.
        self.twist_backend = self.declare_parameter('twist_backend', 'kortex').value
        self.sim_max_joint_rps = self.declare_parameter('sim_max_joint_rps', 0.5).value
        # Kinova "Home" (bent-elbow) pose for the 7-DoF Gen3 — well-conditioned
        # for Cartesian jogging, unlike the all-zeros candle pose (singular).
        self.home_pose = list(self.declare_parameter(
            'home_pose_rad', [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
        ).value)
        self.home_time_s = self.declare_parameter('home_time_s', 6.0).value
        self.jtc_stream_topic = self.declare_parameter(
            'jtc_stream_topic', '/joint_trajectory_controller/joint_trajectory'
        ).value

        # -- ROS entities (reentrant group so the e-stop service and joint_states are
        #    processed WHILE the demo thread blocks on an action result) --
        self._cb = ReentrantCallbackGroup()
        self._jtc_client = ActionClient(
            self, FollowJointTrajectory, self.jtc_action_name, callback_group=self._cb
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, self.gripper_action_name, callback_group=self._cb
        )
        self._joint_state_sub = self.create_subscription(
            JointState, self.joint_states_topic, self._joint_state_cb, 10, callback_group=self._cb
        )
        self._estop_srv = self.create_service(
            Trigger, '~/estop', self._estop_cb, callback_group=self._cb
        )
        self._switch_client = self.create_client(
            SwitchController, self.switch_controller_service, callback_group=self._cb
        )
        self._list_client = self.create_client(
            ListControllers, self.list_controllers_service, callback_group=self._cb
        )
        # The Kinova base LATCHES the last twist command, so we stream continuously
        # while cartesian mode is active (zeros when idle / on watchdog timeout)
        # rather than publishing only on change.
        self._twist_pub = self.create_publisher(Twist, self.twist_topic, 10)
        self._twist_timer = self.create_timer(
            0.05, self._twist_tick, callback_group=self._cb
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        # URDF joint axes/limits (for the sim_jtc differential-IK backend).
        self._joint_info = None
        self._robot_description_sub = self.create_subscription(
            String, '/robot_description', self._robot_description_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self._cb,
        )
        self._jtc_stream_pub = self.create_publisher(
            JointTrajectory, self.jtc_stream_topic, 10
        )

        # -- State --
        self._state_lock = threading.Lock()
        self._have_joint_state = False
        self._current_positions = []
        self._joint_state_mono = 0.0

        self._twist_lock = threading.Lock()
        self._twist_cmd = [0.0] * 6  # vx vy vz wx wy wz (BASE frame, rad/s), clamped
        self._twist_mono = 0.0
        self._cartesian_active = False
        self._last_tick_mono = time.monotonic()

        self._goal_lock = threading.Lock()
        self._active_jtc_goal = None
        self._active_gripper_goal = None

        self._stop = threading.Event()

        self.get_logger().info(
            "KinovaPrimitives ready (dry_run=%s). Software e-stop: %s/estop"
            % (self.dry_run, self.get_fully_qualified_name())
        )

    # ------------------------------------------------------------------ callbacks
    def _joint_state_cb(self, msg):
        ordered = [0.0] * len(self.joint_names)
        for i, name in enumerate(self.joint_names):
            try:
                idx = msg.name.index(name)
            except ValueError:
                return  # a configured joint isn't in this message; ignore it
            if idx >= len(msg.position):
                return
            ordered[i] = msg.position[idx]
        with self._state_lock:
            self._current_positions = ordered
            self._have_joint_state = True
            self._joint_state_mono = time.monotonic()

    def _estop_cb(self, request, response):
        self.get_logger().warn('E-stop service invoked.')
        self.soft_stop()
        response.success = True
        response.message = (
            'Soft-stop dispatched (goals cancelled; controller deactivation requested). '
            'This is NOT a substitute for the hardware E-stop.'
        )
        return response

    # ------------------------------------------------------------------- helpers
    def get_current_positions(self):
        """Latest joint positions ordered by ``joint_names``.

        Returns None if nothing has been received yet OR the last message is
        older than ``joint_state_max_age_s`` — a relative move computed from a
        stale base can travel much farther than the requested step.
        """
        with self._state_lock:
            if not self._have_joint_state:
                return None
            if time.monotonic() - self._joint_state_mono > self.joint_state_max_age_s:
                return None
            return list(self._current_positions)

    def _wait_future(self, future, timeout_s):
        """Block until an rclpy Future is done (completed by the background executor)."""
        deadline = time.monotonic() + timeout_s
        while rclpy.ok():
            if future.done():
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.02)
        return future.done()

    def wait_for_servers(self):
        """Block until the joint-trajectory (required) and gripper (optional) servers are up."""
        self.get_logger().info(
            'Waiting up to %.1fs for action servers...' % self.server_wait_timeout_s
        )
        jtc_ok = self._jtc_client.wait_for_server(timeout_sec=self.server_wait_timeout_s)
        grip_ok = self._gripper_client.wait_for_server(timeout_sec=self.server_wait_timeout_s)
        if not jtc_ok:
            self.get_logger().error(
                "Joint-trajectory server '%s' unavailable. Is the arm bringup running?"
                % self.jtc_action_name
            )
        if not grip_ok:
            self.get_logger().warn(
                "Gripper server '%s' unavailable; gripper steps will be skipped."
                % self.gripper_action_name
            )
        return jtc_ok  # gripper is optional

    # ---------------------------------------------------------------- primitives
    def move_to_joint_positions(self, target, time_s):
        """Move to absolute joint positions (radians) over ``time_s`` seconds (blocking)."""
        if self._stop.is_set():
            self.get_logger().warn('Stop requested; skipping move.')
            return False
        if len(target) != len(self.joint_names):
            self.get_logger().error(
                'Target size %d != joint count %d.' % (len(target), len(self.joint_names))
            )
            return False
        if self.dry_run:
            self.get_logger().info(
                '[DRY RUN] Would move %d joints over %.1fs (no motion sent).'
                % (len(self.joint_names), time_s)
            )
            return True
        if not self._jtc_client.server_is_ready():
            self.get_logger().error(
                "Joint-trajectory server '%s' not ready." % self.jtc_action_name
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in target]
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(seconds=time_s).to_msg()
        goal.trajectory.points = [point]

        send_future = self._jtc_client.send_goal_async(goal)
        if not self._wait_future(send_future, 10.0):
            self.get_logger().error('Timed out sending joint-trajectory goal.')
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Joint-trajectory goal was rejected by the controller.')
            return False
        with self._goal_lock:
            self._active_jtc_goal = goal_handle

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + time_s + 15.0
        ready = False
        while rclpy.ok():
            if result_future.done():
                ready = True
                break
            if self._stop.is_set():
                self.get_logger().warn('Stop requested during motion; cancelling trajectory goal.')
                goal_handle.cancel_goal_async()
                self._wait_future(result_future, 2.0)
                break
            if time.monotonic() > deadline:
                self.get_logger().error('Timed out waiting for trajectory result.')
                break
            time.sleep(0.02)

        with self._goal_lock:
            self._active_jtc_goal = None
        if not ready:
            return False
        wrapped = result_future.result()
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn('Trajectory did not succeed (status %d).' % wrapped.status)
            return False
        self.get_logger().info('Move complete.')
        return True

    def command_gripper(self, position, max_effort):
        """Command the gripper. ``position`` is in gripper joint units; ``max_effort`` is force."""
        if self._stop.is_set():
            return False
        if self.dry_run:
            self.get_logger().info(
                '[DRY RUN] Would command gripper to %.3f (max_effort %.1f).'
                % (position, max_effort)
            )
            return True
        if not self._gripper_client.server_is_ready():
            self.get_logger().warn(
                "Gripper server '%s' not ready; skipping gripper command."
                % self.gripper_action_name
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)

        send_future = self._gripper_client.send_goal_async(goal)
        if not self._wait_future(send_future, 5.0):
            self.get_logger().warn('Timed out sending gripper goal.')
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn('Gripper goal was rejected.')
            return False
        with self._goal_lock:
            self._active_gripper_goal = goal_handle

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + 15.0
        ready = False
        while rclpy.ok():
            if result_future.done():
                ready = True
                break
            if self._stop.is_set():
                goal_handle.cancel_goal_async()
                break
            if time.monotonic() > deadline:
                break
            time.sleep(0.02)

        with self._goal_lock:
            self._active_gripper_goal = None
        if not ready:
            return False
        wrapped = result_future.result()
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            return True
        # Grippers often report ABORTED when they stall on an object; not necessarily an error.
        self.get_logger().warn(
            'Gripper command finished with status %d (a stall on contact is normal).'
            % wrapped.status
        )
        return False

    def nudge_joint(self, joint_index, delta_deg, time_s=None):
        """Nudge one joint by ``delta_deg`` degrees (clamped to ``max_nudge_deg``), blocking.

        Relative to the latest joint_states. Returns False if the index is invalid or
        no joint state has been received yet.
        """
        if joint_index < 0 or joint_index >= len(self.joint_names):
            self.get_logger().error('nudge joint_index %d out of range.' % joint_index)
            return False
        base = self.get_current_positions()
        if base is None:
            self.get_logger().error('No joint_states received yet; cannot nudge.')
            return False
        if abs(delta_deg) > self.max_nudge_deg:
            self.get_logger().warn(
                'nudge of %.1f deg exceeds max %.1f; clamping.'
                % (delta_deg, self.max_nudge_deg)
            )
            delta_deg = math.copysign(self.max_nudge_deg, delta_deg)
        target = list(base)
        target[joint_index] += math.radians(delta_deg)
        return self.move_to_joint_positions(
            target, time_s if time_s is not None else self.move_time_s
        )

    def stop_requested(self):
        """True after a soft-stop until resume() clears it."""
        return self._stop.is_set()

    # ------------------------------------------------------- cartesian (velocity) jog
    def set_twist(self, vx, vy, vz, wx=0.0, wy=0.0, wz=0.0):
        """Set the streamed Cartesian velocity (BASE frame, m/s and rad/s).

        Clamped to ``max_linear_mps`` / ``max_angular_rps`` (a limit <= 0 means
        that axis group is locked out entirely). The command expires after
        ``twist_timeout_s`` (deadman): callers must re-send continuously
        (~10 Hz) or the stream drops to zero. Writes are dropped unless
        cartesian mode is active. Returns the applied linear speed in m/s.
        """
        if self._stop.is_set() or self.max_linear_mps <= 0.0:
            vx = vy = vz = 0.0
        if self._stop.is_set() or self.max_angular_rps <= 0.0:
            wx = wy = wz = 0.0
        lin = math.sqrt(vx * vx + vy * vy + vz * vz)
        if lin > self.max_linear_mps > 0.0:
            s = self.max_linear_mps / lin
            vx, vy, vz, lin = vx * s, vy * s, vz * s, self.max_linear_mps
        ang = math.sqrt(wx * wx + wy * wy + wz * wz)
        if ang > self.max_angular_rps > 0.0:
            s = self.max_angular_rps / ang
            wx, wy, wz = wx * s, wy * s, wz * s
        if self.dry_run and lin > 0.0:
            self.get_logger().info(
                '[DRY RUN] Would stream twist (%.3f, %.3f, %.3f) m/s.' % (vx, vy, vz),
                throttle_duration_sec=2.0,
            )
        with self._twist_lock:
            if not self._cartesian_active:
                return 0.0  # late write after a mode switch: drop it
            self._twist_cmd = [vx, vy, vz, wx, wy, wz]
            self._twist_mono = time.monotonic()
        return lin

    def _base_to_tool(self, cmd):
        """Rotate a base-frame twist into the tool frame (ros2_kortex hardcodes
        CARTESIAN_REFERENCE_FRAME_TOOL). Returns None if TF is missing/stale."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self.tool_frame, self.base_frame, Time()
            )
        except Exception:
            return None
        age = self.get_clock().now() - Time.from_msg(tf.header.stamp)
        if age.nanoseconds > self.tf_max_age_s * 1e9:
            return None
        q = tf.transform.rotation
        quat = (q.x, q.y, q.z, q.w)
        lin = _quat_rotate(quat, cmd[0:3])
        ang = _quat_rotate(quat, cmd[3:6])
        return list(lin) + list(ang)

    def _robot_description_cb(self, msg):
        """Parse joint axes, child links and limits from the URDF (sim backend)."""
        try:
            root = ET.fromstring(msg.data)
            found = {}
            for j in root.findall('joint'):
                name = j.get('name')
                if name not in self.joint_names:
                    continue
                axis_el = j.find('axis')
                axis = (
                    [float(x) for x in axis_el.get('xyz').split()]
                    if axis_el is not None else [1.0, 0.0, 0.0]
                )
                child = j.find('child').get('link')
                lower = upper = None
                if j.get('type') == 'revolute':
                    lim = j.find('limit')
                    if lim is not None:
                        lower = float(lim.get('lower')) if lim.get('lower') else None
                        upper = float(lim.get('upper')) if lim.get('upper') else None
                found[name] = (child, axis, lower, upper)
            if all(n in found for n in self.joint_names):
                self._joint_info = [found[n] for n in self.joint_names]
                self.get_logger().info(
                    'Parsed URDF joint info for %d joints (sim backend ready).'
                    % len(self._joint_info)
                )
            else:
                missing = [n for n in self.joint_names if n not in found]
                self.get_logger().warn('URDF missing joints: %s' % missing)
        except Exception as exc:
            self.get_logger().error('Failed to parse robot_description: %s' % exc)

    def _jacobian(self):
        """Geometric Jacobian (6 x N, base frame) from live TF + URDF axes."""
        info = self._joint_info
        if info is None:
            return None
        try:
            tf_ee = self._tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, Time()
            )
            age = self.get_clock().now() - Time.from_msg(tf_ee.header.stamp)
            if age.nanoseconds > self.tf_max_age_s * 1e9:
                return None  # frozen TF must stop the jog, same as _base_to_tool
            t = tf_ee.transform.translation
            p_ee = numpy.array([t.x, t.y, t.z])
            cols = []
            for child, axis, _, _ in info:
                tf_i = self._tf_buffer.lookup_transform(self.base_frame, child, Time())
                q = tf_i.transform.rotation
                a = numpy.array(_quat_rotate((q.x, q.y, q.z, q.w), axis))
                ti = tf_i.transform.translation
                p_i = numpy.array([ti.x, ti.y, ti.z])
                cols.append(numpy.concatenate((numpy.cross(a, p_ee - p_i), a)))
            return numpy.column_stack(cols)
        except Exception:
            return None

    def _sim_twist_step(self, cmd, dt=0.05):
        """One differential-IK step: base-frame twist -> small JTC position step.

        Damped least squares keeps the solve stable near singularities. The
        step integrates over the same horizon the point is scheduled for
        (2*dt) so commanded speed matches actual speed despite the 2x
        replacement overlap. If any joint would exceed its position limit the
        WHOLE step is dropped — the jog stops at the limit instead of veering
        off the commanded direction. Closed-loop: each step starts from the
        LIVE joint_states, so error does not accumulate.
        """
        q = self.get_current_positions()
        if q is None:
            self.get_logger().warn(
                'Sim backend: no fresh joint_states; not moving.',
                throttle_duration_sec=2.0,
            )
            return
        jac = self._jacobian()
        if jac is None:
            self.get_logger().warn(
                'Sim backend: no Jacobian (URDF/TF missing or stale); not moving.',
                throttle_duration_sec=2.0,
            )
            return
        v = numpy.array(cmd, dtype=float)
        # Weighted DLS: translation is the task, orientation hold is soft — near
        # a singularity the solver spends the available DoF on where the user is
        # pointing instead of rigidly locking the wrist.
        w = numpy.diag([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
        jw = w @ jac
        lam_sq = 0.01  # damping^2: trades tracking accuracy for singularity robustness
        jjt = jw @ jw.T + lam_sq * numpy.eye(6)
        qdot = jw.T @ numpy.linalg.solve(jjt, w @ v)
        qdot = numpy.clip(qdot, -self.sim_max_joint_rps, self.sim_max_joint_rps)
        # Singularity feedback: silence is worse than a warning when the pose
        # simply cannot move the commanded way (e.g. sideways from the candle pose).
        v_lin = numpy.linalg.norm(v[0:3])
        if v_lin > 1e-6:
            achieved = numpy.linalg.norm((jac @ qdot)[0:3])
            if achieved < 0.2 * v_lin:
                self.get_logger().warn(
                    'Sim backend: commanded direction is (near-)unreachable from '
                    'this pose (singularity) — try Home or another direction.',
                    throttle_duration_sec=2.0,
                )
        horizon = 2 * dt  # == time_from_start below
        target = []
        for qi, di, (_, _, lower, upper) in zip(q, qdot, self._joint_info):
            x = qi + di * horizon
            if (lower is not None and x < lower) or (upper is not None and x > upper):
                self.get_logger().warn(
                    'Sim backend: joint limit reached; stopping this jog.',
                    throttle_duration_sec=2.0,
                )
                return
            target.append(float(x))
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start = Duration(seconds=horizon).to_msg()
        msg.points = [point]
        self._jtc_stream_pub.publish(msg)

    def _twist_tick(self):
        """20 Hz streamer: publish the current twist, or zeros when idle/expired.

        Kortex backend publishes while holding the twist lock so a preempted
        tick can never emit a stale nonzero AFTER a zero from
        soft_stop/deactivate. The sim backend computes outside the lock — its
        commands are bounded position steps, so a stale step costs millimetres,
        not a latched velocity.
        """
        self._last_tick_mono = time.monotonic()
        with self._twist_lock:
            if not self._cartesian_active or self.dry_run:
                return
            expired = time.monotonic() - self._twist_mono > self.twist_timeout_s
            cmd = [0.0] * 6 if (expired or self._stop.is_set()) else list(self._twist_cmd)
            if self.twist_backend != 'sim_jtc':
                if any(cmd):
                    rotated = self._base_to_tool(cmd)
                    if rotated is None:
                        self.get_logger().warn(
                            'No fresh %s->%s TF; zeroing twist.'
                            % (self.base_frame, self.tool_frame),
                            throttle_duration_sec=2.0,
                        )
                        cmd = [0.0] * 6
                    else:
                        cmd = rotated
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.linear.z = cmd[0], cmd[1], cmd[2]
                # The Kortex wire format wants angular velocity in deg/s (the
                # driver passes values through unconverted); convert from rad/s.
                msg.angular.x = math.degrees(cmd[3])
                msg.angular.y = math.degrees(cmd[4])
                msg.angular.z = math.degrees(cmd[5])
                self._twist_pub.publish(msg)
                return
        # sim_jtc backend (outside the lock): zeros mean "send nothing" — the
        # trajectory controller simply holds the last position.
        if any(cmd):
            self._sim_twist_step(cmd)

    def tick_age(self):
        """Seconds since the twist streamer last ran — health check for the executor."""
        return time.monotonic() - self._last_tick_mono

    def _publish_zero_twist(self):
        """Immediately command zero velocity (the Kinova base latches the last twist).

        In dry_run only the internal state is zeroed — a dry-run session never
        publishes to the shared command topic.
        """
        with self._twist_lock:
            self._twist_cmd = [0.0] * 6
            self._twist_mono = 0.0
            if self.dry_run:
                return
            try:
                self._twist_pub.publish(Twist())
            except Exception as exc:
                self.get_logger().error('Zero-twist publish failed: %s' % exc)

    def _switch_controllers(self, activate, deactivate, strict):
        """Blocking controller switch; True if the manager reported ok.

        STRICT for deliberate mode switches (an ok that lies is worse than a
        failure); BEST_EFFORT only for stop paths ('deactivate what you can').
        """
        if not self._switch_client.service_is_ready():
            self.get_logger().error(
                "switch_controller service '%s' not ready." % self.switch_controller_service
            )
            return False
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = (
            SwitchController.Request.STRICT if strict
            else SwitchController.Request.BEST_EFFORT
        )
        req.activate_asap = False
        future = self._switch_client.call_async(req)
        if not self._wait_future(future, 3.0):
            self.get_logger().error('switch_controller timed out.')
            return False
        resp = future.result()
        return bool(resp is not None and resp.ok)

    def _controller_is_active(self, name):
        """True/False from /controller_manager/list_controllers; None if unknown."""
        if not self._list_client.service_is_ready():
            return None
        future = self._list_client.call_async(ListControllers.Request())
        if not self._wait_future(future, 2.0):
            return None
        resp = future.result()
        if resp is None:
            return None
        for c in resp.controller:
            if c.name == name:
                return c.state == 'active'
        return False

    def cartesian_active(self):
        """True while cartesian (twist) mode is active."""
        return self._cartesian_active

    def activate_cartesian(self):
        """Switch to velocity jogging: twist_controller in, trajectory controller out."""
        if self._stop.is_set():
            self.get_logger().warn('Soft-stopped; not activating cartesian mode.')
            return False
        self._publish_zero_twist()
        if self.dry_run:
            with self._twist_lock:
                self._cartesian_active = True
            self.get_logger().info('[DRY RUN] Cartesian mode on (controllers untouched).')
            return True
        if self.twist_backend == 'sim_jtc':
            if self._joint_info is None:
                self.get_logger().error(
                    'Sim backend: robot_description not parsed yet; try again shortly.'
                )
                return False
            jtc_active = self._controller_is_active(self.controller_to_deactivate)
            if jtc_active is not True:  # fail CLOSED on unknown, not just on False
                self.get_logger().error(
                    "Sim backend: '%s' is %s — it must be verifiably ACTIVE "
                    '(it executes the IK steps).'
                    % (self.controller_to_deactivate,
                       'inactive' if jtc_active is False
                       else 'unverifiable (controller_manager unreachable)')
                )
                return False
            self.get_logger().warn(
                '[SIM] Cartesian jog via differential IK through the trajectory '
                'controller — intended for FAKE hardware; joint speed clamped to '
                '%.2f rad/s.' % self.sim_max_joint_rps
            )
            with self._twist_lock:
                if self._stop.is_set():
                    return False  # a soft-stop raced the gates: stopped wins
                self._cartesian_active = True
            return True
        if not self._switch_controllers(
            [self.twist_controller_name], [self.controller_to_deactivate], strict=True
        ):
            self.get_logger().error('Failed to activate cartesian (twist) mode.')
            return False
        # Verify: STRICT ok should mean it happened, but the flag must never
        # outrun controller_manager reality.
        if self._controller_is_active(self.twist_controller_name) is False:
            self.get_logger().error('twist_controller did not activate; staying in joint mode.')
            self._switch_controllers([self.controller_to_deactivate], [], strict=False)
            return False
        if self._stop.is_set():
            # A soft-stop raced the switch: roll back, stopped wins.
            self._publish_zero_twist()
            self._switch_controllers(
                [], [self.twist_controller_name, self.controller_to_deactivate],
                strict=False,
            )
            return False
        with self._twist_lock:
            self._cartesian_active = True
        self.get_logger().info('Cartesian (twist) mode ACTIVE — velocity jogging enabled.')
        return True

    def deactivate_cartesian(self):
        """Back to trajectory mode: zero the twist, twist_controller out, JTC in.

        The flag drops FIRST (tick stops publishing, late set_twist writes are
        refused), then the zero goes out, then the controllers switch — the
        zero is guaranteed to be the last command the active controller relays.
        Activating the JTC in the same switch also forces a servoing-mode
        change on the Kinova base, which terminates any residual twist.
        """
        with self._twist_lock:
            self._cartesian_active = False
        self._publish_zero_twist()
        if self.dry_run:
            self.get_logger().info('[DRY RUN] Cartesian mode off (controllers untouched).')
            return True
        if self.twist_backend == 'sim_jtc':
            # Nothing to switch: the trajectory controller stays active and
            # simply holds position once the IK steps stop.
            self.get_logger().info('[SIM] Cartesian mode off.')
            return True
        time.sleep(0.05)  # one controller cycle: let the zero reach the base first
        ok = self._switch_controllers(
            [self.controller_to_deactivate], [self.twist_controller_name], strict=True
        )
        if not ok:
            self.get_logger().error(
                'Failed to switch back to the trajectory controller; check '
                "'ros2 control list_controllers'."
            )
        else:
            self.get_logger().info('Cartesian mode off; trajectory controller active.')
        return ok

    def open_gripper(self):
        return self.command_gripper(self.gripper_open_position, self.gripper_max_effort)

    def close_gripper(self):
        return self.command_gripper(self.gripper_close_position, self.gripper_max_effort)

    def soft_stop(self):
        """Cancel active goals and (optionally) deactivate the motion controller.

        NOT a substitute for the hardware E-stop.
        """
        self._stop.set()
        # Kill any velocity command first — the base latches the last twist.
        # Deactivation alone does NOT stop the base (verified in ros2_kortex:
        # the stop branch only zeroes an internal array), so the zero twist
        # must reach the still-active controller; send it twice with a dwell.
        with self._twist_lock:
            self._cartesian_active = False
        self._publish_zero_twist()
        time.sleep(0.05)
        self._publish_zero_twist()
        self.get_logger().warn('SOFT-STOP: zeroed twist; cancelling active goals.')
        with self._goal_lock:
            goals = [
                g for g in (self._active_jtc_goal, self._active_gripper_goal) if g is not None
            ]
        # Best-effort from here: no single failure may abort the rest of the stop.
        for goal in goals:
            try:
                goal.cancel_goal_async()
            except Exception as exc:
                self.get_logger().error('Failed to cancel a goal: %s' % exc)
        if self.estop_deactivate_controller:
            try:
                if self._switch_client.service_is_ready():
                    req = SwitchController.Request()
                    req.deactivate_controllers = [
                        self.controller_to_deactivate, self.twist_controller_name
                    ]
                    req.strictness = SwitchController.Request.BEST_EFFORT
                    req.activate_asap = False
                    self._switch_client.call_async(req)
                    self.get_logger().warn(
                        "Requested deactivation of '%s' and '%s'."
                        % (self.controller_to_deactivate, self.twist_controller_name)
                    )
                else:
                    self.get_logger().warn(
                        "switch_controller service '%s' not ready; could not deactivate controller."
                        % self.switch_controller_service
                    )
            except Exception as exc:
                self.get_logger().error('Controller deactivation failed: %s' % exc)
        self.get_logger().warn(
            'Soft-stop dispatched. Use the HARDWARE E-STOP for real emergencies.'
        )

    def resume(self):
        """Clear a previous soft-stop and (best-effort) reactivate the motion controller.

        In dry_run mode the real controller state is left untouched — a
        "safe" session must not re-arm the motion controller for everyone else.
        """
        self._stop.clear()
        if self.dry_run:
            self.get_logger().info(
                'dry_run: soft-stop cleared; controller state left untouched.'
            )
            return
        if self.estop_deactivate_controller:
            try:
                if self._switch_client.service_is_ready():
                    req = SwitchController.Request()
                    req.activate_controllers = [self.controller_to_deactivate]
                    req.strictness = SwitchController.Request.BEST_EFFORT
                    req.activate_asap = False
                    self._switch_client.call_async(req)
                    self.get_logger().warn(
                        "Requested reactivation of '%s'." % self.controller_to_deactivate
                    )
                else:
                    self.get_logger().warn(
                        "switch_controller service '%s' not ready; reactivate manually with "
                        "'ros2 control switch_controllers --activate %s'."
                        % (self.switch_controller_service, self.controller_to_deactivate)
                    )
            except Exception as exc:
                self.get_logger().error('Controller reactivation failed: %s' % exc)
        self.get_logger().info('Soft-stop cleared; commands are accepted again.')

    # --------------------------------------------------------------------- demo
    def run_test_demo(self):
        """Demo: read pose -> open -> slow joint nudge and back -> close -> open."""
        self.get_logger().info(
            '==== test_arm demo (%s) ===='
            % ('DRY RUN - no motion will be sent' if self.dry_run else 'LIVE MOTION')
        )
        if not self.wait_for_servers():
            self.get_logger().error('Required action server unavailable; aborting demo.')
            return

        # Optional, best-effort fault clear (off by default; assumes std_srvs/srv/Trigger).
        if self.clear_faults_service:
            cf = self.create_client(Trigger, self.clear_faults_service)
            if cf.wait_for_service(timeout_sec=3.0):
                cf.call_async(Trigger.Request())
                self.get_logger().info(
                    "Sent clear-faults request to '%s'." % self.clear_faults_service
                )
            else:
                self.get_logger().warn(
                    "clear-faults service '%s' not available; skipping."
                    % self.clear_faults_service
                )

        # Wait for the first joint_states.
        base = None
        t0 = time.monotonic()
        while rclpy.ok():
            base = self.get_current_positions()
            if base is not None:
                break
            if time.monotonic() - t0 > self.server_wait_timeout_s:
                break
            time.sleep(0.1)
        if base is None or len(base) != len(self.joint_names):
            self.get_logger().error(
                "No usable joint_states on '%s' (check joint_names); aborting demo."
                % self.joint_states_topic
            )
            return

        # Validate / clamp the nudge.
        if self.nudge_joint_index < 0 or self.nudge_joint_index >= len(self.joint_names):
            self.get_logger().error(
                'nudge_joint_index %d out of range; aborting.' % self.nudge_joint_index
            )
            return
        nudge_deg = self.nudge_deg
        if abs(nudge_deg) > self.max_nudge_deg:
            self.get_logger().warn(
                'nudge_deg %.1f exceeds max %.1f; clamping.' % (nudge_deg, self.max_nudge_deg)
            )
            nudge_deg = math.copysign(self.max_nudge_deg, nudge_deg)

        target = list(base)
        target[self.nudge_joint_index] += nudge_deg * math.pi / 180.0

        self.get_logger().info(
            'Plan: open gripper -> nudge %s by %.1f deg over %.1fs -> return -> close -> open.'
            % (self.joint_names[self.nudge_joint_index], nudge_deg, self.move_time_s)
        )

        if self._stop.is_set():
            return
        self.open_gripper()
        if self._stop.is_set():
            return
        if not self.move_to_joint_positions(target, self.move_time_s):
            self.get_logger().warn('Outbound nudge failed or was skipped.')
        if self._stop.is_set():
            return
        if not self.move_to_joint_positions(base, self.move_time_s):
            self.get_logger().warn('Return move failed or was skipped.')
        if self._stop.is_set():
            return
        self.close_gripper()
        if self._stop.is_set():
            return
        self.open_gripper()
        self.get_logger().info('==== demo complete ====')
