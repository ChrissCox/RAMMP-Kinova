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

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


class KinovaPrimitives(Node):
    """A small library of basic, safe Kinova Gen3 motion primitives."""

    def __init__(self):
        super().__init__('test_arm')

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

        # -- State --
        self._state_lock = threading.Lock()
        self._have_joint_state = False
        self._current_positions = []

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
        """Latest joint positions ordered by ``joint_names``, or None if not yet received."""
        with self._state_lock:
            if not self._have_joint_state:
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

    def open_gripper(self):
        return self.command_gripper(self.gripper_open_position, self.gripper_max_effort)

    def close_gripper(self):
        return self.command_gripper(self.gripper_close_position, self.gripper_max_effort)

    def soft_stop(self):
        """Cancel active goals and (optionally) deactivate the motion controller.

        NOT a substitute for the hardware E-stop.
        """
        self._stop.set()
        self.get_logger().warn('SOFT-STOP: cancelling active goals.')
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
                    req.deactivate_controllers = [self.controller_to_deactivate]
                    req.strictness = SwitchController.Request.BEST_EFFORT
                    req.activate_asap = False
                    self._switch_client.call_async(req)
                    self.get_logger().warn(
                        "Requested deactivation of '%s'." % self.controller_to_deactivate
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
