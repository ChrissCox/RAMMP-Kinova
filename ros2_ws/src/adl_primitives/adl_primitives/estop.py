"""Automatic software e-stop for the jog_ui twist path.

Runs as its OWN process, launched automatically alongside ``jog_ui`` by
``jog_ui.launch.py`` (that separation is the point: it survives jog_ui dying).
jog_ui publishes ``std_msgs/Bool`` heartbeats on ``/jog_ui/heartbeat``:
``True`` beats come from the same 20 Hz tick that runs jog_ui's in-process
deadman (and only when jog_ui is NOT in dry_run — a dry-run session commands
nothing, so its liveness must not vouch for anyone); a ``False`` beat is
jog_ui's clean-shutdown announcement, which DISARMS this node quietly.

If True beats have been seen and then STOP for longer than
``heartbeat_timeout_s`` with no False, the jog process (or its executor) died
uncleanly, and this node closes the one gap no in-process code can:

1. publishes zero twists FIRST — the Kinova base latches its last twist
   command and deactivation alone does not stop it, so the zero must reach the
   (possibly still active) twist controller;
2. then requests a controller switch: twist_controller out, the fallback
   joint-trajectory controller in (the servoing-mode change also terminates
   any residual twist at the base), and escalates loudly if the switch is not
   confirmed.

It fires once per heartbeat loss and re-arms automatically when True beats
return (e.g. jog_ui relaunched).

Operating rules:
- Run at most ONE live (non-dry-run) jog_ui at a time: multiple live sessions
  share the heartbeat topic and would mask each other's death.
- If this node starts AFTER jog_ui already died, it does nothing until a fresh
  heartbeat arrives — verify the arm manually in that case.

This is an automatic backstop for UNCLEAN process death. The hardware E-stop
remains authoritative — total power/PC loss is still its job.
"""

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class EStop(Node):
    """Deliberately boring: count seconds, and stop the arm if the counter runs out."""

    def __init__(self):
        super().__init__('estop')
        self.heartbeat_topic = self.declare_parameter(
            'heartbeat_topic', '/jog_ui/heartbeat'
        ).value
        self.twist_topic = self.declare_parameter(
            'twist_topic', '/twist_controller/commands'
        ).value
        self.twist_controller_name = self.declare_parameter(
            'twist_controller_name', 'twist_controller'
        ).value
        self.fallback_controller = self.declare_parameter(
            'fallback_controller', 'joint_trajectory_controller'
        ).value
        self.switch_controller_service = self.declare_parameter(
            'switch_controller_service', '/controller_manager/switch_controller'
        ).value
        self.heartbeat_timeout_s = self.declare_parameter('heartbeat_timeout_s', 0.5).value

        self._twist_pub = self.create_publisher(Twist, self.twist_topic, 10)
        self._switch_client = self.create_client(
            SwitchController, self.switch_controller_service
        )
        self._last_beat = None  # monotonic time of the last True beat; None = never seen
        self._armed = False
        self._switch_confirmed = False
        self._confirm_timer = None
        self.create_subscription(Bool, self.heartbeat_topic, self._beat_cb, 10)
        self.create_timer(0.1, self._check)
        self.get_logger().info(
            "Software e-stop up: watching '%s' (timeout %.2fs). On loss: zero '%s', "
            "then switch '%s' out / '%s' in."
            % (self.heartbeat_topic, self.heartbeat_timeout_s, self.twist_topic,
               self.twist_controller_name, self.fallback_controller)
        )

    def _beat_cb(self, msg):
        if not msg.data:
            # Clean-shutdown announcement from jog_ui: disarm without firing.
            if self._armed:
                self.get_logger().info('Clean shutdown announced; e-stop disarmed.')
            self._armed = False
            self._last_beat = None
            return
        first = self._last_beat is None
        self._last_beat = time.monotonic()
        if not self._armed:
            self._armed = True
            self.get_logger().info(
                'Heartbeat detected; e-stop ARMED.' if first
                else 'Heartbeat returned; e-stop re-armed.'
            )

    def _check(self):
        if not self._armed or self._last_beat is None:
            return
        if time.monotonic() - self._last_beat <= self.heartbeat_timeout_s:
            return
        self._armed = False  # fire once; fresh heartbeats re-arm
        # Nothing may abort the fire: state is already safe (disarmed), and a
        # crashed e-stop node is a silently missing backstop.
        try:
            self._fire()
        except Exception as exc:
            self.get_logger().error(
                'E-stop fire path raised %s — zero twist may not have been sent. '
                'VERIFY THE ARM IS STOPPED; use the hardware E-stop if not.' % exc
            )

    def _fire(self):
        self.get_logger().error(
            'HEARTBEAT LOST for > %.2fs (unclean jog death): commanding zero twist, '
            'then deactivating %s.'
            % (self.heartbeat_timeout_s, self.twist_controller_name)
        )
        # Zeros FIRST, repeatedly, while the twist controller may still be
        # active: the base latches its last twist and deactivation alone does
        # not transmit a stop.
        for _ in range(4):
            try:
                self._twist_pub.publish(Twist())
            except Exception as exc:
                self.get_logger().error('Zero-twist publish failed: %s' % exc)
            time.sleep(0.05)
        req = SwitchController.Request()
        req.activate_controllers = [self.fallback_controller]
        req.deactivate_controllers = [self.twist_controller_name]
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = False
        if self._switch_client.service_is_ready():
            self._switch_confirmed = False
            future = self._switch_client.call_async(req)
            future.add_done_callback(self._switch_done)
            if self._confirm_timer is not None:
                self._confirm_timer.cancel()
            self._confirm_timer = self.create_timer(2.5, self._confirm_check)
            self.get_logger().warn(
                "Switch requested: '%s' out, '%s' in — awaiting confirmation."
                % (self.twist_controller_name, self.fallback_controller)
            )
        else:
            self.get_logger().error(
                'switch_controller service unavailable — zero twist was sent, but '
                'VERIFY THE ARM IS STOPPED; use the hardware E-stop if not.'
            )

    def _switch_done(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(
                'switch_controller call failed: %s — VERIFY THE ARM IS STOPPED; '
                'use the hardware E-stop if not.' % exc
            )
            return
        if resp is not None and resp.ok:
            self._switch_confirmed = True
            self.get_logger().info(
                'Controller switch CONFIRMED; the arm is holding position.'
            )
        else:
            self.get_logger().error(
                'controller_manager refused the switch — VERIFY THE ARM IS STOPPED; '
                'use the hardware E-stop if not.'
            )

    def _confirm_check(self):
        if self._confirm_timer is not None:
            self._confirm_timer.cancel()
        if not self._switch_confirmed:
            self.get_logger().error(
                'No controller-switch confirmation within 2.5s — VERIFY THE ARM IS '
                'STOPPED; use the hardware E-stop if not.'
            )


def main(args=None):
    rclpy.init(args=args)
    node = EStop()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.get_logger().warn(
                'Software e-stop exiting — the process-death backstop is OFFLINE.'
            )
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
