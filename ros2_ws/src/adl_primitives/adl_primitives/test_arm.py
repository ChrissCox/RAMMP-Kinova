"""Entry point: run the test_arm demo with a software e-stop.

Uses a MultiThreadedExecutor spinning in the background so the ~/estop service and
joint_states keep being processed WHILE the demo thread blocks on an action result.
"""

import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from adl_primitives.kinova_primitives import KinovaPrimitives


def main(args=None):
    # Handle SIGINT ourselves: rclpy's default handler invalidates the context
    # before `except KeyboardInterrupt` runs, which would leave soft_stop()
    # unable to cancel goals or deactivate the controller.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KinovaPrimitives()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_test_demo()
        node.get_logger().info(
            "Demo finished. E-stop service '%s/estop' stays active. Press Ctrl-C to exit."
            % node.get_fully_qualified_name()
        )
        # Keep the node alive so the e-stop service remains available.
        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        node.get_logger().warn('Ctrl-C received; soft-stopping.')
        node.soft_stop()
        time.sleep(0.5)  # let the cancels / deactivation request flush before shutdown
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)  # let the spin thread exit before destroying the node
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
