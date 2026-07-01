"""Entry point: run the test_arm demo with a software e-stop.

Uses a MultiThreadedExecutor spinning in the background so the ~/estop service and
joint_states keep being processed WHILE the demo thread blocks on an action result.
"""

import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from adl_primitives.kinova_primitives import KinovaPrimitives


def main(args=None):
    rclpy.init(args=args)
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
