#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"

#include "adl_primitives/kinova_primitives.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<adl_primitives::KinovaPrimitives>();

  // Best-effort soft-stop on Ctrl-C. NOTE: once shutdown begins, cancellations may not
  // reach the controller. To reliably halt a motion in progress, call the ~/estop service
  // from another terminal, or use the HARDWARE E-stop.
  rclcpp::on_shutdown([node]() {node->softStop();});

  // Multi-threaded so the e-stop service / joint_states are processed WHILE the demo
  // thread is blocked waiting on an action result.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spin_thread([&executor]() {executor.spin();});

  node->runHelloDemo();

  RCLCPP_INFO(
    node->get_logger(),
    "Demo finished. E-stop service '%s/estop' stays active. Press Ctrl-C to exit.",
    node->get_fully_qualified_name());

  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
