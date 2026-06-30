#ifndef ADL_PRIMITIVES__KINOVA_PRIMITIVES_HPP_
#define ADL_PRIMITIVES__KINOVA_PRIMITIVES_HPP_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "control_msgs/action/gripper_command.hpp"
#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace adl_primitives
{

/// A small library of basic, safe Kinova Gen3 motion primitives (Milestone 1).
///
/// Commands the arm through ros2_kortex's ros2_control interfaces:
///   * joints  -> control_msgs/action/FollowJointTrajectory
///   * gripper -> control_msgs/action/GripperCommand
/// and provides a software e-stop (`~/estop`, std_srvs/srv/Trigger).
///
/// SAFETY: `dry_run` defaults to true (nothing moves). The software e-stop is a
/// convenience, NOT a substitute for the hardware E-stop — the Gen3 has no brakes.
class KinovaPrimitives : public rclcpp::Node
{
public:
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using GoalHandleFJT = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;
  using GripperCommand = control_msgs::action::GripperCommand;
  using GoalHandleGC = rclcpp_action::ClientGoalHandle<GripperCommand>;

  explicit KinovaPrimitives(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /// Block until the joint-trajectory (required) and gripper (optional) servers are up.
  /// Returns true if the required joint-trajectory server is available.
  bool waitForServers();

  /// Latest joint positions ordered by `joint_names_`. False until joint_states arrive.
  bool getCurrentPositions(std::vector<double> & out);

  /// Move to absolute joint positions (radians) over `time_s` seconds (blocking).
  bool moveToJointPositions(const std::vector<double> & target_positions, double time_s);

  /// Gripper commands. `position` is in the gripper's joint units; `max_effort` is force.
  bool commandGripper(double position, double max_effort);
  bool openGripper();
  bool closeGripper();

  /// Software soft-stop: cancel active goals and (optionally) deactivate the motion
  /// controller. NOT a substitute for the hardware E-stop.
  void softStop();

  /// Milestone-1 demo: read pose -> open -> small slow joint nudge and back -> close -> open.
  void runHelloDemo();

  bool stopRequested() const { return stop_requested_.load(); }

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void estopCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  // -- Parameters --
  std::vector<std::string> joint_names_;
  std::string jtc_action_name_;
  std::string gripper_action_name_;
  std::string joint_states_topic_;
  std::string switch_controller_service_;
  std::string controller_to_deactivate_;
  std::string clear_faults_service_;
  bool dry_run_ {true};
  bool estop_deactivate_controller_ {true};
  int nudge_joint_index_ {6};
  double nudge_deg_ {8.0};
  double max_nudge_deg_ {20.0};
  double move_time_s_ {5.0};
  double gripper_open_position_ {0.0};
  double gripper_close_position_ {0.7};
  double gripper_max_effort_ {20.0};
  double server_wait_timeout_s_ {10.0};

  // -- ROS entities --
  rclcpp::CallbackGroup::SharedPtr reentrant_group_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr jtc_client_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_service_;
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switch_controller_client_;

  // -- State --
  std::mutex state_mutex_;
  bool have_joint_state_ {false};
  std::vector<double> current_positions_;   // ordered by joint_names_

  std::mutex goal_mutex_;
  GoalHandleFJT::SharedPtr active_jtc_goal_;
  GoalHandleGC::SharedPtr active_gripper_goal_;

  std::atomic<bool> stop_requested_ {false};
};

}  // namespace adl_primitives

#endif  // ADL_PRIMITIVES__KINOVA_PRIMITIVES_HPP_
