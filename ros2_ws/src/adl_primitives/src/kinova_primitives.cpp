#include "adl_primitives/kinova_primitives.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rmw/qos_profiles.h"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

using namespace std::chrono_literals;

namespace adl_primitives
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
}  // namespace

KinovaPrimitives::KinovaPrimitives(const rclcpp::NodeOptions & options)
: rclcpp::Node("hello_arm", options)
{
  joint_names_ = this->declare_parameter<std::vector<std::string>>(
    "joint_names",
    {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"});
  jtc_action_name_ = this->declare_parameter<std::string>(
    "joint_trajectory_action", "/joint_trajectory_controller/follow_joint_trajectory");
  gripper_action_name_ = this->declare_parameter<std::string>(
    "gripper_action", "/robotiq_gripper_controller/gripper_cmd");
  joint_states_topic_ = this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
  switch_controller_service_ = this->declare_parameter<std::string>(
    "switch_controller_service", "/controller_manager/switch_controller");
  controller_to_deactivate_ = this->declare_parameter<std::string>(
    "controller_to_deactivate", "joint_trajectory_controller");
  clear_faults_service_ = this->declare_parameter<std::string>("clear_faults_service", "");

  dry_run_ = this->declare_parameter<bool>("dry_run", true);
  estop_deactivate_controller_ = this->declare_parameter<bool>("estop_deactivate_controller", true);
  nudge_joint_index_ = this->declare_parameter<int>("nudge_joint_index", 6);
  nudge_deg_ = this->declare_parameter<double>("nudge_deg", 8.0);
  max_nudge_deg_ = this->declare_parameter<double>("max_nudge_deg", 20.0);
  move_time_s_ = this->declare_parameter<double>("move_time_s", 5.0);
  gripper_open_position_ = this->declare_parameter<double>("gripper_open_position", 0.0);
  gripper_close_position_ = this->declare_parameter<double>("gripper_close_position", 0.7);
  gripper_max_effort_ = this->declare_parameter<double>("gripper_max_effort", 20.0);
  server_wait_timeout_s_ = this->declare_parameter<double>("server_wait_timeout_s", 10.0);

  reentrant_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  jtc_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
    this, jtc_action_name_, reentrant_group_);
  gripper_client_ = rclcpp_action::create_client<GripperCommand>(
    this, gripper_action_name_, reentrant_group_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = reentrant_group_;
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, rclcpp::QoS(10),
    std::bind(&KinovaPrimitives::jointStateCallback, this, std::placeholders::_1),
    sub_options);

  estop_service_ = this->create_service<std_srvs::srv::Trigger>(
    "~/estop",
    std::bind(&KinovaPrimitives::estopCallback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, reentrant_group_);

  switch_controller_client_ = this->create_client<controller_manager_msgs::srv::SwitchController>(
    switch_controller_service_, rmw_qos_profile_services_default, reentrant_group_);

  RCLCPP_INFO(
    get_logger(), "KinovaPrimitives ready (dry_run=%s). Software e-stop: %s/estop",
    dry_run_ ? "true" : "false", this->get_fully_qualified_name());
}

void KinovaPrimitives::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::vector<double> ordered(joint_names_.size(), 0.0);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto it = std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
    if (it == msg->name.end()) {
      return;  // a configured joint is not in this message; ignore it
    }
    const size_t idx = static_cast<size_t>(std::distance(msg->name.begin(), it));
    if (idx >= msg->position.size()) {
      return;
    }
    ordered[i] = msg->position[idx];
  }
  std::lock_guard<std::mutex> lk(state_mutex_);
  current_positions_ = std::move(ordered);
  have_joint_state_ = true;
}

bool KinovaPrimitives::getCurrentPositions(std::vector<double> & out)
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  if (!have_joint_state_) {
    return false;
  }
  out = current_positions_;
  return true;
}

bool KinovaPrimitives::waitForServers()
{
  const auto timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(server_wait_timeout_s_));

  RCLCPP_INFO(get_logger(), "Waiting up to %.1fs for action servers...", server_wait_timeout_s_);
  const bool jtc_ok = jtc_client_->wait_for_action_server(timeout);
  const bool grip_ok = gripper_client_->wait_for_action_server(timeout);

  if (!jtc_ok) {
    RCLCPP_ERROR(
      get_logger(), "Joint-trajectory server '%s' unavailable. Is the arm bringup running?",
      jtc_action_name_.c_str());
  }
  if (!grip_ok) {
    RCLCPP_WARN(
      get_logger(), "Gripper server '%s' unavailable; gripper steps will be skipped.",
      gripper_action_name_.c_str());
  }
  return jtc_ok;  // gripper is optional for Milestone 1
}

bool KinovaPrimitives::moveToJointPositions(const std::vector<double> & target, double time_s)
{
  if (stop_requested_.load()) {
    RCLCPP_WARN(get_logger(), "Stop requested; skipping move.");
    return false;
  }
  if (target.size() != joint_names_.size()) {
    RCLCPP_ERROR(
      get_logger(), "Target size %zu != joint count %zu.", target.size(), joint_names_.size());
    return false;
  }

  if (dry_run_) {
    RCLCPP_INFO(
      get_logger(), "[DRY RUN] Would move %zu joints over %.1fs (no motion sent).",
      joint_names_.size(), time_s);
    return true;
  }

  if (!jtc_client_->action_server_is_ready()) {
    RCLCPP_ERROR(get_logger(), "Joint-trajectory server '%s' not ready.", jtc_action_name_.c_str());
    return false;
  }

  FollowJointTrajectory::Goal goal;
  goal.trajectory.joint_names = joint_names_;
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = target;
  point.velocities.assign(joint_names_.size(), 0.0);
  point.time_from_start = rclcpp::Duration::from_seconds(time_s);
  goal.trajectory.points.push_back(point);

  auto goal_future = jtc_client_->async_send_goal(goal);
  if (goal_future.wait_for(10s) != std::future_status::ready) {
    RCLCPP_ERROR(get_logger(), "Timed out sending joint-trajectory goal.");
    return false;
  }
  auto goal_handle = goal_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(get_logger(), "Joint-trajectory goal was rejected by the controller.");
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    active_jtc_goal_ = goal_handle;
  }

  auto result_future = jtc_client_->async_get_result(goal_handle);
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(time_s + 15.0));

  bool ready = false;
  while (rclcpp::ok()) {
    if (result_future.wait_for(50ms) == std::future_status::ready) {
      ready = true;
      break;
    }
    if (stop_requested_.load()) {
      RCLCPP_WARN(get_logger(), "Stop requested during motion; cancelling trajectory goal.");
      jtc_client_->async_cancel_goal(goal_handle);
      result_future.wait_for(2s);
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      RCLCPP_ERROR(get_logger(), "Timed out waiting for trajectory result.");
      break;
    }
  }
  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    active_jtc_goal_.reset();
  }
  if (!ready) {
    return false;
  }

  const auto wrapped = result_future.get();
  if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_WARN(
      get_logger(), "Trajectory did not succeed (result code %d).",
      static_cast<int>(wrapped.code));
    return false;
  }
  RCLCPP_INFO(get_logger(), "Move complete.");
  return true;
}

bool KinovaPrimitives::commandGripper(double position, double max_effort)
{
  if (stop_requested_.load()) {
    return false;
  }

  if (dry_run_) {
    RCLCPP_INFO(
      get_logger(), "[DRY RUN] Would command gripper to %.3f (max_effort %.1f).",
      position, max_effort);
    return true;
  }

  if (!gripper_client_->action_server_is_ready()) {
    RCLCPP_WARN(
      get_logger(), "Gripper server '%s' not ready; skipping gripper command.",
      gripper_action_name_.c_str());
    return false;
  }

  GripperCommand::Goal goal;
  goal.command.position = position;
  goal.command.max_effort = max_effort;

  auto goal_future = gripper_client_->async_send_goal(goal);
  if (goal_future.wait_for(5s) != std::future_status::ready) {
    RCLCPP_WARN(get_logger(), "Timed out sending gripper goal.");
    return false;
  }
  auto goal_handle = goal_future.get();
  if (!goal_handle) {
    RCLCPP_WARN(get_logger(), "Gripper goal was rejected.");
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    active_gripper_goal_ = goal_handle;
  }

  auto result_future = gripper_client_->async_get_result(goal_handle);
  const auto deadline = std::chrono::steady_clock::now() + 15s;
  bool ready = false;
  while (rclcpp::ok()) {
    if (result_future.wait_for(50ms) == std::future_status::ready) {
      ready = true;
      break;
    }
    if (stop_requested_.load()) {
      gripper_client_->async_cancel_goal(goal_handle);
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      break;
    }
  }
  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    active_gripper_goal_.reset();
  }
  if (!ready) {
    return false;
  }

  const auto wrapped = result_future.get();
  if (wrapped.code == rclcpp_action::ResultCode::SUCCEEDED) {
    return true;
  }
  // Grippers often report ABORTED when they stall on an object; not necessarily an error.
  RCLCPP_WARN(
    get_logger(), "Gripper command finished with code %d (a stall on contact is normal).",
    static_cast<int>(wrapped.code));
  return false;
}

bool KinovaPrimitives::openGripper()
{
  return commandGripper(gripper_open_position_, gripper_max_effort_);
}

bool KinovaPrimitives::closeGripper()
{
  return commandGripper(gripper_close_position_, gripper_max_effort_);
}

void KinovaPrimitives::softStop()
{
  stop_requested_.store(true);
  RCLCPP_WARN(get_logger(), "SOFT-STOP: cancelling active goals.");

  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    if (active_jtc_goal_) {
      jtc_client_->async_cancel_goal(active_jtc_goal_);
    }
    if (active_gripper_goal_) {
      gripper_client_->async_cancel_goal(active_gripper_goal_);
    }
  }

  if (estop_deactivate_controller_ && switch_controller_client_) {
    if (switch_controller_client_->service_is_ready()) {
      auto req = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
      req->deactivate_controllers.push_back(controller_to_deactivate_);
      req->strictness = controller_manager_msgs::srv::SwitchController::Request::BEST_EFFORT;
      req->activate_asap = false;
      switch_controller_client_->async_send_request(req);
      RCLCPP_WARN(get_logger(), "Requested deactivation of '%s'.", controller_to_deactivate_.c_str());
    } else {
      RCLCPP_WARN(
        get_logger(), "switch_controller service '%s' not ready; could not deactivate controller.",
        switch_controller_service_.c_str());
    }
  }

  RCLCPP_WARN(get_logger(), "Soft-stop dispatched. Use the HARDWARE E-STOP for real emergencies.");
}

void KinovaPrimitives::estopCallback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  RCLCPP_WARN(get_logger(), "E-stop service invoked.");
  softStop();
  response->success = true;
  response->message =
    "Soft-stop dispatched (goals cancelled; controller deactivation requested). "
    "This is NOT a substitute for the hardware E-stop.";
}

void KinovaPrimitives::runHelloDemo()
{
  RCLCPP_INFO(
    get_logger(), "==== hello_arm demo (%s) ====",
    dry_run_ ? "DRY RUN - no motion will be sent" : "LIVE MOTION");

  if (!waitForServers()) {
    RCLCPP_ERROR(get_logger(), "Required action server unavailable; aborting demo.");
    return;
  }

  // Optional, best-effort fault clear (off by default; assumes std_srvs/srv/Trigger).
  if (!clear_faults_service_.empty()) {
    auto cf = this->create_client<std_srvs::srv::Trigger>(clear_faults_service_);
    if (cf->wait_for_service(3s)) {
      cf->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
      RCLCPP_INFO(get_logger(), "Sent clear-faults request to '%s'.", clear_faults_service_.c_str());
    } else {
      RCLCPP_WARN(
        get_logger(), "clear-faults service '%s' not available; skipping.",
        clear_faults_service_.c_str());
    }
  }

  // Wait for the first joint_states.
  std::vector<double> base;
  const auto t0 = std::chrono::steady_clock::now();
  while (rclcpp::ok() && !getCurrentPositions(base)) {
    if (std::chrono::steady_clock::now() - t0 > std::chrono::duration<double>(server_wait_timeout_s_)) {
      break;
    }
    std::this_thread::sleep_for(100ms);
  }
  if (base.size() != joint_names_.size()) {
    RCLCPP_ERROR(
      get_logger(), "No usable joint_states on '%s' (check joint_names); aborting demo.",
      joint_states_topic_.c_str());
    return;
  }

  // Validate / clamp the nudge.
  if (nudge_joint_index_ < 0 || nudge_joint_index_ >= static_cast<int>(joint_names_.size())) {
    RCLCPP_ERROR(get_logger(), "nudge_joint_index %d out of range; aborting.", nudge_joint_index_);
    return;
  }
  double nudge_deg = nudge_deg_;
  if (std::abs(nudge_deg) > max_nudge_deg_) {
    RCLCPP_WARN(
      get_logger(), "nudge_deg %.1f exceeds max %.1f; clamping.", nudge_deg, max_nudge_deg_);
    nudge_deg = std::copysign(max_nudge_deg_, nudge_deg);
  }

  std::vector<double> target = base;
  target[static_cast<size_t>(nudge_joint_index_)] += nudge_deg * kDegToRad;

  RCLCPP_INFO(
    get_logger(),
    "Plan: open gripper -> nudge %s by %.1f deg over %.1fs -> return -> close -> open.",
    joint_names_[static_cast<size_t>(nudge_joint_index_)].c_str(), nudge_deg, move_time_s_);

  if (stop_requested_.load()) {return;}
  openGripper();

  if (stop_requested_.load()) {return;}
  if (!moveToJointPositions(target, move_time_s_)) {
    RCLCPP_WARN(get_logger(), "Outbound nudge failed or was skipped.");
  }

  if (stop_requested_.load()) {return;}
  if (!moveToJointPositions(base, move_time_s_)) {
    RCLCPP_WARN(get_logger(), "Return move failed or was skipped.");
  }

  if (stop_requested_.load()) {return;}
  closeGripper();

  if (stop_requested_.load()) {return;}
  openGripper();

  RCLCPP_INFO(get_logger(), "==== demo complete ====");
}

}  // namespace adl_primitives
