cmake_minimum_required(VERSION 3.8)
project(adl_primitives)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(rmw REQUIRED)

set(dependencies
  rclcpp
  rclcpp_action
  control_msgs
  trajectory_msgs
  sensor_msgs
  std_srvs
  controller_manager_msgs
  rmw
)

add_executable(hello_arm
  src/hello_arm_node.cpp
  src/kinova_primitives.cpp
)
target_include_directories(hello_arm PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(hello_arm ${dependencies})

install(TARGETS hello_arm
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
