"""MuJoCo-backed bringup: physics replaces the fake hardware, stack unchanged.

    ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie   # once
    ros2 launch mujoco_sim mujoco_bringup.launch.py

Starts: robot_state_publisher (kortex URDF, for TF/Foxglove), the
mujoco_ros2_control controller manager (MuJoCo physics inside), the same three
controllers as the kortex bringup, foxglove_bridge, and (mirror:=true) a
rosbridge websocket for the Windows mirror viewer.

jog_ui / curobo_planner / primitives connect to the identical controller
topics — launch them separately as usual, with use_sim_time:=true.

Args:
    mujoco_model:=<path>    generated scene (default ~/.ros/mujoco_sim/scene_gen3.xml)
    headless:=true          no on-Jetson MuJoCo window (default true)
    mirror:=false           also start rosbridge (ws://:9090) for mirror_viewer
    description_file:=...   kortex xacro for TF, if your layout differs
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            'mujoco_model',
            default_value=os.path.expanduser('~/.ros/mujoco_sim/scene_gen3.xml')),
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('mirror', default_value='false'),
        DeclareLaunchArgument(
            'description_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('kortex_description'), 'robots', 'gen3.xacro']),
            description='kortex xacro used ONLY for TF/visualization'),
    ]

    # Display URDF (TF + Foxglove); ros2_control tags in it are ignored by RSP.
    display_description = ParameterValue(
        Command([
            'xacro ', LaunchConfiguration('description_file'),
            ' robot_ip:=192.168.1.10 use_fake_hardware:=true',
            ' gripper:=robotiq_2f_85 dof:=7 vision:=false sim_gazebo:=false',
        ]),
        value_type=str)

    # Control URDF: just the ros2_control block pointing at MuJoCo.
    control_description = ParameterValue(
        Command([
            'xacro ',
            PathJoinSubstitution(
                [FindPackageShare('mujoco_sim'), 'urdf', 'mujoco_control.urdf.xacro']),
            ' mujoco_model:=', LaunchConfiguration('mujoco_model'),
            ' headless:=', LaunchConfiguration('headless'),
        ]),
        value_type=str)

    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare('mujoco_sim'), 'config', 'controllers.yaml'])

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': display_description,
                     'use_sim_time': True}])

    # mujoco_ros2_control ships its OWN controller-manager executable that
    # embeds the MuJoCo physics loop (publishes /clock every step).
    cm = Node(
        package='mujoco_ros2_control', executable='ros2_control_node',
        output='screen', emulate_tty=True,
        parameters=[{'robot_description': control_description,
                     'use_sim_time': True},
                    controllers_yaml])

    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name, '-c', '/controller_manager'],
                    output='screen')

    foxglove = Node(package='foxglove_bridge', executable='foxglove_bridge',
                    name='foxglove_bridge', output='screen')

    rosbridge = Node(
        package='rosbridge_server', executable='rosbridge_websocket',
        name='rosbridge_websocket', output='screen',
        condition=IfCondition(LaunchConfiguration('mirror')))

    return LaunchDescription(args + [
        rsp, cm,
        spawner('joint_state_broadcaster'),
        spawner('joint_trajectory_controller'),
        spawner('robotiq_gripper_controller'),
        foxglove, rosbridge,
    ])
