"""One-command cuRobo sim demo: fake Gen3 bringup + foxglove_bridge + planner.

    ros2 launch curobo_planner curobo_demo.launch.py

Brings up everything needed to watch cuRobo plan around the obstacle course in
Foxglove. Then, in another terminal:
    ros2 run curobo_planner goto "go to the bottle"

Args:
    use_fake_hardware:=true   (default) simulate the arm; false = real robot
    robot_ip:=192.168.1.10    only used when use_fake_hardware:=false
    foxglove:=true            also start foxglove_bridge (default true)
    execute:=true             send planned trajectories to the controller
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument('use_fake_hardware', default_value='true'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.1.10'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('execute', default_value='true'),
    ]

    scene_file = PathJoinSubstitution(
        [FindPackageShare('curobo_planner'), 'config', 'scene.yaml'])

    gen3_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('kortex_bringup'), 'launch', 'gen3.launch.py'])),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
            'gripper': 'robotiq_2f_85',
            'launch_rviz': 'false',
        }.items(),
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    planner = Node(
        package='curobo_planner',
        executable='planner',
        name='curobo_planner',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'scene_file': scene_file,
            'execute': ParameterValue(LaunchConfiguration('execute'), value_type=bool),
        }],
    )

    return LaunchDescription(args + [gen3_bringup, foxglove, planner])
