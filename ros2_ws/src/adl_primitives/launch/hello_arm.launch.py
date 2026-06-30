"""Launch the Milestone-1 `hello_arm` demo node.

Defaults to dry_run:=true (connects and prints the plan, but does NOT move the arm).
Set dry_run:=false to command real motion, after the safety checklist in the README.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_params = PathJoinSubstitution(
        [FindPackageShare("adl_primitives"), "config", "hello_arm.yaml"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "dry_run",
            default_value="true",
            description="If true, connect and print the plan but DO NOT move the arm.",
        ),
        DeclareLaunchArgument("nudge_deg", default_value="8.0"),
        DeclareLaunchArgument("nudge_joint_index", default_value="6"),
        DeclareLaunchArgument("move_time_s", default_value="5.0"),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to a parameters YAML file.",
        ),
    ]

    hello_arm = Node(
        package="adl_primitives",
        executable="hello_arm",
        name="hello_arm",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "dry_run": ParameterValue(
                    LaunchConfiguration("dry_run"), value_type=bool
                ),
                "nudge_deg": ParameterValue(
                    LaunchConfiguration("nudge_deg"), value_type=float
                ),
                "nudge_joint_index": ParameterValue(
                    LaunchConfiguration("nudge_joint_index"), value_type=int
                ),
                "move_time_s": ParameterValue(
                    LaunchConfiguration("move_time_s"), value_type=float
                ),
            },
        ],
    )

    return LaunchDescription(declared_args + [hello_arm])
