"""Launch the `jog_ui` browser jog panel.

Defaults to dry_run:=true (clicks are logged, the arm does NOT move).
Set dry_run:=false to command real motion, after the safety checklist in the README.
Open http://<jetson-ip>:8080 in a browser on the same LAN.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_params = PathJoinSubstitution(
        [FindPackageShare("adl_primitives"), "config", "jog_ui.yaml"]
    )

    declared_args = [
        DeclareLaunchArgument(
            "dry_run",
            default_value="true",
            description="If true, accept and log UI commands but DO NOT move the arm.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to a parameters YAML file.",
        ),
    ]

    jog_ui = Node(
        package="adl_primitives",
        executable="jog_ui",
        name="jog_ui",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("params_file"),
            {
                # dry_run is deliberately launch-arg-only: the safety gate must be
                # explicit on the command line, never buried in a YAML edit.
                # (ui_port and everything else live in the params file.)
                "dry_run": ParameterValue(
                    LaunchConfiguration("dry_run"), value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription(declared_args + [jog_ui])
