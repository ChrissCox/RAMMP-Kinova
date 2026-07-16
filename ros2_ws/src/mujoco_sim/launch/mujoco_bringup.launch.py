"""The whole RAMMP stack, one command:

    ros2 launch mujoco_sim mujoco_bringup.launch.py

Starts:
  * robot_state_publisher (kortex URDF -> TF)
  * mujoco_ros2_control controller manager (MuJoCo physics inside), wrapped
    in xvfb-run: the camera publisher needs a GL display, headless Jetsons
    don't have one
  * the joint trajectory + gripper controllers
  * rosbridge websocket (ws://:9090) — the Windows mirror viewer and the
    voice app both ride it
  * rammp_perception detectors: fixed scene_cam + eye-in-hand d405
  * the cuRobo planner (takes ~15 s to warm up; watch for "planner ready")

Then talk to it:  ros2 run curobo_planner goto     (or the voice app)

Prerequisite (once, after every scene.yaml edit that adds/moves geometry):
    ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie

Args:
    mujoco_model:=<path>       generated scene (default ~/.ros/mujoco_sim/scene_gen3.xml)
    description_file:=...      kortex xacro for TF, if your layout differs
    enable_finetune:=false     ~2x faster plans, still collision-free, less smooth
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            'mujoco_model',
            default_value=os.path.expanduser('~/.ros/mujoco_sim/scene_gen3.xml')),
        DeclareLaunchArgument(
            'description_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('kortex_description'), 'robots', 'gen3.xacro']),
            description='kortex xacro used ONLY for TF/visualization'),
        DeclareLaunchArgument(
            'enable_finetune', default_value='true',
            description='planner trajectory polish; false = ~2x faster plans'),
        DeclareLaunchArgument(
            'brain_model', default_value='claude-haiku-4-5',
            description='claude-haiku-4-5 = fastest (sub-second decisions), '
                        'claude-sonnet-4-6 / claude-opus-4-8 = deeper '
                        'reasoning for hard multi-step tasks'),
        DeclareLaunchArgument(
            'brain_thinking', default_value='false',
            description='adaptive thinking (slower, smarter) — NOT '
                        'supported by haiku; pair with sonnet/opus'),
    ]

    # Display URDF (TF); ros2_control tags in it are ignored by RSP.
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
            ' headless:=true',
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
    # embeds the MuJoCo physics loop (publishes /clock every step). xvfb-run
    # gives the camera renderer the GL display a headless boot lacks.
    cm = Node(
        package='mujoco_ros2_control', executable='ros2_control_node',
        output='screen', emulate_tty=True,
        prefix='xvfb-run -a -s "-screen 0 1280x800x24"',
        parameters=[{'robot_description': control_description,
                     'use_sim_time': True},
                    controllers_yaml])

    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name, '-c', '/controller_manager'],
                    output='screen')

    rosbridge = Node(
        package='rosbridge_server', executable='rosbridge_websocket',
        name='rosbridge_websocket', output='screen')

    scene_detector = Node(
        package='rammp_perception', executable='detector',
        output='screen', emulate_tty=True,
        parameters=[{'use_sim_time': True}])

    d405_detector = Node(
        package='rammp_perception', executable='detector',
        name='d405_detector', output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'rgb_topic': '/d405/color',
            'depth_topic': '/d405/depth',
            'info_topic': '/d405/camera_info',
            'camera_attached_frame': 'bracelet_link',
        }])

    planner = Node(
        package='curobo_planner', executable='planner',
        output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'enable_finetune': ParameterValue(
                LaunchConfiguration('enable_finetune'), value_type=bool),
        }])

    # AnyGrasp proposer: loads the licensed detector once (~10 s), then
    # answers /grasp_proposer/request on demand from the live D405. If the
    # venv/license is unavailable it stays up and answers with a named
    # error — the planner's geometric grasps are the fallback.
    grasp_proposer = Node(
        package='rammp_perception', executable='grasp_proposer',
        output='screen', emulate_tty=True,
        respawn=True, respawn_delay=5.0,   # a CUDA OOM once killed it —
        parameters=[{'use_sim_time': True}])   # come back, don't stay dead

    # The brain: Claude picks tools from live circumstance (/rammp/task).
    # Without ANTHROPIC_API_KEY it degrades to a planner passthrough.
    brain = Node(
        package='curobo_planner', executable='brain',
        output='screen', emulate_tty=True,
        parameters=[{'use_sim_time': True,
                     'model': LaunchConfiguration('brain_model'),
                     'thinking': ParameterValue(
                         LaunchConfiguration('brain_thinking'),
                         value_type=bool)}])

    return LaunchDescription(args + [
        rsp, cm,
        spawner('joint_state_broadcaster'),
        spawner('joint_trajectory_controller'),
        spawner('robotiq_gripper_controller'),
        rosbridge,
        scene_detector, d405_detector, grasp_proposer, planner, brain,
    ])
