from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Path substitutions
    urdf_path = PathJoinSubstitution([
        FindPackageShare("hamr_description"),
        "urdf",
        "hamr.urdf.xacro"
    ])

    gazebo_config_path = PathJoinSubstitution([
        FindPackageShare("hamr_bringup"),
        "config",
        "gazebo_bridge.yaml"
    ])

    rviz_config_path = PathJoinSubstitution([
        FindPackageShare("hamr_description"),
        "rviz",
        "urdf_config.rviz"
    ])

    gz_world_path = PathJoinSubstitution([
        FindPackageShare("hamr_bringup"),
        "worlds",
        "empty_world.sdf"
    ])

    return LaunchDescription([
        # Launch Gazebo with world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py"
                ])
            ]),
            launch_arguments={"gz_args": [gz_world_path, " -r"]}.items()
        ),

        # Spawn robot in GZ from robot_description topic
        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=["-topic", "robot_description"],
            output="screen"
        ),

        # Robot State Publisher with xacro
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": Command(["xacro ", urdf_path])
            }],
            output="screen"
        ),

        # Bridge config
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            parameters=[{"config_file": gazebo_config_path}],
            output="screen"
        ),

        # RViz with configuration
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config_path],
            output="screen"
        ),

        # HAMR Controller Node
        Node(
            package="hamr_control",
            executable="hamr_controller",
            output="screen"
        ),
    ])
