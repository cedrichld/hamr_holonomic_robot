import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)

def generate_launch_description():
    # ----------------------------------------------------------------------
    # Paths and directories
    # ----------------------------------------------------------------------
    home = os.environ.get("HOME", "/home/kartik")
    maps_dir = os.path.join(home, "maps")
    pcd_dir = os.path.join(maps_dir, "pcd")

    # Make sure directories exist
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    # Timestamped RTAB-Map database path
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    db_path = os.path.join(maps_dir, f"rtabmap_{ts}.db")

    # --- Check required external packages ---
    try:
        realsense_share = get_package_share_directory("realsense2_camera")
    except PackageNotFoundError:
        return LaunchDescription([
            LogInfo(msg="[env_mapping] ERROR: package 'realsense2_camera' not found. "
                        "Install it with: 'sudo apt install ros-${ROS_DISTRO}-realsense2-camera'")
        ])

    try:
        rtabmap_share = get_package_share_directory("rtabmap_launch")
    except PackageNotFoundError:
        return LaunchDescription([
            LogInfo(msg="[env_mapping] ERROR: package 'rtabmap_launch' not found. "
                        "Install it with: 'sudo apt install ros-${ROS_DISTRO}-rtabmap-ros ros-${ROS_DISTRO}-rtabmap-launch'")
        ])

    # RViz config (if you created it)
    pkg_share = get_package_share_directory("hamr_pcl_generator")
    rviz_config_path = os.path.join(pkg_share, "config", "mapping_view.rviz")
    rviz_args = []
    if os.path.exists(rviz_config_path):
        rviz_args = ["--display-config", rviz_config_path]

    # External launch files
    realsense_launch = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch",
        "rs_launch.py",
    )

    rtabmap_launch = os.path.join(
        get_package_share_directory("rtabmap_launch"),
        "launch",
        "rtabmap.launch.py",
    )

    # ----------------------------------------------------------------------
    # Nodes / included launches
    # ----------------------------------------------------------------------

    # RealSense D455 node (realsense2_camera)
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments={
            "pointcloud.enable": "true",
            "align_depth.enable": "true",
            "depth_fps": "60",
            "rgb_fps": "60",
            "initial_reset": "true",
        }.items(),
    )

    # RTAB-Map SLAM node
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rtabmap_launch),
        launch_arguments={
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "frame_id": "camera_link",
            "approx_sync": "true",
            "database_path": db_path,
        }.items(),
    )

    # RViz2 (with config if available)
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
    )

    # Your PCL mapper node (saves final colored PCD on shutdown)
    pcl_mapper = Node(
        package="hamr_pcl_generator",
        executable="pcl_mapper",
        name="pcl_mapper",
        output="screen",
        parameters=[{
            "input_topic": "/rtabmap/cloud_map",
            "target_frame": "map",
            "output_dir": pcd_dir,
            "filename_prefix": "hamr_room_",
            "save_on_shutdown": True,
        }],
    )

    return LaunchDescription([
        realsense,
        rtabmap,
        rviz,
        pcl_mapper,
    ])
