import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _get_latest_rtabmap_db(maps_dir: str) -> str:
    """
    Pick the most recent rtabmap_*.db in maps_dir.
    Fallback: maps_dir/rtabmap.db if none found.
    """
    if not os.path.exists(maps_dir):
        raise RuntimeError(f"Maps directory does not exist: {maps_dir}")

    candidates = []
    for fname in os.listdir(maps_dir):
        if fname.startswith("rtabmap_") and fname.endswith(".db"):
            full = os.path.join(maps_dir, fname)
            candidates.append(full)

    if candidates:
        # Sort by filename (timestamp in name) or mtime, pick latest
        candidates.sort()  # filenames contain timestamp, so lexical sort works
        latest = candidates[-1]
        return latest

    # Fallback to a default file if no timestamped DBs exist
    fallback = os.path.join(maps_dir, "rtabmap.db")
    if os.path.exists(fallback):
        return fallback

    raise RuntimeError(
        f"No RTAB-Map database found in {maps_dir}. "
        f"Run env_mapping.py first to create a map."
    )


def generate_launch_description():
    # ------------------------------------------------------------------
    # Paths / dirs
    # ------------------------------------------------------------------
    home = os.environ.get("HOME", "/home/kartik")
    maps_dir = os.path.join(home, "maps")

    # Figure out which DB to use for localization (latest one)
    db_path = _get_latest_rtabmap_db(maps_dir)
    print(f"[env_localization] Using RTAB-Map DB: {db_path}")

    # RViz config (same as mapping)
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

    # ------------------------------------------------------------------
    # Nodes / includes
    # ------------------------------------------------------------------

    # 1) RealSense D455 (same settings as mapping)
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

    # 2) RTAB-Map in localization mode
    #    Key bits:
    #      localization:=true
    #      rtabmap_args: disable incremental mapping, init WM from DB
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rtabmap_launch),
        launch_arguments={
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "frame_id": "camera_link",
            "approx_sync": "true",
            "database_path": db_path,
            "localization": "true",
            "rtabmap_args": "--Rtabmap/StartNewMap false "
                            "--Mem/IncrementalMemory false "
                            "--Mem/InitWMWithAllNodes true",
        }.items(),
    )

    # 3) RViz2 (same mapping view)
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
    )

    # No pcl_mapper here — we're only localizing against an existing map.

    return LaunchDescription([
        realsense,
        rtabmap,
        rviz,
    ])
