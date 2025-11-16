from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false'
        ),

        # RealSense D455
        Node(
            package='realsense2_camera',
            executable='rs_launch.py',
            name='d455',
            output='screen',
            parameters=[{
                'pointcloud.enable': True,
                'rgb_camera.profile': '640x480x30',
                'depth_module.profile': '640x480x30',
                'align_depth.enable': True,
                'enable_sync': True,
                'enable_rgbd': True,
            }]
        ),

        # RTAB-Map RGB-D SLAM
        Node(
            package='rtabmap_launch',
            executable='rtabmap.launch.py',
            name='rtabmap',
            output='screen',
            parameters=[{
                # frame setup
                'frame_id': 'camera_link',
                'map_frame_id': 'map',
                'odom_frame_id': 'odom',

                # Realsense topics
                'rgb_topic': '/camera/camera/color/image_raw',
                'depth_topic': '/camera/camera/depth/image_rect_raw',
                'camera_info_topic': '/camera/camera/color/camera_info',

                'approx_sync': True,
                'subscribe_depth': True,
                'subscribe_rgb': True,

                'rtabmap_args': '--delete_db_on_start',   # fresh map each run
            }]
        )
    ])
