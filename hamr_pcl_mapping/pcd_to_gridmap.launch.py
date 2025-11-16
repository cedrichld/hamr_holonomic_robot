import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('hamr_pcl_mapping')

    maps_dir = os.path.join(pkg_share, 'maps')
    config_path = os.path.join(pkg_share, 'config', 'pcd_to_gridmap.yaml')

    return LaunchDescription([
        Node(
            package='grid_map_pcl',
            executable='grid_map_pcl_loader_node',
            name='grid_map_pcl_loader_node',
            output='screen',
            parameters=[{
                # PCD + output grid_map bag in same folder
                'folder_path': maps_dir,
                'pcd_filename': 'global_map.pcd',
                'output_grid_map': 'global_elevation.bag',

                'configFilePath_': config_path,
                'config_file_path': config_path,
            }]
        )
    ])
