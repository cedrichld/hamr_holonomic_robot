from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():


    folder_path = LaunchConfiguration('folder_path')
    pcd_filename = LaunchConfiguration('pcd_filename')
    output_grid_map = LaunchConfiguration('output_grid_map')
    
    declare_folder_path = DeclareLaunchArgument(
        'folder_path', default_value='/home/cedric/ros2_ws/hamr_ws/src/rosbags/',
        description='.pcd folder path'
    )
    declare_pcd_filename = DeclareLaunchArgument(
        'pcd_filename', default_value='vicon_area.pcd',
        description='.pcd file name'
    )
    declare_output_grid_map = DeclareLaunchArgument(
        'output_grid_map', default_value='elevation_map.bag',
        description='output .bag folder (which contains grid_map topic)'
    )

    node_params = [
        {'folder_path': folder_path},
        {'pcd_filename': pcd_filename},
        {'map_rosbag_topic': 'grid_map'},
        {'output_grid_map': output_grid_map},
        {'map_frame': 'map'},
        {'map_layer_name': 'elevation'},
        {'prefix': ''},
        {'set_verbosity_to_debug': False}
    ]

    pcl_loader_node = Node(
        package='grid_map_pcl',
        executable='grid_map_pcl_loader_node',
        name='grid_map_pcl_loader_node',
        output='screen',
        parameters=node_params
    )

    ld = LaunchDescription()

    ld.add_action(declare_folder_path)
    ld.add_action(declare_pcd_filename)
    ld.add_action(declare_output_grid_map)

    ld.add_action(pcl_loader_node)

    return ld