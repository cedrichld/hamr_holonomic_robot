## Adaptation from GridMap Documentation Demo Launch file by ANYbotics

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Find the grid_map_demos package share directory
    grid_map_demos_dir = get_package_share_directory('grid_map_demos')
    hamr_bringup_dir = get_package_share_directory('hamr_bringup')

    # Declare launch configuration variables that can access the launch arguments values
    testing_terrain = LaunchConfiguration('testing_terrain')
    visualize_rviz = LaunchConfiguration('visualize_rviz')
    img_px_res = LaunchConfiguration('img_px_res') # pixels
    map_width = LaunchConfiguration('map_width') # meters
    map_height = LaunchConfiguration('map_height') # meters
    filters_config_file = LaunchConfiguration('filters_config')
    visualization_config_file = LaunchConfiguration('visualization_config')
    rviz_config_file = LaunchConfiguration('rviz_config')

    declare_testing_terrain = DeclareLaunchArgument(
        'testing_terrain', default_value='true',
        description='Use demo terrain.png (true) or custom off-road PNG (false)'
    )
    declare_visualize_rviz = DeclareLaunchArgument(
        'visualize_rviz', default_value='true',
        description='Launch RViz2 with a given config'
    )
    declare_img_px_res = DeclareLaunchArgument(
        'img_px_res', default_value='1025',
        description='Heightmap image pixel resolution (square)'
    )
    declare_map_width = DeclareLaunchArgument(
        'map_width', default_value='45.0', # meters
        description='Map width in meters'
    )
    declare_map_height = DeclareLaunchArgument(
        'map_height', default_value='4.5', # meters
        description='Map height range (min to max) in meters'
    )

    img_px_res_str = LaunchConfiguration('img_px_res')
    terrain_filename = PythonExpression([
        '"terrain1_" + str(', img_px_res_str, ') + ".png"'
    ])

    terrain_path = PathJoinSubstitution([
        FindPackageShare('hamr_bringup'),
        'terrain_assets',
        'heightmaps',
        'off_road_maps',
        terrain_filename,
    ])


    # To appropriately orient the terrain map
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='terrain_frame_rot90',
        arguments=[
        #    x    y    z    qx   qy   qz            qw            parent child
            '0', '0', '0', '0', '0', '0.70710678', '0.70710678', 'map', 'terrain_map'
        ]
    )

    declare_filters_config_file_cmd = DeclareLaunchArgument(
        'filters_config',
        default_value=os.path.join(
            hamr_bringup_dir, 'config', 'filters_demo_filter_chain.yaml'),
        description='Full path to the filter chain config file to use')

    declare_visualization_config_file_cmd = DeclareLaunchArgument(
        'visualization_config',
        default_value=os.path.join(
            hamr_bringup_dir, 'config', 'filters_demo.yaml'),
        description='Full path to the Gridmap visualization config file to use')
    
    '''
    The filters_demo.yaml:
    ---------------------
    image_to_gridmap:
        ros__parameters:
            image_topic: "/image"
            resolution: 0.02
            map_frame_id: "map"
            min_height: -0.5
            max_height: 1.0

        grid_map_visualization:
        ros__parameters:
            grid_map_topic: /filtered_map
            grid_map_visualizations: [surface_normals, traversability_grid] 
            surface_normals:
            type: vectors
            params:
                layer_prefix: normal_vectors_
                position_layer: elevation
                scale: 0.06
                line_width: 0.005
                color: 15600153 # red
            traversability_grid:
            type: occupancy_grid
            params:
                layer: traversability
                data_min: 0.0
                data_max: 1.0
    ----------------------
    '''

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(
            grid_map_demos_dir, 'rviz', 'filters_demo.rviz'),
        description='Full path to the RVIZ config file to use')

    # Declare node actions
    grid_map_filter_demo_node = Node(
        package='grid_map_demos',
        executable='filters_demo',
        name='grid_map_filters',
        output='screen',
        parameters=[filters_config_file]
    )

    # ============================================================================
    # IfCondition(testing_terrain):
    image_publisher_node_testing = Node(
        condition=IfCondition(testing_terrain),
        package='grid_map_demos',
        executable='image_publisher.py',
        name='image_publisher',
        output='screen',
        parameters=[{
            'image_path': os.path.join(grid_map_demos_dir, 'data', 'terrain.png'),
            'topic': 'image'
        }]
    )
    image_to_gridmap_demo_node_testing = Node(
        condition=IfCondition(testing_terrain),
        package='hamr_control_cpp',
        executable='image_to_gridmap',
        name='image_to_gridmap',
        output='screen',
        parameters=[visualization_config_file]
    )

    # else:
    image_publisher_node = Node(
        condition=UnlessCondition(testing_terrain),
        package='grid_map_demos',
        executable='image_publisher.py',
        name='image_publisher',
        output='screen',
        parameters=[{
            'image_path': terrain_path,
            'topic': 'image'
        }]
    )
    image_to_gridmap_demo_node = Node(
        condition=UnlessCondition(testing_terrain),
        package='hamr_control_cpp',
        executable='image_to_gridmap',
        name='image_to_gridmap',
        output='screen',
        parameters=[{
            'image_topic': "/image",
            'map_frame_id': 'terrain_map',
            'min_height': 0.0, # m
            'max_height': map_height, # m
            'map_width': map_width,
            'img_px_res': img_px_res
        }] # meters per cell -> img_res / map_xy: 45m / 1025px
    )
    # ========================================================
    
    grid_map_visualization_node = Node(
        package='grid_map_visualization',
        executable='grid_map_visualization',
        name='grid_map_visualization',
        output='screen',
        parameters=[visualization_config_file]
    )
        
    rviz2_node = Node(
        condition=IfCondition(visualize_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    # Create the launch description and populate
    ld = LaunchDescription()
    

    # Add launch arguments to the launch description
    ld.add_action(declare_testing_terrain)
    ld.add_action(declare_visualize_rviz)
    ld.add_action(declare_img_px_res)
    ld.add_action(declare_map_width)
    ld.add_action(declare_map_height)
    # Config files
    ld.add_action(declare_filters_config_file_cmd)
    ld.add_action(declare_visualization_config_file_cmd)
    ld.add_action(declare_rviz_config_file_cmd)

    # To appropriately orient the terrain map
    ld.add_action(static_tf)
    
    # Add node actions to the launch description
    ld.add_action(grid_map_filter_demo_node)

    ld.add_action(image_publisher_node)
    ld.add_action(image_to_gridmap_demo_node)
    ld.add_action(image_publisher_node_testing)
    ld.add_action(image_to_gridmap_demo_node_testing)

    ld.add_action(grid_map_visualization_node)
    ld.add_action(rviz2_node)

    return ld
