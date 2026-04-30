from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os
 
def generate_launch_description():
    nav2_params = os.path.join(
        get_package_share_directory('navigation'),
        'config',
        'finalcostmap.yaml'
    )
    slam_params = os.path.join(
        get_package_share_directory('navigation'),
        'config',
        'slam_params.yaml'
    )
    slam_launch = os.path.join(
        get_package_share_directory('slam_toolbox'),
        'launch',
        'online_async_launch.py'
    )
 
    return LaunchDescription([
        # TEMPORARY: Remove once real odometry (wheel encoders/IMU) is available
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'odom', 'base_link']
        ),
 
        # SLAM Toolbox — dynamically publishes map -> odom TF and /map topic
        # Do NOT publish map -> odom statically anywhere else or it will conflict
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments={
                'use_sim_time': 'false',
                'slam_params_file': slam_params,
                'use_composition': 'False',
                'autostart': 'True',
            }.items()
        ),

        # Local costmap — launched standalone since controller_server is excluded.
        # PushRosNamespace ensures the node registers as /local_costmap/costmap
        # so the lifecycle manager can find it at local_costmap/costmap/get_state.
        GroupAction([
            PushRosNamespace('local_costmap'),
            Node(
                package='nav2_costmap_2d',
                executable='nav2_costmap_2d',
                name='costmap',
                output='screen',
                parameters=[nav2_params],
            ),
        ]),

        # Nav2 nodes — controller_server intentionally excluded
        # Robot is driven by APF planner node publishing directly to /cmd_vel
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_params]
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_params]
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_params]
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[nav2_params]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[nav2_params]
        ),
    ])
