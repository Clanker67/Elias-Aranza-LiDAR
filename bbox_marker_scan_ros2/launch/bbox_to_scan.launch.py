from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('bbox_marker_scan_ros2'),
        'config',
        'bbox_to_scan.params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('marker_topic', default_value='/bbox_markers'),
        DeclareLaunchArgument('scan_topic', default_value='/bbox_fake_scan'),

        Node(
            package='bbox_marker_scan_ros2',
            executable='bbox_marker_to_scan',
            name='bbox_marker_to_scan',
            output='screen',
            parameters=[LaunchConfiguration('params_file'), {
                'marker_topic': LaunchConfiguration('marker_topic'),
                'scan_topic': LaunchConfiguration('scan_topic'),
            }],
        )
    ])
