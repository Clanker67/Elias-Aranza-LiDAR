from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Full path to ouster_ros driver launch file
    ouster_launch = os.path.join(get_package_share_directory('ouster_ros'), 'launch', 'driver.launch.py')


    # Full path to pcl_tutorial launch file
    pcl_launch = os.path.join(
    get_package_share_directory('pcl_tutorial'),
    'launch',
    'minimal_processing_node_Santora.launch.py')
    
    return LaunchDescription([
        #1) Launch the Ouster LiDAR driver
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ouster_launch),
        ),

        # 2) Static transform from laser_sensor_frame -> laser_data_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_pub',
            output='screen',
            arguments=['0', '0', '0.5', '0', '0', '0', '1',
                       'laser_sensor_frame', 'laser_data_frame']
        ),
        
        # 2.1) Static transform from laser_sensor_frame -> laser_data_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_pub2',
            output='screen',
            arguments=['0', '0', '.5', '0', '0', '0', '1',
                       'os_lidar', 'laser_sensor_frame']
        ),

        # 3) Launch point cloud processing
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pcl_launch),
        ),
    ])

