from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Full path to ouster_ros driver launch file
    ouster_launch = os.path.join(
        get_package_share_directory('ouster_ros'),
        'launch',
        'driver.launch.py'
    )

    # Full path to pcl_tutorial launch file
    pcl_launch = os.path.join(
        get_package_share_directory('pcl_tutorial'),
        'launch',
        'minimal_processing_node_Santora.launch.py'
    )

    return LaunchDescription([
       
       IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ouster_launch),
        ),

       
      #  Node(
        #    package='tf2_ros',
        #    executable='static_transform_publisher',
        #    name='map_to_odom',
        #    output='screen',
        #    arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom']
       # ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'odom', 'base_link']
        ),

        # Mounting transform: base_link -> os_sensor
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_os_lidar',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'base_link', 'os_sensor']
        ),

        # Static transform from laser_sensor_frame -> laser_data_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_pub',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', '1',
                       'laser_sensor_frame', 'laser_data_frame']
        ),

        # Static transform from os_lidar -> laser_sensor_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_pub2',
            output='screen',
            arguments=['0', '0', '-.8', '0', '0', '0', '1',
                       'os_lidar', 'laser_sensor_frame']
        ),

        # Direct transform os_lidar -> laser_data_frame
  #      Node(
   #         package='tf2_ros',
   #         executable='static_transform_publisher',
   #         name='os_lidar_to_laser_data_frame',
   #         output='screen',
    #        arguments=['0', '0', '-.8', '0', '0', '0', '1',
    #                   'os_lidar', 'laser_data_frame']
    #    ),

        # Launch point cloud processing
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pcl_launch),
        ),

        # MarkerArray -> LaserScan
  #      Node(
   #         package='bbox_marker_scan_ros2',
    #        executable='bbox_marker_to_scan',
     #       name='bbox_marker_to_scan',
      #      output='screen',
       #     parameters=[{
        #        'marker_topic': '/bounding_boxes',
         #       'scan_topic': '/fake_scan',
          #      'frame_id': 'base_link'
           # }]
        #),
    ])
