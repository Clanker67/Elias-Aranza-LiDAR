from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pcl_tutorial',
            executable='minimal_pcl_tutorial',
            name='minimal_example_node',
            # prefix='valgrind --leak-check=yes ',      # Uncomment to run valgrind 
            output='screen',
            parameters=[
                {"cloud_topic": "/ouster/points"},
                {"world_frame": "base_link"},
                {"camera_frame": "os_lidar"},
                {"voxel_leaf_size": 0.03},          # Size in meters of voxel grid filter
                {"x_filter_min": -2.5},
		{"x_filter_max": 2.5},
		{"y_filter_min": -6.0},
		{"y_filter_max": 6.0},
		{"z_filter_min": -2.0},
		{"z_filter_max": 3.0},
		{"cluster_tolerance": .28},
		{"cluster_min_size": 20},
		{"cluster_max_size": 200},
               # {"x_filter_min": -2.5},              # Dimensions in meters to crop point cloud
               # {"x_filter_max": 2.5},
               # {"y_filter_min": -6.0},
               # {"y_filter_max": 6.0},
               # {"z_filter_min": -2.0},
               # {"z_filter_max": 3.0},
                {"plane_max_iterations": 100.0},      # RANSAC max iteration value
                {"plane_distance_threshold": 0.04},  # in meters
              #  {"cluster_tolerance": .28},         # in meters
              #  {"cluster_min_size": 20},            # number of points
              #  {"cluster_max_size": 200}
            ]
        )
     ])
