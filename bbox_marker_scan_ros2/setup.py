from setuptools import find_packages, setup

package_name = 'bbox_marker_scan_ros2'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bbox_to_scan.launch.py']),
        ('share/' + package_name + '/config', ['config/bbox_to_scan.params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elias',
    maintainer_email='elias@example.com',
    description='Convert 2D bounding-box MarkerArray outlines into LaserScan.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bbox_marker_to_scan = bbox_marker_scan_ros2.bbox_marker_to_scan:main',
        ],
    },
)
