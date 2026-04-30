from setuptools import find_packages, setup

package_name = 'camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sd1',
    maintainer_email='sd1@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'yolo_detector_node = camera.yolo_detector_node:main',
        'yolopv2_lane_node = camera.yolopv2_lane_node_xyz:main',
        'lane_to_scan_node = camera.lane_to_scan_node:main',
        ],
    },
)
