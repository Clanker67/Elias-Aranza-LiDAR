# bbox_marker_scan_ros2

ROS 2 Python package that converts `visualization_msgs/msg/MarkerArray` box outlines into a synthetic `sensor_msgs/msg/LaserScan`.

## What it expects
This is built for markers like the ones you pasted:
- namespace: `bbox_2d_outline`
- marker type: `LINE_STRIP` or `LINE_LIST`
- points already in the scan frame, usually `base_link`

It ignores markers like:
- `bbox_height`
- text markers
- delete markers

## What it does
For every bbox outline marker, the node:
1. pulls the 2D `x,y` edges from the marker points
2. builds line segments
3. raycasts across the requested angular field of view
4. publishes the nearest hit per ray as a `LaserScan`

## Package layout
```text
bbox_marker_scan_ros2/
├── bbox_marker_scan_ros2/
│   ├── __init__.py
│   └── bbox_marker_to_scan.py
├── config/
│   └── bbox_to_scan.params.yaml
├── launch/
│   └── bbox_to_scan.launch.py
├── resource/
│   └── bbox_marker_scan_ros2
├── package.xml
├── setup.cfg
├── setup.py
└── README.md
```

## Build
```bash
cd ~/ros2_ws/src
cp -r /mnt/data/bbox_marker_scan_ros2 .
cd ..
colcon build --packages-select bbox_marker_scan_ros2
source install/setup.bash
```

## Run
```bash
ros2 launch bbox_marker_scan_ros2 bbox_to_scan.launch.py
```

## Run with your own topics
```bash
ros2 launch bbox_marker_scan_ros2 bbox_to_scan.launch.py \
  marker_topic:=/your_bbox_marker_topic \
  scan_topic:=/your_fake_scan_topic
```

## Run directly
```bash
ros2 run bbox_marker_scan_ros2 bbox_marker_to_scan --ros-args \
  -p marker_topic:=/bbox_markers \
  -p scan_topic:=/fake_scan \
  -p target_namespace:=bbox_2d_outline \
  -p frame_id:=base_link
```

## Main parameters
- `marker_topic`: input `MarkerArray`
- `scan_topic`: output `LaserScan`
- `target_namespace`: which marker namespace to use
- `frame_id`: override output frame; empty means use marker frame
- `use_marker_header_stamp`: use input marker stamp instead of node time
- `angle_min`, `angle_max`, `angle_increment`: scan geometry
- `range_min`, `range_max`: valid scan range
- `publish_inf_for_no_return`: publish `inf` when no ray hit exists

## Notes for SLAM
- the bbox markers need to already be in the frame your SLAM expects
- if your SLAM is using `base_scan`, either publish there or remap/transform accordingly
- this creates a geometric fake scan from box edges only, not from raw point cloud density
