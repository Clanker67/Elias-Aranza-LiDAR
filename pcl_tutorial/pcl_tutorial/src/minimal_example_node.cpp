/**
 * Minimal working example of PCL and ROS 2
 *
 * CHANGES:
 * - ground plane is segmented and removed from the scan crop
 * - /aroundbox publishes nonground scan cloud
 * - bounding boxes come from nonground scan cloud
 * - fake_scan comes from nonground scan cloud
 */

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/header.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <array>
#include <algorithm>

class MinimalPointCloudProcessor : public rclcpp::Node
{
public:
  MinimalPointCloudProcessor()
  : Node("minimal_point_cloud_processor",
         rclcpp::NodeOptions()
           .allow_undeclared_parameters(true)
           .automatically_declare_parameters_from_overrides(true))
  {
    RCLCPP_INFO(this->get_logger(), "Setting up publishers");

    voxel_grid_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("voxel_cluster", 1);
    lanebox_pub_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("lanebox", 1);
    right_pub_      = this->create_publisher<sensor_msgs::msg::PointCloud2>("rightside", 1);
    left_pub_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("leftside", 1);
    stop_pub_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("stopsign", 1);
    stop_pub_2      = this->create_publisher<sensor_msgs::msg::PointCloud2>("stopsign2", 1);
    around_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("aroundbox", 1);

    bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "bounding_boxes", rclcpp::SensorDataQoS());
    scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
      "fake_scan", rclcpp::SensorDataQoS());
    ranked_scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
      "ranked_object_scan", rclcpp::SensorDataQoS());

    RCLCPP_INFO(this->get_logger(), "Getting parameters");

    rclcpp::Parameter cloud_topic_param, world_frame_param, camera_frame_param, scan_frame_param;
    rclcpp::Parameter voxel_leaf_size_param, x_filter_min_param, x_filter_max_param;
    rclcpp::Parameter y_filter_min_param, y_filter_max_param, z_filter_min_param, z_filter_max_param;
    rclcpp::Parameter cluster_tolerance_param, cluster_min_size_param, cluster_max_size_param;
    rclcpp::Parameter scan_x_min_param, scan_x_max_param;
    rclcpp::Parameter scan_y_min_param, scan_y_max_param;
    rclcpp::Parameter scan_z_min_param, scan_z_max_param;
    rclcpp::Parameter ground_distance_threshold_param, ground_max_iterations_param;

    this->get_parameter_or("cloud_topic", cloud_topic_param, rclcpp::Parameter("", "/points"));
    this->get_parameter_or("world_frame", world_frame_param, rclcpp::Parameter("", "map"));
    this->get_parameter_or("camera_frame", camera_frame_param, rclcpp::Parameter("", "map"));
    this->get_parameter_or("scan_frame", scan_frame_param, rclcpp::Parameter("", "base_link"));

    this->get_parameter_or("voxel_leaf_size", voxel_leaf_size_param, rclcpp::Parameter("", 0.08));
    this->get_parameter_or("x_filter_min", x_filter_min_param, rclcpp::Parameter("", 1.0));
    this->get_parameter_or("x_filter_max", x_filter_max_param, rclcpp::Parameter("", 120.0));
    this->get_parameter_or("y_filter_min", y_filter_min_param, rclcpp::Parameter("", -25.0));
    this->get_parameter_or("y_filter_max", y_filter_max_param, rclcpp::Parameter("", 10.0));
    this->get_parameter_or("z_filter_min", z_filter_min_param, rclcpp::Parameter("", -1.0));
    this->get_parameter_or("z_filter_max", z_filter_max_param, rclcpp::Parameter("", 8.0));

    this->get_parameter_or("cluster_tolerance", cluster_tolerance_param, rclcpp::Parameter("", 1.0));
    this->get_parameter_or("cluster_min_size", cluster_min_size_param, rclcpp::Parameter("", 6));
    this->get_parameter_or("cluster_max_size", cluster_max_size_param, rclcpp::Parameter("", 2000));

    this->get_parameter_or("scan_x_min", scan_x_min_param, rclcpp::Parameter("", -3.0));
    this->get_parameter_or("scan_x_max", scan_x_max_param, rclcpp::Parameter("", 3.0));
    this->get_parameter_or("scan_y_min", scan_y_min_param, rclcpp::Parameter("", -1.5));
    this->get_parameter_or("scan_y_max", scan_y_max_param, rclcpp::Parameter("", 6.0));
    this->get_parameter_or("scan_z_min", scan_z_min_param, rclcpp::Parameter("", -.7));
    this->get_parameter_or("scan_z_max", scan_z_max_param, rclcpp::Parameter("", 1.5));

    this->get_parameter_or("ground_distance_threshold", ground_distance_threshold_param, rclcpp::Parameter("", 0.03));
    this->get_parameter_or("ground_max_iterations", ground_max_iterations_param, rclcpp::Parameter("", 100));

    cloud_topic     = cloud_topic_param.as_string();
    world_frame     = world_frame_param.as_string();
    camera_frame    = camera_frame_param.as_string();
    scan_frame      = scan_frame_param.as_string();

    voxel_leaf_size = static_cast<float>(voxel_leaf_size_param.as_double());
    x_filter_min    = x_filter_min_param.as_double();
    x_filter_max    = x_filter_max_param.as_double();
    y_filter_min    = y_filter_min_param.as_double();
    y_filter_max    = y_filter_max_param.as_double();
    z_filter_min    = z_filter_min_param.as_double();
    z_filter_max    = z_filter_max_param.as_double();

    cluster_tolerance_ = static_cast<float>(cluster_tolerance_param.as_double());
    cluster_min_size_  = cluster_min_size_param.as_int();
    cluster_max_size_  = cluster_max_size_param.as_int();

    scan_x_min_ = scan_x_min_param.as_double();
    scan_x_max_ = scan_x_max_param.as_double();
    scan_y_min_ = scan_y_min_param.as_double();
    scan_y_max_ = scan_y_max_param.as_double();
    scan_z_min_ = scan_z_min_param.as_double();
    scan_z_max_ = scan_z_max_param.as_double();

    ground_distance_threshold_ = static_cast<float>(ground_distance_threshold_param.as_double());
    ground_max_iterations_     = ground_max_iterations_param.as_int();

    RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", cloud_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "world_frame: %s", world_frame.c_str());
    RCLCPP_INFO(this->get_logger(), "scan_frame: %s", scan_frame.c_str());
    RCLCPP_INFO(
      this->get_logger(),
      "Scan crop: x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]",
      scan_x_min_, scan_x_max_, scan_y_min_, scan_y_max_, scan_z_min_, scan_z_max_);
    RCLCPP_INFO(
      this->get_logger(),
      "Ground segmentation: distance_threshold=%.3f max_iterations=%d",
      ground_distance_threshold_, ground_max_iterations_);

    cloud_subscriber_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic,
        rclcpp::SensorDataQoS(),
        std::bind(&MinimalPointCloudProcessor::cloud_callback, this, std::placeholders::_1));

    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr voxel_grid_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lanebox_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr right_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr left_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stop_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stop_pub_2;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr around_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr ranked_scan_pub_;

  std::string cloud_topic;
  std::string world_frame;
  std::string camera_frame;
  std::string scan_frame;

  float voxel_leaf_size;
  float x_filter_min, x_filter_max;
  float y_filter_min, y_filter_max;
  float z_filter_min, z_filter_max;

  float scan_x_min_, scan_x_max_;
  float scan_y_min_, scan_y_max_;
  float scan_z_min_, scan_z_max_;

  float cluster_tolerance_;
  int cluster_min_size_;
  int cluster_max_size_;

  float ground_distance_threshold_;
  int ground_max_iterations_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> br_;

  struct Box2D
  {
    float min_x, max_x;
    float min_y, max_y;
  };

  static float boxCenterDistance(const Box2D& b)
  {
    const float cx = 0.5f * (b.min_x + b.max_x);
    const float cy = 0.5f * (b.min_y + b.max_y);
    return std::sqrt(cx * cx + cy * cy);
  }

  static int countFiniteRanges(const sensor_msgs::msg::LaserScan& scan)
  {
    int count = 0;
    for (const auto &r : scan.ranges)
    {
      if (std::isfinite(r)) {
        ++count;
      }
    }
    return count;
  }

  void publishPointCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher,
    const pcl::PointCloud<pcl::PointXYZI>& point_cloud,
    const std::string& frame_id)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(point_cloud, msg);
    msg.header.frame_id = frame_id;
    msg.header.stamp = this->get_clock()->now();
    publisher->publish(msg);
  }

  pcl::PointCloud<pcl::PointXYZI> removeGroundPlane(
    const pcl::PointCloud<pcl::PointXYZI>& input_cloud) const
  {
    if (input_cloud.empty()) {
      return input_cloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>(input_cloud));
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(ground_max_iterations_);
    seg.setDistanceThreshold(ground_distance_threshold_);
    seg.setInputCloud(cloud_ptr);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
      return input_cloud;
    }

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    pcl::PointCloud<pcl::PointXYZI> nonground_cloud;
    extract.setInputCloud(cloud_ptr);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(nonground_cloud);

    return nonground_cloud;
  }

  bool raySegmentIntersection(
    float ray_dx, float ray_dy,
    float x1, float y1,
    float x2, float y2,
    float& out_t) const
  {
    const float sx = x2 - x1;
    const float sy = y2 - y1;

    const float denom = ray_dx * sy - ray_dy * sx;

    if (std::fabs(denom) < 1e-6f)
      return false;

    const float qpx = x1;
    const float qpy = y1;

    const float t = (qpx * sy - qpy * sx) / denom;
    const float u = (qpx * ray_dy - qpy * ray_dx) / denom;

    if (t >= 0.0f && u >= 0.0f && u <= 1.0f)
    {
      out_t = t;
      return true;
    }

    return false;
  }

  sensor_msgs::msg::LaserScan pointCloudToScan(
    const pcl::PointCloud<pcl::PointXYZI>& cloud,
    const std_msgs::msg::Header& header) const
  {
    sensor_msgs::msg::LaserScan scan;
    scan.header = header;

    scan.angle_min = -static_cast<float>(M_PI);
    scan.angle_max =  static_cast<float>(M_PI);
    scan.angle_increment = 0.008726646f;  // 0.5 deg
    scan.time_increment = 0.0f;
    scan.scan_time = 0.1f;
    scan.range_min = 0.05f;
    scan.range_max = 30.0f;

    const int beam_count =
      static_cast<int>(std::floor((scan.angle_max - scan.angle_min) / scan.angle_increment)) + 1;

    scan.ranges.assign(beam_count, std::numeric_limits<float>::infinity());
    scan.intensities.assign(beam_count, 0.0f);

    for (const auto& pt : cloud.points)
    {
      const float x = pt.x;
      const float y = pt.y;
      const float r = std::hypot(x, y);

      if (r < scan.range_min || r > scan.range_max)
        continue;

      const float angle = std::atan2(y, x);
      if (angle < scan.angle_min || angle > scan.angle_max)
        continue;

      const int idx = static_cast<int>((angle - scan.angle_min) / scan.angle_increment);
      if (idx < 0 || idx >= beam_count)
        continue;

      if (!std::isfinite(scan.ranges[idx]) || r < scan.ranges[idx])
      {
        scan.ranges[idx] = r;
      }
      scan.intensities[idx] = 1.0f;

      const int spread = 6;
      for (int k = -spread; k <= spread; ++k)
      {
        const int j = idx + k;
        if (j >= 0 && j < beam_count)
        {
          if (!std::isfinite(scan.ranges[j]) || r < scan.ranges[j])
          {
            scan.ranges[j] = r;
          }
          scan.intensities[j] = 1.0f;
        }
      }
    }

    return scan;
  }

  sensor_msgs::msg::LaserScan boxesToScan(
    const std::vector<Box2D>& boxes,
    const std_msgs::msg::Header& header,
    bool sort_by_distance = false) const
  {
    sensor_msgs::msg::LaserScan scan;
    scan.header = header;

    scan.angle_min = -static_cast<float>(M_PI);
    scan.angle_max =  static_cast<float>(M_PI);
    scan.angle_increment = 0.008726646f;  // 0.5 deg
    scan.time_increment = 0.0f;
    scan.scan_time = 0.1f;
    scan.range_min = 0.05f;
    scan.range_max = 30.0f;

    const int beam_count =
      static_cast<int>(std::floor((scan.angle_max - scan.angle_min) / scan.angle_increment)) + 1;

    scan.ranges.assign(beam_count, std::numeric_limits<float>::infinity());
    scan.intensities.assign(beam_count, 0.0f);

    std::vector<Box2D> boxes_local = boxes;
    if (sort_by_distance)
    {
      std::sort(
        boxes_local.begin(), boxes_local.end(),
        [](const Box2D& a, const Box2D& b)
        {
          return boxCenterDistance(a) < boxCenterDistance(b);
        });
    }

    for (int i = 0; i < beam_count; ++i)
    {
      const float angle = scan.angle_min + i * scan.angle_increment;
      const float dx = std::cos(angle);
      const float dy = std::sin(angle);

      float best_range = scan.range_max;
      bool hit = false;

      for (const auto& b : boxes_local)
      {
        const float inflate_x = 0.15f;
        const float inflate_y = 0.15f;

        Box2D bi;
        bi.min_x = b.min_x - inflate_x;
        bi.max_x = b.max_x + inflate_x;
        bi.min_y = b.min_y - inflate_y;
        bi.max_y = b.max_y + inflate_y;

        const std::array<std::array<float, 4>, 4> edges = {{
          {bi.min_x, bi.min_y, bi.max_x, bi.min_y},
          {bi.max_x, bi.min_y, bi.max_x, bi.max_y},
          {bi.max_x, bi.max_y, bi.min_x, bi.max_y},
          {bi.min_x, bi.max_y, bi.min_x, bi.min_y}
        }};

        for (const auto& e : edges)
        {
          float t;
          if (raySegmentIntersection(dx, dy, e[0], e[1], e[2], e[3], t))
          {
            if (t >= scan.range_min && t <= scan.range_max)
            {
              if (t < best_range)
              {
                best_range = t;
                hit = true;
              }
            }
          }
        }
      }

      if (hit)
      {
        const int spread = 3;
        for (int k = -spread; k <= spread; ++k)
        {
          const int j = i + k;
          if (j >= 0 && j < beam_count)
          {
            if (!std::isfinite(scan.ranges[j]) || best_range < scan.ranges[j])
            {
              scan.ranges[j] = best_range;
            }
            scan.intensities[j] = 1.0f;
          }
        }
      }
    }

    return scan;
  }

  std::vector<Box2D> transformWorldBoxesToScanFrame(
    const std::vector<Box2D>& world_boxes,
    const geometry_msgs::msg::TransformStamped& scan_from_world_tf) const
  {
    std::vector<Box2D> scan_boxes;
    scan_boxes.reserve(world_boxes.size());

    const double tx = scan_from_world_tf.transform.translation.x;
    const double ty = scan_from_world_tf.transform.translation.y;

    const double qx = scan_from_world_tf.transform.rotation.x;
    const double qy = scan_from_world_tf.transform.rotation.y;
    const double qz = scan_from_world_tf.transform.rotation.z;
    const double qw = scan_from_world_tf.transform.rotation.w;

    const double yaw = std::atan2(
      2.0 * (qw * qz + qx * qy),
      1.0 - 2.0 * (qy * qy + qz * qz));

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    for (const auto& b : world_boxes)
    {
      std::array<std::pair<double, double>, 4> corners = {{
        {b.min_x, b.min_y},
        {b.max_x, b.min_y},
        {b.max_x, b.max_y},
        {b.min_x, b.max_y}
      }};

      double min_x = std::numeric_limits<double>::max();
      double max_x = -std::numeric_limits<double>::max();
      double min_y = std::numeric_limits<double>::max();
      double max_y = -std::numeric_limits<double>::max();

      for (const auto& p : corners)
      {
        const double xw = p.first;
        const double yw = p.second;

        const double dx = xw - tx;
        const double dy = yw - ty;

        const double xs =  c * dx + s * dy;
        const double ys = -s * dx + c * dy;

        min_x = std::min(min_x, xs);
        max_x = std::max(max_x, xs);
        min_y = std::min(min_y, ys);
        max_y = std::max(max_y, ys);
      }

      Box2D sb;
      sb.min_x = static_cast<float>(min_x);
      sb.max_x = static_cast<float>(max_x);
      sb.min_y = static_cast<float>(min_y);
      sb.max_y = static_cast<float>(max_y);

      if (sb.max_x < scan_x_min_ || sb.min_x > scan_x_max_ ||
          sb.max_y < scan_y_min_ || sb.min_y > scan_y_max_)
      {
        continue;
      }

      scan_boxes.push_back(sb);
    }

    return scan_boxes;
  }

  std::vector<Box2D> publishBoundingBoxes(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const std_msgs::msg::Header& header)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    std::vector<Box2D> boxes;

    if (!cloud || cloud->empty())
    {
      bbox_pub_->publish(marker_array);
      return boxes;
    }

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(cluster_min_size_);
    ec.setMaxClusterSize(cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    int id = 0;

    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZI> cluster;
      cluster.reserve(indices.indices.size());

      for (int idx : indices.indices)
        cluster.push_back((*cloud)[idx]);

      pcl::PointXYZI min_pt, max_pt;
      pcl::getMinMax3D(cluster, min_pt, max_pt);

      const float dx = std::max(0.01f, (max_pt.x - min_pt.x));
      const float dy = std::max(0.01f, (max_pt.y - min_pt.y));
      const float dz = std::max(0.01f, (max_pt.z - min_pt.z));

      Box2D b;
      b.min_x = min_pt.x;
      b.max_x = max_pt.x;
      b.min_y = min_pt.y;
      b.max_y = max_pt.y;
      boxes.push_back(b);

      const float z_floor = min_pt.z;

      visualization_msgs::msg::Marker outline;
      outline.header = header;
      outline.ns = "bbox_2d_outline";
      outline.id = id++;
      outline.type = visualization_msgs::msg::Marker::LINE_STRIP;
      outline.action = visualization_msgs::msg::Marker::ADD;
      outline.pose.orientation.w = 1.0;
      outline.scale.x = 0.06;
      outline.color.a = 1.0;
      outline.color.r = 0.0;
      outline.color.g = 1.0;
      outline.color.b = 0.0;

      geometry_msgs::msg::Point p;
      p.x = min_pt.x; p.y = min_pt.y; p.z = z_floor; outline.points.push_back(p);
      p.x = max_pt.x; p.y = min_pt.y; p.z = z_floor; outline.points.push_back(p);
      p.x = max_pt.x; p.y = max_pt.y; p.z = z_floor; outline.points.push_back(p);
      p.x = min_pt.x; p.y = max_pt.y; p.z = z_floor; outline.points.push_back(p);
      p.x = min_pt.x; p.y = min_pt.y; p.z = z_floor; outline.points.push_back(p);

      outline.lifetime.sec = 0;
      outline.lifetime.nanosec = 200000000;
      marker_array.markers.push_back(outline);

      visualization_msgs::msg::Marker label;
      label.header = header;
      label.ns = "bbox_height";
      label.id = id++;
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.action = visualization_msgs::msg::Marker::ADD;
      label.pose.position.x = (min_pt.x + max_pt.x) * 0.5f;
      label.pose.position.y = (min_pt.y + max_pt.y) * 0.5f;
      label.pose.position.z = z_floor + 0.25f;
      label.pose.orientation.w = 1.0;
      label.scale.z = 0.35f;
      label.color.a = 1.0f;
      label.color.r = 1.0f;
      label.color.g = 1.0f;
      label.color.b = 1.0f;

      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2)
         << "W:" << dy << "  L:" << dx << "  H:" << dz;

      label.text = ss.str();
      label.lifetime.sec = 0;
      label.lifetime.nanosec = 200000000;
      marker_array.markers.push_back(label);
    }

    bbox_pub_->publish(marker_array);
    return boxes;
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr recent_cloud)
  {
    auto start = std::chrono::high_resolution_clock::now();

    geometry_msgs::msg::TransformStamped world_transform;
    try
    {
      world_transform = tf_buffer_->lookupTransform(
        world_frame,
        recent_cloud->header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(3));
    }
    catch (const tf2::TransformException &ex)
    {
      RCLCPP_ERROR(this->get_logger(), "World transform error: %s", ex.what());
      return;
    }

    sensor_msgs::msg::PointCloud2 world_cloud_msg;
    pcl_ros::transformPointCloud(world_frame, world_transform, *recent_cloud, world_cloud_msg);

    pcl::PointCloud<pcl::PointXYZI> world_cloud;
    pcl::fromROSMsg(world_cloud_msg, world_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>(world_cloud));
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud_voxel_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter_world;
    voxel_filter_world.setInputCloud(world_cloud_ptr);
    voxel_filter_world.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_filter_world.filter(*world_cloud_voxel_filtered);

    pcl::PointCloud<pcl::PointXYZI> around_xyz_filtered_cloud;
    pcl::CropBox<pcl::PointXYZI> crop_all;
    crop_all.setInputCloud(world_cloud_voxel_filtered);
    crop_all.setMin(Eigen::Vector4f(-1.0f, -6.0f, -2.0f, 0.0f));
    crop_all.setMax(Eigen::Vector4f( 1.0f,  6.0f,  3.0f, 0.0f));
    crop_all.filter(around_xyz_filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI> xyz_filtered_cloud;
    pcl::CropBox<pcl::PointXYZI> crop;
    crop.setInputCloud(world_cloud_voxel_filtered);
    crop.setMin(Eigen::Vector4f(x_filter_min, y_filter_min, z_filter_min, 0.0f));
    crop.setMax(Eigen::Vector4f(x_filter_max, y_filter_max, z_filter_max, 0.0f));
    crop.filter(xyz_filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI> right_xyz_filtered_cloud;
    pcl::CropBox<pcl::PointXYZI> cropr;
    cropr.setInputCloud(world_cloud_voxel_filtered);
    cropr.setMin(Eigen::Vector4f(0, -10, -15, 0));
    cropr.setMax(Eigen::Vector4f(10, 10, 15, 0));
    cropr.filter(right_xyz_filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI> left_xyz_filtered_cloud;
    pcl::CropBox<pcl::PointXYZI> cropl;
    cropl.setInputCloud(world_cloud_voxel_filtered);
    cropl.setMin(Eigen::Vector4f(-10, -10, -15, 0));
    cropl.setMax(Eigen::Vector4f(0, 10, 15, 0));
    cropl.filter(left_xyz_filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI> stop_xyz_filtered_cloud;
    pcl::CropBox<pcl::PointXYZI> crops;
    crops.setInputCloud(world_cloud_voxel_filtered);
    crops.setMin(Eigen::Vector4f(0, 0, -15, 0));
    crops.setMax(Eigen::Vector4f(10, 10, 15, 0));
    crops.filter(stop_xyz_filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI> stop2_xyz_filtered_cloud;
    pcl::PassThrough<pcl::PointXYZI> passs;
    passs.setInputCloud(stop_xyz_filtered_cloud.makeShared());
    passs.setFilterFieldName("intensity");
    passs.setFilterLimits(4500, 50000);
    passs.filter(stop2_xyz_filtered_cloud);

    publishPointCloud(voxel_grid_pub_, *world_cloud_voxel_filtered, world_frame);
    publishPointCloud(lanebox_pub_, xyz_filtered_cloud, world_frame);
    publishPointCloud(right_pub_, right_xyz_filtered_cloud, world_frame);
    publishPointCloud(left_pub_, left_xyz_filtered_cloud, world_frame);
    publishPointCloud(stop_pub_, stop_xyz_filtered_cloud, world_frame);
    publishPointCloud(stop_pub_2, stop2_xyz_filtered_cloud, world_frame);

    geometry_msgs::msg::TransformStamped scan_transform;
    try
    {
      scan_transform = tf_buffer_->lookupTransform(
        scan_frame,
        recent_cloud->header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(3));
    }
    catch (const tf2::TransformException &ex)
    {
      RCLCPP_ERROR(this->get_logger(), "Scan transform error: %s", ex.what());
      return;
    }

    sensor_msgs::msg::PointCloud2 scan_cloud_msg_ros;
    pcl_ros::transformPointCloud(scan_frame, scan_transform, *recent_cloud, scan_cloud_msg_ros);

    pcl::PointCloud<pcl::PointXYZI> scan_cloud_raw;
    pcl::fromROSMsg(scan_cloud_msg_ros, scan_cloud_raw);

    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan_cloud_raw));
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_cloud_voxel_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter_scan;
    voxel_filter_scan.setInputCloud(scan_cloud_ptr);
    voxel_filter_scan.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_filter_scan.filter(*scan_cloud_voxel_filtered);

    pcl::PointCloud<pcl::PointXYZI> scan_filtered_cloud;
    pcl::CropBox<pcl::PointXYZI> scan_crop;
    scan_crop.setInputCloud(scan_cloud_voxel_filtered);
    scan_crop.setMin(Eigen::Vector4f(scan_x_min_, scan_y_min_, scan_z_min_, 0.0f));
    scan_crop.setMax(Eigen::Vector4f(scan_x_max_, scan_y_max_, scan_z_max_, 0.0f));
    scan_crop.filter(scan_filtered_cloud);

    pcl::PointCloud<pcl::PointXYZI> scan_nonground_cloud = removeGroundPlane(scan_filtered_cloud);

    publishPointCloud(around_pub_, scan_nonground_cloud, scan_frame);

    std_msgs::msg::Header scan_hdr;
    scan_hdr.frame_id = scan_frame;
    scan_hdr.stamp = this->get_clock()->now();

    auto scan_boxes = publishBoundingBoxes(scan_nonground_cloud.makeShared(), scan_hdr);

    auto fake_scan_msg = pointCloudToScan(scan_nonground_cloud, scan_hdr);
    scan_pub_->publish(fake_scan_msg);

    ranked_scan_pub_->publish(boxesToScan(scan_boxes, scan_hdr, true));

    auto stop = std::chrono::high_resolution_clock::now();
    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    RCLCPP_INFO(
      this->get_logger(),
      "Time (msec): %ld | scan boxes: %zu | fake_scan finite hits: %d | raw scan pts: %zu | nonground pts: %zu | lane pts: %zu | voxel pts: %zu",
      t_ms.count(),
      scan_boxes.size(),
      countFiniteRanges(fake_scan_msg),
      scan_filtered_cloud.size(),
      scan_nonground_cloud.size(),
      xyz_filtered_cloud.size(),
      world_cloud_voxel_filtered->size());
  }
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MinimalPointCloudProcessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
