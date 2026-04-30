#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

class SORFilterNode : public rclcpp::Node
{
public:
    SORFilterNode()
    : Node("sor_filter_node")
    {
        // Subscription using SensorDataQoS (BestEffort)
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points",
            rclcpp::SensorDataQoS(),
            std::bind(&SORFilterNode::callback, this, std::placeholders::_1)
        );

        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/points_denoised",
            rclcpp::SystemDefaultsQoS()
        );

        RCLCPP_INFO(this->get_logger(), "SOR Filter Node started.");
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(24);
        sor.setStddevMulThresh(1.5);
        sor.filter(*filtered);

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*filtered, output);

        output.header = msg->header; // keep same timestamp & frame

        pub_->publish(output);
    }

    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SORFilterNode>());
    rclcpp::shutdown();
    return 0;
}

