#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <chrono>
#include <string>

using namespace std::chrono_literals;

class PointsLocalFilteredNode :public rclcpp::Node 
{
    public:
    PointsLocalFilteredNode() : Node("points_local_filtered_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        input_topic_=declare_parameter<std::string>("input_topic","/camera/camera/depth/color/points");
        output_topic_=declare_parameter<std::string>("output_topic","/points_local_filtered");
        target_frame_=declare_parameter<std::string>("target_frame","camera_depth_optical_frame");//"base_link");
        tf_timeout_ms_=declare_parameter<int>("tf_timeout_ms",500);
        crop_min_x_=declare_parameter<double>("crop_min_x",0.0);
        crop_max_x_=declare_parameter<double>("crop_max_x",10.0);
        crop_min_y_=declare_parameter<double>("crop_min_y",-5.0);
        crop_max_y_=declare_parameter<double>("crop_max_y",5.0);
        crop_min_z_=declare_parameter<double>("crop_min_z",-0.5);
        crop_max_z_=declare_parameter<double>("crop_max_z",1.5);
        use_passthrough_z_=declare_parameter<bool>("use_passthrough_z",true);
        pass_min_z_=declare_parameter<double>("pass_min_z",-0.5);
        pass_max_z_=declare_parameter<double>("pass_max_z",2.0);
        voxel_leaf_=declare_parameter<double>("voxel_leaf",0.05);
        use_sor_=declare_parameter<bool>("use_sor",true);
        sor_mean_k_=declare_parameter<int>("sor_mean_k",50);
        sor_stddev_mul_=declare_parameter<double>("sor_stddev_mul",1.0);
        sub_=create_subscription<sensor_msgs::msg::PointCloud2>(input_topic_, rclcpp::SensorDataQoS(), std::bind(&PointsLocalFilteredNode::cb, this, std::placeholders::_1));
        pub_=create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::QoS(10));
        RCLCPP_INFO(get_logger(), "Points_Local_Filtered:");
        RCLCPP_INFO(get_logger(), "input topic: %s", input_topic_.c_str());
        RCLCPP_INFO(get_logger(), "output_topic: %s", output_topic_.c_str());
        RCLCPP_INFO(get_logger(), "target_frame: %s", target_frame_.c_str());
        RCLCPP_INFO(get_logger(), "  crop box: x[%.2f, %.2f], y[%.2f, %.2f], z[%.2f, %.2f]",crop_min_x_, crop_max_x_, crop_min_y_, crop_max_y_, crop_min_z_, crop_max_z_);
        RCLCPP_INFO(get_logger(), "  voxel_leaf: %.3f", voxel_leaf_);
        RCLCPP_INFO(get_logger(), "  use_sor: %s", use_sor_ ? "true" : "false");
    }
    private:
    void cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 1. Transform to target frame
        sensor_msgs::msg::PointCloud2 cloud_tf = *msg;
        if (!target_frame_.empty() && msg->header.frame_id != target_frame_)
        {
            try
            {
                auto tf = tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id, tf2::TimePointZero,std::chrono::milliseconds(tf_timeout_ms_));
                tf2::doTransform(*msg, cloud_tf, tf);
            }
            catch(const tf2::TransformException &ex)
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF %s -> %s unavailable: %s (dropping frame)", msg->header.frame_id.c_str(), target_frame_.c_str(), ex.what());
                return;
            }
        }
        // 2. Convert to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(cloud_tf, *cloud);
        if(cloud->empty()) return;
        // 3. Remove NaN
        std::vector<int> idx;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
        if(cloud->empty()) return;
        // 4. Crop Box
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f((float)crop_min_x_, (float)crop_min_y_, (float)crop_min_z_, 1.0f));
        crop.setMax(Eigen::Vector4f((float)crop_max_x_, (float)crop_max_y_, (float)crop_max_z_, 1.0f));
        crop.filter(*cropped);
        if(cropped->empty()) return;
        // 5. PassThrough Z
        pcl::PointCloud<pcl::PointXYZ>::Ptr passed = cropped;
        if (use_passthrough_z_) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cropped);
        pass.setFilterFieldName("z");
        pass.setFilterLimits((float)pass_min_z_, (float)pass_max_z_);
        pass.filter(*tmp);
        passed = tmp;
        if (passed->empty()) return;
        }
        // 6. Statistical Outlier Removal
        pcl::PointCloud<pcl::PointXYZ>::Ptr denoised = passed;
        if (use_sor_) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(passed);
        sor.setMeanK(sor_mean_k_);
        sor.setStddevMulThresh(sor_stddev_mul_);
        sor.filter(*tmp);
        denoised = tmp;
        if (denoised->empty()) return;
        }
        // 7. Voxel Grid Downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr voxel(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(denoised);
        vg.setLeafSize((float)voxel_leaf_, (float)voxel_leaf_, (float)voxel_leaf_);
        vg.filter(*voxel);
        if (voxel->empty()) return;
        // 8. Convert to ROS msg and publish
        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(*voxel, out);
        out.header.stamp = msg->header.stamp;
        out.header.frame_id = target_frame_.empty() ? msg->header.frame_id : target_frame_;
        pub_->publish(out);
    }

    // Params
    std::string input_topic_;
    std::string output_topic_;
    std::string target_frame_;
    int tf_timeout_ms_;
    double crop_min_x_;
    double crop_max_x_;
    double crop_min_y_;
    double crop_max_y_;
    double crop_min_z_;
    double crop_max_z_;
    bool use_passthrough_z_;
    double pass_min_z_;
    double pass_max_z_;
    bool use_sor_;
    int sor_mean_k_;
    double sor_stddev_mul_;
    double voxel_leaf_;
    // ROS
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointsLocalFilteredNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}