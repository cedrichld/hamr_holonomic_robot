#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <memory>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>


class PngCostmapPublisher : public rclcpp::Node
{
private:
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr height_pub_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr elevation_pub_;
    grid_map::GridMap elevation_map_;
    bool elevation_ready_ = false;
    
    nav_msgs::msg::OccupancyGrid occupancy_grid_;
    
    // Parameters
    double map_width_m_;
    double map_length_m_;
    double map_height_m_;
    std::string image_path_;
    std::string frame_id_;
    double publish_rate_;
    double resolution_;
    
    int image_width_;
    int image_height_;
    cv::Mat img;
    cv::Mat img_gray_;
    cv::Mat height32;
    bool map_loaded_;
    bool grid_ready_{false};

public:
    PngCostmapPublisher() : Node("png_costmap_publisher"), map_loaded_(false)
    {
        // Declare parameters with default values
        this->declare_parameter("map_width_m", 40.0);
        this->declare_parameter("map_length_m", 40.0);
        this->declare_parameter("map_height_m", 2.0);
        this->declare_parameter("image_path", "/home/cedric/ros2_ws/hamr_ws/src/hamr_bringup/terrain_assets/heightmaps/compa_OR_test_map_257.png");
        this->declare_parameter("frame_id", "map");
        this->declare_parameter("publish_rate", 1.0);
        
        // Get parameters
        this->get_parameter("map_width_m", map_width_m_);
        this->get_parameter("map_length_m", map_length_m_);
        this->get_parameter("map_height_m", map_height_m_);
        this->get_parameter("image_path", image_path_);
        this->get_parameter("frame_id", frame_id_);
        this->get_parameter("publish_rate", publish_rate_);
        
        // Publisher with QoS settings for latching behavior
        rclcpp::QoS qos(1);
        qos.transient_local();  // Equivalent to latched in ROS1
        costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/costmap", qos);
        image_pub_   = this->create_publisher<sensor_msgs::msg::Image>("/costmap_image", qos);
        height_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/height_image", qos);
        auto elev_qos = rclcpp::QoS(1).transient_local().reliable();
        elevation_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>("/elevation_map", elev_qos);
        RCLCPP_INFO(get_logger(), "2.5D map on: %s", elevation_pub_->get_topic_name());
        
        // Load and process the image
        if (loadAndProcessImage())
        {
            // Timer for periodic publishing
            auto timer_period = std::chrono::duration<double>(1.0 / publish_rate_);
            // auto timer_period = std::chrono::milliseconds(5000);
            publish_timer_ = this->create_wall_timer(
                timer_period,
                std::bind(&PngCostmapPublisher::publishCostmap, this)
            );
            
            RCLCPP_INFO(this->get_logger(), "PNG Costmap Publisher initialized successfully");
            RCLCPP_INFO(this->get_logger(), "Map resolution: %f meters/pixel", resolution_);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize PNG Costmap Publisher");
        }
    }
    
    bool loadAndProcessImage()
    {
        try
        {
            img = cv::imread(image_path_, cv::IMREAD_GRAYSCALE);
            if (img.empty()){
                RCLCPP_ERROR(this->get_logger(), "Could not load image from %s", image_path_.c_str());
                return false;
            }
            image_height_ = img.rows;
            image_width_ = img.cols;
            img_gray_ = img.clone();
            height32.create(img_gray_.rows, img_gray_.cols, CV_32FC1);
            img_gray_.convertTo(height32, CV_32F, float_t(map_height_m_) / 255.0f);
            RCLCPP_INFO(this->get_logger(), "Loaded image with dimensions: %dx%d", image_width_, image_height_);
            // Calculate resolution
            resolution_ = map_width_m_ / static_cast<double>(image_width_); 
            // Convert to costmap and create occupancy grid
            createOccupancyGrid(img);
            buildElevationGridMap();
            map_loaded_ = true;
            // if (elevation_ready_) {
            // auto gm_msg_ptr = grid_map::GridMapRosConverter::toMessage(elevation_map_);  
            // RCLCPP_INFO(this->get_logger(),
            //             "Publishing elevation map once. frame=%s len=(%.2f, %.2f) res=%.4f layers=%zu",
            //             gm_msg_ptr->header.frame_id.c_str(),
            //             gm_msg_ptr->info.length_x, gm_msg_ptr->info.length_y,
            //             gm_msg_ptr->info.resolution,
            //             gm_msg_ptr->layers.size());
            // elevation_pub_->publish(std::move(gm_msg_ptr));
            // RCLCPP_INFO(this->get_logger(), "Published elevation map (latched).");
            // }
            return true;
        }
        catch (const cv::Exception& e){
            RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
            return false;
        }
        catch (const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Error loading costmap: %s", e.what());
            return false;
        }
    }
    
    void createOccupancyGrid(const cv::Mat& img)
    {
        // Initialize occupancy grid message
        occupancy_grid_.header.frame_id = frame_id_;
        // Map metadata
        occupancy_grid_.info.resolution = resolution_;
        occupancy_grid_.info.width = image_width_;
        occupancy_grid_.info.height = image_height_;
        // Origin (bottom-left corner of the map)
        occupancy_grid_.info.origin.position.x = -20;
        occupancy_grid_.info.origin.position.y = -20;
        occupancy_grid_.info.origin.position.z = 0.0;
        occupancy_grid_.info.origin.orientation.x = 0.0;
        occupancy_grid_.info.origin.orientation.y = 0.0;
        occupancy_grid_.info.origin.orientation.z = 0.0;
        occupancy_grid_.info.origin.orientation.w = 1.0;
        // Prepare data vector
        occupancy_grid_.data.resize(image_width_ * image_height_);
        // Convert image data to ROS format
        // OpenCV uses top-left origin, ROS uses bottom-left origin
        // So we need to flip the image vertically
        for (int row = 0; row < image_height_; ++row)
        {
            for (int col = 0; col < image_width_; ++col)
            {
                // Get pixel value (0-255)
                uint8_t pixel_value = img.at<uint8_t>(row, col);
                // Convert to costmap value (0-100)
                float_t cost_value = convertToCostmapValue(pixel_value);
                // ROS uses bottom-left origin, so flip vertically
                int ros_row = (image_height_ - 1) - row;
                int index = ros_row * image_width_ + col;
                occupancy_grid_.data[index] = cost_value;
            }
        }
    }

    void buildElevationGridMap()
    {
        // One float layer named "elevation"
        elevation_map_ = grid_map::GridMap({"elevation"});
        elevation_map_.setFrameId(frame_id_);
        elevation_map_.setGeometry(                      // full size in meters + resolution
            grid_map::Length(map_width_m_, map_length_m_), 
            resolution_);
        elevation_map_.setPosition(grid_map::Position(0.0, 0.0)); // center at (0,0)

        // Compute physical extents from the *image*
        const double phys_w = image_width_  * resolution_;
        const double phys_h = image_height_ * resolution_;
        const double x0 = -0.5 * phys_w;
        const double y0 = -0.5 * phys_h;

        // Fill the elevation layer from height32 (CV_32FC1)
        // Flip vertically to convert OpenCV top-left -> map bottom-left
        for (int row = 0; row < image_height_; ++row) {
            for (int col = 0; col < image_width_; ++col) {
            const float h = height32.at<float>(row, col);      // meters
            const int ros_row = (image_height_ - 1) - row;     // vertical flip
            const double x = x0 + (col     + 0.5) * resolution_;
            const double y = y0 + (ros_row + 0.5) * resolution_;
            grid_map::Index idx;
            if (elevation_map_.getIndex(grid_map::Position(x, y), idx)) {
                elevation_map_.at("elevation", idx) = h;
            }
            }
        }

        elevation_ready_ = true;
    }
    
    int8_t convertToCostmapValue(uint8_t grayscale_value)
    {
        // Direct linear mapping (0-255 grayscale to 0-100 cost)
        return static_cast<int8_t>((static_cast<float>(grayscale_value) / 255.0f) * 100.0f);
        
        // Custom thresholded mapping 
        /*
        if (grayscale_value < 50)
            return 0;    // Low areas - free space
        else if (grayscale_value < 150)
            return 50;   // Medium areas - moderate cost
        else
            return 100;  // High areas - high cost/obstacles
        */
        
        // Exponential cost mapping for steep penalties
        /*
        float normalized = static_cast<float>(grayscale_value) / 255.0f;
        return static_cast<int8_t>(std::pow(normalized, 2.0f) * 100.0f);
        */
        
        //Inverse mapping (if darker pixels = lower elevation)
        // return static_cast<int8_t>(((255 - static_cast<float>(grayscale_value)) / 255.0f) * 100.0f);
    }
    
    void publishCostmap()
    {
        if (!map_loaded_) return;
        
        const rclcpp::Time stamp = this->get_clock()->now();
        occupancy_grid_.header.stamp = stamp;
        costmap_pub_->publish(occupancy_grid_);
        
        // Images (viz only)
        std_msgs::msg::Header h; h.stamp = stamp; h.frame_id = frame_id_;
        image_pub_->publish(*cv_bridge::CvImage(h, "mono8",  img).toImageMsg());
        height_pub_->publish(*cv_bridge::CvImage(h, "32FC1", height32 ).toImageMsg());
        
        // GridMap (2.5D source of truth with geometry)
        if (elevation_ready_) {
        elevation_map_.setTimestamp(stamp.nanoseconds());
        auto gm_msg = grid_map::GridMapRosConverter::toMessage(elevation_map_);
        elevation_pub_->publish(std::move(gm_msg));
        }
        // publish_timer_->cancel();
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try{
        auto node = std::make_shared<PngCostmapPublisher>();
        rclcpp::spin(node);
    }
    catch (const std::exception& e){
        RCLCPP_ERROR(rclcpp::get_logger("png_costmap_publisher"), "Exception in main: %s", e.what());
        return -1;
    }
    rclcpp::shutdown();
    return 0;
}