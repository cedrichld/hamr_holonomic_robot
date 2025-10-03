#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <yaml-cpp/yaml.h>
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <cmath>

class MazeRunner : public rclcpp::Node {
public:
    MazeRunner() : rclcpp::Node("maze_runner")
    {
        // Map Parameters
        yaml_path_   = declare_parameter<std::string>("yaml_path", "/home/cedric/ros2_ws/hamr_ws/src/map/map.yaml"); 
        image_path_  = declare_parameter<std::string>("image", "");   
        resolution_  = declare_parameter<double>("resolution", 0.05);
        origin_xyz_  = declare_parameter<std::vector<double>>("origin", {0.0, 0.0, 0.0});
        negate_      = declare_parameter<int>("negate", 0);
        occ_thresh_  = declare_parameter<double>("occupied_thresh", 0.65);
        free_thresh_ = declare_parameter<double>("free_thresh", 0.196);
        frame_id_    = declare_parameter<std::string>("frame_id", "map");
        publish_hz_  = declare_parameter<double>("publish_rate_hz", 2.0);

        width_ = declare_parameter("width", 400);
        height_ = declare_parameter("height", 400);
        origin_x_ = declare_parameter("origin_x", -10.0);
        origin_y_ = declare_parameter("origin_y", -10.0);
        border_walls_ = declare_parameter("border_walls", true);
        occupied_val_= declare_parameter("occupied_val", 100);
        unknown_val_= declare_parameter("unknown_val", -1);
        // obstacles_    = declare_parameter<std::vector<int64_t>>("obstacles", {});
    
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.reliable().transient_local();
        map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/map",qos);

        if (!load_config_and_image()){
            RCLCPP_FATAL(get_logger(), "Failed to load configuration or image");
            return;
        }
        RCLCPP_INFO(get_logger(), "yaml_path='%s' image='%s'", yaml_path_.c_str(), image_path_.c_str());

        
    // build_map();
    // timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]{
    //  map_.header.stamp = now();
    //  map_pub_->publish(map_);
    // });
    // RCLCPP_INFO(get_logger(), "maze_map_publisher up: %dx%d cells @ %.3fm/cell (origin %.2f, %.2f)", frame='%s',
    //             width_, height_, resolution_, origin_x_, origin_y_, frame_id_.c_str());

    auto period = std::chrono::milliseconds((int)std::round(1000.0 / std::max(0.1, publish_hz_)));
    timer_=create_wall_timer(period, [this]{
        grid_.header.stamp = now();
        map_pub_->publish(grid_);
    });
    RCLCPP_INFO(get_logger(), "Publishing map %ux%u @ %.3fm/cell, frame='%s'",
                grid_.info.width, grid_.info.height, grid_.info.resolution, frame_id_.c_str());
    }

private:

    struct PgmImage {
        int width{0}, height{0};
        int max_val{255};
        std::vector<uint16_t> pixels;
    };

    static void skip_comments(std::istream& is){
        while(true){
                int c=is.peek();
                if(c=='#'){ std::string dummy; std::getline(is, dummy); }
                else if (std::isspace(c)) { is.get(); }
                else break;
        }
    }

    static bool read_int(std::istream& is, int& out) {
        skip_comments(is);
        if (!(is >> out)) return false;
        return true;
    }

    static bool load_pgm(const std::string& path, PgmImage& img, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err = "Cannot open PGM: " + path; return false; }

        std::string magic;
        f >> magic;
        if (magic != "P5" && magic != "P2") { err = "Unsupported magic (need P5 or P2): " + magic; return false; }

        int w=0, h=0, maxv=0;
        if (!read_int(f, w) || !read_int(f, h) || !read_int(f, maxv)) {
        err = "Failed to read PGM header"; return false;
        }
        if (w <= 0 || h <= 0 || maxv <= 0) { err = "Invalid PGM header values"; return false; }
        img.width = w; img.height = h; img.max_val = maxv;
        img.pixels.resize((size_t)w * (size_t)h);

        if (magic == "P5") {
        // consume one whitespace after maxval
        f.get();
        if (maxv <= 255) {
            std::vector<unsigned char> buf((size_t)w*h);
            f.read(reinterpret_cast<char*>(buf.data()), buf.size());
            if ((size_t)f.gcount() != buf.size()) { err = "Short read on P5 data"; return false; }
            for (size_t i=0;i<buf.size();++i) img.pixels[i] = buf[i];
        } else {
            // 2 bytes per sample, big endian per spec
            std::vector<unsigned char> buf((size_t)w*h*2);
            f.read(reinterpret_cast<char*>(buf.data()), buf.size());
            if ((size_t)f.gcount() != buf.size()) { err = "Short read on P5(16) data"; return false; }
            for (size_t i=0;i<(size_t)w*h;++i) {
            img.pixels[i] = (uint16_t(buf[2*i]) << 8) | uint16_t(buf[2*i+1]);
            }
        }
        } else { // P2 ASCII
        for (size_t i=0;i<(size_t)w*h;++i) {
            int v=0;
            if (!read_int(f, v)) { err = "Short read on P2 data"; return false; }
            img.pixels[i] = (uint16_t)v;
        }
        }
        return true;
    }

    bool load_config_and_image() {
        std::string image_file = image_path_;
        double res = resolution_;
        std::vector<double> origin = origin_xyz_;
        int negate = negate_;
        double occ_th = occ_thresh_;
        double free_th = free_thresh_;

        if (!yaml_path_.empty()) {
            try {
            YAML::Node y = YAML::LoadFile(yaml_path_);
            std::string image_rel = y["image"].as<std::string>();
            res   = y["resolution"].as<double>();
            origin = y["origin"].as<std::vector<double>>();
            if (y["negate"])          negate   = y["negate"].as<int>();
            if (y["occupied_thresh"]) occ_th   = y["occupied_thresh"].as<double>();
            if (y["free_thresh"])     free_th  = y["free_thresh"].as<double>();

            // Resolve image relative to YAML
            auto slash = yaml_path_.find_last_of("/\\");
            std::string yaml_dir = (slash == std::string::npos) ? std::string(".") : yaml_path_.substr(0, slash);
            image_file = yaml_dir + "/" + image_rel;
            } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to parse YAML '%s': %s", yaml_path_.c_str(), e.what());
            return false;
            }
        } else {
            if (image_file.empty()) {
            RCLCPP_ERROR(get_logger(), "Provide either 'yaml_path' or 'image' (plus resolution/origin/negate/thresholds).");
            return false;
            }
        }

        // Read PGM
        PgmImage img;
        std::string err;
        if (!load_pgm(image_file, img, err)) {
            RCLCPP_ERROR(get_logger(), "%s", err.c_str());
            return false;
        }

        // Build OccupancyGrid
        nav_msgs::msg::OccupancyGrid grid;
        grid.header.frame_id = frame_id_;
        grid.info.resolution = res;
        grid.info.width  = img.width;
        grid.info.height = img.height;
        grid.info.origin.position.x = origin.size() > 0 ? origin[0] : 0.0;
        grid.info.origin.position.y = origin.size() > 1 ? origin[1] : 0.0;
        // yaw (origin[2]) is typically 0 for static maps; grid origin orientation is kept as identity:
        grid.info.origin.orientation.w = 1.0;

        grid.data.resize((size_t)img.width * (size_t)img.height);
        const double invMax = (img.max_val > 0) ? (1.0 / img.max_val) : 1.0;

        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                uint16_t p = img.pixels[(size_t)y * img.width + x];
                // Convert to [0..255] grayscale
                double gray01 = p * invMax;              // 0..1 where 0=black, 1=white
                double occ = negate ? gray01 : (1.0 - gray01); // per map_server: invert if negate=0

                int8_t v = -1; // unknown
                if (occ > occ_thresh_)      v = 100; // occupied
                else if (occ < free_thresh_) v = 0;  // free
                else                         v = -1; // unknown band

                // Flip vertically: PGM origin is top-left; ROS OccupancyGrid expects bottom-left as (0,0)
                int yflip = img.height - 1 - y;
                grid.data[(size_t)yflip * img.width + x] = v;
            }
        }

        grid_ = std::move(grid);
        return true;
    }

    void build_map(){
        map_.header.frame_id = frame_id_;
        map_.info.resolution = resolution_;
        map_.info.width = width_;
        map_.info.height = height_;
        map_.info.origin.position.x = origin_x_;
        map_.info.origin.position.y = origin_y_;
        map_.info.origin.orientation.w = 1.0;

        map_.data.assign(width_ * height_, static_cast<int8_t>(unknown_val_));

        auto idx = [&](int x, int y) {
            return y * width_ + x;
        };

        if (unknown_val_== -1){
        } else {
            std::fill(map_.data.begin(), map_.data.end(), static_cast<int8_t>(0));
        }

        if(border_walls_){
            for(int x = 0; x < width_; ++x) {
                map_.data[idx(x, 0)]             = static_cast<int8_t>(occupied_val_);
                map_.data[idx(x, height_-1)]     = static_cast<int8_t>(occupied_val_);
            }
            for(int y = 0; y < height_; ++y) {
                map_.data[idx(0, y)]             = static_cast<int8_t>(occupied_val_);
                map_.data[idx(width_-1, y)]      = static_cast<int8_t>(occupied_val_);
            }
        }

        if(obstacles_.size() %4 !=0){
            RCLCPP_WARN(get_logger(), "Parameter 'obstacles' lenght is not a multiple of 4, ignoring");
        }
        for (size_t i=0; i+3 < obstacles_.size(); i+=4){
            int ox = static_cast<int>(obstacles_[i+0]);
            int oy = static_cast<int>(obstacles_[i+1]);
            int ow = static_cast<int>(obstacles_[i+2]);
            int oh = static_cast<int>(obstacles_[i+3]);
            for(int y = oy; y<oy+oh; ++y) {
                if(y<0 || y>=height_) continue;
                for(int x = ox; x<ox+ow; ++x) {
                    if(x<0 || x>=width_) continue;
                    map_.data[idx(x,y)] = occupied_val_;
            }
        }
        }

        size_t occ=0;
        for(auto v: map_.data) {
            if(v == static_cast<int8_t>(occupied_val_)) occ++;
            RCLCPP_INFO(get_logger(), "Map has %zu occupied cells", occ);
        }
    }
  

//ROS I/O
rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
rclcpp::TimerBase::SharedPtr timer_;

//Paramters
int width_, height_, occupied_val_, unknown_val_, negate_;
double resolution_, origin_x_, origin_y_, occ_thresh_, free_thresh_;
bool border_walls_;
std::vector<int64_t> obstacles_;
std::vector<double> origin_xyz_;
std::string yaml_path_, image_path_, frame_id_;
double publish_hz_;

nav_msgs::msg::OccupancyGrid map_;
nav_msgs::msg::OccupancyGrid grid_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MazeRunner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}