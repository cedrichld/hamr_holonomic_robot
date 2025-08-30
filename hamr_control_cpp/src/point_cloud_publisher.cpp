// map_and_cloud_publisher.cpp
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <cmath>
#include <cstring> // std::memcpy

class MazeRunner : public rclcpp::Node {
public:
  MazeRunner() : rclcpp::Node("maze_runner") {
    // Map params (yours)
    yaml_path_   = declare_parameter<std::string>("yaml_path", "/home/kartik/hamr_ws/src/hamr_holonomic_robot/map/terrain_traversable.yaml");
    image_path_  = declare_parameter<std::string>("image", "");
    cloud_csv_path_ = declare_parameter<std::string>("cloud_csv", "");  // e.g., /path/to/terrain_cloud.csv
    resolution_  = declare_parameter<double>("resolution", 0.05);
    origin_xyz_  = declare_parameter<std::vector<double>>("origin", {0.0, 0.0, 0.0});
    negate_      = declare_parameter<int>("negate", 0);
    occ_thresh_  = declare_parameter<double>("occupied_thresh", 0.65);
    free_thresh_ = declare_parameter<double>("free_thresh", 0.196);
    frame_id_    = declare_parameter<std::string>("frame_id", "map");
    publish_hz_  = declare_parameter<double>("publish_rate_hz", 2.0);

    // New: hill/point-cloud parameters
    hill_amp_m_   = declare_parameter<double>("hill_amp_m", 0.6);   // peak height
    hill_sigma_m_ = declare_parameter<double>("hill_sigma_m", 2.5); // radius-ish
    cloud_topic_  = declare_parameter<std::string>("cloud_topic", "/terrain_cloud");

    // QoS: latched map
    rclcpp::QoS map_qos(rclcpp::KeepLast(1));
    map_qos.reliable().transient_local();
    map_pub_   = create_publisher<nav_msgs::msg::OccupancyGrid>("/map", map_qos);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic_, 1);

    if (!load_config_and_image()) {
      RCLCPP_FATAL(get_logger(), "Failed to load configuration or image");
      return;
    }
    RCLCPP_INFO(get_logger(), "yaml_path='%s' image='%s'", yaml_path_.c_str(), image_path_.c_str());

    bool ok = false;
    if (!cloud_csv_path_.empty()) {
    ok = build_pointcloud_from_csv(cloud_csv_path_, frame_id_);
    }
    if (!ok) {
    // fallback: synthesize a Gaussian hill from the grid (optional)
    build_hill_pointcloud_from_grid();
    }

    auto period = std::chrono::milliseconds((int)std::round(1000.0 / std::max(0.1, publish_hz_)));
    timer_ = create_wall_timer(period, [this]{
      // publish map (latched-style) and cloud continuously
      grid_.header.stamp = now();
      map_pub_->publish(grid_);

      sensor_msgs::msg::PointCloud2 pc2 = cloud_msg_;
      pc2.header.stamp = now();
      cloud_pub_->publish(pc2);
    });

    RCLCPP_INFO(get_logger(), "Publishing map %ux%u @ %.3fm/cell, frame='%s'",
                grid_.info.width, grid_.info.height, grid_.info.resolution, frame_id_.c_str());
    RCLCPP_INFO(get_logger(), "Publishing cloud '%s' with %zu points",
                cloud_topic_.c_str(), size_t(cloud_msg_.width));
  }

private:
  struct PgmImage {
    int width{0}, height{0};
    int max_val{255};
    std::vector<uint16_t> pixels;
  };

  // ---- PGM helpers (unchanged from yours, condensed here) ----
  static void skip_comments(std::istream& is){
    while (true) {
      int c=is.peek();
      if (c=='#') { std::string dummy; std::getline(is, dummy); }
      else if (std::isspace(c)) { is.get(); }
      else break;
    }
  }
  static bool read_int(std::istream& is, int& out) { skip_comments(is); return bool(is >> out); }

  static bool load_pgm(const std::string& path, PgmImage& img, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "Cannot open PGM: " + path; return false; }
    std::string magic; f >> magic;
    if (magic != "P5" && magic != "P2") { err = "Unsupported magic (need P5 or P2): " + magic; return false; }
    int w=0, h=0, maxv=0;
    if (!read_int(f, w) || !read_int(f, h) || !read_int(f, maxv)) { err = "Failed to read PGM header"; return false; }
    if (w<=0 || h<=0 || maxv<=0) { err = "Invalid PGM header values"; return false; }
    img.width=w; img.height=h; img.max_val=maxv;
    img.pixels.resize(size_t(w)*size_t(h));
    if (magic=="P5") {
      f.get(); // consume one whitespace
      if (maxv<=255) {
        std::vector<unsigned char> buf(size_t(w)*size_t(h));
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (size_t(f.gcount()) != buf.size()) { err="Short read on P5 data"; return false; }
        for (size_t i=0;i<buf.size();++i) img.pixels[i] = buf[i];
      } else {
        std::vector<unsigned char> buf(size_t(w)*size_t(h)*2);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (size_t(f.gcount()) != buf.size()) { err="Short read on P5(16) data"; return false; }
        for (size_t i=0;i<size_t(w)*size_t(h);++i)
          img.pixels[i] = (uint16_t(buf[2*i])<<8) | uint16_t(buf[2*i+1]);
      }
    } else { // P2
      for (size_t i=0;i<size_t(w)*size_t(h);++i) {
        int v=0; if (!read_int(f,v)) { err="Short read on P2 data"; return false; }
        img.pixels[i] = uint16_t(v);
      }
    }
    return true;
  }

  static std::vector<int8_t> gray_to_occ(const PgmImage& img, int negate, double occ_th, double free_th) {
    std::vector<int8_t> out(size_t(img.width)*size_t(img.height), -1);
    const double invMax = img.max_val>0 ? (1.0/img.max_val) : 1.0;
    for (int y=0;y<img.height;++y) {
      for (int x=0;x<img.width;++x) {
        uint16_t p = img.pixels[size_t(y)*img.width + x];
        double gray01 = p * invMax;               // 0..1, 0=black
        double occ = negate ? gray01 : (1.0 - gray01); // like map_server
        int8_t v = -1;
        if      (occ > occ_th)      v = 100;
        else if (occ < free_th)     v = 0;
        else                        v = -1;
        // flip y so (0,0) is bottom-left
        int yflip = img.height - 1 - y;
        out[size_t(yflip)*img.width + x] = v;
      }
    }
    return out;
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
        auto slash = yaml_path_.find_last_of("/\\");
        std::string yaml_dir = (slash == std::string::npos) ? "." : yaml_path_.substr(0, slash);
        image_file = yaml_dir + "/" + image_rel;
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to parse YAML '%s': %s", yaml_path_.c_str(), e.what());
        return false;
      }
    } else if (image_file.empty()) {
      RCLCPP_ERROR(get_logger(), "Provide either 'yaml_path' or 'image' (plus resolution/origin/negate/thresholds).");
      return false;
    }

    PgmImage img; std::string err;
    if (!load_pgm(image_file, img, err)) {
      RCLCPP_ERROR(get_logger(), "%s", err.c_str());
      return false;
    }

    grid_.header.frame_id = frame_id_;
    grid_.info.resolution = res;
    grid_.info.width      = img.width;
    grid_.info.height     = img.height;
    grid_.info.origin.position.x = origin.size()>0 ? origin[0] : 0.0;
    grid_.info.origin.position.y = origin.size()>1 ? origin[1] : 0.0;
    grid_.info.origin.orientation.w = 1.0;

    grid_.data = gray_to_occ(img, negate, occ_th, free_th);
    return true;
  }

  // Build a Gaussian hill point set over free interior cells; publish as PointCloud2
  void build_hill_pointcloud_from_grid() {
    const auto W = int(grid_.info.width);
    const auto H = int(grid_.info.height);
    const double res = grid_.info.resolution;

    // Center of map in meters (map origin + half span)
    const double cx = grid_.info.origin.position.x + (W-1) * res * 0.5;
    const double cy = grid_.info.origin.position.y + (H-1) * res * 0.5;

    // Collect points
    std::vector<float> buffer; buffer.reserve(size_t(W)*size_t(H)*3);
    auto idx = [&](int x,int y){ return size_t(y)*size_t(W)+size_t(x); };

    for (int y=0;y<H;++y) {
      for (int x=0;x<W;++x) {
        int8_t v = grid_.data[idx(x,y)];
        // skip occupied and unknown; keep only clear free cells
        if (v != 0) continue;

        const double wx = grid_.info.origin.position.x + x * res;
        const double wy = grid_.info.origin.position.y + y * res;
        // Gaussian bump around (cx,cy)
        const double dx = wx - cx, dy = wy - cy;
        const double r2 = (dx*dx + dy*dy) / (2.0 * hill_sigma_m_ * hill_sigma_m_);
        const double wz = hill_amp_m_ * std::exp(-r2); // meters high

        buffer.push_back(float(wx));
        buffer.push_back(float(wy));
        buffer.push_back(float(wz));
      }
    }

    // Fill PointCloud2
    cloud_msg_.header.frame_id = frame_id_;
    cloud_msg_.height = 1;
    cloud_msg_.width  = static_cast<uint32_t>(buffer.size()/3);
    cloud_msg_.is_bigendian = false;
    cloud_msg_.is_dense = true;

    sensor_msgs::msg::PointField f_x, f_y, f_z;
    f_x.name="x"; f_x.offset=0;  f_x.datatype=sensor_msgs::msg::PointField::FLOAT32; f_x.count=1;
    f_y.name="y"; f_y.offset=4;  f_y.datatype=sensor_msgs::msg::PointField::FLOAT32; f_y.count=1;
    f_z.name="z"; f_z.offset=8;  f_z.datatype=sensor_msgs::msg::PointField::FLOAT32; f_z.count=1;
    cloud_msg_.fields = {f_x, f_y, f_z};
    cloud_msg_.point_step = 12;
    cloud_msg_.row_step   = cloud_msg_.point_step * cloud_msg_.width;

    cloud_storage_.resize(cloud_msg_.width * 3 * sizeof(float));
    std::memcpy(cloud_storage_.data(), buffer.data(), cloud_storage_.size());
    cloud_msg_.data = cloud_storage_;
  }


   bool build_pointcloud_from_csv(const std::string& csv_path, const std::string& frame_id) {
        std::ifstream f(csv_path);
        if (!f) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open cloud CSV: %s", csv_path.c_str());
            return false;
        }
        std::vector<float> xyz; xyz.reserve(1<<20); // pre-reserve ~large

        std::string line;
        auto is_header = [](const std::string& s)->bool {
            // treat as header if contains any alphabetic char
            for (char c : s) if (std::isalpha(static_cast<unsigned char>(c))) return true;
            return false;
        };

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            // skip comments
            if (!line.empty() && line[0] == '#') continue;
            // skip header rows
            if (is_header(line)) continue;

            // split on comma OR whitespace
            std::vector<std::string> toks;
            {
            std::string cur;
            for (char c : line) {
                if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
                if (!cur.empty()) { toks.push_back(cur); cur.clear(); }
                } else {
                cur.push_back(c);
                }
            }
            if (!cur.empty()) toks.push_back(cur);
            }
            if (toks.size() < 3) continue;

            try {
            float x = std::stof(toks[0]);
            float y = std::stof(toks[1]);
            float z = std::stof(toks[2]);
            xyz.push_back(x); xyz.push_back(y); xyz.push_back(z);
            } catch (...) {
            // ignore malformed row
            }
        }

        const size_t npts = xyz.size()/3;
        if (npts == 0) {
            RCLCPP_ERROR(this->get_logger(), "No points parsed from %s", csv_path.c_str());
            return false;
        }

        // Fill PointCloud2
        cloud_msg_.header.frame_id = frame_id;
        cloud_msg_.height = 1;
        cloud_msg_.width  = static_cast<uint32_t>(npts);
        cloud_msg_.is_bigendian = false;
        cloud_msg_.is_dense = true;

        sensor_msgs::msg::PointField fx, fy, fz;
        fx.name="x"; fx.offset=0;  fx.datatype=sensor_msgs::msg::PointField::FLOAT32; fx.count=1;
        fy.name="y"; fy.offset=4;  fy.datatype=sensor_msgs::msg::PointField::FLOAT32; fy.count=1;
        fz.name="z"; fz.offset=8;  fz.datatype=sensor_msgs::msg::PointField::FLOAT32; fz.count=1;
        cloud_msg_.fields = {fx, fy, fz};
        cloud_msg_.point_step = 12;
        cloud_msg_.row_step   = cloud_msg_.point_step * cloud_msg_.width;

        cloud_storage_.resize(npts * 3 * sizeof(float));
        std::memcpy(cloud_storage_.data(), xyz.data(), cloud_storage_.size());
        cloud_msg_.data = cloud_storage_;

        RCLCPP_INFO(this->get_logger(), "Loaded cloud: %s (%zu points)", csv_path.c_str(), npts);
        return true;
    }

  // ROS I/O
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Params
  int width_, height_, occupied_val_, unknown_val_, negate_;
  double resolution_, origin_x_, origin_y_, occ_thresh_, free_thresh_;
  bool border_walls_;
  std::vector<double> origin_xyz_;
  std::string yaml_path_, image_path_, frame_id_;
  double publish_hz_;
  std::string cloud_csv_path_;
  sensor_msgs::msg::PointCloud2 cloud_msg_;
  std::vector<uint8_t> cloud_storage_;

  // Hill params
  double hill_amp_m_{0.6};
  double hill_sigma_m_{2.5};
  std::string cloud_topic_{"/terrain_cloud"};

  // Messages / storage
  nav_msgs::msg::OccupancyGrid grid_;
//   sensor_msgs::msg::PointCloud2 cloud_msg_;
//   std::vector<uint8_t> cloud_storage_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MazeRunner>());
  rclcpp::shutdown();
  return 0;
}
