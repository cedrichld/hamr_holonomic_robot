#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

#include <queue>
#include <vector>
#include <cmath>
#include <limits>
#include <optional>
#include <algorithm>
#include <cstdint>
#include <string>

struct PQNode {
  int idx;        // linear index (r*W + c)
  double f, g, h; // f=g+h
  int parent;     // previous idx in path (-1 if none)
  bool operator<(const PQNode& o) const { return f > o.f; } // min-heap
};

class AStarPlanner : public rclcpp::Node {
public:
  AStarPlanner() : rclcpp::Node("astar_on_costmap") {
    // --- Parameters ---
    this->declare_parameter<std::string>("costmap_topic", "/costmap");
    this->declare_parameter<std::string>("elevation_topic", "/elevation_map");
    this->declare_parameter<std::string>("frame_id", "map");
    this->declare_parameter<std::string>("goal_topic", "/goal_pose"); // RViz2 default
    this->declare_parameter<bool>("also_listen_legacy_goal", true);   // RViz1 compat

    // Movement & constraints
    this->declare_parameter<double>("max_tilt_deg", 20.0); // degrees
    this->declare_parameter<double>("max_step_m", 0.08);   // max per-edge elevation jump
    this->declare_parameter<double>("w_occ", 1.0);         // weight on occupancy [0..1]
    this->declare_parameter<double>("w_low", 0.5);         // prefer lower elevations
    this->declare_parameter<double>("w_slope", 0.5);       // slope penalty
    this->declare_parameter<double>("diag_penalty", 1.0);  // tiny tie-breaker on diagonals

    // Get params
    this->get_parameter("costmap_topic", costmap_topic_);
    this->get_parameter("elevation_topic", elevation_topic_);
    this->get_parameter("frame_id", frame_id_);
    this->get_parameter("goal_topic", goal_topic_);
    this->get_parameter("also_listen_legacy_goal", also_listen_legacy_goal_);

    this->get_parameter("max_tilt_deg", max_tilt_deg_);
    this->get_parameter("max_step_m", max_step_m_);
    this->get_parameter("w_occ", w_occ_);
    this->get_parameter("w_low", w_low_);
    this->get_parameter("w_slope", w_slope_);
    this->get_parameter("diag_penalty", diag_penalty_);

    max_slope_ = std::tan(max_tilt_deg_ * M_PI / 180.0);

    // --- Subs & pubs ---
    costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      costmap_topic_, rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&AStarPlanner::onCostmap, this, std::placeholders::_1));

    elev_sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
      elevation_topic_, rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&AStarPlanner::onElevation, this, std::placeholders::_1));

    start_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", 10, std::bind(&AStarPlanner::onStart, this, std::placeholders::_1));

    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      goal_topic_, 10, std::bind(&AStarPlanner::onGoal, this, std::placeholders::_1));

    if (also_listen_legacy_goal_) {
      legacy_goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/move_base_simple/goal", 10, std::bind(&AStarPlanner::onGoal, this, std::placeholders::_1));
    }

    path_pub_  = this->create_publisher<nav_msgs::msg::Path>("/astar_path", 1);
    debug_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/astar_debug", 1);

    RCLCPP_INFO(this->get_logger(),
      "A* ready. frame='%s'. Start: /initialpose, Goal: '%s'%s",
      frame_id_.c_str(), goal_topic_.c_str(),
      also_listen_legacy_goal_ ? " (+ /move_base_simple/goal)" : "");
  }

private:
  // --- Data ---
  nav_msgs::msg::OccupancyGrid::SharedPtr grid_;
  grid_map::GridMap elevation_;
  bool have_grid_ = false;
  bool have_elevation_ = false;

  std::optional<geometry_msgs::msg::Pose> start_pose_;
  std::optional<geometry_msgs::msg::Pose> goal_pose_;

  // --- Params ---
  std::string costmap_topic_, elevation_topic_, frame_id_, goal_topic_;
  bool also_listen_legacy_goal_{true};
  double max_tilt_deg_{}, max_step_m_{}, max_slope_{};
  double w_occ_{}, w_low_{}, w_slope_{}, diag_penalty_{};

  // --- ROS ---
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr elev_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr legacy_goal_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr debug_pub_;

  // --- Callbacks ---
  void onCostmap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    grid_ = msg;
    have_grid_ = true;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Got costmap: %ux%u @ %.3fm",
                         grid_->info.width, grid_->info.height, grid_->info.resolution);
    tryPlan();
  }

  void onElevation(const grid_map_msgs::msg::GridMap::SharedPtr msg) {
    grid_map::GridMapRosConverter::fromMessage(*msg, elevation_);
    have_elevation_ = elevation_.exists("elevation");
    RCLCPP_INFO(this->get_logger(), "Got elevation map. 'elevation' layer %s",
                have_elevation_ ? "present" : "MISSING");
    tryPlan();
  }

  void onStart(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    if (msg->header.frame_id != frame_id_) {
      RCLCPP_WARN(this->get_logger(), "Start in frame '%s' (expected '%s'). Proceeding.",
                  msg->header.frame_id.c_str(), frame_id_.c_str());
    }
    start_pose_ = msg->pose.pose;
    RCLCPP_INFO(this->get_logger(), "Got start: (%.2f, %.2f)",
                start_pose_->position.x, start_pose_->position.y);
    tryPlan();
  }

  void onGoal(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (msg->header.frame_id != frame_id_) {
      RCLCPP_WARN(this->get_logger(), "Goal in frame '%s' (expected '%s'). Proceeding.",
                  msg->header.frame_id.c_str(), frame_id_.c_str());
    }
    goal_pose_ = msg->pose;
    RCLCPP_INFO(this->get_logger(), "Got goal: (%.2f, %.2f)",
                goal_pose_->position.x, goal_pose_->position.y);
    tryPlan();
  }

  // --- Helpers ---
  bool worldToCell(double x, double y, int &c, int &r) const {
    const auto &info = grid_->info;
    const double res = info.resolution;
    const double ox = info.origin.position.x;
    const double oy = info.origin.position.y;
    c = static_cast<int>(std::floor((x - ox) / res));
    r = static_cast<int>(std::floor((y - oy) / res));
    return !(c < 0 || r < 0 || c >= static_cast<int>(info.width) || r >= static_cast<int>(info.height));
  }

  void cellCenter(int c, int r, double &x, double &y) const {
    const auto &info = grid_->info;
    const double res = info.resolution;
    const double ox = info.origin.position.x;
    const double oy = info.origin.position.y;
    x = ox + (c + 0.5) * res;
    y = oy + (r + 0.5) * res;
  }

  inline int idxOf(int c, int r) const { return r * static_cast<int>(grid_->info.width) + c; }

  bool elevAtCell(int c, int r, double &elev) const {
    if (!have_elevation_) return false;
    double x, y; cellCenter(c, r, x, y);
    grid_map::Position p(x, y);
    if (!elevation_.isInside(p)) return false;

    // GridMap API that returns float
    float val = elevation_.atPosition("elevation", p, grid_map::InterpolationMethods::INTER_NEAREST);
    if (!std::isfinite(val)) return false;
    elev = static_cast<double>(val);
    return true;
  }

  double occCost01(int c, int r) const {
    const int v = grid_->data[idxOf(c, r)];
    if (v < 0)    return 1.0; // unknown as costly
    if (v >= 100) return 1.0; // lethal
    return std::clamp(v / 100.0, 0.0, 1.0);
  }

  // Main planning trigger
  void tryPlan() {
    if (!(have_grid_ && have_elevation_ && start_pose_ && goal_pose_)) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "Waiting: grid=%d elev=%d start=%d goal=%d",
        have_grid_, have_elevation_, start_pose_.has_value(), goal_pose_.has_value());
      return;
    }

    int cs, rs, cg, rg;
    if (!worldToCell(start_pose_->position.x, start_pose_->position.y, cs, rs)) {
      RCLCPP_WARN(this->get_logger(), "Start out of grid.");
      return;
    }
    if (!worldToCell(goal_pose_->position.x, goal_pose_->position.y, cg, rg)) {
      RCLCPP_WARN(this->get_logger(), "Goal out of grid.");
      return;
    }

    const int W = static_cast<int>(grid_->info.width);
    const int H = static_cast<int>(grid_->info.height);
    const double res = grid_->info.resolution;

    // Elevation range for normalization
    double min_e = +std::numeric_limits<double>::infinity();
    double max_e = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        double e;
        if (elevAtCell(c, r, e)) {
          if (e < min_e) min_e = e;
          if (e > max_e) max_e = e;
        }
      }
    }
    if (!std::isfinite(min_e) || !std::isfinite(max_e) || max_e <= min_e) {
      min_e = 0.0; max_e = 1.0;
    }
    auto normElev01 = [&](double e) {
      return std::clamp((e - min_e) / (max_e - min_e), 0.0, 1.0);
    };

    // A* data
    std::priority_queue<PQNode> open;
    std::vector<double> g(W * H, std::numeric_limits<double>::infinity());
    std::vector<int> parent(W * H, -1);
    std::vector<uint8_t> closed(W * H, 0);

    const int start_idx = idxOf(cs, rs);
    const int goal_idx  = idxOf(cg, rg);

    auto hfun = [&](int c, int r) {
      const double dx = (c - cg);
      const double dy = (r - rg);
      return std::sqrt(dx*dx + dy*dy);
    };

    g[start_idx] = 0.0;
    open.push(PQNode{start_idx, hfun(cs, rs), 0.0, hfun(cs, rs), -1});

    // For debug: explored trail
    std::vector<geometry_msgs::msg::Point> explored;

    // 8-connected neighbor offsets
    const int dc[8] = {+1,-1, 0, 0, +1,+1,-1,-1};
    const int dr[8] = { 0, 0,+1,-1, +1,-1,+1,-1};

    while (!open.empty()) {
      PQNode cur = open.top(); open.pop();
      if (closed[cur.idx]) continue;
      closed[cur.idx] = 1;

      int cc = cur.idx % W;
      int rr = cur.idx / W;

      // debug trail
      double wx, wy; cellCenter(cc, rr, wx, wy);
      geometry_msgs::msg::Point pt; pt.x = wx; pt.y = wy; pt.z = 0.05;
      explored.push_back(pt);

      if (cur.idx == goal_idx) {
        // reconstruct
        std::vector<int> path_idx;
        for (int v = cur.idx; v >= 0; v = parent[v]) path_idx.push_back(v);
        std::reverse(path_idx.begin(), path_idx.end());
        publishPath(path_idx);
        publishDebug(explored);
        RCLCPP_INFO(this->get_logger(),
          "A* done. Path nodes = %zu, cost = %.3f", path_idx.size(), cur.g);
        return;
      }

      double e_cur;
      if (!elevAtCell(cc, rr, e_cur)) continue;

      for (int k = 0; k < 8; ++k) {
        const int nc = cc + dc[k];
        const int nr = rr + dr[k];
        if (nc < 0 || nr < 0 || nc >= W || nr >= H) continue;

        const int nidx = idxOf(nc, nr);
        if (closed[nidx]) continue;

        // occupancy
        double occ = occCost01(nc, nr);
        if (occ >= 0.99) continue; // blocked

        // elevation & slope constraints
        double e_nbr;
        if (!elevAtCell(nc, nr, e_nbr)) continue;

        const double step = std::fabs(e_nbr - e_cur);
        if (step > max_step_m_) continue;

        const bool diagonal = (dc[k] != 0 && dr[k] != 0);
        const double dist = diagonal ? std::sqrt(2.0) * res : res;

        const double slope = step / dist;
        if (slope > max_slope_) continue;

        // cost terms
        double base = dist;
        if (diagonal) base += diag_penalty_ * 0.001; // tiny tie-breaker

        const double elev_term  = w_low_   * normElev01(e_nbr); // prefer low
        const double occ_term   = w_occ_   * occ;               // avoid high cost
        const double slope_term = w_slope_ * slope;             // avoid steep

        const double move_cost = base + elev_term + occ_term + slope_term;

        const double tentative = g[cur.idx] + move_cost;
        if (tentative < g[nidx]) {
          g[nidx] = tentative;
          parent[nidx] = cur.idx;
          const double h = hfun(nc, nr);
          open.push(PQNode{nidx, tentative + h, tentative, h, cur.idx});
        }
      }
    }

    RCLCPP_WARN(this->get_logger(), "A*: no path found.");
    publishPath({}); // clear
    publishDebug(explored);
  }

  void publishPath(const std::vector<int>& path_idx) {
    nav_msgs::msg::Path path;
    path.header.stamp = this->now();
    path.header.frame_id = frame_id_;
    if (!grid_) { path_pub_->publish(path); return; }

    const int W = static_cast<int>(grid_->info.width);
    for (int idx : path_idx) {
      int c = idx % W;
      int r = idx / W;
      double x, y; cellCenter(c, r, x, y);
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = 0.0;
      ps.pose.orientation.w = 1.0;
      path.poses.push_back(ps);
    }
    path_pub_->publish(path);
  }

  void publishDebug(const std::vector<geometry_msgs::msg::Point>& explored) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp = this->now();
    m.ns = "astar_debug";
    m.id = 1;
    m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.05; m.scale.y = 0.05; m.scale.z = 0.05;
    m.color.a = 0.6;  m.color.r = 0.0;  m.color.g = 0.0;  m.color.b = 1.0;
    m.points = explored;
    debug_pub_->publish(m);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AStarPlanner>());
  rclcpp::shutdown();
  return 0;
}
