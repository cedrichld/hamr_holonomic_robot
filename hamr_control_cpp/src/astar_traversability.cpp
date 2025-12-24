#include <rclcpp/rclcpp.hpp>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/GridMap.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <queue>
#include <vector>
#include <limits>
#include <cmath>
#include <string>
#include <optional>

#include <algorithm>
#include <unordered_set>

// quat -> yaw (radians)
double quat_to_yaw(const geometry_msgs::msg::Quaternion& q_in) {
    const double n = std::sqrt(q_in.x*q_in.x + q_in.y*q_in.y +
                               q_in.z*q_in.z + q_in.w*q_in.w);
    if (n == 0.0) return 0.0; // fallback
    const double x = q_in.x / n;
    const double y = q_in.y / n;
    const double z = q_in.z / n;
    const double w = q_in.w / n;

    const double siny_cosp = 2.0 * (w*z + x*y);
    const double cosy_cosp = 1.0 - 2.0 * (y*y + z*z);
    return std::atan2(siny_cosp, cosy_cosp); // in [-pi, pi]
}

// yaw -> quat
geometry_msgs::msg::Quaternion yaw_to_quat(double yaw) {
    geometry_msgs::msg::Quaternion q;
    const double h = 0.5 * yaw;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(h);
    q.w = std::cos(h);
    return q;
}

class TraversabilityAStarNode final : public rclcpp::Node {
public:
  TraversabilityAStarNode() : Node("traversability_astar")
  {
    // -----------------------
    // Parameters
    // -----------------------
    traversability_topic_   = declare_parameter<std::string>("traversability_topic", "/filtered_map");
    traversability_layer_   = declare_parameter<std::string>("traversability_layer", "traversability");

    // Interpretation: traversability [0-1] higher = more traverable
    traversability_threshold_ = declare_parameter<double>("traversability_threshold", 0.35);
    alpha_traversability_     = declare_parameter<double>("alpha_traversability", 5.0);

    allow_diagonal_         = declare_parameter<bool>("allow_diagonal", true);
    heuristic_weight_       = declare_parameter<double>("heuristic_weight", 0.1);

    // Inflation not used rn
    using_inflation_  = declare_parameter<bool>("using_inflation", false);
    inflation_radius_ = declare_parameter<double>("inflation_radius", 0.5);
    inflation_weight_ = declare_parameter<double>("inflation_weight", 5.0);
    inflation_decay_  = declare_parameter<double>("inflation_decay", 0.15);

    // Anytime Repairing A*(ARA*)
    use_anytime_           = declare_parameter<bool>("use_anytime", true);
    epsilon_start_         = declare_parameter<double>("epsilon_start", 3.0);
    epsilon_min_           = declare_parameter<double>("epsilon_min", 1.0);
    epsilon_step_          = declare_parameter<double>("epsilon_step", 0.25);
    max_expansions_total_  = declare_parameter<int>("max_expansions_total", 1000000);
    publish_on_improve_    = declare_parameter<bool>("publish_on_improve", true);
    min_improvement_ratio_ = declare_parameter<double>("min_improvement_ratio", 0.05); // eg 0.01 for 1%


    unknown_is_obstacle_    = declare_parameter<bool>("unknown_is_obstacle", true);

    use_tf_start_           = declare_parameter<bool>("use_tf_start", true);
    map_frame_              = declare_parameter<std::string>("map_frame", "map");
    grid_map_frame_         = declare_parameter<std::string>("grid_map_frame", "terrain_map");
    base_frame_             = declare_parameter<std::string>("base_frame", "base_link");

    // replanning behavior
    replan_on_new_map_      = declare_parameter<bool>("replan_on_new_map", true);

    // -----------------------
    // TF
    // -----------------------
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // -----------------------
    // Publishers (latched-ish QoS)
    // -----------------------
    rclcpp::QoS latched_qos(1);
    latched_qos.transient_local();
    latched_qos.reliable();

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/astar/path", latched_qos);
    explored_pub_ = create_publisher<visualization_msgs::msg::Marker>("/astar/explored", 1);

    // -----------------------
    // Subscribers
    // -----------------------
    gridmap_sub_ = create_subscription<grid_map_msgs::msg::GridMap>(
      traversability_topic_,
      rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&TraversabilityAStarNode::onGridMap, this, std::placeholders::_1));

    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose",
      1,
      std::bind(&TraversabilityAStarNode::onGoal, this, std::placeholders::_1));

    start_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose",
      1,
      std::bind(&TraversabilityAStarNode::onStart, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "Traversability A* ready. Subscribing to %s (layer=%s).",
      traversability_topic_.c_str(), traversability_layer_.c_str());
  }

private:
  // ============================================================
  // Map handling
  // ============================================================
  void onGridMap(const grid_map_msgs::msg::GridMap::SharedPtr msg)
  {
    grid_map::GridMap map;
    try {
      grid_map::GridMapRosConverter::fromMessage(*msg, map);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "GridMapRosConverter::fromMessage failed: %s", e.what());
      return;
    }

    if (!map.exists(traversability_layer_)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "Received GridMap but layer '%s' does not exist. Available layers: [%s]",
        traversability_layer_.c_str(),
        joinLayers(map.getLayers()).c_str());
      return;
    }

    grid_map_ = std::move(map);
    have_map_ = true;

    // Plan if we already have a goal (and optionally replan on map updates)
    if (have_goal_ && replan_on_new_map_) {
      planAndPublish();
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "GridMap frame_id: %s | planner map_frame: %s | grid_map_frame param: %s",
      grid_map_.getFrameId().c_str(), map_frame_.c_str(), grid_map_frame_.c_str());

  }

  static std::string joinLayers(const std::vector<std::string>& layers)
  {
    std::string out;
    for (size_t i = 0; i < layers.size(); ++i) {
      out += layers[i];
      if (i + 1 < layers.size()) out += ", ";
    }
    return out;
  }

  // ============================================================
  // Start/Goal handling
  // ============================================================
  void onGoal(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    goal_pose_ = *msg;
    have_goal_ = true;

    if (have_map_) {
      planAndPublish();
    }
  }

  void onStart(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    start_pose_ = msg->pose.pose;
    start_pose_frame_ = msg->header.frame_id.empty() ? map_frame_ : msg->header.frame_id;
    have_start_pose_ = true;

    if (!use_tf_start_ && have_goal_ && have_map_) {
      planAndPublish();
    }
  }

  bool getStartPoseInMapFrame(geometry_msgs::msg::PoseStamped& start_out)
  {
    start_out.header.stamp = now();
    start_out.header.frame_id = map_frame_;

    if (use_tf_start_) {
      // Use TF: map_frame -> base_frame
      geometry_msgs::msg::TransformStamped tf;
      try {
        tf = tf_buffer_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "TF lookup failed (%s -> %s): %s",
          map_frame_.c_str(), base_frame_.c_str(), e.what());
        return false;
      }
      start_out.pose.position.x = tf.transform.translation.x;
      start_out.pose.position.y = tf.transform.translation.y;
      start_out.pose.position.z = tf.transform.translation.z;
      start_out.pose.orientation = tf.transform.rotation;
      return true;
    }

    // Else use /initialpose
    if (!have_start_pose_) return false;

    geometry_msgs::msg::PoseStamped sp;
    sp.header.stamp = now();
    sp.header.frame_id = start_pose_frame_;
    sp.pose = start_pose_;

    if (sp.header.frame_id == map_frame_) {
      start_out.pose = sp.pose;
      return true;
    }

    // Transform into map frame
    try {
      geometry_msgs::msg::TransformStamped tf =
        tf_buffer_->lookupTransform(map_frame_, sp.header.frame_id, tf2::TimePointZero);
      tf2::doTransform(sp, start_out, tf);
      return true;
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Start pose transform failed (%s -> %s): %s",
        sp.header.frame_id.c_str(), map_frame_.c_str(), e.what());
      return false;
    }
  }

  bool getGoalPoseInMapFrame(geometry_msgs::msg::PoseStamped& goal_out)
  {
    if (!have_goal_) return false;

    goal_out = goal_pose_;

    if (goal_out.header.frame_id.empty()) {
      goal_out.header.frame_id = map_frame_;
      return true;
    }

    if (goal_out.header.frame_id == map_frame_) {
      return true;
    }

    // Transform
    try {
      geometry_msgs::msg::TransformStamped tf =
        tf_buffer_->lookupTransform(map_frame_, goal_out.header.frame_id, tf2::TimePointZero);
      geometry_msgs::msg::PoseStamped tmp = goal_out;
      tf2::doTransform(tmp, goal_out, tf);
      return true;
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Goal pose transform failed (%s -> %s): %s",
        goal_pose_.header.frame_id.c_str(), map_frame_.c_str(), e.what());
      return false;
    }
  }

  // ============================================================
  // A* internals
  // ============================================================
  struct NodeRec {
    float g = std::numeric_limits<float>::infinity();
    float h = 0.0f;
    int parent = -1;
    bool inited = false;
  };

  struct PQItem {
    int idx;
    float f;
    bool operator<(const PQItem& o) const { return f > o.f; } // min-heap via max-heap
  };

  void planAndPublish()
  {
    if (!have_map_) return;

    geometry_msgs::msg::PoseStamped start_map, goal_map;
    if (!getStartPoseInMapFrame(start_map)) return;
    if (!getGoalPoseInMapFrame(goal_map)) return;

    // Convert world -> grid_map::Index
    grid_map::Position ps(start_map.pose.position.x, start_map.pose.position.y);
    grid_map::Position pg(goal_map.pose.position.x, goal_map.pose.position.y);
    const double goal_yaw = quat_to_yaw(goal_map.pose.orientation);

    grid_map::Position ps_gm, pg_gm;
    if (!transformPosition(map_frame_, grid_map_frame_, ps, ps_gm)) return;
    if (!transformPosition(map_frame_, grid_map_frame_, pg, pg_gm)) return;


    grid_map::Index is, ig;
    if (!grid_map_.getIndex(ps_gm, is)) {
      RCLCPP_WARN(get_logger(), "Start is outside GridMap bounds.");
      return;
    }
    if (!grid_map_.getIndex(pg_gm, ig)) {
      RCLCPP_WARN(get_logger(), "Goal is outside GridMap bounds.");
      return;
    }


    grid_map::Position pg_back_gm;
    grid_map_.getPosition(ig, pg_back_gm);

    grid_map::Position pg_back_map;
    transformPosition(grid_map_frame_, map_frame_, pg_back_gm, pg_back_map);

    RCLCPP_INFO(get_logger(),
      "Goal map: (%.3f, %.3f) -> gm_index: (%d, %d) -> back map: (%.3f, %.3f)",
      pg.x(), pg.y(), ig(0), ig(1), pg_back_map.x(), pg_back_map.y());


    const int size_x = grid_map_.getSize()(0);
    const int size_y = grid_map_.getSize()(1);

    const auto toLinear = [size_y](int ix, int iy) {
      return ix * size_y + iy;
    };
    const auto toIndex2 = [size_y](int idx) {
      int ix = idx / size_y;
      int iy = idx % size_y;
      return std::pair<int,int>(ix, iy);
    };

    const int start_idx = toLinear(is(0), is(1));
    const int goal_idx  = toLinear(ig(0), ig(1));

    // Run A*
    std::vector<int> path_linear;
    std::vector<int> explored_linear;

    // bool ok = astar(size_x, size_y, start_idx, goal_idx, toIndex2, toLinear, path_linear, explored_linear); old

    bool ok = false;
    if (use_anytime_) {
      ok = araStar(size_x, size_y, start_idx, goal_idx, toIndex2, toLinear, path_linear, explored_linear, goal_yaw);
    } else {
      ok = astar(size_x, size_y, start_idx, goal_idx, toIndex2, toLinear, path_linear, explored_linear, goal_yaw);
    }


    if (!ok) {
      RCLCPP_WARN(get_logger(), "A* failed to find a path.");
      publishExplored(explored_linear, size_x, size_y);
      return;
    }

    publishPath(path_linear, size_x, size_y, goal_yaw);
    publishExplored(explored_linear, size_x, size_y);
  }

  bool transformPosition(const std::string& from,
                       const std::string& to,
                       const grid_map::Position& pin,
                       grid_map::Position& pout)
  {
    geometry_msgs::msg::PointStamped p_in, p_out;
    p_in.header.stamp = now();
    p_in.header.frame_id = from;
    p_in.point.x = pin.x();
    p_in.point.y = pin.y();
    p_in.point.z = 0.0;

    try {
      auto tf = tf_buffer_->lookupTransform(to, from, tf2::TimePointZero);
      tf2::doTransform(p_in, p_out, tf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "transformPosition %s->%s failed: %s", from.c_str(), to.c_str(), e.what());
      return false;
    }

    pout.x() = p_out.point.x;
    pout.y() = p_out.point.y;
    return true;
  }



  template <typename ToIndex2, typename ToLinear>
  bool astar(
    int size_x,
    int size_y,
    int start_idx,
    int goal_idx,
    ToIndex2 toIndex2,
    ToLinear toLinear,
    std::vector<int>& out_path,
    std::vector<int>& out_explored,
    double goal_yaw
  )
  {
    const int N = size_x * size_y;

    std::vector<NodeRec> rec(N);
    std::vector<bool> closed(N, false);
    std::priority_queue<PQItem> open;

    auto heuristic = [&](int a, int b) -> float {
      auto [ax, ay] = toIndex2(a);
      auto [bx, by] = toIndex2(b);
      const float dx = float(ax - bx);
      const float dy = float(ay - by);
      return float(heuristic_weight_) * std::sqrt(dx*dx + dy*dy) * float(grid_map_.getResolution());
    };

    auto isTraversable = [&](int idx) -> bool {
      auto [ix, iy] = toIndex2(idx);
      grid_map::Index gi(ix, iy);

      // Validity check
      if (!grid_map_.isValid(gi, traversability_layer_)) {
        return !unknown_is_obstacle_;
      }

      const float t = grid_map_.at(traversability_layer_, gi);
      if (!std::isfinite(t)) return !unknown_is_obstacle_;

      return (t >= float(traversability_threshold_));
    };

    auto travValue = [&](int idx) -> float {
      auto [ix, iy] = toIndex2(idx);
      grid_map::Index gi(ix, iy);
      if (!grid_map_.isValid(gi, traversability_layer_)) return 0.0f;
      float t = grid_map_.at(traversability_layer_, gi);
      if (!std::isfinite(t)) return 0.0f;

      // If layer isn't [0,1], clamp or remap here.
      if (t < 0.0f) t = 0.0f;
      if (t > 1.0f) t = 1.0f;
      return t;
    };

    // Initialize start
    if (!isTraversable(start_idx)) {
      RCLCPP_WARN(get_logger(), "Start cell is not traversable.");
      return false;
    }
    if (!isTraversable(goal_idx)) {
      RCLCPP_WARN(get_logger(), "Goal cell is not traversable.");
      return false;
    }

    auto isBlocked = [&](int idx) -> bool {
      return !isTraversable(idx);
    };

    rec[start_idx].g = 0.0f;
    rec[start_idx].h = heuristic(start_idx, goal_idx);
    rec[start_idx].parent = -1;
    rec[start_idx].inited = true;

    open.push({start_idx, rec[start_idx].g + rec[start_idx].h});

    // Neighbor deltas
    std::vector<std::pair<int,int>> deltas4 {{1,0},{-1,0},{0,1},{0,-1}};
    std::vector<std::pair<int,int>> deltas8 {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    const auto& deltas = allow_diagonal_ ? deltas8 : deltas4;

    while (!open.empty()) {
      int cur = open.top().idx;
      open.pop();

      if (closed[cur]) continue;
      closed[cur] = true;
      out_explored.push_back(cur);

      if (cur == goal_idx) {
        reconstruct(rec, goal_idx, out_path);
        return true;
      }

      auto [cx, cy] = toIndex2(cur);

      for (const auto& [dxi, dyi] : deltas) {
        const int nx = cx + dxi;
        const int ny = cy + dyi;
        if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y) continue;

        int nbr = toLinear(nx, ny);
        if (closed[nbr]) continue;
        if (!isTraversable(nbr)) continue;

        const bool diagonal = (dxi != 0 && dyi != 0);
        const float step_d = float(grid_map_.getResolution()) * (diagonal ? 1.41421356237f : 1.0f);

        const float t = travValue(nbr);
        // const float mult = 1.0f + float(alpha_traversability_) * (1.0f - t);
        // const float edge_cost = step_d * mult;

        float difficulty = 1.0f - t; // inverted semantics
        difficulty = std::clamp(difficulty, 0.0f, 1.0f);

        float mult = 1.0f + float(alpha_traversability_) * difficulty;
        const float pen = using_inflation_ ? clearancePenalty(nbr, size_x, size_y, toIndex2, toLinear, isBlocked) : 0;
        float edge_cost = step_d * mult;

        const float g_new = rec[cur].g + edge_cost * (1.0f + pen);

        if (!rec[nbr].inited || g_new < rec[nbr].g) {
          rec[nbr].inited = true;
          rec[nbr].g = g_new;
          rec[nbr].h = heuristic(nbr, goal_idx);
          rec[nbr].parent = cur;

          open.push({nbr, rec[nbr].g + rec[nbr].h});
        }
      }
    }

    return false;
  }

  template <typename ToIndex2, typename ToLinear>
  bool araStar(
    int size_x,
    int size_y,
    int start_idx,
    int goal_idx,
    ToIndex2 toIndex2,
    ToLinear toLinear,
    std::vector<int>& out_path,
    std::vector<int>& out_explored,
    double goal_yaw
  )
  {
    const int N = size_x * size_y;

    std::vector<NodeRec> rec(N);
    std::vector<bool> closed(N, false);

    // OPEN bookkeeping (priority queue can contain duplicates; in_open rebuilds)
    std::vector<bool> in_open(N, false);
    std::priority_queue<PQItem> open;
    std::unordered_set<int> incons;

    auto heuristic = [&](int a, int b) -> float {
      auto [ax, ay] = toIndex2(a);
      auto [bx, by] = toIndex2(b);
      const float dx = float(ax - bx);
      const float dy = float(ay - by);
      // Keep heuristic admissible - let epsilon provide greediness
      return std::sqrt(dx*dx + dy*dy) * float(grid_map_.getResolution());
    };

    auto isTraversable = [&](int idx) -> bool {
      auto [ix, iy] = toIndex2(idx);
      grid_map::Index gi(ix, iy);

      if (!grid_map_.isValid(gi, traversability_layer_)) {
        return !unknown_is_obstacle_;
      }

      const float t = grid_map_.at(traversability_layer_, gi);
      if (!std::isfinite(t)) return !unknown_is_obstacle_;

      return (t >= float(traversability_threshold_));
    };

    auto travValue = [&](int idx) -> float {
      auto [ix, iy] = toIndex2(idx);
      grid_map::Index gi(ix, iy);
      if (!grid_map_.isValid(gi, traversability_layer_)) return 0.0f;
      float t = grid_map_.at(traversability_layer_, gi);
      if (!std::isfinite(t)) return 0.0f;
      return std::clamp(t, 0.0f, 1.0f);
    };

    if (!isTraversable(start_idx)) {
      RCLCPP_WARN(get_logger(), "Start cell is not traversable.");
      return false;
    }
    if (!isTraversable(goal_idx)) {
      RCLCPP_WARN(get_logger(), "Goal cell is not traversable.");
      return false;
    }

    // Initialize start
    rec[start_idx].g = 0.0f;
    rec[start_idx].h = heuristic(start_idx, goal_idx);
    rec[start_idx].parent = -1;
    rec[start_idx].inited = true;

    auto key = [&](int idx, double eps) -> float {
      return rec[idx].g + float(eps) * rec[idx].h;
    };

    open.push({start_idx, key(start_idx, epsilon_start_)});
    in_open[start_idx] = true;

    // neighbor deltas
    std::vector<std::pair<int,int>> deltas4 {{1,0},{-1,0},{0,1},{0,-1}};
    std::vector<std::pair<int,int>> deltas8 {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    const auto& deltas = allow_diagonal_ ? deltas8 : deltas4;

    float best_goal_g = std::numeric_limits<float>::infinity();
    bool have_solution = false;

    int expansions = 0;
    double eps = epsilon_start_;

    auto maybePublish = [&](float new_goal_g) {
      if (!publish_on_improve_) return;

      // de-spam: only publish if relative improvement exceeds threshold
      if (have_solution && min_improvement_ratio_ > 0.0) {
        const float ratio = (best_goal_g - new_goal_g) / std::max(1e-6f, best_goal_g);
        if (ratio < float(min_improvement_ratio_)) return;
      }

      std::vector<int> tmp_path;
      reconstruct(rec, goal_idx, tmp_path);
      publishPath(tmp_path, size_x, size_y, goal_yaw);
    };

    auto improvePath = [&]() {
      while (!open.empty() && expansions < max_expansions_total_) {

        // ARA* termination condition for current eps:
        // stop if best possible f in OPEN can't beat current best goal g
        if (have_solution && open.top().f >= best_goal_g) {
          return;
        }

        int cur = open.top().idx;
        open.pop();

        // pq duplicates: only expand if it's still in OPEN and not CLOSED
        if (closed[cur]) continue;
        if (!in_open[cur]) continue;

        in_open[cur] = false;
        closed[cur] = true;
        out_explored.push_back(cur);
        expansions++;

        auto [cx, cy] = toIndex2(cur);

        for (const auto& [dxi, dyi] : deltas) {
          const int nx = cx + dxi;
          const int ny = cy + dyi;
          if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y) continue;

          int nbr = toLinear(nx, ny);
          if (!isTraversable(nbr)) continue;

          const bool diagonal = (dxi != 0 && dyi != 0);
          const float step_d = float(grid_map_.getResolution()) * (diagonal ? 1.41421356237f : 1.0f);

          // Your current semantics: traversability layer is goodness in [0,1]
          const float t = travValue(nbr);
          float difficulty = 1.0f - t;
          difficulty = std::clamp(difficulty, 0.0f, 1.0f);

          const float mult = 1.0f + float(alpha_traversability_) * difficulty;
          const float edge_cost = step_d * mult;

          const float g_new = rec[cur].g + edge_cost;

          if (!rec[nbr].inited || g_new < rec[nbr].g) {
            rec[nbr].inited = true;
            rec[nbr].g = g_new;
            rec[nbr].h = heuristic(nbr, goal_idx);
            rec[nbr].parent = cur;

            if (closed[nbr]) {
              incons.insert(nbr);
            } else {
              open.push({nbr, key(nbr, eps)});
              in_open[nbr] = true;
            }
          }
        }

        // Improvement check: if goal got a better g, publish immediately
        if (rec[goal_idx].inited && rec[goal_idx].g < best_goal_g) {
          float new_goal_g = rec[goal_idx].g;
          have_solution = true;
          maybePublish(new_goal_g);
          best_goal_g = new_goal_g;
        }
      }
    };

    // Main ARA* loop: repeatedly improve then reduce eps
    while (expansions < max_expansions_total_) {
      // Try to improve for current eps
      std::fill(closed.begin(), closed.end(), false);
      improvePath();

      if (!have_solution) {
        // No solution found yet: keep same eps and continue (or bail)
        if (open.empty()) break;
      } else {
        // If eps already at min and OPEN can't beat current best, stop
        if (eps <= epsilon_min_) {
          break;
        }
      }

      // Decrease eps
      eps = std::max(epsilon_min_, eps - epsilon_step_);

      // Rebuild OPEN = OPEN ∪ INCONS, update priorities for new eps
      std::priority_queue<PQItem> new_open;

      // Add all nodes currently marked in_open (still pending)
      for (int i = 0; i < N; ++i) {
        if (in_open[i] && rec[i].inited) {
          new_open.push({i, key(i, eps)});
        }
      }

      // Add INCONS nodes
      for (int idx : incons) {
        if (!rec[idx].inited) continue;
        in_open[idx] = true;
        new_open.push({idx, key(idx, eps)});
      }
      incons.clear();

      open = std::move(new_open);
    }

    if (!have_solution) return false;

    reconstruct(rec, goal_idx, out_path);
    return true;
  }



  static void reconstruct(const std::vector<NodeRec>& rec, int goal, std::vector<int>& out)
  {
    out.clear();
    int cur = goal;
    while (cur >= 0) {
      out.push_back(cur);
      cur = rec[cur].parent;
    }
    std::reverse(out.begin(), out.end());
  }

  // ============================================================
  // Inflation - not being used right now
  // ============================================================
  
  template <typename ToIndex2, typename ToLinear>
  float clearancePenalty(int cell_idx, int size_x, int size_y, ToIndex2 toIndex2, ToLinear toLinear,
                        const std::function<bool(int)>& isBlocked) const
  {
    const float r = static_cast<float>(grid_map_.getResolution());
    const int rad_cells = static_cast<int>(std::ceil(inflation_radius_ / r));
    const float max_dist = static_cast<float>(inflation_radius_);

    auto [cx, cy] = toIndex2(cell_idx);

    float best = std::numeric_limits<float>::infinity();

    // search in a square window, compute Euclidean distance in meters
    for (int dx = -rad_cells; dx <= rad_cells; ++dx) {
      for (int dy = -rad_cells; dy <= rad_cells; ++dy) {
        int nx = cx + dx;
        int ny = cy + dy;
        if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y) continue;

        int j = toLinear(nx, ny);
        if (!isBlocked(j)) continue;

        float dist = r * std::sqrt(float(dx*dx + dy*dy));
        if (dist < best) best = dist;
      }
    }

    if (!std::isfinite(best) || best >= max_dist) return 0.0f;

    // Smooth exponential decay penalty (bigger when closer)
    // penalty in [0, inflation_weight_] approximately
    const float decay = std::max(1e-3f, static_cast<float>(inflation_decay_));
    float pen = static_cast<float>(inflation_weight_) * std::exp(-(best / decay));
    return pen;
  }



  // ============================================================
  // Publishing
  // ============================================================
  void publishPath(const std::vector<int>& path_linear, int size_x, int size_y, double goal_yaw)
  {
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = map_frame_;

    for (int idx : path_linear)
    {
      // Recover grid indices (row = ix, col = iy).
      int ix = idx / size_y;
      int iy = idx % size_y;

      // manual bounds check using size_x / size_y
      if (ix < 0 || ix >= size_x || iy < 0 || iy >= size_y) {
        continue;
      }

      grid_map::Index gi(ix, iy);

      // Elevation at this cell (grid_map frame).
      float z = 0.0f;
      if (grid_map_.isValid(gi, "elevation_inpainted")) {
        z = grid_map_.at("elevation_inpainted", gi);
        if (!std::isfinite(z)) {
          z = 0.0f;
        }
      }

      // Position (x,y) in grid_map frame.
      grid_map::Position p_gm;
      if (!grid_map_.getPosition(gi, p_gm)) {
        continue;  // out of bounds or invalid
      }

      // Transform (x,y) into map frame.
      grid_map::Position p_map;
      if (!transformPosition(grid_map_frame_, map_frame_, p_gm, p_map)) {
        continue;
      }

      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = p_map.x();
      ps.pose.position.y = p_map.y();
      ps.pose.position.z = z; // path follows terrain height

      ps.pose.orientation = yaw_to_quat(goal_yaw);
      path.poses.push_back(ps);
    }

    path_pub_->publish(path);
  }


  void publishExplored(const std::vector<int>& explored_linear, int size_x, int size_y)
  {
    visualization_msgs::msg::Marker m;
    m.header.stamp = now();
    m.header.frame_id = map_frame_;
    m.ns = "astar_explored";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::POINTS;
    m.action = visualization_msgs::msg::Marker::ADD;

    // scale = point size
    m.scale.x = 0.03;
    m.scale.y = 0.03;

    // color
    m.color.a = 0.8;
    m.color.r = 0.1;
    m.color.g = 0.9;
    m.color.b = 0.2;

    m.points.reserve(explored_linear.size());

    for (int idx : explored_linear) {
      int ix = idx / size_y;
      int iy = idx % size_y;

      grid_map::Index gi(ix, iy);
      grid_map::Position p_gm;
      if (!grid_map_.getPosition(gi, p_gm)) continue;

      grid_map::Position p_map;
      if (!transformPosition(grid_map_frame_, map_frame_, p_gm, p_map)) continue;

      geometry_msgs::msg::Point pt;
      pt.x = p_map.x();
      pt.y = p_map.y();

      pt.z = 0.0;
      m.points.push_back(pt);
    }

    explored_pub_->publish(m);
  }

private:
  // Params
  std::string traversability_topic_;
  std::string traversability_layer_;
  double traversability_threshold_;
  double alpha_traversability_;
  bool allow_diagonal_;
  double heuristic_weight_;

  bool using_inflation_;
  double inflation_radius_;
  double inflation_weight_;
  double inflation_decay_;

  bool use_anytime_{true};
  double epsilon_start_{3.0};
  double epsilon_min_{1.0};
  double epsilon_step_{0.25};
  int max_expansions_total_{200000};
  bool publish_on_improve_{true};
  double min_improvement_ratio_{0.0};


  bool unknown_is_obstacle_;
  bool use_tf_start_;
  std::string map_frame_;
  std::string grid_map_frame_;
  std::string base_frame_;
  bool replan_on_new_map_;

  // TF
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  // ROS interfaces
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr gridmap_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr start_sub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr explored_pub_;

  // State
  grid_map::GridMap grid_map_;
  bool have_map_ = false;

  geometry_msgs::msg::PoseStamped goal_pose_;
  bool have_goal_ = false;

  geometry_msgs::msg::Pose start_pose_;
  std::string start_pose_frame_;
  bool have_start_pose_ = false;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TraversabilityAStarNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}