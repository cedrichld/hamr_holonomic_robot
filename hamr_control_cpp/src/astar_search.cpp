#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/utils.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>


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

class AStarPlanner : public rclcpp::Node {
public:
    AStarPlanner() : rclcpp::Node("astar_planner") {
        RCLCPP_INFO(this->get_logger(), "A* Planner node initialized");

        use_prm_ = declare_parameter<bool>("use_prm", true);
        heuristic_weight_ = declare_parameter<double>("heuristic_weight", 1.0);
        inflation_radius_m_ = declare_parameter<double>("inflation_radius_m", 0.2);
        allow_diagonal_ = declare_parameter<bool>("allow_diagonal", true);
        
        use_tf_start_ = declare_parameter<bool>("use_tf_start", true);
        map_frame_    = declare_parameter<std::string>("map_frame",  "map");
        base_frame_   = declare_parameter<std::string>("base_frame", "base_link");

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this](){
            if(!use_tf_start_) return;
            try{
                auto tf = tf_buffer_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
                start_pose_tf_.header = tf.header;
                start_pose_tf_.pose.position.x = tf.transform.translation.x;
                start_pose_tf_.pose.position.y = tf.transform.translation.y;
                start_pose_tf_.pose.position.z = 0.0;
                start_pose_tf_.pose.orientation = tf.transform.rotation;
                have_start_tf_ = true;
                // if(use_tf_start_ && have_map_ && have_goal_) planPath();
            } catch (tf2::TransformException & ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Could not get transform: %s", ex.what());
                have_start_tf_ = false;
            }
        });

        rclcpp::QoS latched(1); 
        latched.reliable().transient_local();
        
        path_pub_ = create_publisher<nav_msgs::msg::Path>("/astar/path", latched);
        explored_pub_ = create_publisher<visualization_msgs::msg::Marker>("/astar/explored", 1);
        
        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(1).transient_local().reliable(),
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ onMap(m); });
            
        goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 1, [this](geometry_msgs::msg::PoseStamped::SharedPtr g){ onGoal(g); });
            
        start_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 1, [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr s){ onStart(s); });

        if (use_prm_) {
            prm_nodes_sub_ = create_subscription<visualization_msgs::msg::Marker>(
                "/prm/nodes", rclcpp::QoS(1).transient_local().reliable(),
                [this](visualization_msgs::msg::Marker::SharedPtr m){ onPrmNodes(m); });
                
            prm_adj_sub_ = create_subscription<std_msgs::msg::Int32MultiArray>(
                "/prm/adjacency", rclcpp::QoS(1).transient_local().reliable(),
                [this](std_msgs::msg::Int32MultiArray::SharedPtr a){ onPrmAdjacency(a); });
        }

        RCLCPP_INFO(this->get_logger(), "A* Planner configured: use_prm=%s, heuristic_weight=%.2f", 
                    use_prm_ ? "true" : "false", heuristic_weight_);
    }

private:
    bool use_prm_;
    double heuristic_weight_;
    double inflation_radius_m_;
    bool allow_diagonal_;

    nav_msgs::msg::OccupancyGrid map_;
    std::vector<uint8_t> inflated_;
    bool have_map_ = false;

    geometry_msgs::msg::PoseStamped start_, goal_;
    bool have_start_ = false, have_goal_ = false;

    struct PrmNode { double x, y; };
    std::vector<PrmNode> prm_nodes_;
    std::vector<std::vector<int>> prm_adj_;
    bool have_prm_ = false;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr start_sub_;
    rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr prm_nodes_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr prm_adj_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr explored_pub_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    bool use_tf_start_{true};
    std::string map_frame_{"map"};
    std::string base_frame_{"base_link"};
    geometry_msgs::msg::PoseStamped start_pose_tf_;
    bool have_start_tf_{false}; 
    rclcpp::TimerBase::SharedPtr tf_timer_;

    bool getStartPose(geometry_msgs::msg::PoseStamped & out) {
        if(use_tf_start_){
            if(!have_start_tf_) return false;
            out = start_pose_tf_;
            return true;
        } else {
            if(!have_start_) return false;
            out = start_;
            return true;
        }
    }

    struct GridNode {
        int x, y;
        double yaw;
        double g_cost, h_cost;
        int parent_idx;
        
        double f_cost() const { return g_cost + h_cost; }
        
        bool operator>(const GridNode& other) const {
            return f_cost() > other.f_cost();
        }
    };

    inline bool worldToMap(double wx, double wy, int &mx, int &my) const {
        double ox = map_.info.origin.position.x;
        double oy = map_.info.origin.position.y;
        double r = map_.info.resolution;
        mx = (int)std::floor((wx - ox) / r);
        my = (int)std::floor((wy - oy) / r);
        return mx >= 0 && my >= 0 && (unsigned)mx < map_.info.width && (unsigned)my < map_.info.height;
    }

    inline void mapToWorld(int mx, int my, double &wx, double &wy) const {
        wx = map_.info.origin.position.x + (mx + 0.5) * map_.info.resolution;
        wy = map_.info.origin.position.y + (my + 0.5) * map_.info.resolution;
    }

    inline int coordToIndex(int x, int y) const {
        return y * (int)map_.info.width + x;
    }

    inline void indexToCoord(int idx, int &x, int &y) const {
        x = idx % (int)map_.info.width;
        y = idx / (int)map_.info.width;
    }

    inline bool isOccupied(int mx, int my) const {
        if (mx < 0 || my < 0 || (unsigned)mx >= map_.info.width || (unsigned)my >= map_.info.height)
            return true;
        return inflated_[my * map_.info.width + mx] > 0;
    }

    void inflateMap() {
        inflated_.assign(map_.info.width * map_.info.height, 0);
        int R = (int)std::ceil(inflation_radius_m_ / map_.info.resolution);
        int W = (int)map_.info.width, H = (int)map_.info.height;
        
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int8_t v = map_.data[y * W + x];
                if (v < 0 || v >= 65) {
                    for (int dy = -R; dy <= R; ++dy) {
                        int yy = y + dy;
                        if (yy < 0 || yy >= H) continue;
                        int dxmax = (int)std::floor(std::sqrt((double)R * R - dy * dy));
                        int x0 = std::max(0, x - dxmax);
                        int x1 = std::min(W - 1, x + dxmax);
                        std::fill(inflated_.begin() + yy * W + x0, inflated_.begin() + yy * W + x1 + 1, 1);
                    }
                }
            }
        }
    }

    double heuristic(int x1, int y1, int x2, int y2) const {
        double dx = (double)(x2 - x1) * map_.info.resolution;
        double dy = (double)(y2 - y1) * map_.info.resolution;
        return std::sqrt(dx * dx + dy * dy) * heuristic_weight_;
    }

    std::vector<std::pair<int, int>> getNeighbors(int x, int y) const {
        std::vector<std::pair<int, int>> neighbors;
        
        static const std::vector<std::pair<int, int>> moves = {
            {-1, 0}, {1, 0}, {0, -1}, {0, 1},
            {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
        };
        
        int max_neighbors = allow_diagonal_ ? 8 : 4;
        
        for (int i = 0; i < max_neighbors; ++i) {
            int nx = x + moves[i].first;
            int ny = y + moves[i].second;
            
            if (nx >= 0 && ny >= 0 && (unsigned)nx < map_.info.width && (unsigned)ny < map_.info.height) {
                if (!isOccupied(nx, ny)) {
                    neighbors.emplace_back(nx, ny);
                }
            }
        }
        return neighbors;
    }

    nav_msgs::msg::Path gridAStar(int start_x, int start_y, double start_yaw, int goal_x, int goal_y, double goal_yaw) {
        nav_msgs::msg::Path path;
        path.header.frame_id = map_.header.frame_id;
        path.header.stamp = now();

        std::priority_queue<GridNode, std::vector<GridNode>, std::greater<GridNode>> open_set;
        std::unordered_map<int, GridNode> all_nodes;
        std::unordered_set<int> closed_set;
        std::vector<int> explored_cells;

        int start_idx = coordToIndex(start_x, start_y);
        int goal_idx = coordToIndex(goal_x, goal_y);

        GridNode start_node;
        start_node.x = start_x;
        start_node.y = start_y;
        start_node.yaw = start_yaw;
        start_node.g_cost = 0.0;
        start_node.h_cost = heuristic(start_x, start_y, goal_x, goal_y);
        start_node.parent_idx = -1;

        open_set.push(start_node);
        all_nodes[start_idx] = start_node;

        while (!open_set.empty()) {
            GridNode current = open_set.top();
            // current.yaw = goal_yaw; // !!! compute specific yaw if needed
            open_set.pop();
            
            int current_idx = coordToIndex(current.x, current.y);
            
            if (closed_set.find(current_idx) != closed_set.end()) {
                continue;
            }
            
            closed_set.insert(current_idx);
            explored_cells.push_back(current_idx);

            if (current.x == goal_x && current.y == goal_y) {
                std::vector<std::pair<int, int>> path_coords;
                int idx = current_idx;
                
                while (idx != -1) {
                    int x, y;
                    indexToCoord(idx, x, y);
                    path_coords.emplace_back(x, y);
                    idx = all_nodes[idx].parent_idx;
                }
                
                std::reverse(path_coords.begin(), path_coords.end());
                
                for (const auto& coord : path_coords) {
                    geometry_msgs::msg::PoseStamped pose;
                    pose.header = path.header;
                    mapToWorld(coord.first, coord.second, pose.pose.position.x, pose.pose.position.y);
                    pose.pose.orientation = yaw_to_quat(current.yaw);
                    path.poses.push_back(pose);
                }
                
                publishExplored(explored_cells);
                RCLCPP_INFO(get_logger(), "Grid A* found path with %zu waypoints, explored %zu cells", 
                           path.poses.size(), explored_cells.size());
                return path;
            }

            auto neighbors = getNeighbors(current.x, current.y);
            
            for (const auto& neighbor : neighbors) {
                int nx = neighbor.first, ny = neighbor.second;
                int neighbor_idx = coordToIndex(nx, ny);
                
                if (closed_set.find(neighbor_idx) != closed_set.end()) {
                    continue;
                }
                
                double move_cost = (nx != current.x && ny != current.y) ? 
                                   map_.info.resolution * 1.414 : map_.info.resolution;
                double tentative_g = current.g_cost + move_cost;
                
                if (all_nodes.find(neighbor_idx) == all_nodes.end() || 
                    tentative_g < all_nodes[neighbor_idx].g_cost) {
                    
                    GridNode neighbor_node;
                    neighbor_node.x = nx;
                    neighbor_node.y = ny;
                    // neighbor_node.yaw = goal_yaw; // !!! compute specific yaw if needed
                    neighbor_node.g_cost = tentative_g;
                    neighbor_node.h_cost = heuristic(nx, ny, goal_x, goal_y);
                    neighbor_node.parent_idx = current_idx;
                    
                    all_nodes[neighbor_idx] = neighbor_node;
                    open_set.push(neighbor_node);
                }
            }
        }

        publishExplored(explored_cells);
        RCLCPP_WARN(get_logger(), "Grid A* failed to find path, explored %zu cells", explored_cells.size());
        return path;
    }

    nav_msgs::msg::Path prmAStar() {
        nav_msgs::msg::Path path;
        path.header.frame_id = map_.header.frame_id;
        path.header.stamp = now();

        if (!have_prm_) {
            RCLCPP_WARN(get_logger(), "PRM data not available");
            return path;
        }

        geometry_msgs::msg::PoseStamped start_pose;
        if(!getStartPose(start_pose)) return path;
        double start_x = start_pose.pose.position.x;
        double start_y = start_pose.pose.position.y;
        double start_yaw = quat_to_yaw(start_pose.pose.orientation);
        double goal_x = goal_.pose.position.x;
        double goal_y = goal_.pose.position.y;
        double goal_yaw = quat_to_yaw(goal_.pose.orientation);

        int start_node = findNearestNode(start_x, start_y);
        int goal_node = findNearestNode(goal_x, goal_y);

        if (start_node == -1 || goal_node == -1) {
            RCLCPP_WARN(get_logger(), "Cannot find valid start or goal nodes in PRM");
            return path;
        }

        struct PrmAStarNode {
            int idx;
            double g_cost, h_cost;
            int parent;
            double yaw;
            
            double f_cost() const { return g_cost + h_cost; }
            bool operator>(const PrmAStarNode& other) const {
                return f_cost() > other.f_cost();
            }
        };

        std::priority_queue<PrmAStarNode, std::vector<PrmAStarNode>, std::greater<PrmAStarNode>> open_set;
        std::unordered_map<int, PrmAStarNode> all_nodes;
        std::unordered_set<int> closed_set;

        PrmAStarNode start_astar;
        start_astar.yaw = start_yaw;
        start_astar.idx = start_node;
        start_astar.g_cost = 0.0;
        start_astar.h_cost = euclideanDistance(prm_nodes_[start_node].x, prm_nodes_[start_node].y,
                                               prm_nodes_[goal_node].x, prm_nodes_[goal_node].y) * heuristic_weight_;
        start_astar.parent = -1;

        open_set.push(start_astar);
        all_nodes[start_node] = start_astar;

        while (!open_set.empty()) {
            PrmAStarNode current = open_set.top();
            current.yaw = goal_yaw; // !!! compute specific yaw if needed
            open_set.pop();

            if (closed_set.find(current.idx) != closed_set.end()) {
                continue;
            }

            closed_set.insert(current.idx);

            if (current.idx == goal_node) {
                std::vector<int> path_indices;
                int idx = current.idx;
                
                while (idx != -1) {
                    path_indices.push_back(idx);
                    idx = all_nodes[idx].parent;
                }
                
                std::reverse(path_indices.begin(), path_indices.end());

                geometry_msgs::msg::PoseStamped start_pose;
                start_pose.header = path.header;
                start_pose.pose.position.x = start_x;
                start_pose.pose.position.y = start_y;
                start_pose.pose.orientation = goal_.pose.orientation; // !!! or compute specific yaw if needed
                path.poses.push_back(start_pose);

                for (int idx : path_indices) {
                    geometry_msgs::msg::PoseStamped pose;
                    pose.header = path.header;
                    pose.pose.position.x = prm_nodes_[idx].x;
                    pose.pose.position.y = prm_nodes_[idx].y;
                    pose.pose.orientation = goal_.pose.orientation; // !!! or compute specific yaw if needed
                    path.poses.push_back(pose);
                }

                geometry_msgs::msg::PoseStamped goal_pose;
                goal_pose.header = path.header;
                goal_pose.pose.position.x = goal_x;
                goal_pose.pose.position.y = goal_y;
                goal_pose.pose.orientation = goal_.pose.orientation;
                path.poses.push_back(goal_pose);

                RCLCPP_INFO(get_logger(), "PRM A* found path with %zu waypoints, explored %zu nodes", 
                           path.poses.size(), closed_set.size());
                return path;
            }

            for (int neighbor_idx : prm_adj_[current.idx]) {
                if (closed_set.find(neighbor_idx) != closed_set.end()) {
                    continue;
                }

                double edge_cost = euclideanDistance(prm_nodes_[current.idx].x, prm_nodes_[current.idx].y,
                                                     prm_nodes_[neighbor_idx].x, prm_nodes_[neighbor_idx].y);
                double tentative_g = current.g_cost + edge_cost;

                if (all_nodes.find(neighbor_idx) == all_nodes.end() || 
                    tentative_g < all_nodes[neighbor_idx].g_cost) {
                    
                    PrmAStarNode neighbor_node;
                    neighbor_node.idx = neighbor_idx;
                    neighbor_node.g_cost = tentative_g;
                    neighbor_node.h_cost = euclideanDistance(prm_nodes_[neighbor_idx].x, prm_nodes_[neighbor_idx].y,
                                                             prm_nodes_[goal_node].x, prm_nodes_[goal_node].y) * heuristic_weight_;
                    neighbor_node.parent = current.idx;
                    
                    all_nodes[neighbor_idx] = neighbor_node;
                    open_set.push(neighbor_node);
                }
            }
        }

        RCLCPP_WARN(get_logger(), "PRM A* failed to find path, explored %zu nodes", closed_set.size());
        return path;
    }

    double euclideanDistance(double x1, double y1, double x2, double y2) const {
        double dx = x2 - x1;
        double dy = y2 - y1;
        return std::sqrt(dx * dx + dy * dy);
    }

    int findNearestNode(double x, double y) const {
        int best_idx = -1;
        double best_dist = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < prm_nodes_.size(); ++i) {
            double dist = euclideanDistance(x, y, prm_nodes_[i].x, prm_nodes_[i].y);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = (int)i;
            }
        }
        return best_idx;
    }

    void publishExplored(const std::vector<int>& explored) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = map_.header.frame_id;
        marker.header.stamp = now();
        marker.ns = "astar";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::POINTS;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.color.r = 0.0f; marker.color.g = 1.0f; marker.color.b = 0.0f; marker.color.a = 0.5f;

        for (int idx : explored) {
            int x, y;
            indexToCoord(idx, x, y);
            geometry_msgs::msg::Point p;
            mapToWorld(x, y, p.x, p.y);
            p.z = 0.05;
            marker.points.push_back(p);
        }
        
        explored_pub_->publish(marker);
    }

    void planPath() {
        // if (!have_map_ || !have_start_ || !have_goal_) {
        //     RCLCPP_WARN(get_logger(), "Missing data: map=%s, start=%s, goal=%s", 
        //                have_map_ ? "✓" : "✗", have_start_ ? "✓" : "✗", have_goal_ ? "✓" : "✗");
        //     return;
        // }

        if (!have_map_ || !have_goal_) {
            RCLCPP_WARN(get_logger(), "Missing data: map=%s, goal=%s", 
                       have_map_ ? "✓" : "✗", have_goal_ ? "✓" : "✗");
            return;
        }
        geometry_msgs::msg::PoseStamped start_pose;
        if(!getStartPose(start_pose)){
            RCLCPP_WARN(get_logger(), "Start pose not available yet");
            return;
        }

        if (use_prm_ && !have_prm_) {
            RCLCPP_WARN(get_logger(), "PRM mode enabled but PRM data not available");
            return;
        }

        auto start_time = now();
        nav_msgs::msg::Path path;

        if (use_prm_) {
            path = prmAStar();
        } else {
            int start_x, start_y, goal_x, goal_y;
            double start_yaw = quat_to_yaw(start_pose.pose.orientation);
            double goal_yaw = quat_to_yaw(goal_.pose.orientation);
            if (!worldToMap(start_pose.pose.position.x, start_pose.pose.position.y, start_x, start_y) ||
                !worldToMap(goal_.pose.position.x, goal_.pose.position.y, goal_x, goal_y)) {
                RCLCPP_ERROR(get_logger(), "Start or goal outside map bounds");
                return;
            }
            
            if (isOccupied(start_x, start_y) || isOccupied(goal_x, goal_y)) {
                RCLCPP_ERROR(get_logger(), "Start or goal in occupied space");
                return;
            }
            
            path = gridAStar(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw);
        }

        auto end_time = now();
        double planning_time = (end_time - start_time).seconds() * 1000.0;

        if (!path.poses.empty()) {
            path_pub_->publish(path);
            RCLCPP_INFO(get_logger(), "Published path with %zu poses (%.2f ms)", 
                       path.poses.size(), planning_time);
        } else {
            RCLCPP_WARN(get_logger(), "No path found (%.2f ms)", planning_time);
        }
    }

    void onMap(nav_msgs::msg::OccupancyGrid::SharedPtr m) {
        if (have_map_) return;
        map_ = *m;
        have_map_ = true;
        inflateMap();
        RCLCPP_INFO(get_logger(), "Map received and inflated (%.2f m radius)", inflation_radius_m_);
        planPath();
    }

    void onStart(geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr s) {
        start_.header = s->header;
        start_.pose = s->pose.pose;
        have_start_ = true;
        RCLCPP_INFO(get_logger(), "Start pose set: (%.2f, %.2f)", 
                   start_.pose.position.x, start_.pose.position.y);
        planPath();
    }

    void onGoal(geometry_msgs::msg::PoseStamped::SharedPtr g) {
        goal_ = *g;
        have_goal_ = true;
        RCLCPP_INFO(get_logger(), "Goal pose set: (%.2f, %.2f, %.2f)", 
                   goal_.pose.position.x, goal_.pose.position.y, quat_to_yaw(goal_.pose.orientation));
        planPath();
    }

    void onPrmNodes(visualization_msgs::msg::Marker::SharedPtr m) {
        prm_nodes_.clear();
        prm_nodes_.reserve(m->points.size());
        
        for (const auto& point : m->points) {
            prm_nodes_.push_back({point.x, point.y});
        }
        
        RCLCPP_INFO(get_logger(), "Received %zu PRM nodes", prm_nodes_.size());
        
        if (have_prm_ && prm_adj_.size() == prm_nodes_.size()) {
            planPath();
        }
    }

    void onPrmAdjacency(std_msgs::msg::Int32MultiArray::SharedPtr a) {
        prm_adj_.clear();
        
        size_t idx = 0;
        while (idx < a->data.size()) {
            int num_neighbors = a->data[idx++];
            std::vector<int> neighbors;
            
            for (int i = 0; i < num_neighbors; ++i) {
                if (idx < a->data.size()) {
                    neighbors.push_back(a->data[idx++]);
                }
            }
            prm_adj_.push_back(neighbors);
        }
        
        have_prm_ = true;
        RCLCPP_INFO(get_logger(), "Received PRM adjacency for %zu nodes", prm_adj_.size());
        
        if (prm_adj_.size() == prm_nodes_.size()) {
            planPath();
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AStarPlanner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}