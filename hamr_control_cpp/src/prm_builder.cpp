#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <algorithm>
#include <random>
#include <cmath>
#include <nanoflann.hpp>

struct KDCloud{
    struct Pt { double x,y; };
    std::vector<Pt> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const{
        return (dim==0)?pts[idx].x:pts[idx].y;
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

class PrmBuilder : public rclcpp::Node {
    public:

    PrmBuilder() : rclcpp::Node("prm_builder") {
        RCLCPP_INFO(this->get_logger(), "PrmBuilder node has been initialized.");

        samples_ = declare_parameter<int>("samples", 10000);
        k_neighbors_ = declare_parameter<int>("k_neighbors", 6);
        inflattion_m_ = declare_parameter<double>("inflation_radius_m", 0.75);
        occ_threshold_ = declare_parameter<int>("occupied_threshold", 65);
        allow_unknown_ = declare_parameter<bool>("allow_unknown", false);
        seed_ = declare_parameter<int>("seed", 0);
        publish_inflated_map_ = declare_parameter<bool>("publish_inflated_map", true);

        rclcpp::QoS latched(1); latched.reliable().transient_local();
        nodes_pub_ = create_publisher<visualization_msgs::msg::Marker>("/prm/nodes", latched);
        adj_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("/prm/adjacency", latched);
        edges_pub_ = create_publisher<visualization_msgs::msg::Marker>("/prm/edges", latched);
        inflated_map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/map_inflated", latched);
        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(1).transient_local().reliable(),
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ onMap(m); });
        
        RCLCPP_INFO(this->get_logger(), "PrmBuilder initialized with parameters: samples=%d, k_neighbors=%d, inflation_radius_m=%.2f, occupied_threshold=%d, allow_unknown=%s, seed=%d",
            samples_, k_neighbors_, inflattion_m_, occ_threshold_, allow_unknown_ ? "true" : "false", seed_);
        
        publishTestMarker();
    }

    private:
    int samples_, k_neighbors_, occ_threshold_, seed_;
    double inflattion_m_;
    bool allow_unknown_;

    nav_msgs::msg::OccupancyGrid map_;
    std::vector<uint8_t> inflated_;
    bool have_map_{false};

    struct Node { double x, y; };
    std::vector<Node> nodes_;
    std::vector<std::vector<int>> adj_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr nodes_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr adj_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edges_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr inflated_map_pub_;
    bool publish_inflated_map_{true};

    inline bool worldToMap(double wx, double wy, int &mx, int &my) const {
        double ox = map_.info.origin.position.x;
        double oy = map_.info.origin.position.y;
        double r  = map_.info.resolution;
        mx = (int)std::floor((wx - ox)/r);
        my = (int)std::floor((wy - oy)/r);
        return mx>=0 && my>=0 && (unsigned)mx<map_.info.width && (unsigned)my<map_.info.height;
    }

    inline bool occRaw(int mx, int my) const {
        int8_t v = map_.data[my*map_.info.width + mx];
        if (v<0) return !allow_unknown_;
        return v >= occ_threshold_;
    }

    void inflate() {
        inflated_.assign(map_.info.width * map_.info.height, 0);
        int R = (int)std::ceil(inflattion_m_ / map_.info.resolution);
        int W = (int)map_.info.width, H=(int)map_.info.height;
        for (int y=0; y<H; ++y) for (int x=0; x<W; ++x) {
            if (!occRaw(x,y)) continue;
            for (int dy=-R; dy<=R; ++dy) {
                int yy=y+dy; if (yy<0||yy>=H) continue;
                int dxmax=(int)std::floor(std::sqrt((double)R*R - dy*dy));
                int x0=std::max(0, x-dxmax), x1=std::min(W-1, x+dxmax);
                std::fill(inflated_.begin()+yy*W+x0, inflated_.begin()+yy*W+x1+1, 1);
            }
        }
    }

    bool lineFree(double x0, double y0, double x1, double y1) const {
        int mx0, my0, mx1, my1;
        if (!worldToMap(x0,y0,mx0,my0)) return false;
        if (!worldToMap(x1,y1,mx1,my1)) return false;
        int dx = std::abs(mx1-mx0), sx = mx0<mx1?1:-1;
        int dy = -std::abs(my1-my0), sy = my0<my1?1:-1;
        int err = dx + dy, x=mx0, y=my0;
        while (true) {
            if (inflated_[y*(int)map_.info.width + x]) return false;
            if (x==mx1 && y==my1) break;
            int e2 = 2*err;
            if (e2 >= dy){ err += dy; x += sx; }
            if (e2 <= dx){ err += dx; y += sy; }
        }
        return true;
    }

    void onMap(nav_msgs::msg::OccupancyGrid::SharedPtr m) {
        if (have_map_) return;
        map_ = *m;
        have_map_ = true;
        
        RCLCPP_INFO(this->get_logger(), "Map received - Frame: %s, Size: %dx%d, Origin: (%.2f, %.2f)", 
                    map_.header.frame_id.c_str(),
                    map_.info.width, map_.info.height,
                    map_.info.origin.position.x, map_.info.origin.position.y);
        
        inflate();
        RCLCPP_INFO(this->get_logger(), "Received map (%d x %d, res=%.3f m/cell), inflating obstacles by %.2f m",
            map_.info.width, map_.info.height, map_.info.resolution, inflattion_m_);
        publishInflatedMap();
        buildPRM();
        publishGraph();
    }

    double freeAreaM2() const {
        const int W = (int)map_.info.width, H = (int)map_.info.height;
        const double Acell = map_.info.resolution * map_.info.resolution;
        size_t free_cnt = 0;
        for (int i=0;i<W*H;++i) if (inflated_[i]==0) ++free_cnt;
        return free_cnt * Acell;
    }

    double connectRadius(int n) const {
        const int d = 2;
        const double zeta2 = M_PI;
        const double mu_free = freeAreaM2();
        const double gamma = 2.2 * std::sqrt((1.0 + 1.0/d) * mu_free / zeta2);
        double rn = gamma * std::sqrt(std::log(std::max(n,2)) / (double)std::max(n,2));
        rn = std::min(rn, 3.0);
        rn = std::max(rn, 3.0*map_.info.resolution);
        return rn;
    }

    void buildPRM() {
        nodes_.clear(); adj_.clear();
        nodes_.reserve(samples_);
        
        const double ox  = map_.info.origin.position.x;
        const double oy  = map_.info.origin.position.y;
        const double res = map_.info.resolution;
        const double maxx = ox + map_.info.width  * res;
        const double maxy = oy + map_.info.height * res;

        std::mt19937 rng(seed_ ? seed_ : (int)now().nanoseconds());
        std::uniform_real_distribution<double> UX(ox, maxx), UY(oy, maxy);

        int added=0, attempts=0;
        while (added < samples_ && attempts < samples_*30) {
            ++attempts;
            double x = UX(rng), y = UY(rng);
            int mx,my; if (!worldToMap(x,y,mx,my)) continue;
            if (inflated_[my*(int)map_.info.width + mx]) continue;
            nodes_.push_back({x,y}); ++added;
        }
        
        RCLCPP_INFO(get_logger(), "Generated %zu nodes (attempts=%d)", nodes_.size(), attempts);
        if (nodes_.size() > 0) {
            RCLCPP_INFO(get_logger(), "Node range: (%.2f, %.2f) to (%.2f, %.2f)", 
                       nodes_[0].x, nodes_[0].y, nodes_.back().x, nodes_.back().y);
        }

        KDCloud cloud;
        cloud.pts.reserve(nodes_.size());
        for(auto &n : nodes_) cloud.pts.push_back({n.x,n.y});
        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KDCloud>, KDCloud, 2, size_t>;

        KDTree index(2, cloud, {10});
        index.buildIndex();

        const int n = (int)nodes_.size();
        const int k_target = (int)std::ceil(4.1 * std::log((double)std::max(n,2)));
        const int Kquery   = std::max(k_target*3 + 1, 2);

        adj_.assign(nodes_.size(), {});
        std::vector<size_t> idx(Kquery);
        std::vector<double> dist2(Kquery);

        for (size_t i=0;i<nodes_.size();++i) {
            double q[2] = { nodes_[i].x, nodes_[i].y };
            size_t found = index.knnSearch(q, Kquery, idx.data(), dist2.data());
            int connected = 0;    
            for (size_t t=0; t<found && connected < k_target; ++t) {
                size_t j = idx[t];
                if (j == i) continue;
                if (j <  i) continue;
                if (!lineFree(nodes_[i].x, nodes_[i].y, nodes_[j].x, nodes_[j].y)) continue;
                adj_[i].push_back((int)j);
                adj_[(size_t)j].push_back((int)i);
                ++connected;
            }
        }
        
        size_t total_edges = 0;
        for (const auto& adj_list : adj_) {
            total_edges += adj_list.size();
        }
        RCLCPP_INFO(get_logger(), "Connectivity complete (PRM* k=%d, n=%d, edges=%zu)", k_target, n, total_edges/2);
    }

    void publishGraph() {
        if (nodes_.empty()) {
            RCLCPP_WARN(get_logger(), "No nodes to publish!");
            return;
        }
        
        RCLCPP_INFO(get_logger(), "Publishing markers in frame: %s", map_.header.frame_id.c_str());

        visualization_msgs::msg::Marker m;
        m.header.frame_id = map_.header.frame_id;
        m.header.stamp = now();
        m.ns = "prm";
        m.id = 1;
        m.type = visualization_msgs::msg::Marker::POINTS;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.20;
        m.scale.y = 0.20;
        m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 1.0f;
        m.lifetime = rclcpp::Duration::from_nanoseconds(0);

        m.points.reserve(nodes_.size());
        for (const auto &n : nodes_) {
            geometry_msgs::msg::Point p;
            p.x = n.x; p.y = n.y; p.z = 0.1;
            m.points.push_back(p);
        }

        nodes_pub_->publish(m);
        RCLCPP_INFO(get_logger(), "Published %zu node markers", m.points.size());

        std_msgs::msg::Int32MultiArray adj;
        adj.layout.dim.resize(1);
        adj.layout.dim[0].label = "flattened_adjacency";
        adj.layout.dim[0].size  = adj_.size();
        for (auto &nbrs : adj_) {
            adj.data.push_back((int)nbrs.size());
            adj.data.insert(adj.data.end(), nbrs.begin(), nbrs.end());
        }
        adj_pub_->publish(adj);

        visualization_msgs::msg::Marker mk;
        mk.header = m.header;
        mk.ns = "prm"; mk.id=0;
        mk.type = visualization_msgs::msg::Marker::LINE_LIST;
        mk.action = visualization_msgs::msg::Marker::ADD;
        mk.scale.x = 0.05;
        mk.color.r = 0.0f; mk.color.g = 0.8f; mk.color.b = 1.0f; mk.color.a = 1.0f;
        mk.lifetime = rclcpp::Duration::from_nanoseconds(0);

        for (size_t i=0;i<adj_.size();++i) {
            for (int j : adj_[i]) if ((size_t)j > i) {
                geometry_msgs::msg::Point a,b;
                a.x=nodes_[i].x; a.y=nodes_[i].y; a.z=0.1;
                b.x=nodes_[j].x; b.y=nodes_[j].y; b.z=0.1;
                mk.points.push_back(a); mk.points.push_back(b);
            }
        }
        edges_pub_->publish(mk);

        RCLCPP_INFO(get_logger(), "Published %zu edge markers (%zu edges)", 
                    mk.points.size(), mk.points.size()/2);
    }

    void publishInflatedMap() {
        if (!publish_inflated_map_) return;

        nav_msgs::msg::OccupancyGrid out = map_;
        out.header.stamp = now();
        out.data.resize((size_t)map_.info.width * map_.info.height);

        const int W = (int)map_.info.width, H = (int)map_.info.height;
        for (int i = 0; i < W*H; ++i) {
            if (inflated_[i]) {
                out.data[i] = 100;                  
            } else {
                int8_t v = map_.data[i];
                if (v < 0 && !allow_unknown_) out.data[i] = -1;  
                else                           out.data[i] = 0;  
            }
        }
        inflated_map_pub_->publish(out);
    }

    void publishTestMarker() {
        visualization_msgs::msg::Marker test;
        test.header.frame_id = "map";
        test.header.stamp = now();
        test.ns = "test";
        test.id = 999;
        test.type = visualization_msgs::msg::Marker::SPHERE;
        test.action = visualization_msgs::msg::Marker::ADD;
        
        test.pose.position.x = 0.0;
        test.pose.position.y = 0.0; 
        test.pose.position.z = 0.5;
        test.pose.orientation.w = 1.0;
        
        test.scale.x = 0.5;
        test.scale.y = 0.5; 
        test.scale.z = 0.5;
        
        test.color.r = 1.0;
        test.color.g = 1.0;
        test.color.b = 0.0;
        test.color.a = 1.0;
        test.lifetime = rclcpp::Duration::from_nanoseconds(0);
        
        nodes_pub_->publish(test);
        RCLCPP_INFO(get_logger(), "Published test marker");
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PrmBuilder>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}