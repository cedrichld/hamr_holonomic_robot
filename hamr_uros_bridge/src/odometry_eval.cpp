#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <Eigen/Dense>
#include <deque>
#include <cmath>
#include <algorithm>

using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::PoseWithCovarianceStamped;
using nav_msgs::msg::Path;

namespace {

inline double wrap_pi(double a) {
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a < 0) a += 2.0 * M_PI;
  return a - M_PI;
}

inline double yaw_from_quat(double x, double y, double z, double w) {
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

struct Sample {
  rclcpp::Time t;
  double x, y, yaw;
};

} // namespace

class OdomViconEvalNode : public rclcpp::Node {
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      PoseWithCovarianceStamped, PoseStamped>;

public:
  OdomViconEvalNode() : Node("odometry_eval_cpp") {
    // ---- Parameters ----
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/robot_pose");
    vicon_topic_ = declare_parameter<std::string>("vicon_topic", "/HAMR_Base/pose");
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "mocap");
    odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
    double slop_ms = declare_parameter<double>("sync_slop_ms", 30.0);
    calib_seconds_ = declare_parameter<double>("calib_seconds", 5.0);
    publish_paths_ = declare_parameter<bool>("publish_paths", true);

    // ---- Publishers ----
    if (publish_paths_) {
      pub_path_odom_   = create_publisher<Path>("/hamr_eval/odom_path", 1);
      pub_path_vicon_  = create_publisher<Path>("/hamr_eval/vicon_path", 1);
      pub_path_align_  = create_publisher<Path>("/hamr_eval/aligned_odom_path", 1);

      path_odom_.header.frame_id  = map_frame_id_;
      path_vicon_.header.frame_id = map_frame_id_;
      path_align_.header.frame_id = map_frame_id_;
    }

    // ---- Subscribers + Synchronizer ----
    sub_odom_.subscribe(this, odom_topic_);
    sub_vicon_.subscribe(this, vicon_topic_);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(100), sub_odom_, sub_vicon_);
    sync_->setInterMessageLowerBound(0, rclcpp::Duration::from_seconds(0.0));
    sync_->registerCallback(
        std::bind(&OdomViconEvalNode::cbSync, this,
                  std::placeholders::_1, std::placeholders::_2));

    slop_ = rclcpp::Duration::from_nanoseconds(
        static_cast<int64_t>(slop_ms * 1e6));

    RCLCPP_INFO(get_logger(), "HAMR Odom↔Vicon eval (C++) started. Topics: %s , %s",
                odom_topic_.c_str(), vicon_topic_.c_str());
  }

private:
  // --- Core callback (synced messages) ---
  void cbSync(const PoseWithCovarianceStamped::ConstSharedPtr& odom_msg,
              const PoseStamped::ConstSharedPtr& vicon_msg) {

    // Extract odom
    const auto& po = odom_msg->pose.pose;
    double ox = po.position.x;
    double oy = po.position.y;
    double oyaw = yaw_from_quat(po.orientation.x, po.orientation.y,
                                po.orientation.z, po.orientation.w);

    // Extract vicon
    const auto& pv = vicon_msg->pose;
    double vx = pv.position.x;
    double vy = pv.position.y;
    double vyaw = yaw_from_quat(pv.orientation.x, pv.orientation.y,
                                pv.orientation.z, pv.orientation.w);

    // rclcpp::Time t = odom_msg->header.stamp;
    rclcpp::Time t(odom_msg->header.stamp, RCL_ROS_TIME);

    // Append buffers
    buf_odom_.push_back({t, ox, oy, oyaw});
    buf_vicon_.push_back({t, vx, vy, vyaw});

    // Keep memory bounded
    const size_t MAX_KEEP = 50000;
    if (buf_odom_.size() > MAX_KEEP) {
      buf_odom_.erase(buf_odom_.begin(), buf_odom_.begin() + (MAX_KEEP/5));
      buf_vicon_.erase(buf_vicon_.begin(), buf_vicon_.begin() + (MAX_KEEP/5));
    }

    // Perform one-shot alignment after calib window
    if (!aligned_) {
      const rclcpp::Time t0 = buf_odom_.front().t;
      if ((t - t0) > rclcpp::Duration::from_seconds(calib_seconds_)) {
        estimateAlignment();  // sets T_map_from_odom_
        aligned_ = true;
        RCLCPP_INFO(get_logger(), "Estimated SE(2) alignment (odom -> %s).", map_frame_id_.c_str());
      }
    }

    // Publish paths + stats
    if (publish_paths_) {
      publishPathsAndStats();
    }
  }

  // --- Estimate SE(2) using weighted Procrustes on (x,y) ---
  void estimateAlignment() {
    const size_t n = std::min(buf_odom_.size(), buf_vicon_.size());
    if (n < 5) return;

    Eigen::MatrixXd X(2, n), Y(2, n);
    for (size_t i = 0; i < n; ++i) {
      X(0, i) = buf_odom_[i].x; X(1, i) = buf_odom_[i].y;
      Y(0, i) = buf_vicon_[i].x; Y(1, i) = buf_vicon_[i].y;
    }
    // Center
    Eigen::Vector2d x_c = X.rowwise().mean();
    Eigen::Vector2d y_c = Y.rowwise().mean();
    Eigen::MatrixXd X0 = X.colwise() - x_c;
    Eigen::MatrixXd Y0 = Y.colwise() - y_c;

    // 2x2 covariance
    Eigen::Matrix2d S = X0 * Y0.transpose();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();
    Eigen::Matrix2d R = U * V.transpose();
    if (R.determinant() < 0) {
      // Fix improper rotation (reflection)
      U.col(1) *= -1.0;
      R = U * V.transpose();
    }
    Eigen::Vector2d t = y_c - R * x_c;

    // Save as 3x3 homogeneous
    T_map_from_odom_.setIdentity();
    T_map_from_odom_.block<2,2>(0,0) = R;
    T_map_from_odom_.block<2,1>(0,2) = t;
  }

  // --- Apply T to a point ---
  inline void applyT(const Eigen::Matrix3d& T, double x, double y, double& xo, double& yo) const {
    xo = T(0,0)*x + T(0,1)*y + T(0,2);
    yo = T(1,0)*x + T(1,1)*y + T(1,2);
  }

  void publishPathsAndStats() {
    // const auto now = this->now();
    const auto now = rclcpp::Clock(RCL_ROS_TIME).now();

    if (buf_odom_.empty() || buf_vicon_.empty()) return;
    const auto& o = buf_odom_.back();
    const auto& v = buf_vicon_.back();

    double ax = o.x, ay = o.y;
    if (aligned_) applyT(T_map_from_odom_, o.x, o.y, ax, ay);

    // Paths
    PoseStamped ps_odom, ps_vicon, ps_align;
    ps_odom.header.stamp = now;  ps_vicon.header.stamp = now; ps_align.header.stamp = now;
    ps_odom.header.frame_id = map_frame_id_;
    ps_vicon.header.frame_id = map_frame_id_;
    ps_align.header.frame_id = map_frame_id_;
    ps_odom.pose.position.x = o.x; ps_odom.pose.position.y = o.y;
    ps_vicon.pose.position.x = v.x; ps_vicon.pose.position.y = v.y;
    ps_align.pose.position.x = ax;  ps_align.pose.position.y = ay;

    auto set_qz = [](PoseStamped& ps, double yaw){
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = std::sin(0.5*yaw);
      ps.pose.orientation.w = std::cos(0.5*yaw);
    };
    set_qz(ps_odom, o.yaw);
    set_qz(ps_vicon, v.yaw);
    set_qz(ps_align, o.yaw); // keep odom yaw for aligned path viz

    if (publish_paths_) {
      path_odom_.header.stamp = now;
      path_vicon_.header.stamp = now;
      path_align_.header.stamp = now;
      path_odom_.poses.push_back(ps_odom);
      path_vicon_.poses.push_back(ps_vicon);
      path_align_.poses.push_back(ps_align);
      // Trim if very long
      const size_t MAXP = 5000;
      auto trim = [&](Path& p){
        if (p.poses.size() > MAXP) {
          p.poses.erase(p.poses.begin(), p.poses.begin() + (MAXP/5));
        }
      };
      trim(path_odom_); trim(path_vicon_); trim(path_align_);
      pub_path_odom_->publish(path_odom_);
      pub_path_vicon_->publish(path_vicon_);
      pub_path_align_->publish(path_align_);
    }

    // --- Metrics ---
    const double dx = ax - v.x;
    const double dy = ay - v.y;
    const double ate = std::hypot(dx, dy);

    // RPE over last k samples
    const int k = 10;
    double rpe = std::numeric_limits<double>::quiet_NaN();
    if (path_align_.poses.size() > static_cast<size_t>(k) &&
        path_vicon_.poses.size() > static_cast<size_t>(k)) {
      const auto& a0 = path_align_.poses[path_align_.poses.size()-k-1].pose.position;
      const auto& a1 = path_align_.poses.back().pose.position;
      const auto& v0 = path_vicon_.poses[path_vicon_.poses.size()-k-1].pose.position;
      const auto& v1 = path_vicon_.poses.back().pose.position;
      rpe = std::hypot((a1.x - a0.x) - (v1.x - v0.x),
                       (a1.y - a0.y) - (v1.y - v0.y));
    }

    const double dyaw = wrap_pi(o.yaw - v.yaw);

    // Throttled log
    if ((now - last_stat_time_) > rclcpp::Duration::from_seconds(2.0)) {
      RCLCPP_INFO(get_logger(), "ATE=%.03f m | RPE(%d)=%.03f m | dYaw=%.1f deg%s",
                  ate, k, rpe, dyaw * 180.0/M_PI, aligned_ ? " | aligned" : " | aligning...");
      last_stat_time_ = now;
    }
  }

private:
  // Params
  std::string odom_topic_, vicon_topic_, map_frame_id_, odom_frame_id_;
  rclcpp::Duration slop_{0,0};
  double calib_seconds_{5.0};
  bool publish_paths_{true};

  // Sync
  message_filters::Subscriber<PoseWithCovarianceStamped> sub_odom_;
  message_filters::Subscriber<PoseStamped>               sub_vicon_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Buffers
  std::deque<Sample> buf_odom_, buf_vicon_;

  // Alignment
  bool aligned_{false};
  Eigen::Matrix3d T_map_from_odom_{Eigen::Matrix3d::Identity()};

  // Paths
  rclcpp::Publisher<Path>::SharedPtr pub_path_odom_;
  rclcpp::Publisher<Path>::SharedPtr pub_path_vicon_;
  rclcpp::Publisher<Path>::SharedPtr pub_path_align_;
  Path path_odom_, path_vicon_, path_align_;

  // Stats
  // rclcpp::Time last_stat_time_{};
  rclcpp::Time last_stat_time_{rclcpp::Clock(RCL_ROS_TIME).now()};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomViconEvalNode>());
  rclcpp::shutdown();
  return 0;
}
