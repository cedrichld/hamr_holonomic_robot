# hamr_holonomic_robot
## Installation steps (ROS2 Jazzy Ubuntu 24.04):
1. Create ros2_ws: 
```bash
mkdir -p ~/ros2_ws/
cd ~/ros2_ws
mkdir src/
cd src/
git clone https://github.com/cedrichld/hamr_holonomic_robot.git
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
``` 
Smart to add to ~/.bashrc: 
```bash 
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```
2. Build ros2_ws:
```bash
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```
Smart to add to ~/.bashrc:
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```
3. Export (required for the STLs):
```bash
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix hamr_description)/share":$GZ_SIM_RESOURCE_PATH 
```
Smart to add to ~/.bashrc: 
```bash
echo "sexport GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix hamr_description)/share":$GZ_SIM_RESOURCE_PATH" >> ~/.bashrc
```
## Run Simulation (must have sourced correctly before)
```bash
ros2 launch hamr_bringup hamr.launch.xml
ros2 run reference_trajectory waypoint_traj 
```
For not waypoint_traj is just a set of waypoints that we discretize in smaller steps:
```python
points = np.array([ # x, y, yaw
            [0.0, 0.0, 0.0], # SQUARE
            [5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0]
])
```