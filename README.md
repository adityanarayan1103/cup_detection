# Cup Detection Package

ROS package for autonomous cup detection and approach behavior for the PuppyPi robot. The robot uses computer vision to detect cups and autonomously walks toward them, stopping at a safe distance.

## Features

- **Autonomous Cup Detection**: Uses OpenCV with color and shape-based detection
- **Smart Approach**: PID-controlled movement for smooth approach and stopping
- **Configurable**: Adjustable distance thresholds and detection parameters
- **ROS Integration**: Standard ROS services and topics for control
- **Debug Visualization**: Publishes annotated images showing detection results

## Installation

The package is already set up in your workspace. Build it with:

```bash
cd /Users/adityanarayanverma/Desktop/puppipi/ros1_ws
catkin_make --pkg cup_detection
source devel/setup.bash
```

## Quick Start

### Basic Usage

1. **Launch the node**:
```bash
roslaunch cup_detection cup_detection.launch
```

2. **Start detection** (in another terminal):
```bash
# Enter detection mode
rosservice call /cup_detection/enter

# Enable autonomous movement
rosservice call /cup_detection/set_running "data: true"
```

3. **Place a light-colored cup** in front of the robot and watch it approach!

4. **Stop when done**:
```bash
# Disable movement
rosservice call /cup_detection/set_running "data: false"

# Exit detection mode
rosservice call /cup_detection/exit
```

### Auto-Start Mode

Launch with automatic start:
```bash
roslaunch cup_detection cup_detection.launch auto_start:=true
```

This will automatically enter detection mode and start movement.

## ROS API

### Services

- **`~enter`** (`std_srvs/Trigger`)
  - Enters cup detection mode and subscribes to camera
  - Initializes robot pose

- **`~exit`** (`std_srvs/Trigger`)
  - Exits detection mode and stops the robot
  - Cleans up resources

- **`~set_running`** (`std_srvs/SetBool`)
  - Enable/disable autonomous movement
  - `data: true` → start moving toward cup
  - `data: false` → stop moving

- **`~set_distance_threshold`** (`interfaces/SetFloat64`)
  - Set stopping distance (cup radius in pixels)
  - Default: 100 pixels
  - Example: `rosservice call /cup_detection/set_distance_threshold "data: 120.0"`

### Topics

#### Subscribed
- **`/usb_cam/image_raw`** (`sensor_msgs/Image`)
  - Camera feed input

#### Published
- **`/puppy_control/velocity`** (`puppy_control/Velocity`)
  - Movement commands for the robot

- **`/puppy_control/pose`** (`puppy_control/Pose`)
  - Pose adjustments for the robot

- **`~image_result`** (`sensor_msgs/Image`)
  - Debug image with detection visualization
  - View with: `rosrun image_view image_view image:=/cup_detection/image_result`

## Configuration

### Launch File Parameters

```bash
roslaunch cup_detection cup_detection.launch \
  node_name:=cup_detection \
  auto_start:=false \
  stop_radius:=100 \
  camera_topic:=/usb_cam/image_raw \
  image_width:=640 \
  image_height:=480
```

### Detection Parameters

Edit `scripts/cup_detection_node.py` to tune:

- **Color Range** (`CupDetector.__init__`):
  - `lower_hsv`: Lower HSV threshold (default: [0, 0, 180] for white)
  - `upper_hsv`: Upper HSV threshold (default: [180, 30, 255])

- **Distance Thresholds** (`CupDetectionNode.__init__`):
  - `min_radius`: 20px (too far, move fast)
  - `target_radius`: 85px (approaching, slow down)
  - `stop_radius`: 100px (close enough, stop)

- **Movement Speeds**:
  - `base_forward_velocity`: 12.0 (fast approach)
  - `approach_velocity`: 8.0 (slow approach)

- **PID Gains**:
  - `x_pid`: P=0.0015, I=0.0001, D=0.00005 (horizontal alignment)
  - `forward_pid`: P=0.15, I=0.001, D=0.01 (forward control)

## How It Works

### Detection Algorithm

1. **Color Filtering**: Converts image to HSV and filters for light colors (white, cream, light blue)
2. **Morphological Cleanup**: Removes noise using erosion and dilation
3. **Contour Detection**: Finds object boundaries in the filtered image
4. **Shape Analysis**: Scores candidates based on:
   - Circularity (cups viewed from top are circular)
   - Aspect ratio (cups from side view are taller or square-ish)
   - Size (minimum area threshold)
5. **Best Match**: Selects highest-scoring candidate as the cup

### Movement Control

1. **Horizontal Alignment**: PID controller keeps cup centered in camera view
2. **Distance Estimation**: Uses detected circle radius as distance proxy
3. **Speed Control**: 
   - Far away (radius < 20px): Fast forward (12.0)
   - Medium (20-85px): Progressive slowdown
   - Close (85-100px): Slow approach (8.0)
   - Very close (≥100px): **STOP**

## Troubleshooting

### Robot doesn't detect the cup

- Ensure cup is light-colored (white, cream, light blue)
- Check camera feed: `rosrun image_view image_view image:=/usb_cam/image_raw`
- View detection results: `rosrun image_view image_view image:=/cup_detection/image_result`
- Adjust HSV thresholds in code if needed

### Robot moves but doesn't approach cup

- Check that `set_running` is called with `data: true`
- Verify cup is detected (green circle in image_result)
- Check velocity commands: `rostopic echo /puppy_control/velocity`

### Robot doesn't stop at cup

- Cup might be too small/far → increase `stop_radius`
- Adjust stopping threshold: `rosservice call /cup_detection/set_distance_threshold "data: 120.0"`

### Robot behavior is jerky

- Reduce PID gains (especially P term) in code
- Decrease forward velocities

## Testing Tips

1. **Start Simple**: Use a plain white cup on a dark surface
2. **Good Lighting**: Ensure adequate lighting for camera
3. **Clear Path**: Remove obstacles between robot and cup
4. **Gradual Tuning**: Adjust one parameter at a time
5. **Monitor Logs**: Watch terminal output for detection status

## Example Testing Session

```bash
# Terminal 1: Launch the node
roslaunch cup_detection cup_detection.launch

# Terminal 2: Control the robot
rosservice call /cup_detection/enter
rosservice call /cup_detection/set_running "data: true"

# Terminal 3: Monitor detection (optional)
rostopic echo /cup_detection/image_result

# When done
rosservice call /cup_detection/set_running "data: false"
rosservice call /cup_detection/exit
```

## Dependencies

- ROS (tested with ROS Noetic)
- OpenCV (python3-opencv)
- NumPy (python3-numpy)
- puppy_control package
- interfaces package
- common package (PID controller)

## License

MIT

## Author

PuppyPi Developer Team
