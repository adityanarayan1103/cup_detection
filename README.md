# Cup Detection PoC - YOLOv8 Object Detection for PuppyPi

> **Proof of Concept**: Autonomous cup detection and tracking for the PuppyPi quadruped robot using YOLOv8 deep learning.

## Overview

This ROS package enables the PuppyPi robot to:
- **Detect cups** in real-time using YOLOv8 object detection
- **Track and approach** detected cups autonomously
- **Provide visual feedback** via RGB LEDs
- **Integrate** with the robot's movement and pose control systems

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   USB Camera    │───▶│  cup_detection   │───▶│  Robot Control  │
│  /usb_cam/      │    │     (YOLOv8)     │    │  /puppy_control │
│   image_raw     │    │                  │    │    /velocity    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   RGB LEDs   │
                       │  (Feedback)  │
                       └──────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **YOLOv8 Detection** | Uses YOLOv8n model for real-time cup detection (COCO class 41) |
| **PID Control** | Smooth movement using dual PID controllers for yaw and approach |
| **RGB LED Sync** | Visual feedback: 🟢 Green (ready), 🔵 Blue (tracking), 🟠 Orange (searching) |
| **Heartbeat** | Safety mechanism - auto-stops if no heartbeat received within 5s |
| **Action Groups** | Proper pose initialization via Stand.d6ac action |

## Installation

### Prerequisites
```bash
# On robot (Raspberry Pi)
pip3 install ultralytics==8.0.196 --no-deps
pip3 install opencv-python torchvision pandas matplotlib seaborn thop py-cpuinfo
```

### Build
```bash
cd ~/puppypi/src
git clone https://github.com/adityanarayan1103/cup_detection.git
cd ~/puppypi
catkin_make --pkg cup_detection
source devel/setup.bash
```

## Usage

### Launch
```bash
# PyTorch ARM fix required
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Option 1: With existing camera
rosrun cup_detection cup_detection_node.py

# Option 2: Standalone with camera
roslaunch cup_detection cup_detection.launch
```

### Services

| Service | Type | Description |
|---------|------|-------------|
| `~enter` | Trigger | Enter detection mode, subscribe to camera |
| `~exit` | Trigger | Exit detection mode, cleanup |
| `~set_running` | SetBool | Start/stop autonomous movement |
| `~heartbeat` | SetBool | Keep-alive signal (5s timeout) |

### Example Workflow
```bash
# 1. Enter detection mode
rosservice call /cup_detection/enter

# 2. Start autonomous tracking
rosservice call /cup_detection/set_running "data: true"

# 3. View detection output
rqt_image_view /cup_detection/image_result

# 4. Stop and exit
rosservice call /cup_detection/set_running "data: false"
rosservice call /cup_detection/exit
```

## Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/usb_cam/image_raw` | sensor_msgs/Image | Subscribe | Camera input |
| `~image_result` | sensor_msgs/Image | Publish | Annotated detection output |
| `/puppy_control/velocity` | Velocity | Publish | Movement commands |
| `/puppy_control/pose` | Pose | Publish | Pose adjustments |
| `/ros_robot_controller/set_rgb` | RGBsState | Publish | LED control |

## Configuration

### Movement Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_radius` | 150 px | Distance threshold to stop |
| `min_radius` | 30 px | Minimum detection radius |
| `base_forward_velocity` | 8.0 | Max forward speed |
| `approach_velocity` | 4.0 | Slow approach speed |

### PID Tuning
```python
x_pid = PID(P=0.002, I=0.0001, D=0.0005)  # Yaw control
z_pid = PID(P=0.003, I=0.001, D=0.0)       # Pitch control
```

## LED States

| Color | State | Meaning |
|-------|-------|---------|
| 🟢 Green | Ready | Node active, waiting or target reached |
| 🔵 Blue | Tracking | Cup detected and being tracked |
| 🟠 Orange | Searching | Running but no cup detected |
| ⚫ Off | Inactive | Node not running or exited |

## Known Issues

1. **Camera Device**: Camera is on `/dev/video1`, requires symlink:
   ```bash
   sudo ln -sf /dev/video1 /dev/video0
   ```

2. **PyTorch TLS Error**: On ARM devices, set `LD_PRELOAD`:
   ```bash
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
   ```

3. **YOLO Model Download**: First run downloads ~6MB model (requires internet)

## Future Improvements

- [ ] Support for multiple object classes (not just cups)
- [ ] Configurable target class via ROS parameter
- [ ] Distance estimation using depth camera
- [ ] Integration with navigation stack
- [ ] Model switching (YOLOv8n/s/m)

## License

MIT License - See LICENSE file

## Author

Aditya Narayan Verma
