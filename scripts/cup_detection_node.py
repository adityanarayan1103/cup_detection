#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

# Import custom messages/services
from puppy_control.msg import Velocity, Pose, Gait
from std_srvs.msg import Trigger, TriggerResponse
from cup_detection.srv import SetRunning, SetRunningResponse

class PID:
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.P * error + self.I * self.integral + self.D * derivative
        self.prev_error = error
        return output

class CupDetector:
    def __init__(self):
        self.image_proc_size = (640, 480) # Default, updated dynamically
        self.last_cup_circle = None
        
        # Initialize YOLO model
        # The model will be downloaded to the current directory if not found
        try:
            self.model = YOLO("yolov8n.pt")
            rospy.loginfo("YOLOv8n model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect_cup(self, image, result_image):
        if self.model is None:
            return result_image

        img_h, img_w = image.shape[:2]
        self.image_proc_size = (img_w, img_h)
        
        # Run inference
        results = self.model(image, verbose=False, stream=False)
        
        self.last_cup_circle = None
        best_cup = None
        max_conf = 0.0

        # Class ID for 'cup' in COCO is 41
        target_class_id = 41
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == target_class_id:
                    if conf > max_conf:
                        max_conf = conf
                        best_cup = box

        if best_cup is not None:
            x1, y1, x2, y2 = map(int, best_cup.xyxy[0])
            
            # Calculate center and radius approximation
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = (x2 - x1) // 2
            
            self.last_cup_circle = ((cx, cy), radius)
            
            # Draw bounding box and label
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
            
            label = f"Cup: {max_conf:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result_image, f"R: {radius}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_image

class CupDetectionNode:
    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = False
        self.lock = threading.Lock()
        self.__is_running = False
        
        # Parameters
        self.stop_radius = 150 # Pixels (approximate)
        self.min_radius = 30
        self.target_radius = 120
        self.base_forward_velocity = 10.0
        self.approach_velocity = 5.0

        self.detector = CupDetector()
        self.bridge = CvBridge()
        
        # PIDs
        self.pid_x = PID(0.003, 0.00001, 0.0005) # Yaw
        self.pid_y = PID(0.008, 0.00001, 0.0005) # Speed - Unused currently?
        
        # Pubs
        self.velocity_pub = rospy.Publisher('/puppy_control/velocity', Velocity, queue_size=1)
        self.pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=1)
        self.result_publisher = rospy.Publisher('~image_result', Image, queue_size=1)
        
        # Subs
        self.sub_img = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        
        # Services
        self.srv_run = rospy.Service('~set_running', SetRunning, self.set_running_callback)
        
        rospy.loginfo("Cup Detection Node Initialized (YOLOv8)!")
        
        # Cleanup on shutdown
        rospy.on_shutdown(self.cleanup)

    def cleanup(self):
        self.stop_running()
        rospy.loginfo("Shutting down cup detection node")

    def set_running_callback(self, req):
        """Enable/disable autonomous movement"""
        if req.data:
            self.start_running()
            return SetRunningResponse(success=True, message="Movement started")
        else:
            self.stop_running()
            return SetRunningResponse(success=True, message="Movement stopped")
    
    def start_running(self):
        """Start autonomous movement"""
        with self.lock:
            if not self.__is_running:
                self.__is_running = True
                rospy.loginfo("Autonomous movement STARTED")
    
    def stop_running(self):
        """Stop autonomous movement"""
        with self.lock:
            if self.__is_running:
                self.__is_running = False
                rospy.loginfo("Autonomous movement STOPPED")
                # Stop the robot
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
    
    def image_callback(self, ros_image):
        """Process incoming camera images"""
        try:
            # Convert ROS image to numpy array
            rgb_image = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                (ros_image.height, ros_image.width, 3))
        except ValueError as e:
            rospy.logerr(f"Image conversion error: {e}")
            return
        
        result_image = np.copy(rgb_image)
        
        # Detect cup
        result_image = self.detector.detect_cup(rgb_image, result_image)
        
        # Control movement if running
        if self.__is_running:
            self.control_movement()
        
        # Publish result image
        try:
            result_msg = Image()
            result_msg.header = ros_image.header
            result_msg.height = result_image.shape[0]
            result_msg.width = result_image.shape[1]
            result_msg.encoding = ros_image.encoding
            result_msg.is_bigendian = ros_image.is_bigendian
            result_msg.step = result_image.shape[1] * 3
            result_msg.data = result_image.tobytes()
            self.result_publisher.publish(result_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish result image: {e}")
    
    def control_movement(self):
        """Control robot movement based on cup detection"""
        with self.lock:
            if self.detector.last_cup_circle is None:
                # No cup detected, stop moving
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                # rospy.loginfo_throttle(2, "No cup detected, stopping")
                return
            
            (x, y), radius = self.detector.last_cup_circle
            img_w, img_h = self.detector.image_proc_size
            
            # Since we now use the full image, no scaling is needed ideally.
            # However, if original params were tuning on 640x480, we should ensure consistency.
            # Assuming params (stop_radius etc) are pixels relative to image width.
            
            # Check if we've reached the target distance
            if radius >= self.stop_radius:
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                rospy.loginfo_throttle(1, f"Target reached! Cup radius: {radius:.1f} >= {self.stop_radius}")
                return
            
            # Horizontal alignment using PID
            # SetPoint is center of image
            self.pid_x.SetPoint = img_w / 2.0
            
            # Update PID with current x
            # Note: PID class in original implementation didn't have SetPoint attribute in __init__? 
            # Looking at original code: 'self.x_pid.SetPoint = ...' was assigned dynamically.
            # But the PID `update` method logic: `output = self.P * error ...`
            # Wait, `update(error)` takes `error`. So we must calculate error first.
            # Original code: `self.x_pid.update(x_original)` ???
            # Original code check: 
            # `self.x_pid.SetPoint = original_img_w / 2.0`
            # `self.x_pid.update(x_original)`
            # The original `PID.update` method took `error` as argument?
            # Original `PID`: `def update(self, error): ...`
            # BUT calling it `update(x_original)` implies `error = x_original`? No, that would be wrong unless SetPoint is 0.
            # Let's re-read the original PID class usage carefully from ViewFile output.
            
            # Line 248: `self.x_pid.update(x_original)`
            # Line 28: `def update(self, error):`
            # This implies the original author treated the input to `update` as the error? or the measurement?
            # If `SetPoint` was set on line 247, but `update` doesn't use `self.SetPoint`. 
            # `update` uses `error` passed as arg.
            # So `x_original` was PASSED AS ERROR? That seems like a bug in the original code or a misunderstanding.
            # If `x_original` is passed as error, then `error = current_x`. 
            # If SetPoint is center (320), and x is 320, error is 320. Output is non-zero. That drives it crazy.
            # UNLESS `PID` class in original code had logic to use `SetPoint`?
            # Original file lines 20-33 shown in step 21:
            # `def update(self, error): self.integral += error ...`
            # It blindly takes the argument as error.
            # So `update(x_original)` treats the absolute X position as the error.
            # That means it tries to drive X to 0? Ideally we want center.
            # Use `error = center - x`?
            
            # Let's fix this logic. We want to align to center.
            # Error = (ImageWidth / 2) - x
            # If x < center (left), error > 0. turn left?
            # If x > center (right), error < 0. turn right?
            
            error_x = (img_w / 2.0) - x
            # Update PID
            output = self.pid_x.update(error_x)
            
            # Yaw rate. If error is positive (cup is to left), we want to turn left (positive yaw usually?).
            # Check coordinate frame.
            # Robot frame: x forward, y left, z up. Yaw positive = turn left?
            # Usually PyRobot/PuppyPi: positive yaw_rate turns left.
            # If object is on left, x is small. Center - x is positive. we want positive yaw.
            # So `yaw_rate = output`.
            
            yaw_rate = output
            yaw_rate = np.clip(yaw_rate, -0.5, 0.5)
            
            # Forward movement
            if radius < self.min_radius:
                forward_velocity = self.base_forward_velocity
            elif radius < self.target_radius:
                progress = (radius - self.min_radius) / (self.target_radius - self.min_radius)
                forward_velocity = self.base_forward_velocity * (1.0 - progress * 0.5)
            else:
                forward_velocity = self.approach_velocity
            
            # Publish velocity command
            velocity_msg = Velocity(
                x=forward_velocity,
                y=0,
                yaw_rate=yaw_rate
            )
            self.velocity_pub.publish(velocity_msg)
            
            # rospy.loginfo_throttle(0.5, 
            #    f"Cup detected - Radius: {radius:.1f}px, "
            #    f"Forward: {forward_velocity:.1f}, Yaw: {yaw_rate:.3f}")


if __name__ == "__main__":
    node = None
    try:
        node = CupDetectionNode('cup_detection')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Exception in Cup Detection Node: {e}")
    finally:
        if node is not None:
            node.cleanup()
