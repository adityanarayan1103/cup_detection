#!/usr/bin/env python3
# encoding: utf-8
# Enhanced Cup Detection with YOLOv8 and object_tracking features
# Features: RGB LED sync, heartbeat mechanism, enter/exit services, action groups

import cv2
import math
import rospy
import threading
import numpy as np
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from puppy_control.msg import Velocity, Pose
from puppy_control.srv import SetRunActionName
from ros_robot_controller.msg import RGBState, RGBsState
from ultralytics import YOLO


class PID:
    """PID Controller with clear method"""
    def __init__(self, P=0.0, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.SetPoint = 0.0
        self.output = 0.0
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_value):
        error = self.SetPoint - current_value
        self.integral += error
        derivative = error - self.prev_error
        self.output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return self.output

    def clear(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.output = 0.0


class CupDetector:
    """YOLO-based cup detector"""
    def __init__(self):
        self.image_proc_size = (640, 480)
        self.last_cup_circle = None
        self.model = None
        self.target_class_id = 41  # COCO class ID for 'cup'
        
        try:
            self.model = YOLO("yolov8n.pt")
            rospy.loginfo("YOLOv8n model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")

    def detect_cup(self, image, result_image):
        if self.model is None:
            return result_image

        img_h, img_w = image.shape[:2]
        self.image_proc_size = (img_w, img_h)
        
        results = self.model(image, verbose=False, stream=False)
        
        self.last_cup_circle = None
        best_cup = None
        max_conf = 0.0

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == self.target_class_id and conf > max_conf:
                    max_conf = conf
                    best_cup = box

        if best_cup is not None:
            x1, y1, x2, y2 = map(int, best_cup.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = (x2 - x1) // 2
            
            self.last_cup_circle = ((cx, cy), radius)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result_image, f"Cup: {max_conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_image


class CupDetectionNode:
    """Enhanced Cup Detection Node with object_tracking features"""
    
    # Default standing pose
    Stand = {
        'roll': math.radians(0),
        'pitch': math.radians(0),
        'yaw': 0.0,
        'height': -10,
        'x_shift': -0.5,
        'stance_x': 0,
        'stance_y': 0,
        'run_time': 300
    }

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        
        self.detector = CupDetector()
        self.image_sub = None
        self.result_publisher = rospy.Publisher('~image_result', Image, queue_size=1)
        
        # Movement publishers
        self.velocity_pub = rospy.Publisher('/puppy_control/velocity', Velocity, queue_size=1)
        self.pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=1)
        
        # PID controllers
        self.x_pid = PID(P=0.002, I=0.0001, D=0.0005)  # Yaw control
        self.z_pid = PID(P=0.003, I=0.001, D=0.0)       # Pitch control
        
        self.PuppyPose = self.Stand.copy()
        self.__isRunning = False
        self.lock = threading.RLock()
        
        # Movement parameters
        self.stop_radius = 150
        self.min_radius = 30
        self.base_forward_velocity = 8.0
        self.approach_velocity = 4.0
        
        # RGB LED control
        self.rgb_pub = rospy.Publisher('/ros_robot_controller/set_rgb', RGBsState, queue_size=1)
        
        # Heartbeat mechanism
        self.heartbeat_timer = None
        self.heartbeat_timeout = 5.0  # seconds
        
        # Action group service
        self.runActionGroup_srv = None
        try:
            rospy.wait_for_service('/puppy_control/runActionGroup', timeout=5)
            self.runActionGroup_srv = rospy.ServiceProxy('/puppy_control/runActionGroup', SetRunActionName)
            rospy.loginfo("/puppy_control/runActionGroup service available")
        except rospy.ROSException:
            rospy.logwarn("/puppy_control/runActionGroup service not available")
        
        # Services
        self.enter_srv = rospy.Service('~enter', Trigger, self.enter_callback)
        self.exit_srv = rospy.Service('~exit', Trigger, self.exit_callback)
        self.set_running_srv = rospy.Service('~set_running', SetBool, self.set_running_callback)
        self.heartbeat_srv = rospy.Service('~heartbeat', SetBool, self.heartbeat_callback)
        
        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("Cup Detection Node Initialized (Enhanced YOLOv8)!")

    # --- Service Callbacks ---
    
    def enter_callback(self, req):
        """Enter cup detection mode"""
        rospy.loginfo("Entering cup detection mode")
        if self.image_sub is not None:
            self.image_sub.unregister()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.init_movement()
        self.reset_movement()
        self.set_rgb_leds(0, 255, 0)  # Green = ready
        return TriggerResponse(success=True, message="Entered cup detection mode")

    def exit_callback(self, req):
        """Exit cup detection mode"""
        rospy.loginfo("Exiting cup detection mode")
        self.cleanup()
        return TriggerResponse(success=True, message="Exited cup detection mode")

    def set_running_callback(self, req):
        """Enable/disable autonomous movement"""
        if req.data:
            self.start_running()
            return SetBoolResponse(success=True, message="Movement started")
        else:
            self.stop_running()
            return SetBoolResponse(success=True, message="Movement stopped")

    def heartbeat_callback(self, req):
        """Heartbeat to keep node alive"""
        with self.lock:
            if self.heartbeat_timer:
                self.heartbeat_timer.cancel()
            
            if req.data:
                self.heartbeat_timer = threading.Timer(self.heartbeat_timeout, self.heartbeat_timeout_handler)
                self.heartbeat_timer.start()
        
        return SetBoolResponse(success=req.data, message="Heartbeat received" if req.data else "Heartbeat stopped")

    def heartbeat_timeout_handler(self):
        """Called when heartbeat times out"""
        rospy.logwarn("Heartbeat timeout! Calling exit service.")
        try:
            exit_proxy = rospy.ServiceProxy('~exit', Trigger)
            exit_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call exit service: {e}")

    # --- Movement Control ---
    
    def start_running(self):
        with self.lock:
            if not self.__isRunning:
                self.__isRunning = True
                self.set_rgb_leds(0, 0, 255)  # Blue = tracking
                rospy.loginfo("Movement control started")

    def stop_running(self):
        with self.lock:
            if self.__isRunning:
                self.__isRunning = False
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                self.reset_pose()
                self.set_rgb_leds(0, 255, 0)  # Green = standby
                rospy.loginfo("Movement control stopped")

    def reset_pose(self):
        with self.lock:
            self.PuppyPose = self.Stand.copy()
            pose_msg = Pose(
                stance_x=self.PuppyPose['stance_x'],
                stance_y=self.PuppyPose['stance_y'],
                x_shift=self.PuppyPose['x_shift'],
                height=self.PuppyPose['height'],
                roll=self.PuppyPose['roll'],
                pitch=self.PuppyPose['pitch'],
                yaw=self.PuppyPose['yaw'],
                run_time=self.PuppyPose['run_time']
            )
            self.pose_pub.publish(pose_msg)

    def reset_movement(self):
        with self.lock:
            self.x_pid.clear()
            self.z_pid.clear()
            self.reset_pose()

    def init_movement(self):
        if self.runActionGroup_srv:
            try:
                self.runActionGroup_srv('Stand.d6ac', True)
                rospy.loginfo("Initialized to Stand pose")
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to initialize pose: {e}")

    # --- RGB LED Control ---
    
    def set_rgb_leds(self, r, g, b):
        rgb_msg = RGBsState()
        rgb_msg.data = [
            RGBState(id=1, r=r, g=g, b=b),
            RGBState(id=2, r=r, g=g, b=b)
        ]
        self.rgb_pub.publish(rgb_msg)

    # --- Image Processing ---
    
    def image_callback(self, ros_image):
        try:
            rgb_image = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                (ros_image.height, ros_image.width, 3))
        except ValueError as e:
            rospy.logerr(f"Image conversion error: {e}")
            return

        result_image = np.copy(rgb_image)
        result_image = self.detector.detect_cup(rgb_image, result_image)

        if self.__isRunning:
            self.control_movement()

        # Publish result
        result_msg = Image()
        result_msg.header = ros_image.header
        result_msg.height = result_image.shape[0]
        result_msg.width = result_image.shape[1]
        result_msg.encoding = ros_image.encoding
        result_msg.is_bigendian = ros_image.is_bigendian
        result_msg.step = result_image.shape[1] * 3
        result_msg.data = result_image.tobytes()
        self.result_publisher.publish(result_msg)

    def control_movement(self):
        with self.lock:
            if self.detector.last_cup_circle is None:
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                self.set_rgb_leds(255, 100, 0)  # Orange = searching
                return

            (x, y), radius = self.detector.last_cup_circle
            img_w, img_h = self.detector.image_proc_size

            # Set LED to blue when tracking
            self.set_rgb_leds(0, 0, 255)

            # Check if target reached
            if radius >= self.stop_radius:
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                self.set_rgb_leds(0, 255, 0)  # Green = reached
                rospy.loginfo_throttle(2, f"Target reached! Radius: {radius}")
                return

            # Horizontal alignment (yaw)
            self.x_pid.SetPoint = img_w / 2.0
            self.x_pid.update(x)
            yaw_rate = -np.clip(self.x_pid.output, -0.5, 0.5)

            # Forward velocity based on distance
            if radius < self.min_radius:
                forward_vel = self.base_forward_velocity
            else:
                progress = (radius - self.min_radius) / (self.stop_radius - self.min_radius)
                forward_vel = self.base_forward_velocity * (1.0 - progress * 0.7)
                forward_vel = max(forward_vel, self.approach_velocity)

            self.velocity_pub.publish(Velocity(x=forward_vel, y=0, yaw_rate=yaw_rate))

    # --- Cleanup ---
    
    def cleanup(self):
        with self.lock:
            if self.image_sub is not None:
                self.image_sub.unregister()
                self.image_sub = None
            
            if self.heartbeat_timer:
                self.heartbeat_timer.cancel()
                self.heartbeat_timer = None
            
            self.__isRunning = False
            self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
            
            if self.runActionGroup_srv:
                try:
                    self.runActionGroup_srv('Stand.d6ac', True)
                except:
                    pass
            
            self.set_rgb_leds(0, 0, 0)
            rospy.loginfo("Cleanup completed")


if __name__ == "__main__":
    try:
        node = CupDetectionNode('cup_detection')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Exception in Cup Detection Node: {e}")
