#!/usr/bin/env python3
# encoding: utf-8
# Cup Detection and Autonomous Approach Node for PuppyPi Robot

import cv2
import math
import rospy
import threading
import numpy as np
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolRequest, SetBoolResponse
from interfaces.srv import SetFloat64, SetFloat64Request, SetFloat64Response
from puppy_control.msg import Velocity, Pose
from puppy_control.srv import SetRunActionName
from common import PID


class CupDetector:
    """Detects cups in images using color and shape-based computer vision"""
    
    def __init__(self):
        self.image_proc_size = (320, 240)
        self.last_cup_circle = None
        
        # HSV color range for light-colored cups (white, cream, light blue)
        # These values target light/white objects
        self.lower_hsv = np.array([0, 0, 180])
        self.upper_hsv = np.array([180, 30, 255])
        
        # Detection parameters
        self.min_area = 500
        self.max_area = 50000
        
    def detect_cup(self, image, result_image):
        """
        Detect cup in image using color filtering and shape detection
        Returns: (x, y, radius) of detected cup, or None if not found
        """
        img_h, img_w = image.shape[:2]
        image_resized = cv2.resize(image, self.image_proc_size)
        
        # Convert to HSV for better color detection
        image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
        
        # Create mask for light-colored objects
        mask = cv2.inRange(image_hsv, self.lower_hsv, self.upper_hsv)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area]
        
        if not valid_contours:
            self.last_cup_circle = None
            return result_image
        
        # Find the best cup candidate
        cup_circle = None
        best_score = 0
        
        for contour in valid_contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate aspect ratio (cups are typically taller than wide or circular)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            # Try to fit a circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circularity = area / (math.pi * radius * radius) if radius > 0 else 0
            
            # Score based on circularity and reasonable aspect ratio
            # Cups from top view are circular; from side view have aspect ratio near 1.0-1.5
            score = 0
            if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for a cup
                score += 3
            if circularity > 0.6:  # Reasonably circular
                score += 5
            if radius > 15:  # Not too small
                score += 2
                
            if score > best_score:
                best_score = score
                cup_circle = ((cx, cy), radius)
        
        if cup_circle and best_score >= 5:  # Minimum threshold for valid cup
            self.last_cup_circle = cup_circle
            (x, y), r = cup_circle
            
            # Scale back to original image size
            x = int(x / self.image_proc_size[0] * img_w)
            y = int(y / self.image_proc_size[1] * img_h)
            r = int(r / self.image_proc_size[0] * img_w)
            
            # Draw detection on result image
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(result_image, f'Cup R:{r}', (x - 50, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            self.last_cup_circle = None
        
        return result_image


class CupDetectionNode:
    """ROS node for cup detection and autonomous approach"""
    
    # Initial standing pose
    Stand = {
        'roll': math.radians(0),
        'pitch': math.radians(0),
        'yaw': 0.000,
        'height': -10,
        'x_shift': -0.5,
        'stance_x': 0,
        'stance_y': 0,
        'run_time': 300
    }
    
    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        rospy.loginfo("Cup Detection Node Initialized")
        
        # Detection
        self.detector = CupDetector()
        self.image_sub = None
        self.result_publisher = rospy.Publisher('~image_result', Image, queue_size=1)
        
        # Movement control
        self.velocity_pub = rospy.Publisher('/puppy_control/velocity', Velocity, queue_size=1)
        self.pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=1)
        
        # PID controllers for smooth movement
        self.x_pid = PID.PID(P=0.0015, I=0.0001, D=0.00005)  # Horizontal alignment
        self.forward_pid = PID.PID(P=0.15, I=0.001, D=0.01)  # Forward movement
        
        # Pose tracking
        self.puppy_pose = self.Stand.copy()
        
        # State management
        self.__is_running = False
        self.lock = threading.RLock()
        
        # Distance thresholds (based on cup radius in pixels)
        self.min_radius = 20   # Too far, keep moving forward
        self.target_radius = 85  # Target distance, slow down
        self.stop_radius = 100   # Stop here, close enough
        
        # Movement parameters
        self.base_forward_velocity = 12.0
        self.approach_velocity = 8.0
        
        # Services
        self.enter_srv = rospy.Service('~enter', Trigger, self.enter_callback)
        self.exit_srv = rospy.Service('~exit', Trigger, self.exit_callback)
        self.set_running_srv = rospy.Service('~set_running', SetBool, self.set_running_callback)
        self.set_distance_srv = rospy.Service('~set_distance_threshold', SetFloat64, self.set_distance_callback)
        
        # Action group service for pose control
        self.action_group_srv = rospy.ServiceProxy('/puppy_control/runActionGroup', SetRunActionName)
        rospy.loginfo("Waiting for /puppy_control/runActionGroup service...")
        try:
            rospy.wait_for_service('/puppy_control/runActionGroup', timeout=10)
            rospy.loginfo("/puppy_control/runActionGroup service is available")
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to connect to /puppy_control/runActionGroup service: {e}")
            self.action_group_srv = None
        
        rospy.on_shutdown(self.shutdown_callback)
        rospy.loginfo("Cup Detection Node ready")
    
    def shutdown_callback(self):
        rospy.loginfo("Shutting down Cup Detection Node")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources and stop the robot"""
        if self.image_sub is not None:
            self.image_sub.unregister()
            rospy.loginfo("Image subscriber unregistered")
        
        self.stop_running()
        self.reset_pose()
        rospy.loginfo("Cleanup completed")
    
    def enter_callback(self, req):
        """Enter service - start detection and movement"""
        rospy.loginfo("Entering cup detection mode")
        
        if self.image_sub is not None:
            self.image_sub.unregister()
        
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.init_movement()
        self.reset_movement()
        
        return TriggerResponse(success=True, message="Entered cup detection mode")
    
    def exit_callback(self, req):
        """Exit service - stop everything"""
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
    
    def set_distance_callback(self, req):
        """Set the target stopping distance (cup radius in pixels)"""
        self.stop_radius = int(req.data)
        rospy.loginfo(f"Stop radius set to {self.stop_radius} pixels")
        return SetFloat64Response(success=True, message=f"Distance threshold set to {self.stop_radius}")
    
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
    
    def reset_pose(self):
        """Reset to standing pose"""
        with self.lock:
            self.puppy_pose = self.Stand.copy()
            pose_msg = Pose(
                stance_x=self.puppy_pose['stance_x'],
                stance_y=self.puppy_pose['stance_y'],
                x_shift=self.puppy_pose['x_shift'],
                height=self.puppy_pose['height'],
                roll=self.puppy_pose['roll'],
                pitch=self.puppy_pose['pitch'],
                yaw=self.puppy_pose['yaw'],
                run_time=self.puppy_pose['run_time']
            )
            self.pose_pub.publish(pose_msg)
            rospy.loginfo("Pose reset to standing position")
    
    def reset_movement(self):
        """Reset PID controllers and movement state"""
        with self.lock:
            self.x_pid.clear()
            self.forward_pid.clear()
            self.reset_pose()
            rospy.loginfo("Movement controllers reset")
    
    def init_movement(self):
        """Initialize robot to standing position"""
        if self.action_group_srv:
            try:
                rospy.loginfo("Initializing to Stand pose")
                self.action_group_srv('Stand.d6ac', True)
                rospy.loginfo("Stand pose initialized")
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to initialize pose: {e}")
        else:
            rospy.logwarn("Action group service not available")
    
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
        """Control robot movement based on cup detection"""
        with self.lock:
            if self.detector.last_cup_circle is None:
                # No cup detected, stop moving
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                rospy.loginfo_throttle(2, "No cup detected, stopping")
                return
            
            (x, y), radius = self.detector.last_cup_circle
            img_w, img_h = self.detector.image_proc_size
            
            # Scale to original image dimensions
            original_img_w = rospy.get_param('~original_image_width', 640)
            original_img_h = rospy.get_param('~original_image_height', 480)
            
            x_original = x / img_w * original_img_w
            radius_original = radius / img_w * original_img_w
            
            # Check if we've reached the target distance
            if radius_original >= self.stop_radius:
                self.velocity_pub.publish(Velocity(x=0, y=0, yaw_rate=0))
                rospy.loginfo_throttle(1, f"Target reached! Cup radius: {radius_original:.1f} >= {self.stop_radius}")
                # Optionally stop autonomous mode
                # self.stop_running()
                return
            
            # Horizontal alignment using PID
            self.x_pid.SetPoint = original_img_w / 2.0
            self.x_pid.update(x_original)
            yaw_rate = -self.x_pid.output  # Negative for correct direction
            yaw_rate = np.clip(yaw_rate, -0.3, 0.3)
            
            # Forward movement based on distance
            if radius_original < self.min_radius:
                # Far away, move at base speed
                forward_velocity = self.base_forward_velocity
            elif radius_original < self.target_radius:
                # Getting closer, slow down proportionally
                progress = (radius_original - self.min_radius) / (self.target_radius - self.min_radius)
                forward_velocity = self.base_forward_velocity * (1.0 - progress * 0.5)
            else:
                # Very close, move slowly
                forward_velocity = self.approach_velocity
            
            # Publish velocity command
            velocity_msg = Velocity(
                x=forward_velocity,
                y=0,
                yaw_rate=yaw_rate
            )
            self.velocity_pub.publish(velocity_msg)
            
            rospy.loginfo_throttle(0.5, 
                f"Cup detected - Radius: {radius_original:.1f}px, "
                f"Forward: {forward_velocity:.1f}, Yaw: {yaw_rate:.3f}")


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
