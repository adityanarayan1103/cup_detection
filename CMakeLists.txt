cmake_minimum_required(VERSION 3.0.2)
project(cup_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_srvs
  puppy_control
  interfaces
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    sensor_msgs 
    std_srvs 
    puppy_control 
    interfaces
    std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/cup_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
