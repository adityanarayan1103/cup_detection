cmake_minimum_required(VERSION 3.0.2)
project(cup_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  puppy_control
  message_generation
)

add_service_files(
  FILES
  SetRunning.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  puppy_control
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge puppy_control
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/cup_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
