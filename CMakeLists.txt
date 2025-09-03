cmake_minimum_required(VERSION 3.8)
project(posetrack)

# Default to Release build for better performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc dnn imgcodecs videoio)

# Custom message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeypointTrack.msg"
  DEPENDENCIES
  vision_msgs
  geometry_msgs
  std_msgs
)

# Output OpenCV info
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Set DNN paths
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")
set(ONNXRuntime_INCLUDE_DIRS "/usr/local/onnxruntime/include")
set(ONNXRuntime_LIBRARIES "/usr/local/onnxruntime/lib/libonnxruntime.so")

include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)

# Build BYTETracker library
add_library(bytetrack_lib SHARED
  src/bytetrack/BYTETracker.cpp
  src/bytetrack/kalmanFilter.cpp
  src/bytetrack/lapjv.cpp
  src/bytetrack/STrack.cpp
  src/bytetrack/utils.cpp
)

target_link_libraries(bytetrack_lib
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

target_include_directories(bytetrack_lib
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    src/bytetrack
)

# Main executable
set(SOURCES
  src/yolov11_pose_track.cc
  src/yolov11_pose.cc
  src/OSNet.cpp
)

add_executable(yolov11_pose_track_node ${SOURCES})

# Explicitly add message-generated include directory
target_include_directories(yolov11_pose_track_node PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

# 修改点1: 使用 ament_target_dependencies 并移除 PRIVATE 关键字
ament_target_dependencies(yolov11_pose_track_node
  rclcpp
  sensor_msgs
  cv_bridge
  vision_msgs
  image_transport
  geometry_msgs
  std_msgs
  rosidl_default_runtime
  rosidl_typesupport_cpp
)

# 修改点2: 使用无关键字形式的 target_link_libraries
target_link_libraries(yolov11_pose_track_node
  bytetrack_lib
  ${OpenCV_LIBS}
  ${rclcpp_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${vision_msgs_LIBRARIES}
  dnn
  pthread
  rt
  dl
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)

# 修改安装目标部分
install(TARGETS yolov11_pose_track_node bytetrack_lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install models
install(DIRECTORY 
  models/
  DESTINATION share/${PROJECT_NAME}/models
)

# Install message definitions
install(
  DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}/msg
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(rosidl_typesupport_cpp)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
