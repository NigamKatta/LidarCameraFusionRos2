cmake_minimum_required(VERSION 3.8)
project(sensorfusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
## Add dependencies
set(dependencies
  rclcpp
  rclpy
  sensor_msgs
  std_msgs
  OpenCV
  PCL
  Boost
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

## Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

ament_package()

add_executable(lidar_node src/interpolated_lidar.cpp)
ament_target_dependencies(lidar_node ${dependencies})
target_link_libraries(lidar_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(camera_node src/lidar_camera_node.cpp)
ament_target_dependencies(camera_node ${dependencies})
target_link_libraries(camera_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

