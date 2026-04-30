cmake_minimum_required(VERSION 3.8)
project(pcl_filters)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # comment these back out once copyright/license is added
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(${PCL_INCLUDE_DIRS})

add_executable(sor_filter src/sor_filter.cpp)
ament_target_dependencies(
  sor_filter
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)

target_link_libraries(sor_filter ${PCL_LIBRARIES})

install(
  TARGETS sor_filter
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
