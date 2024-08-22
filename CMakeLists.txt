cmake_minimum_required(VERSION 3.5)
project(cudnn_tutorial)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(CUDA REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(
  ${CUDA_INCLUDE_DIRS} 
  include
)

link_directories(/usr/local/cuda/lib64)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(cudnnActivation src/cudnnActivation.cpp)
target_link_libraries(cudnnActivation ${CUDA_LIBRARIES} cublas cudnn)
ament_target_dependencies(cudnnActivation rclcpp)

add_executable(cudnnSimpleDnn src/cudnnSimpleDnn.cpp)
target_link_libraries(cudnnSimpleDnn ${CUDA_LIBRARIES} cublas cudnn)
ament_target_dependencies(cudnnSimpleDnn rclcpp)

add_executable(cudnnSimpleCnn src/cudnnSimpleCnn.cpp)
target_link_libraries(cudnnSimpleCnn ${CUDA_LIBRARIES} cublas cudnn)
ament_target_dependencies(cudnnSimpleCnn rclcpp)

add_executable(cudnnSaveLoadModel src/cudnnSaveLoadModel.cpp)
target_link_libraries(cudnnSaveLoadModel ${CUDA_LIBRARIES} cublas cudnn)
ament_target_dependencies(cudnnSaveLoadModel rclcpp)

add_executable(cudnnSimpleRnn src/cudnnSimpleRnn.cpp)
target_link_libraries(cudnnSimpleRnn ${CUDA_LIBRARIES} cublas cudnn)
ament_target_dependencies(cudnnSimpleRnn rclcpp)

install(TARGETS
  cudnnActivation
  cudnnSimpleDnn
  cudnnSimpleCnn
  cudnnSaveLoadModel
  cudnnSimpleRnn
  DESTINATION lib/${PROJECT_NAME})


ament_package()
