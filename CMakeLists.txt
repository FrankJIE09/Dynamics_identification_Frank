cmake_minimum_required(VERSION 3.10)
project(dynamics_parameter_estimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Eigen3 (header-only)
find_package(Eigen3 REQUIRED NO_MODULE)

# Pinocchio
find_package(pinocchio REQUIRED)

# URDF 默认路径（相对于项目根目录）
set(URDF_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/urdf/AR5-5_07R-W4C4A2.urdf")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/dynamics_parameter_estimation.cpp")
  add_executable(dynamics_parameter_estimation src/dynamics_parameter_estimation.cpp)
  target_include_directories(dynamics_parameter_estimation PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(dynamics_parameter_estimation PRIVATE Eigen3::Eigen pinocchio::pinocchio)
  target_compile_definitions(dynamics_parameter_estimation PRIVATE URDF_FILE_PATH="${URDF_FILE_PATH}" PINOCCHIO_WITHOUT_EIGEN_TENSOR_SUPPORT)
  if(MSVC)
    target_compile_definitions(dynamics_parameter_estimation PRIVATE _USE_MATH_DEFINES)
  endif()
endif()

# ---- step2_dynamics_parameter_estimation ----
add_executable(step2_dynamics_parameter_estimation
  src/step2_dynamics_parameter_estimation.cpp
)
target_include_directories(step2_dynamics_parameter_estimation PRIVATE
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(step2_dynamics_parameter_estimation PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
)
target_compile_definitions(step2_dynamics_parameter_estimation PRIVATE
  URDF_FILE_PATH="${URDF_FILE_PATH}"
  PINOCCHIO_WITHOUT_EIGEN_TENSOR_SUPPORT
)
if(MSVC)
  target_compile_definitions(step2_dynamics_parameter_estimation PRIVATE _USE_MATH_DEFINES)
endif()

# ---- step2.2_dynamics_parameter_estimation_joint ----
add_executable(step2_2_dynamics_parameter_estimation_joint
  src/step2.2_dynamics_parameter_estimation_joint.cpp
)
target_include_directories(step2_2_dynamics_parameter_estimation_joint PRIVATE
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(step2_2_dynamics_parameter_estimation_joint PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
)
target_compile_definitions(step2_2_dynamics_parameter_estimation_joint PRIVATE
  URDF_FILE_PATH="${URDF_FILE_PATH}"
  PINOCCHIO_WITHOUT_EIGEN_TENSOR_SUPPORT
)
if(MSVC)
  target_compile_definitions(step2_2_dynamics_parameter_estimation_joint PRIVATE _USE_MATH_DEFINES)
endif()

# ---- step3_torque_comparison_identified_urdf ----
add_executable(step3_torque_comparison_identified_urdf
  src/step3_torque_comparison_identified_urdf.cpp
)
target_include_directories(step3_torque_comparison_identified_urdf PRIVATE
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(step3_torque_comparison_identified_urdf PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
)
target_compile_definitions(step3_torque_comparison_identified_urdf PRIVATE
  PINOCCHIO_WITHOUT_EIGEN_TENSOR_SUPPORT
)
if(MSVC)
  target_compile_definitions(step3_torque_comparison_identified_urdf PRIVATE _USE_MATH_DEFINES)
endif()

# ---- step1 数据采集程序：依赖 Rokae SDK (rokae/robot.h, print_helper.hpp)，需在 xCoreSDK 工程内编译 ----
# 若已设置 ROKAE_SDK_DIR，可取消注释以下块并指定 include/lib
# add_executable(step1_dynamics_data_collection src/step1_dynamics_data_collection.cpp)
# add_executable(step1_dynamics_data_collection_joint src/step1_dynamics_data_collection_joint.cpp)
# target_include_directories(step1_dynamics_data_collection PRIVATE $ENV{ROKAE_SDK_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(step1_dynamics_data_collection_joint PRIVATE $ENV{ROKAE_SDK_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(step1_dynamics_data_collection PRIVATE ...)
# target_link_libraries(step1_dynamics_data_collection_joint PRIVATE ...)
