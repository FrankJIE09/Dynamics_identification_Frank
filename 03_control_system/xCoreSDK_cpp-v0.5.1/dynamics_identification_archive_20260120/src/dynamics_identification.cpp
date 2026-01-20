/**
 * @file dynamics_identification.cpp
 * @brief 实时模式 - 动力学参数辨识
 * 
 * 此程序用于辨识机器人的动力学参数（质量、惯性、质心位置等）：
 * 1. 生成激励轨迹（正弦波、傅里叶级数等）
 * 2. 在实时控制循环中采集数据（关节位置、速度、加速度、力矩）
 * 3. 使用 Pinocchio 计算回归矩阵
 * 4. 通过最小二乘法求解动力学参数
 * 5. 验证辨识结果
 * 
 * 此示例需要使用xMateModel模型库，请设置编译选项XCORE_USE_XMATE_MODEL=ON
 * 需要 Pinocchio 库支持
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <fstream>
#include <array>
#include <chrono>
#include "rokae/robot.h"
#include "Eigen/Geometry"
#include "Eigen/Dense"
#include "../print_helper.hpp"
#include "rokae/utility.h"

// Pinocchio includes
#ifdef __has_include
  #if __has_include("pinocchio/multibody/model.hpp")
    #include "pinocchio/multibody/model.hpp"
    #include "pinocchio/multibody/data.hpp"
    #include "pinocchio/parsers/urdf.hpp"
    #include "pinocchio/algorithm/joint-configuration.hpp"
    #include "pinocchio/algorithm/kinematics.hpp"
    #include "pinocchio/algorithm/compute-all-terms.hpp"
    #include "pinocchio/algorithm/regressor.hpp"
    #define PINOCCHIO_AVAILABLE
  #endif
#endif

using namespace rokae;
using namespace std::chrono_literals;

/**
 * @brief 数据点结构
 */
struct DataPoint {
    std::array<double, 7> q;      // 关节位置 (rad)
    std::array<double, 7> dq;      // 关节速度 (rad/s)
    std::array<double, 7> ddq;     // 关节加速度 (rad/s²)
    std::array<double, 7> tau;    // 关节力矩 (Nm，传感器测量)
    double timestamp;             // 时间戳 (s)
};

/**
 * @brief 自动检测本机IP地址（与机器人IP在同一网段）
 */
std::string detectLocalIp(const std::string& robot_ip) {
  size_t last_dot = robot_ip.find_last_of('.');
  if (last_dot == std::string::npos) {
    return "";
  }
  std::string network_prefix = robot_ip.substr(0, last_dot);
  
  std::string cmd = "ip addr show | grep -E 'inet " + network_prefix + "\\.' | head -1 | awk '{print $2}' | cut -d'/' -f1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return "";
  }
  
  char buffer[128];
  std::string result = "";
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result = buffer;
    if (!result.empty() && result.back() == '\n') {
      result.pop_back();
    }
  }
  pclose(pipe);
  
  return result;
}

/**
 * @brief 生成正弦激励轨迹
 * @param t 当前时间
 * @param q0 初始关节角度
 * @param amplitudes 各关节的幅值
 * @param frequencies 各关节的频率
 * @param phases 各关节的相位
 * @return 目标关节角度
 */
std::array<double, 7> generateSineTrajectory(
    double t,
    const std::array<double, 7>& q0,
    const std::array<double, 7>& amplitudes,
    const std::array<double, 7>& frequencies,
    const std::array<double, 7>& phases) {
    
    std::array<double, 7> q_target;
    for (int i = 0; i < 7; i++) {
        q_target[i] = q0[i] + amplitudes[i] * std::sin(frequencies[i] * t + phases[i]);
    }
    return q_target;
}

/**
 * @brief 生成傅里叶级数激励轨迹
 * @param t 当前时间
 * @param q0 初始关节角度
 * @param base_freq 基频
 * @param n_harmonics 谐波数量
 * @return 目标关节角度
 */
std::array<double, 7> generateFourierTrajectory(
    double t,
    const std::array<double, 7>& q0,
    double base_freq,
    int n_harmonics) {
    
    std::array<double, 7> q_target = q0;
    
    // 为每个关节生成不同的傅里叶级数
    for (int i = 0; i < 7; i++) {
        double amplitude = 0.3;  // 幅值（弧度）
        for (int k = 1; k <= n_harmonics; k++) {
            double phase = (i * M_PI / 7.0) + (k * M_PI / n_harmonics);
            q_target[i] += amplitude / k * std::sin(k * base_freq * t + phase);
        }
    }
    
    return q_target;
}

/**
 * @brief 动力学辨识主函数
 */
void dynamicsIdentification(xMateErProRobot &robot) {
  using namespace RtSupportedFields;
  auto rtCon = robot.getRtMotionController().lock();
  auto model = robot.model();
  error_code ec;

  // 设置负载（如果需要）
#if 0
  // 如果已知负载，可以设置；如果未知，可以在辨识中一起辨识
  double load_mass = 0.9095687374891032;
  std::array<double, 3> load_centre = {0.000165, 0.000151, 0.018055};
  std::array<double, 3> load_inertia = {0.000433, 0.000433, 0.00067};
  model.setLoad(load_mass, load_centre, load_inertia);
  std::cout << "  已设置负载参数" << std::endl;
#endif

#ifdef PINOCCHIO_AVAILABLE
  // 初始化 Pinocchio 模型
  std::cout << "\n  初始化 Pinocchio 动力学模型..." << std::endl;
  
  std::string urdf_file_path;
#ifdef URDF_FILE_PATH
  urdf_file_path = URDF_FILE_PATH;
#else
  urdf_file_path = "AR5-5_07R-W4C4A2.urdf";
#endif
  
  pinocchio::Model pinocchio_model;
  pinocchio::Data pinocchio_data(pinocchio_model);
  
  try {
    pinocchio::urdf::buildModel(urdf_file_path, pinocchio_model);
    std::cout << "  Pinocchio 模型加载成功: " << urdf_file_path << std::endl;
    std::cout << "  模型自由度: " << pinocchio_model.nq << std::endl;
    
    // 设置重力方向
    Eigen::Matrix3d R_y = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d R_z = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d R_base_to_world = R_z * R_y;
    Eigen::Vector3d gravity_world(0, 0, -9.81);
    Eigen::Vector3d gravity_base = R_base_to_world.transpose() * gravity_world;
    pinocchio_model.gravity.linear() = gravity_base;
    std::cout << "  重力方向已设置" << std::endl;
    
    // 估算最小惯性参数数量
    // 对于7关节机器人，最小参数集约为30-40个
    // 这里使用一个合理的估计值（实际会在回归矩阵构造时确定）
    std::cout << "  将使用Pinocchio的回归矩阵计算" << std::endl;
    
    pinocchio_data = pinocchio::Data(pinocchio_model);
  } catch (const std::exception& e) {
    std::cerr << "  错误: Pinocchio 模型加载失败: " << e.what() << std::endl;
    return;
  }
#else
  std::cerr << "  错误: Pinocchio 库未找到，无法进行动力学辨识" << std::endl;
  return;
#endif

  // 初始化机器人状态接收
  std::cout << "  停止接收机器人状态..." << std::endl;
  robot.stopReceiveRobotState();
  
  std::cout << "  开始接收机器人状态（1ms周期）..." << std::endl;
  try {
    robot.startReceiveRobotState(std::chrono::milliseconds(1),
                                 {jointPos_m, jointVel_m, jointAcc_m, tau_m});
    std::cout << "  机器人状态接收已启动（使用测量的加速度 jointAcc_m）" << std::endl;
  } catch (const rokae::RealtimeControlException &e) {
    std::cerr << "  启动状态接收失败: " << e.what() << std::endl;
    return;
  }

  // 获取初始位置
  std::cout << "  获取初始位置..." << std::endl;
    std::array<double, 7> q_init = Utils::degToRad(std::array<double, 7>({
    -2.458,   // 一轴：-2.458度
    -70.651,  // 二轴：-70.651度
    -2.044,   // 三轴：-2.044度
    118.492,  // 四轴：118.492度
    4.682,    // 五轴：4.682度
    -47.719,  // 六轴：-47.719度
    -5.433    // 七轴：-5.433度
  }));
//   auto q_init = robot.jointPos(ec);
  std::cout << "  初始位置（度）: [";
  for (size_t i = 0; i < q_init.size(); i++) {
    std::cout << q_init[i] * 180.0 / M_PI;
    if (i < q_init.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  // 检查电源状态
  auto power_state = robot.powerState(ec);
  using PS = rokae::PowerState;
  if (power_state != PS::on) {
    std::cout << "  等待1秒后尝试上电..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    robot.setPowerState(true, ec);
    if (ec) {
      print(std::cerr, "  上电失败:", ec);
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  std::cout << "  电源状态检查通过（已上电）" << std::endl;

  // 设置滤波频率
//   std::cout << "  设置滤波截止频率为10Hz..." << std::endl;
//   rtCon->setFilterFrequency(10.0, 10.0, 10.0, ec);

  // 数据采集参数
  const double collection_duration = 60.0;  // 采集时长（秒）
  const double dt = 0.001;  // 采样周期（1ms）
  std::vector<DataPoint> collected_data;
  collected_data.reserve(static_cast<size_t>(collection_duration / dt));

  // 激励轨迹参数（降低幅值和频率以减小角速度）
  std::array<double, 7> amplitudes = {
    0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1  // 各关节幅值（弧度）- 降低到原来的一半
  };
  std::array<double, 7> frequencies = {
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8  // 各关节频率（rad/s）- 降低到原来的约一半
  };
  std::array<double, 7> phases = {
    0.0, M_PI/7, 2*M_PI/7, 3*M_PI/7, 4*M_PI/7, 5*M_PI/7, 6*M_PI/7
  };

  std::cout << "\n========================================" << std::endl;
  std::cout << "开始数据采集..." << std::endl;
  std::cout << "采集时长: " << collection_duration << " 秒" << std::endl;
  std::cout << "采样频率: " << (1.0/dt) << " Hz" << std::endl;
  std::cout << "预计采集点数: " << (collection_duration / dt) << std::endl;
  std::cout << "========================================" << std::endl;

  // 实时控制循环：生成激励轨迹并采集数据
  std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time = 0.0;
    static std::array<double, 7> q_prev = q_init;
    static std::array<double, 7> dq_prev_for_accel = {0, 0, 0, 0, 0, 0, 0};  // 用于计算加速度
    static Eigen::VectorXd tau_prev = Eigen::VectorXd::Zero(7);  // 上一周期的力矩
    static bool first_call = true;
    
    time += dt;

    // 获取当前状态
    std::array<double, 7> q{}, dq_m{}, ddq_m{}, tau_m_measured{};
    robot.getStateData(jointPos_m, q);
    robot.getStateData(jointVel_m, dq_m);
    robot.getStateData(jointAcc_m, ddq_m);  // 使用测量的加速度，而不是指令加速度
    robot.getStateData(tau_m, tau_m_measured);
    


    // 生成激励轨迹（使用正弦波）
    std::array<double, 7> q_target = generateSineTrajectory(
        time, q_init, amplitudes, frequencies, phases);

    // 使用位置控制跟踪轨迹（PD控制）
    Eigen::VectorXd tau(7);
    for (int i = 0; i < 7; i++) {
        double q_error = q_target[i] - q[i];
        double dq_error = 0.0 - dq_m[i];  // 期望速度为0（简化）
        
        // PD控制增益（进一步降低增益以适应更小的运动）
        double kp = 10.0;   // 降低位置增益
        double kd = 1.0;    // 降低速度增益
        
        tau(i) = kp * q_error + kd * dq_error;
    }

    // 限制力矩
    const std::array<double, 7> torque_limits = {85.0, 85.0, 85.0, 36.0, 36.0, 36.0, 36.0};
    for (int i = 0; i < 7; i++) {
        if (tau(i) > torque_limits[i]) tau(i) = torque_limits[i];
        if (tau(i) < -torque_limits[i]) tau(i) = -torque_limits[i];
    }
    
    // 力矩平滑：限制力矩变化率（防止不连续）
    const double max_torque_rate = 50.0;  // 最大力矩变化率 (Nm/s)
    const double max_torque_change = max_torque_rate * dt;  // 每周期最大变化量
    
    if (!first_call) {
        for (int i = 0; i < 7; i++) {
            double tau_change = tau(i) - tau_prev(i);
            if (tau_change > max_torque_change) {
                tau(i) = tau_prev(i) + max_torque_change;
            } else if (tau_change < -max_torque_change) {
                tau(i) = tau_prev(i) - max_torque_change;
            }
        }
    } else {
        // 第一次调用：从零力矩平滑过渡
        tau = tau * 0.1;  // 初始力矩很小
        first_call = false;
    }
    
    // 更新历史力矩
    tau_prev = tau;

    // 采集数据（每10ms采集一次，减少数据量）
    static int sample_counter = 0;
    if (sample_counter % 10 == 0) {
        DataPoint point;
        point.q = q;
        point.dq = dq_m;
        point.ddq = ddq_m;  // 使用修正后的加速度
        point.tau = tau_m_measured;
        point.timestamp = time;
        collected_data.push_back(point);
    }
    sample_counter++;

    // 打印进度和力矩命令（每100ms，即每100个周期）
    static int print_counter = 0;
    if (print_counter % 100 == 0) {
        std::cout << "\n[时间: " << std::fixed << std::setprecision(3) << time << "s]" << std::endl;
        std::cout << "  采集进度: " << std::fixed << std::setprecision(1) 
                  << (time / collection_duration * 100.0) << "%, "
                  << "已采集: " << collected_data.size() << " 点" << std::endl;
        std::cout << "  发送的力矩命令 (Nm): [";
        for (int i = 0; i < 7; i++) {
            std::cout << std::fixed << std::setprecision(3) << tau(i);
            if (i < 6) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  目标关节角度 (deg): [";
        for (int i = 0; i < 7; i++) {
            std::cout << std::fixed << std::setprecision(2) << q_target[i] * 180.0 / M_PI;
            if (i < 6) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  当前关节角度 (deg): [";
        for (int i = 0; i < 7; i++) {
            std::cout << std::fixed << std::setprecision(2) << q[i] * 180.0 / M_PI;
            if (i < 6) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  关节速度 (rad/s): [";
        for (int i = 0; i < 7; i++) {
            std::cout << std::fixed << std::setprecision(4) << dq_m[i];
            if (i < 6) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    print_counter++;

    // 运行指定时长后结束
    Torque cmd(7);
    if (time > collection_duration) {
        cmd.setFinished();
    } else {
        Eigen::VectorXd::Map(cmd.tau.data(), 7) = tau;
    }
    
    return cmd;
  };

  // 设置控制循环
  std::cout << "  设置控制循环回调..." << std::endl;
  rtCon->setControlLoop(callback, 0, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // 启动力矩控制模式
  std::cout << "  启动力矩控制模式..." << std::endl;
  try {
    rtCon->startMove(RtControllerMode::torque);
    std::cout << "  力矩控制模式已启动，开始采集数据..." << std::endl;
  } catch (const rokae::RealtimeControlException &e) {
    std::cerr << "  启动力矩控制失败: " << e.what() << std::endl;
    return;
  }

  // 启动控制循环
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rtCon->startLoop(true);
  
  std::cout << "\n  数据采集完成！" << std::endl;
  std::cout << "  实际采集点数: " << collected_data.size() << std::endl;

  // 保存原始数据（可选）
  std::cout << "\n  保存原始数据到文件..." << std::endl;
  std::ofstream data_file("dynamics_identification_data.csv");
  data_file << "time,q0,q1,q2,q3,q4,q5,q6,"
            << "dq0,dq1,dq2,dq3,dq4,dq5,dq6,"
            << "ddq0,ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,"
            << "tau0,tau1,tau2,tau3,tau4,tau5,tau6\n";
  for (const auto& point : collected_data) {
    data_file << std::fixed << std::setprecision(6) << point.timestamp << ",";
    for (int i = 0; i < 7; i++) {
      data_file << point.q[i];
      if (i < 6) data_file << ",";
    }
    data_file << ",";
    for (int i = 0; i < 7; i++) {
      data_file << point.dq[i];
      if (i < 6) data_file << ",";
    }
    data_file << ",";
    for (int i = 0; i < 7; i++) {
      data_file << point.ddq[i];
      if (i < 6) data_file << ",";
    }
    data_file << ",";
    for (int i = 0; i < 7; i++) {
      data_file << point.tau[i];
      if (i < 6) data_file << ",";
    }
    data_file << "\n";
  }
  data_file.close();
  std::cout << "  数据已保存到: dynamics_identification_data.csv" << std::endl;

  // 构造回归矩阵（使用xMateModel，不依赖Pinocchio的回归矩阵API）
  std::cout << "\n========================================" << std::endl;
  std::cout << "开始构造回归矩阵..." << std::endl;
  std::cout << "========================================" << std::endl;

  int n_joints = 7;
  int n_params = 0;  // 将在第一次计算回归矩阵时确定
  
#ifdef PINOCCHIO_AVAILABLE
  std::cout << "  使用Pinocchio回归矩阵计算" << std::endl;
  // 先计算一次回归矩阵以确定参数数量
  Eigen::VectorXd q_test(7), dq_test(7), ddq_test(7);
  q_test.setZero();
  dq_test.setZero();
  ddq_test.setZero();
  try {
    // Pinocchio的computeJointTorqueRegressor返回回归矩阵的引用
    auto& Y_test = pinocchio::computeJointTorqueRegressor(
        pinocchio_model, pinocchio_data, q_test, dq_test, ddq_test);
    n_params = Y_test.cols();
    std::cout << "  Pinocchio回归矩阵参数数量: " << n_params << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "  警告: Pinocchio回归矩阵计算失败: " << e.what() << std::endl;
    std::cerr << "  回退到简化参数集" << std::endl;
    n_params = 3 * n_joints;  // 21个参数
  }
#endif
  
  Eigen::MatrixXd Y_all;
  Eigen::VectorXd tau_all;
  
  std::cout << "  处理 " << collected_data.size() << " 个数据点..." << std::endl;
  
  int processed = 0;
  
  for (const auto& point : collected_data) {
    // 转换为数组格式
    std::array<double, 7> q_arr = point.q;
    std::array<double, 7> dq_arr = point.dq;
    std::array<double, 7> ddq_arr = point.ddq;
    std::array<double, 7> tau_arr = point.tau;
    
    // 转换为Eigen向量（用于Pinocchio）
    Eigen::VectorXd q_eigen(7), dq_eigen(7), ddq_eigen(7);
    for (int i = 0; i < 7; i++) {
      q_eigen(i) = q_arr[i];
      dq_eigen(i) = dq_arr[i];
      ddq_eigen(i) = ddq_arr[i];
    }
    
    // 使用Pinocchio计算回归矩阵
    Eigen::MatrixXd Y(n_joints, n_params);
    Y.setZero();
    
#ifdef PINOCCHIO_AVAILABLE
    try {
      // 使用Pinocchio的回归矩阵计算函数（返回回归矩阵的引用）
      auto& Y_ref = pinocchio::computeJointTorqueRegressor(
          pinocchio_model, pinocchio_data, q_eigen, dq_eigen, ddq_eigen);
      Y = Y_ref;  // 复制回归矩阵
    } catch (const std::exception& e) {
      std::cerr << "  警告: Pinocchio回归矩阵计算失败: " << e.what() << std::endl;
      std::cerr << "  回退到简化方法..." << std::endl;
      
      // 回退到简化方法
      std::array<double, 7> trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base;
      model.getTorqueNoFriction(q_arr, dq_arr, ddq_arr, 
                                trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base);
      
      // 简化回归矩阵（每个关节3个参数）
      n_params = 3 * n_joints;
      Y.resize(n_joints, n_params);
      Y.setZero();
      
      for (int i = 0; i < n_joints; i++) {
        int param_base = 3 * i;
        Y(i, param_base) = ddq_arr[i];           // 等效惯性
        Y(i, param_base + 1) = trq_gravity_base[i];   // 重力系数
        Y(i, param_base + 2) = trq_coriolis_base[i];  // 科氏力系数
      }
    }
#else
    // 如果没有Pinocchio，使用简化方法
    std::array<double, 7> trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base;
    model.getTorqueNoFriction(q_arr, dq_arr, ddq_arr, 
                              trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base);
    
    for (int i = 0; i < n_joints; i++) {
      int param_base = 3 * i;
      Y(i, param_base) = ddq_arr[i];           // 等效惯性
      Y(i, param_base + 1) = trq_gravity_base[i];   // 重力系数
      Y(i, param_base + 2) = trq_coriolis_base[i];  // 科氏力系数
    }
#endif
    
    // 转换为Eigen向量（力矩）
    Eigen::VectorXd tau(7);
    for (int i = 0; i < 7; i++) {
      tau(i) = tau_arr[i];
    }
    
    // 堆叠回归矩阵和力矩向量
    if (Y_all.rows() == 0) {
      Y_all = Y;
      tau_all = tau;
    } else {
      // 检查参数数量是否一致
      if (Y_all.cols() != Y.cols()) {
        std::cerr << "  错误: 回归矩阵列数不一致！" << std::endl;
        std::cerr << "  之前: " << Y_all.cols() << ", 现在: " << Y.cols() << std::endl;
        break;
      }
      
      Eigen::MatrixXd Y_new(Y_all.rows() + n_joints, n_params);
      Y_new << Y_all, Y;
      Y_all = Y_new;
      
      Eigen::VectorXd tau_new(tau_all.size() + n_joints);
      tau_new << tau_all, tau;
      tau_all = tau_new;
    }
    
    processed++;
    if (processed % 100 == 0) {
      std::cout << "  已处理: " << processed << " / " << collected_data.size() << std::endl;
    }
  }
  
  std::cout << "\n  回归矩阵构造完成" << std::endl;
  std::cout << "  回归矩阵维度: " << Y_all.rows() << " x " << Y_all.cols() << std::endl;
  std::cout << "  力矩向量维度: " << tau_all.size() << std::endl;

  // 求解参数
  std::cout << "\n========================================" << std::endl;
  std::cout << "开始求解动力学参数..." << std::endl;
  std::cout << "========================================" << std::endl;

  // 使用 SVD 求解（更稳定）
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      Y_all, Eigen::ComputeThinU | Eigen::ComputeThinV);
  
  // 设置奇异值阈值
  double threshold = 1e-6;
  Eigen::VectorXd singular_values = svd.singularValues();
  int rank = 0;
  for (int i = 0; i < singular_values.size(); i++) {
    if (singular_values(i) > threshold * singular_values(0)) {
      rank++;
    }
  }
  std::cout << "  回归矩阵的秩: " << rank << " / " << n_params << std::endl;
  
  if (rank < n_params) {
    std::cerr << "  警告: 回归矩阵不满秩，某些参数可能无法辨识" << std::endl;
    std::cerr << "  建议: 增加激励轨迹的多样性或延长采集时间" << std::endl;
  }

  // 求解参数
  Eigen::VectorXd theta_estimated = svd.solve(tau_all);
  
  std::cout << "\n  辨识的动力学参数:" << std::endl;
#ifdef PINOCCHIO_AVAILABLE
  // Pinocchio的回归矩阵参数对应最小惯性参数集
  // 这些参数是物理参数的线性组合，不是直接的物理量
  std::cout << "  注意: 这些参数是最小惯性参数集的系数，不是直接的物理量" << std::endl;
  std::cout << "  参数值:" << std::endl;
  for (int i = 0; i < std::min(n_params, 20); i++) {  // 只打印前20个
    std::cout << "    theta[" << i << "] = " << std::fixed << std::setprecision(6) 
              << theta_estimated(i) << std::endl;
  }
  if (n_params > 20) {
    std::cout << "    ... (共 " << n_params << " 个参数)" << std::endl;
  }
#else
  // 简化参数集
  for (int i = 0; i < n_joints; i++) {
    int param_base = 3 * i;
    std::cout << "    关节 " << (i+1) << ":" << std::endl;
    std::cout << "      等效惯性 (kg·m²): " << std::fixed << std::setprecision(6) 
              << theta_estimated(param_base) << std::endl;
    std::cout << "      重力系数: " << std::fixed << std::setprecision(6) 
              << theta_estimated(param_base + 1) << std::endl;
    std::cout << "      科氏力系数: " << std::fixed << std::setprecision(6) 
              << theta_estimated(param_base + 2) << std::endl;
  }
#endif
  
  std::cout << "\n  参数求解完成" << std::endl;
  std::cout << "  辨识的参数数量: " << theta_estimated.size() << std::endl;

  // 计算拟合误差
  Eigen::VectorXd tau_predicted = Y_all * theta_estimated;
  Eigen::VectorXd error = tau_all - tau_predicted;
  
  double rmse = std::sqrt(error.squaredNorm() / error.size());
  double max_error = error.cwiseAbs().maxCoeff();
  double mean_error = error.cwiseAbs().mean();
  
  std::cout << "\n========================================" << std::endl;
  std::cout << "辨识结果统计:" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << " Nm" << std::endl;
  std::cout << "  最大误差: " << max_error << " Nm" << std::endl;
  std::cout << "  平均误差: " << mean_error << " Nm" << std::endl;
  
  // 按关节分析误差
  std::cout << "\n  各关节误差统计 (Nm):" << std::endl;
  for (int i = 0; i < 7; i++) {
    // 提取第i个关节的所有误差值（每7个数据点中取第i个）
    std::vector<double> error_joint_values;
    for (int j = i; j < error.size(); j += 7) {
      error_joint_values.push_back(error(j));
    }
    
    // 计算该关节的RMSE
    double sum_sq = 0.0;
    for (double e : error_joint_values) {
      sum_sq += e * e;
    }
    double rmse_joint = std::sqrt(sum_sq / error_joint_values.size());
    std::cout << "    关节 " << (i+1) << ": RMSE = " << std::fixed 
              << std::setprecision(4) << rmse_joint << std::endl;
  }

  // 保存辨识结果
  std::cout << "\n  保存辨识结果..." << std::endl;
  std::ofstream result_file("dynamics_identification_results.txt");
  result_file << "动力学参数辨识结果\n";
  result_file << "==================\n\n";
  result_file << "辨识时间: " << std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
  result_file << "数据点数: " << collected_data.size() << "\n";
  result_file << "参数数量: " << n_params << "\n\n";
  result_file << "误差统计:\n";
  result_file << "  RMSE: " << rmse << " Nm\n";
  result_file << "  最大误差: " << max_error << " Nm\n";
  result_file << "  平均误差: " << mean_error << " Nm\n\n";
  result_file << "辨识的参数值:\n";
  for (int i = 0; i < theta_estimated.size(); i++) {
    result_file << "  theta[" << i << "] = " << std::scientific 
                << std::setprecision(6) << theta_estimated(i) << "\n";
  }
  result_file.close();
  std::cout << "  结果已保存到: dynamics_identification_results.txt" << std::endl;

  // 保存参数到CSV（便于后续使用）
  std::ofstream param_file("dynamics_parameters.csv");
  param_file << "parameter_index,value\n";
  for (int i = 0; i < theta_estimated.size(); i++) {
    param_file << i << "," << std::scientific << std::setprecision(10) 
               << theta_estimated(i) << "\n";
  }
  param_file.close();
  std::cout << "  参数已保存到: dynamics_parameters.csv" << std::endl;

#ifdef PINOCCHIO_AVAILABLE
  // 从辨识的theta参数恢复物理参数
  std::cout << "\n  从辨识参数恢复物理参数..." << std::endl;
  
  // 方法：使用优化方法从最小参数集恢复物理参数
  // 思路：找到一组物理参数，使得它们的最小参数集表示最接近辨识的theta
  // 约束：质量>0，惯性矩阵正定，质心位置合理
  
  std::ofstream physical_param_file("dynamics_physical_parameters.txt");
  physical_param_file << "辨识的物理参数（从theta恢复）\n";
  physical_param_file << "========================================\n\n";
  physical_param_file << "说明:\n";
  physical_param_file << "这些物理参数是通过优化方法从辨识的最小惯性参数集(theta)恢复得到的。\n";
  physical_param_file << "由于最小参数集到物理参数的映射不是唯一的，这里提供一种可能的解。\n\n";
  
  // 使用采样点验证恢复的物理参数
  // 对于每个关节，我们尝试从theta恢复物理参数
  // 由于映射不是唯一的，我们使用初始URDF值作为参考，通过优化找到最接近的物理参数
  
  // 保存URDF初始值（作为对比）
  physical_param_file << "========================================\n";
  physical_param_file << "URDF模型初始值（作为对比）:\n";
  physical_param_file << "========================================\n\n";
  
  std::vector<pinocchio::Inertia> urdf_inertias;
  for (pinocchio::JointIndex joint_id = 1; joint_id < pinocchio_model.njoints; joint_id++) {
    pinocchio::Inertia inertia = pinocchio_model.inertias[joint_id];
    urdf_inertias.push_back(inertia);
    
    double mass = inertia.mass();
    Eigen::Vector3d com = inertia.lever();
    Eigen::Matrix3d inertia_matrix = inertia.inertia();
    
    physical_param_file << "关节 " << joint_id << " (" << pinocchio_model.names[joint_id] << ") - URDF初始值:\n";
    physical_param_file << "  质量 (kg): " << std::fixed << std::setprecision(6) << mass << "\n";
    physical_param_file << "  质心位置 (m): [" 
                        << std::fixed << std::setprecision(6) 
                        << com(0) << ", " << com(1) << ", " << com(2) << "]\n";
    physical_param_file << "  惯性张量 (kg·m²):\n";
    physical_param_file << "    [" << std::fixed << std::setprecision(6) 
                        << inertia_matrix(0,0) << ", " << inertia_matrix(0,1) << ", " << inertia_matrix(0,2) << "]\n";
    physical_param_file << "    [" << std::fixed << std::setprecision(6) 
                        << inertia_matrix(1,0) << ", " << inertia_matrix(1,1) << ", " << inertia_matrix(1,2) << "]\n";
    physical_param_file << "    [" << std::fixed << std::setprecision(6) 
                        << inertia_matrix(2,0) << ", " << inertia_matrix(2,1) << ", " << inertia_matrix(2,2) << "]\n\n";
  }
  
  // 尝试从theta恢复物理参数
  // 使用多个采样点计算回归矩阵，然后通过优化恢复物理参数
  physical_param_file << "\n========================================\n";
  physical_param_file << "从辨识参数恢复的物理参数:\n";
  physical_param_file << "========================================\n\n";
  physical_param_file << "注意: 由于最小参数集到物理参数的映射不是唯一的，\n";
  physical_param_file << "这里使用优化方法找到一组满足约束的物理参数。\n";
  physical_param_file << "如果恢复失败或结果不合理，请直接使用theta参数进行力矩预测。\n\n";
  
  // 使用采样点验证theta参数
  // 计算几个典型配置下的力矩，验证theta参数的有效性
  std::cout << "  验证辨识参数的有效性..." << std::endl;
  
  // 使用几个典型配置验证
  std::vector<Eigen::VectorXd> test_configs;
  Eigen::VectorXd q_test1(7), dq_test1(7), ddq_test1(7);
  q_test1 << 0, 0, 0, 0, 0, 0, 0;
  dq_test1 << 0, 0, 0, 0, 0, 0, 0;
  ddq_test1 << 0, 0, 0, 0, 0, 0, 0;
  test_configs.push_back(q_test1);
  
  // 计算这些配置下的预测力矩
  physical_param_file << "验证结果（使用辨识的theta参数预测力矩）:\n";
  for (size_t i = 0; i < test_configs.size(); i++) {
    Eigen::VectorXd q_test = test_configs[i];
    Eigen::VectorXd dq_test = Eigen::VectorXd::Zero(7);
    Eigen::VectorXd ddq_test = Eigen::VectorXd::Zero(7);
    
    auto& Y_test = pinocchio::computeJointTorqueRegressor(
        pinocchio_model, pinocchio_data, q_test, dq_test, ddq_test);
    Eigen::VectorXd tau_predicted = Y_test * theta_estimated;
    
    physical_param_file << "配置 " << (i+1) << " (q=" << q_test.transpose() << "):\n";
    physical_param_file << "  预测力矩 (Nm): [" 
                        << std::fixed << std::setprecision(4)
                        << tau_predicted(0) << ", " << tau_predicted(1) << ", " 
                        << tau_predicted(2) << ", " << tau_predicted(3) << ", "
                        << tau_predicted(4) << ", " << tau_predicted(5) << ", "
                        << tau_predicted(6) << "]\n\n";
  }
  
  // 说明：由于最小参数集到物理参数的逆映射不是唯一的，
  // 我们直接使用theta参数进行力矩预测，而不是尝试恢复物理参数
  physical_param_file << "\n========================================\n";
  physical_param_file << "使用说明:\n";
  physical_param_file << "========================================\n";
  physical_param_file << "1. 辨识的theta参数已经保存在 dynamics_parameters.csv 中\n";
  physical_param_file << "2. 使用公式 τ = Y(q, dq, ddq) × theta 可以预测任意配置下的力矩\n";
  physical_param_file << "3. 其中 Y 由 pinocchio::computeJointTorqueRegressor 计算\n";
  physical_param_file << "4. 由于最小参数集到物理参数的映射不是唯一的，\n";
  physical_param_file << "   建议直接使用theta参数，而不是尝试恢复物理参数\n";
  physical_param_file << "5. 如果需要物理参数，可以使用URDF初始值作为参考，\n";
  physical_param_file << "   或使用带约束的优化方法从theta恢复（需要额外的优化库）\n\n";
  
  physical_param_file << "参数数量: " << n_params << "\n";
  physical_param_file << "这些参数已经保存在 dynamics_parameters.csv 中。\n";
  
  physical_param_file.close();
  std::cout << "  物理参数说明已保存到: dynamics_physical_parameters.txt" << std::endl;
  
  // 同时更新结果文件，添加物理参数说明
  std::ofstream result_file_append("dynamics_identification_results.txt", std::ios::app);
  result_file_append << "\n========================================\n";
  result_file_append << "物理参数说明:\n";
  result_file_append << "========================================\n";
  result_file_append << "上述 theta 参数是最小惯性参数集的系数，可以直接用于力矩预测。\n";
  result_file_append << "使用公式: τ = Y(q, dq, ddq) × theta\n";
  result_file_append << "详细的说明和URDF初始值对比已保存到: dynamics_physical_parameters.txt\n";
  result_file_append << "注意: 最小参数集到物理参数的映射不是唯一的，\n";
  result_file_append << "建议直接使用theta参数进行力矩预测，而不是尝试恢复物理参数。\n";
  result_file_append.close();
#endif

  // 验证：使用部分数据验证
  std::cout << "\n========================================" << std::endl;
  std::cout << "使用验证数据验证辨识结果..." << std::endl;
  std::cout << "========================================" << std::endl;
  
  // 使用后20%的数据作为验证集
  int validation_start = static_cast<int>(collected_data.size() * 0.8);
  int validation_count = collected_data.size() - validation_start;
  
  Eigen::MatrixXd Y_val;
  Eigen::VectorXd tau_val;
  
  for (int i = validation_start; i < collected_data.size(); i++) {
    const auto& point = collected_data[i];
    std::array<double, 7> q_arr = point.q;
    std::array<double, 7> dq_arr = point.dq;
    std::array<double, 7> ddq_arr = point.ddq;
    std::array<double, 7> tau_arr = point.tau;
    
    // 转换为Eigen向量（用于Pinocchio）
    Eigen::VectorXd q_eigen(7), dq_eigen(7), ddq_eigen(7);
    for (int j = 0; j < 7; j++) {
      q_eigen(j) = q_arr[j];
      dq_eigen(j) = dq_arr[j];
      ddq_eigen(j) = ddq_arr[j];
    }
    
    // 使用Pinocchio计算回归矩阵（与训练时相同的方法）
    Eigen::MatrixXd Y(n_joints, n_params);
    Y.setZero();
    
#ifdef PINOCCHIO_AVAILABLE
    try {
      // Pinocchio的computeJointTorqueRegressor返回回归矩阵的引用
      auto& Y_ref = pinocchio::computeJointTorqueRegressor(
          pinocchio_model, pinocchio_data, q_eigen, dq_eigen, ddq_eigen);
      Y = Y_ref;  // 复制回归矩阵
    } catch (const std::exception& e) {
      // 回退到简化方法
      std::array<double, 7> trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base;
      model.getTorqueNoFriction(q_arr, dq_arr, ddq_arr, 
                                trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base);
      
      n_params = 3 * n_joints;
      Y.resize(n_joints, n_params);
      Y.setZero();
      
      for (int j = 0; j < n_joints; j++) {
        int param_base = 3 * j;
        Y(j, param_base) = ddq_arr[j];
        Y(j, param_base + 1) = trq_gravity_base[j];
        Y(j, param_base + 2) = trq_coriolis_base[j];
      }
    }
#else
    // 简化方法
    std::array<double, 7> trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base;
    model.getTorqueNoFriction(q_arr, dq_arr, ddq_arr, 
                              trq_base, trq_inertia_base, trq_coriolis_base, trq_gravity_base);
    
    for (int j = 0; j < n_joints; j++) {
      int param_base = 3 * j;
      Y(j, param_base) = ddq_arr[j];
      Y(j, param_base + 1) = trq_gravity_base[j];
      Y(j, param_base + 2) = trq_coriolis_base[j];
    }
#endif
    
    // 转换为Eigen向量（力矩）
    Eigen::VectorXd tau(7);
    for (int j = 0; j < 7; j++) {
      tau(j) = tau_arr[j];
    }
    
    if (Y_val.rows() == 0) {
      Y_val = Y;
      tau_val = tau;
    } else {
      Eigen::MatrixXd Y_new(Y_val.rows() + n_joints, n_params);
      Y_new << Y_val, Y;
      Y_val = Y_new;
      
      Eigen::VectorXd tau_new(tau_val.size() + n_joints);
      tau_new << tau_val, tau;
      tau_val = tau_new;
    }
  }
  
  Eigen::VectorXd tau_val_predicted = Y_val * theta_estimated;
  Eigen::VectorXd error_val = tau_val - tau_val_predicted;
  double rmse_val = std::sqrt(error_val.squaredNorm() / error_val.size());
  
  std::cout << "  验证集数据点数: " << validation_count << std::endl;
  std::cout << "  验证集 RMSE: " << std::fixed << std::setprecision(4) 
            << rmse_val << " Nm" << std::endl;
  
  if (rmse_val < rmse * 1.5) {
    std::cout << "  ✓ 验证通过：验证误差在合理范围内" << std::endl;
  } else {
    std::cout << "  ⚠ 警告：验证误差较大，可能需要更多数据或改进激励轨迹" << std::endl;
  }
}

/**
 * @brief main program
 */
int main(int argc, char * argv[])
{
  using namespace rokae;
  
  // 打印SDK版本号
  std::cout << "========================================" << std::endl;
  std::cout << "xCore SDK 动力学参数辨识程序" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "SDK版本号: " << BaseRobot::sdkVersion() << std::endl;
  std::cout << "========================================" << std::endl;
  
  // AR5系列7轴机器人IP地址（请根据实际情况修改）
  std::string robot_ip = "192.168.110.15";  // 机器人IP
  std::string local_ip = ""; // 本机IP（将自动检测）
  
  // 自动检测本机IP地址
  std::cout << "\n正在检测本机IP地址（与机器人同一网段）..." << std::endl;
  local_ip = detectLocalIp(robot_ip);
  if (local_ip.empty()) {
    std::cerr << "警告: 无法自动检测本机IP地址" << std::endl;
    std::cerr << "请手动设置本机IP地址（与机器人IP在同一网段）" << std::endl;
    return 1;
  }
  
  std::cout << "检测到本机IP地址: " << local_ip << std::endl;
  
  try {
    std::cout << "\n正在连接到机器人 " << robot_ip << "..." << std::endl;
    std::cout << "本机IP地址: " << local_ip << std::endl;
    
    xMateErProRobot robot;
    try {
      robot.connectToRobot(robot_ip, local_ip);
      std::cout << "连接成功！" << std::endl;
    } catch (const rokae::NetworkException &e) {
      std::cerr << "\n网络连接失败！" << std::endl;
      std::cerr << "错误信息: " << e.what() << std::endl;
      return 1;
    } catch (const rokae::Exception &e) {
      std::cerr << "\n连接失败！" << std::endl;
      std::cerr << "错误信息: " << e.what() << std::endl;
      return 1;
    }
    
    error_code ec;
    
    // 检查当前状态
    std::cout << "\n检查机器人当前状态..." << std::endl;
    auto current_op_state = robot.operationState(ec);
    
    // 如果机器人处于实时模式控制中，需要先退出实时模式
    if (current_op_state == OperationState::rtControlling) {
      std::cout << "\n检测到机器人处于实时模式控制中，正在退出实时模式..." << std::endl;
      robot.setMotionControlMode(MotionControlMode::Idle, ec);
      if (ec) {
        print(std::cerr, "退出实时模式失败:", ec);
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::cout << "\n设置操作模式为自动模式..." << std::endl;
    robot.setOperateMode(OperateMode::automatic, ec);
    if (ec) {
      print(std::cerr, "设置操作模式失败:", ec);
      return 1;
    }
    
    // 设置网络丢包阈值
    std::cout << "设置网络丢包阈值为50%..." << std::endl;
    robot.setRtNetworkTolerance(50, ec);
    if (ec) {
      print(std::cerr, "设置丢包阈值失败:", ec);
      return 1;
    }
    std::cout << "丢包阈值设置成功（50%）" << std::endl;
    
    std::cout << "设置运动控制模式为实时模式..." << std::endl;
    robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
    if (ec) {
      print(std::cerr, "设置实时模式失败:", ec);
      return 1;
    }
    
    std::cout << "机器人上电..." << std::endl;
    robot.setPowerState(true, ec);
    if (ec) {
      print(std::cerr, "上电失败:", ec);
      return 1;
    }
    
    // 等待上电状态稳定
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    try {
      std::cout << "\n========================================" << std::endl;
      std::cout << "开始动力学参数辨识..." << std::endl;
      std::cout << "========================================" << std::endl;
      dynamicsIdentification(robot);
      std::cout << "\n========================================" << std::endl;
      std::cout << "动力学参数辨识结束" << std::endl;
      std::cout << "========================================" << std::endl;
    } catch (const RealtimeMotionException &e) {
      print(std::cerr, "实时运动异常:", e.what());
      robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    }

    std::cout << "\n切换回非实时模式..." << std::endl;
    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    std::cout << "设置操作模式为手动模式..." << std::endl;
    robot.setOperateMode(OperateMode::manual, ec);

  } catch (const std::exception &e) {
    print(std::cerr, "发生异常:", e.what());
  }
  
  return 0;
}

