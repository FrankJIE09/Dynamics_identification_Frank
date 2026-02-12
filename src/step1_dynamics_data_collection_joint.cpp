/**
 * @file step1_dynamics_data_collection.cpp
 * @brief 实时模式 - 动力学参数辨识数据采集
 * 
 * 此程序用于采集机器人动力学辨识所需的数据：
 * 1. 生成激励轨迹（正弦波、傅里叶级数等）
 * 2. 在实时控制循环中采集数据（关节位置、速度、加速度、力矩）
 * 3. 保存数据到CSV文件
 * 
 * 此示例需要使用xMateModel模型库，请设置编译选项XCORE_USE_XMATE_MODEL=ON
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
#include "../print_helper.hpp"
#include "rokae/utility.h"

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
 * @brief 数据采集主函数
 */
void collectDynamicsData(xMateErProRobot &robot, const std::string& output_file = "dynamics_identification_data.csv") {
  using namespace RtSupportedFields;
  auto rtCon = robot.getRtMotionController().lock();
  auto model = robot.model();
  error_code ec;

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
    -0.458,   // 一轴：-2.458度
    -0.651,  // 二轴：-70.651度
    -0.044,   // 三轴：-2.044度
    0.492,  // 四轴：118.492度
    0.682,    // 五轴：4.682度
    -0.719,  // 六轴：-47.719度
    -0.433    // 七轴：-5.433度
  }));
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

  // 数据采集参数
  const double collection_duration = 60.0;  // 采集时长（秒）
  const double dt = 0.001;  // 采样周期（1ms）
  std::vector<DataPoint> collected_data;
  collected_data.reserve(static_cast<size_t>(collection_duration / dt));

  // 激励轨迹参数（降低幅值和频率以减小角速度）
  std::array<double, 7> amplitudes = {
    0.45, 0.45, 0.45, 0.5, 0.5, 0.5, 0.5  // 各关节幅值（弧度）
  };
  std::array<double, 7> frequencies = {
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8  // 各关节频率（rad/s）
  };
  std::array<double, 7> phases = {
    0.0, M_PI/7, 2*M_PI/7, 3*M_PI/7, 4*M_PI/7, 5*M_PI/7, 6*M_PI/7
  };

  std::cout << "\n========================================" << std::endl;
  std::cout << "开始数据采集..." << std::endl;
  std::cout << "采集时长: " << collection_duration << " 秒" << std::endl;
  std::cout << "采样频率: " << (1.0/dt) << " Hz" << std::endl;
  std::cout << "预计采集点数: " << (collection_duration / dt) << std::endl;
  std::cout << "输出文件: " << output_file << std::endl;
  std::cout << "========================================" << std::endl;

  // 实时控制循环：关节位置方式下发激励轨迹并采集数据
  std::function<JointPosition(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time = 0.0;

    time += dt;

    // 获取当前状态
    std::array<double, 7> q{}, dq_m{}, ddq_m{}, tau_m_measured{};
    robot.getStateData(jointPos_m, q);
    robot.getStateData(jointVel_m, dq_m);
    robot.getStateData(jointAcc_m, ddq_m);
    robot.getStateData(tau_m, tau_m_measured);

    // 生成激励轨迹（正弦波）作为目标关节位置
    std::array<double, 7> q_target = generateSineTrajectory(
        time, q_init, amplitudes, frequencies, phases);

    // 采集数据（每10ms采集一次，减少数据量）
    static int sample_counter = 0;
    if (sample_counter % 10 == 0) {
        DataPoint point;
        point.q = q;
        point.dq = dq_m;
        point.ddq = ddq_m;
        point.tau = tau_m_measured;
        point.timestamp = time;
        collected_data.push_back(point);
    }
    sample_counter++;

    // 打印进度（每100ms）
    static int print_counter = 0;
    if (print_counter % 100 == 0) {
        std::cout << "\n[时间: " << std::fixed << std::setprecision(3) << time << "s]" << std::endl;
        std::cout << "  采集进度: " << std::fixed << std::setprecision(1)
                  << (time / collection_duration * 100.0) << "%, "
                  << "已采集: " << collected_data.size() << " 点" << std::endl;
    }
    print_counter++;

    // 返回关节位置指令；到时后结束
    JointPosition cmd(7);
    if (time > collection_duration) {
        cmd.setFinished();
    } else {
        for (int i = 0; i < 7; i++) {
            cmd.joints[i] = q_target[i];
        }
    }
    return cmd;
  };

  // 设置控制循环
  std::cout << "  设置控制循环回调..." << std::endl;
  rtCon->setControlLoop(callback, 0, true);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // 启动关节位置控制模式
  std::cout << "  启动关节位置控制模式..." << std::endl;
  try {
    rtCon->startMove(RtControllerMode::jointPosition);
    std::cout << "  关节位置控制模式已启动，开始采集数据..." << std::endl;
  } catch (const rokae::RealtimeControlException &e) {
    std::cerr << "  启动关节位置控制失败: " << e.what() << std::endl;
    return;
  }

  // 启动控制循环
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rtCon->startLoop(true);
  
  std::cout << "\n  数据采集完成！" << std::endl;
  std::cout << "  实际采集点数: " << collected_data.size() << std::endl;

  // 保存原始数据到CSV文件
  std::cout << "\n  保存原始数据到文件..." << std::endl;
  std::ofstream data_file(output_file);
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
  std::cout << "  数据已保存到: " << output_file << std::endl;
  std::cout << "  数据点数: " << collected_data.size() << std::endl;
}

/**
 * @brief main program
 */
int main(int argc, char * argv[])
{
  using namespace rokae;
  
  // 打印SDK版本号
  std::cout << "========================================" << std::endl;
  std::cout << "xCore SDK 动力学数据采集程序" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "SDK版本号: " << BaseRobot::sdkVersion() << std::endl;
  std::cout << "========================================" << std::endl;
  
  // 解析命令行参数
  std::string robot_ip = "10.17.0.110";  // 默认机器人IP
  std::string output_file = "dynamics_identification_data.csv";  // 默认输出文件
  
  if (argc > 1) {
    robot_ip = argv[1];
  }
  if (argc > 2) {
    output_file = argv[2];
  }
  
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
  std::cout << "机器人IP地址: " << robot_ip << std::endl;
  std::cout << "输出文件: " << output_file << std::endl;
  
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
      std::cout << "开始数据采集..." << std::endl;
      std::cout << "========================================" << std::endl;
      collectDynamicsData(robot, output_file);
      std::cout << "\n========================================" << std::endl;
      std::cout << "数据采集结束" << std::endl;
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

