/**
 * @file torque_control.cpp
 * @brief 实时模式 - 直接力矩控制
 * 此示例需要使用xMateModel模型库，请设置编译选项XCORE_USE_XMATE_MODEL=ON
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 * Information in this file is the intellectual property of Rokae Technology Co., Ltd,
 * And may contains trade secrets that must be stored and viewed confidentially.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include "rokae/robot.h"
#include "Eigen/Geometry"
#include "../print_helper.hpp"
#include "rokae/utility.h"

using namespace rokae;

/**
 * @brief 自动检测本机IP地址（与机器人IP在同一网段）
 * @param robot_ip 机器人IP地址
 * @return 本机IP地址，如果检测失败则返回空字符串
 */
std::string detectLocalIp(const std::string& robot_ip) {
  // 提取机器人IP的网段（例如：192.168.110.15 -> 192.168.110）
  size_t last_dot = robot_ip.find_last_of('.');
  if (last_dot == std::string::npos) {
    return "";
  }
  std::string network_prefix = robot_ip.substr(0, last_dot);
  
  // 使用系统命令获取本机IP地址
  std::string cmd = "ip addr show | grep -E 'inet " + network_prefix + "\\.' | head -1 | awk '{print $2}' | cut -d'/' -f1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return "";
  }
  
  char buffer[128];
  std::string result = "";
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result = buffer;
    // 去除换行符
    if (!result.empty() && result.back() == '\n') {
      result.pop_back();
    }
  }
  pclose(pipe);
  
  return result;
}

/**
 * @brief 力矩控制. 注意:
 * 1) 力矩值不要超过机型的限制条件(见手册);
 * 2) 初次运行时请手握急停开关, 避免机械臂非预期运动造成碰撞
 */
void torqueControl(xMateErProRobot &robot) {
  using namespace RtSupportedFields;
  auto rtCon = robot.getRtMotionController().lock();
  auto model = robot.model();
  error_code ec;
  std::array<double, 7> q_drag = Utils::degToRad(std::array<double, 7>({
    -2.458,   // 一轴：-2.458度
    -70.651,  // 二轴：-70.651度
    -2.044,   // 三轴：-2.044度
    118.492,  // 四轴：118.492度
    4.682,    // 五轴：4.682度
    -47.719,  // 六轴：-47.719度
    -5.433    // 七轴：-5.433度
  }));
  q_drag = {0, M_PI/6, 0, M_PI/3, 0, 0, 0 };

#if 0
  // 设置负载, 请根据实际情况设置，确保安全
  // 设置负载后，动力学计算结果（重力补偿等）会相应改变
  double load_mass = 1.0;  // 负载质量，单位：kg
  std::array<double, 3> load_centre = {0.0, 0.0, 0.1};  // 负载质心，单位：m（相对于法兰）
  std::array<double, 3> load_inertia = {0.01, 0.01, 0.01};  // 负载惯量，单位：kg·m²
  model.setLoad(load_mass, load_centre, load_inertia);
  std::cout << "  已设置负载参数: 质量=" << load_mass << "kg, 质心=[" 
            << load_centre[0] << ", " << load_centre[1] << ", " << load_centre[2] << "]m" << std::endl;
#endif

  std::cout << "  停止接收机器人状态..." << std::endl;
  robot.stopReceiveRobotState();
  
  std::cout << "  开始接收机器人状态（1ms周期）..." << std::endl;
  try {
    robot.startReceiveRobotState(std::chrono::milliseconds(1),
                                 {jointPos_m, jointVel_m, jointAcc_c, tcpPose_m});
    std::cout << "  机器人状态接收已启动" << std::endl;
  } catch (const rokae::RealtimeControlException &e) {
    std::cerr << "  启动状态接收失败: " << e.what() << std::endl;
    std::cerr << "  可能原因:" << std::endl;
    std::cerr << "    1. UDP socket无法绑定到本机IP地址" << std::endl;
    std::cerr << "    2. 本机IP地址配置不正确" << std::endl;
    std::cerr << "    3. 端口被占用" << std::endl;
    std::cerr << "    4. 防火墙阻止了UDP连接" << std::endl;
    std::cerr << "  解决方法:" << std::endl;
    std::cerr << "    - 使用 'ip addr show' 命令查看本机实际IP地址" << std::endl;
    std::cerr << "    - 确保本机IP与机器人IP在同一网段" << std::endl;
    std::cerr << "    - 检查防火墙设置，允许UDP端口通信" << std::endl;
    throw; // 重新抛出异常
  } catch (const std::exception &e) {
    std::cerr << "  启动状态接收时发生异常: " << e.what() << std::endl;
    throw; // 重新抛出异常
  }

  // 运动到拖拽位置
  std::cout << "  运动到拖拽位置..." << std::endl;
  
  // 获取当前位置
  auto current_pos = robot.jointPos(ec);
  std::cout << "  当前位置（度）: [";
  for (size_t i = 0; i < current_pos.size(); i++) {
    std::cout << current_pos[i] * 180.0 / M_PI;
    if (i < current_pos.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  
  // 打印目标角度
  std::cout << "  目标角度（度）: [";
  for (size_t i = 0; i < q_drag.size(); i++) {
    std::cout << q_drag[i] * 180.0 / M_PI;
    if (i < q_drag.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  目标角度（弧度）: [";
  for (size_t i = 0; i < q_drag.size(); i++) {
    std::cout << q_drag[i];
    if (i < q_drag.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  运动速度比例: 0.2 (20%)" << std::endl;
  
  rtCon->MoveJ(0.2, robot.jointPos(ec), q_drag);
  std::cout << "  已到达拖拽位置" << std::endl;

  // 控制模式为力矩控制
  std::cout << "  设置控制模式为力矩控制..." << std::endl;
  
  // 检查电源状态，确保机器人已上电
  auto power_state = robot.powerState(ec);
  if (ec) {
    print(std::cerr, "  获取电源状态失败:", ec);
    return;
  }
  
  using PS = rokae::PowerState;
  if (power_state != PS::on) {
    std::cerr << "  警告: 机器人未上电，当前电源状态: ";
    switch(power_state) {
      case PS::off: std::cerr << "下电"; break;
      case PS::estop: std::cerr << "急停"; break;
      case PS::gstop: std::cerr << "安全门打开"; break;
      default: std::cerr << "未知"; break;
    }
    std::cerr << std::endl;
    std::cout << "  等待1秒后尝试上电..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // 尝试上电
    std::cout << "  正在上电..." << std::endl;
    robot.setPowerState(true, ec);
    if (ec) {
      print(std::cerr, "  上电失败:", ec);
      std::cerr << "  请检查机器人状态（急停、安全门等）" << std::endl;
      return;
    }
    
    // 再次检查电源状态
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    power_state = robot.powerState(ec);
    if (ec) {
      print(std::cerr, "  获取电源状态失败:", ec);
      return;
    }
    
    if (power_state != PS::on) {
      std::cerr << "  错误: 上电后仍未能上电，当前电源状态: ";
      switch(power_state) {
        case PS::off: std::cerr << "下电"; break;
        case PS::estop: std::cerr << "急停"; break;
        case PS::gstop: std::cerr << "安全门打开"; break;
        default: std::cerr << "未知"; break;
      }
      std::cerr << std::endl;
      std::cerr << "  请检查机器人状态（急停按钮、安全门等）" << std::endl;
      return;
    }
    std::cout << "  上电成功！" << std::endl;
  }
  
  std::cout << "  电源状态检查通过（已上电）" << std::endl;
  
  // 等待一小段时间确保状态稳定
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  // 设置滤波截止频率为10Hz（用于平滑指令）
  std::cout << "  设置滤波截止频率为10Hz..." << std::endl;
  rtCon->setFilterFrequency(10.0, 10.0, 10.0, ec);
  if (ec) {
    print(std::cerr, "  设置滤波频率失败:", ec);
    return;
  }
  std::cout << "  滤波频率设置成功（关节: 10Hz, 笛卡尔: 10Hz, 力矩: 10Hz）" << std::endl;
  
  try {
    rtCon->startMove(RtControllerMode::torque);
    std::cout << "  力矩控制模式已启动" << std::endl;
  } catch (const rokae::RealtimeControlException &e) {
    std::cerr << "  启动力矩控制失败: " << e.what() << std::endl;
    std::cerr << "  可能原因:" << std::endl;
    std::cerr << "    1. 机器人电源状态不正确" << std::endl;
    std::cerr << "    2. 操作模式不正确（需要自动模式）" << std::endl;
    std::cerr << "    3. 运动控制模式不正确（需要实时模式）" << std::endl;
    throw; // 重新抛出异常
  }

  // Compliance parameters (阻抗控制参数)
  const double translational_stiffness{200.0};  // 平移刚度 [N/m]
  const double rotational_stiffness{5.0};     // 旋转刚度 [Nm/rad]
  const double damping_ratio{0.7};            // 阻尼比（0.7表示临界阻尼的70%）
  
  Eigen::MatrixXd stiffness(6, 6);
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  
  // 设置阻尼矩阵（临界阻尼公式：damping = 2 * damping_ratio * sqrt(stiffness)）
  Eigen::MatrixXd damping(6, 6);
  damping.setZero();
  damping.topLeftCorner(3, 3) << damping_ratio * 2.0 * sqrt(translational_stiffness) *
    Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << damping_ratio * 2.0 * sqrt(rotational_stiffness) *
    Eigen::MatrixXd::Identity(3, 3);
  
  std::cout << "  阻抗控制参数设置:" << std::endl;
  std::cout << "    平移刚度: " << translational_stiffness << " N/m" << std::endl;
  std::cout << "    旋转刚度: " << rotational_stiffness << " Nm/rad" << std::endl;
  std::cout << "    阻尼比: " << damping_ratio << std::endl;
  std::cout << "    平移阻尼: " << (damping_ratio * 2.0 * sqrt(translational_stiffness)) << " N·s/m" << std::endl;
  std::cout << "    旋转阻尼: " << (damping_ratio * 2.0 * sqrt(rotational_stiffness)) << " Nm·s/rad" << std::endl;

  std::array<double, 16> init_position {};
  Eigen::Matrix<double, 6, 7> jacobian;
  Utils::postureToTransArray(robot.posture(rokae::CoordinateType::flangeInBase, ec), init_position);

  std::function<Torque(void)> callback = [&]{
    using namespace RtSupportedFields;
    static double time=0;
    time += 0.001;
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(init_position.data()).transpose());
    Eigen::Vector3d position_d(initial_transform.translation());
    Eigen::Quaterniond orientation_d(initial_transform.linear());

    std::array<double, 7> q{}, dq_m{}, ddq_c{};
    std::array<double, 16> pos_m {};

    // 接收设置为true, 回调函数中可以直接读取
    robot.getStateData(jointPos_m, q);
    robot.getStateData(jointVel_m, dq_m);
    robot.getStateData(jointAcc_c, ddq_c);

    std::array<double, 42> jacobian_array = model.jacobian(q);
    std::array<double, 7> gravity_array = model.getTorque(q, dq_m, ddq_c, TorqueType::gravity);
    std::array<double, 7> friction_array = model.getTorque(q, dq_m, ddq_c, TorqueType::friction);

    // convert to Eigen
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> friction(friction_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 6>> jacobian_(jacobian_array.data());
    jacobian = jacobian_.transpose();
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q_mat(q.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq_mat(dq_m.data());
    robot.getStateData(tcpPose_m, pos_m);
    Eigen::Affine3d transform(Eigen::Matrix4d::Map(pos_m.data()).transpose());
    Eigen::Vector3d position(transform.translation());
    Eigen::Quaterniond orientation(transform.linear());

    // compute error to desired equilibrium pose
    // position error
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d;

    // orientation error
    // "difference" quaternion
    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
      orientation.coeffs() << -orientation.coeffs();
    }
    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
    // Transform to base frame
    error.tail(3) << -transform.linear() * error.tail(3);

    // compute control
    Eigen::VectorXd tau_d(7);

    // 计算笛卡尔空间速度（通过雅可比矩阵将关节速度映射到笛卡尔空间）
    Eigen::VectorXd cartesian_velocity = jacobian * dq_mat;

    // cartesian space impedance calculate && map to joint space
    // 阻抗控制公式: tau = J^T * (-K * error - D * velocity)
    // 其中 K 是刚度矩阵，D 是阻尼矩阵，error 是位置/姿态误差，velocity 是笛卡尔空间速度
    tau_d << jacobian.transpose() * (-stiffness * error - damping * cartesian_velocity);

    // 限制力矩大小，防止力矩过大导致机械臂失控
    // AR5系列力矩限制：轴1-3: 85Nm, 轴4-7: 36Nm
    const std::array<double, 7> torque_limits = {85.0, 85.0, 85.0, 36.0, 36.0, 36.0, 36.0};
    bool torque_limited = false;
    for (int i = 0; i < 7; i++) {
      if (tau_d(i) > torque_limits[i]) {
        tau_d(i) = torque_limits[i];
        torque_limited = true;
      } else if (tau_d(i) < -torque_limits[i]) {
        tau_d(i) = -torque_limits[i];
        torque_limited = true;
      }
    }

    // 输出控制扭矩（每100ms输出一次，避免输出过多）
    static int print_counter = 0;
    if (print_counter % 100 == 0) {  // 每100ms输出一次
      std::cout << "\n[时间: " << std::fixed << std::setprecision(3) << time << "s]" << std::endl;
      std::cout << "  控制扭矩 (Nm): [";
      for (int i = 0; i < 7; i++) {
        std::cout << std::fixed << std::setprecision(3) << tau_d(i);
        if (i < 6) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      if (torque_limited) {
        std::cout << "  ⚠️  警告: 力矩已限制到安全范围！" << std::endl;
      }
      
      // 输出位置误差
      std::cout << "  位置误差 (m): [";
      for (int i = 0; i < 3; i++) {
        std::cout << std::fixed << std::setprecision(4) << error(i);
        if (i < 2) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      // 输出姿态误差
      std::cout << "  姿态误差 (rad): [";
      for (int i = 3; i < 6; i++) {
        std::cout << std::fixed << std::setprecision(4) << error(i);
        if (i < 5) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      // 输出当前关节速度
      std::cout << "  关节速度 (rad/s): [";
      for (int i = 0; i < 7; i++) {
        std::cout << std::fixed << std::setprecision(3) << dq_m[i];
        if (i < 6) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      // 输出当前关节位置
      std::cout << "  关节位置 (度): [";
      for (int i = 0; i < 7; i++) {
        std::cout << std::fixed << std::setprecision(2) << q[i] * 180.0 / M_PI;
        if (i < 6) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      // 输出重力补偿力矩
      std::cout << "  重力补偿力矩 (Nm): [";
      for (int i = 0; i < 7; i++) {
        std::cout << std::fixed << std::setprecision(3) << gravity_array[i];
        if (i < 6) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      // 输出摩擦力矩
      std::cout << "  摩擦力矩 (Nm): [";
      for (int i = 0; i < 7; i++) {
        std::cout << std::fixed << std::setprecision(3) << friction_array[i];
        if (i < 6) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
    }
    print_counter++;

    Torque cmd(7);
    Eigen::VectorXd::Map(cmd.tau.data(), 7) = tau_d;

    if(time > 30){
      cmd.setFinished();
    }
    return cmd;
  };

  // 由于需要在callback里读取状态数据, 这里useStateDataInLoop = true
  // 并且调用startReceiveRobotState()时, 设定的发送周期是1ms
  std::cout << "  设置控制循环回调..." << std::endl;
  rtCon->setControlLoop(callback, 0, true);
  std::cout << "  开始控制循环（运行30秒）..." << std::endl;
  rtCon->startLoop(true);
  std::cout << "  控制循环结束" << std::endl;
}

/**
 * @brief 发送0力矩. 力控模型准确的情况下, 机械臂应保持静止不动
 */
template <unsigned short DoF>
void zeroTorque(Cobot<DoF> &robot) {
  error_code ec;
  std::array<double, 7> q_drag = Utils::degToRad(std::array<double, 7>({
    -2.458,   // 一轴：-2.458度
    -70.651,  // 二轴：-70.651度
    -2.044,   // 三轴：-2.044度
    118.492,  // 四轴：118.492度
    4.682,    // 五轴：4.682度
    -47.719,  // 六轴：-47.719度
    -5.433    // 七轴：-5.433度
  }));
  // std::array<double,7> q_drag = {0, M_PI/6, 0, M_PI/3, 0, M_PI/2, 0 };
  auto rtCon = robot.getRtMotionController().lock();

  // 运动到拖拽位置
  rtCon->MoveJ(0.2, robot.jointPos(ec), q_drag);

  // 控制模式为力矩控制
  rtCon->startMove(RtControllerMode::torque);
  Torque cmd {};
  cmd.tau.resize(DoF);

  std::function<Torque(void)> callback = [&]() {
    static double time=0;
    time += 0.001;
    if(time > 30){
      cmd.setFinished();
    }
    return cmd;
  };

  rtCon->setControlLoop(callback);
  rtCon->startLoop();
  print(std::cout, "力矩控制结束");
}

/**
 * @brief main program
 */
int main() {
  using namespace rokae;
  
  // 打印SDK版本号
  std::cout << "========================================" << std::endl;
  std::cout << "xCore SDK 力矩控制示例程序" << std::endl;
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
    std::cerr << "可以使用以下命令查看本机IP地址:" << std::endl;
    std::cerr << "  ip addr show | grep -E 'inet 192\\.168\\.110\\.'" << std::endl;
    std::cerr << "\n如果自动检测失败，请在代码中手动设置 local_ip" << std::endl;
    return 1;
  }
  
  std::cout << "检测到本机IP地址: " << local_ip << std::endl;
  
  try {
    std::cout << "\n正在连接到机器人 " << robot_ip << "..." << std::endl;
    std::cout << "本机IP地址: " << local_ip << std::endl;
    
    // 先创建机器人对象，然后连接（这样可以更好地捕获连接异常）
    xMateErProRobot robot; // AR5系列7轴机器人
    try {
      robot.connectToRobot(robot_ip, local_ip);
      std::cout << "连接成功！" << std::endl;
    } catch (const rokae::NetworkException &e) {
      std::cerr << "\n网络连接失败！" << std::endl;
      std::cerr << "错误信息: " << e.what() << std::endl;
      std::cerr << "\n请检查:" << std::endl;
      std::cerr << "  1. 机器人IP地址是否正确: " << robot_ip << std::endl;
      std::cerr << "  2. 本机IP地址是否正确: " << local_ip << std::endl;
      std::cerr << "  3. 机器人和本机是否在同一网络" << std::endl;
      std::cerr << "  4. 机器人是否已开机" << std::endl;
      std::cerr << "  5. 防火墙是否阻止了连接" << std::endl;
      return 1;
    } catch (const rokae::Exception &e) {
      std::cerr << "\n连接失败！" << std::endl;
      std::cerr << "错误信息: " << e.what() << std::endl;
      std::cerr << "可能原因: SDK版本与控制器版本不匹配，或SDK未授权" << std::endl;
      return 1;
    } catch (const std::exception &e) {
      std::cerr << "\n连接时发生异常！" << std::endl;
      std::cerr << "错误信息: " << e.what() << std::endl;
      return 1;
    }
    
    error_code ec;
    
    // 检查当前状态
    std::cout << "\n检查机器人当前状态..." << std::endl;
    auto current_mode = robot.operateMode(ec);
    if (!ec) {
      std::cout << "  当前操作模式: " << current_mode << std::endl;
    }
    
    auto current_power = robot.powerState(ec);
    if (!ec) {
      std::cout << "  当前电源状态: ";
      using PS = rokae::PowerState;
      switch(current_power) {
        case PS::on: std::cout << "上电"; break;
        case PS::off: std::cout << "下电"; break;
        case PS::estop: std::cout << "急停"; break;
        case PS::gstop: std::cout << "安全门打开"; break;
        default: std::cout << "未知"; break;
      }
      std::cout << std::endl;
    }
    
    auto current_op_state = robot.operationState(ec);
    if (!ec) {
      std::cout << "  当前运行状态: " << current_op_state << std::endl;
    }
    
    // 如果机器人处于实时模式控制中，需要先退出实时模式
    if (current_op_state == OperationState::rtControlling) {
      std::cout << "\n检测到机器人处于实时模式控制中，正在退出实时模式..." << std::endl;
      robot.setMotionControlMode(MotionControlMode::Idle, ec);
      if (ec) {
        print(std::cerr, "退出实时模式失败:", ec);
        return 1;
      }
      std::cout << "已退出实时模式，等待机器人状态变为空闲..." << std::endl;
      // 等待机器人状态变为空闲
      for (int i = 0; i < 50; i++) {  // 最多等待5秒
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        current_op_state = robot.operationState(ec);
        if (current_op_state == OperationState::idle || current_op_state == OperationState::unknown) {
          std::cout << "机器人状态已变为空闲" << std::endl;
          break;
        }
      }
    }
    
    std::cout << "\n设置操作模式为自动模式..." << std::endl;
    robot.setOperateMode(OperateMode::automatic, ec);
    if (ec) {
      print(std::cerr, "设置操作模式失败:", ec);
      return 1;
    }
    std::cout << "操作模式设置成功！" << std::endl;
    
    std::cout << "设置运动控制模式为实时模式..." << std::endl;
    robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
    if (ec) {
      print(std::cerr, "设置实时模式失败:", ec);
      return 1;
    }
    std::cout << "实时模式设置成功！" << std::endl;
    
    std::cout << "机器人上电..." << std::endl;
    robot.setPowerState(true, ec);
    if (ec) {
      print(std::cerr, "上电失败:", ec);
      return 1;
    }
    std::cout << "上电成功！" << std::endl;
    
    // 等待上电状态稳定
    std::cout << "等待上电状态稳定..." << std::endl;
    for (int i = 0; i < 20; i++) {  // 最多等待2秒
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      auto power_state = robot.powerState(ec);
      if (!ec && power_state == PowerState::on) {
        std::cout << "上电状态已稳定" << std::endl;
        break;
      }
      if (i == 19) {
        std::cout << "警告: 等待上电状态超时，但继续尝试" << std::endl;
      }
    }
    
    // 等待机器人状态变为idle，确保可以开始实时控制
   
    
    try {
      std::cout << "\n========================================" << std::endl;
      std::cout << "开始力矩控制..." << std::endl;
      std::cout << "========================================" << std::endl;
      torqueControl(robot);
      std::cout << "\n========================================" << std::endl;
      std::cout << "力矩控制结束" << std::endl;
      std::cout << "========================================" << std::endl;
    } catch (const RealtimeMotionException &e) {
      print(std::cerr, "实时运动异常:", e.what());
      // 发生错误, 切换回非实时模式
      robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    }

    std::cout << "\n切换回非实时模式..." << std::endl;
    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    std::cout << "设置操作模式为手动模式..." << std::endl;
    robot.setOperateMode(OperateMode::manual, ec);
    std::cout << "程序执行完成" << std::endl;

  } catch (const std::exception &e) {
    print(std::cerr, "发生异常:", e.what());
    return 1;
  }
  return 0;
}