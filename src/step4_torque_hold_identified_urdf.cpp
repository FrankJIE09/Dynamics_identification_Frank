/**
 * @file step4_torque_hold_identified_urdf.cpp
 * @brief Step4：用辨识 URDF + Pinocchio 计算动力学力矩，关节力矩控制使机械臂保持当前关节角
 *
 * 用法: ./step4_torque_hold_identified_urdf [URDF路径]
 * 默认 URDF: AR5-5_07R-W4C4A2_identified_joint.urdf（需在当前目录或通过参数指定）
 * 控制律（WBC QP 形式）：
 *   任务层：期望关节加速度 qdd_des = Kp*(q_d - q) - Kv*dq
 *   逆动力学：tau_des = rnea(q, dq, qdd_des) = M*qdd_des + h
 *   力矩约束 QP：min (1/2)||tau - tau_des||^2  s.t.  tau_min <= tau <= tau_max  → 解为 clip(tau_des)
 *   q_d 为启动时记录的关节角
 *
 * 需连接真实机器人，编译选项 XCORE_USE_XMATE_MODEL=ON，并链接 Pinocchio
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>

#include "rokae/robot.h"
#include "rokae/utility.h"
#include "../print_helper.hpp"
#include "Eigen/Dense"
// 先包含 Eigen Tensor，避免 Pinocchio 头文件中使用 Eigen::Tensor 时未定义（ROS Pinocchio 依赖 Tensor）
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef __has_include
  #if __has_include("pinocchio/multibody/model.hpp")
    #include "pinocchio/multibody/model.hpp"
    #include "pinocchio/multibody/data.hpp"
    #include "pinocchio/parsers/urdf.hpp"
    #include "pinocchio/algorithm/rnea.hpp"
    #include "pinocchio/algorithm/crba.hpp"
    #define PINOCCHIO_AVAILABLE
  #endif
#endif

using namespace rokae;

static std::string detectLocalIp(const std::string& robot_ip) {
  size_t last_dot = robot_ip.find_last_of('.');
  if (last_dot == std::string::npos) return "";
  std::string network_prefix = robot_ip.substr(0, last_dot);
  std::string cmd = "ip addr show | grep -E 'inet " + network_prefix + "\\.' | head -1 | awk '{print $2}' | cut -d'/' -f1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "";
  char buffer[128];
  std::string result;
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result = buffer;
    if (!result.empty() && result.back() == '\n') result.pop_back();
  }
  pclose(pipe);
  return result;
}

/** Step4 控制参数，可从 config/step4_torque_hold.yaml 加载；Kp/Kv 每轴独立；q_d 为期望关节角(rad) */
struct Step4Config {
  std::string urdf_file;  // 若在 yaml 中设置则优先使用
  std::array<double, 7> q_d_rad = {
    4.643*M_PI/180., -47.964*M_PI/180., 9.252*M_PI/180., 84.650*M_PI/180.,
    7.803*M_PI/180., 21.312*M_PI/180., -39.733*M_PI/180.
  };
  std::array<double, 7> Kp = {50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0};
  std::array<double, 7> Kv = {14.14, 14.14, 14.14, 14.14, 14.14, 14.14, 14.14};
  std::array<double, 7> pos_deadzone = {0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002};  // 每关节独立 (rad)
  double run_seconds = 60.0;
  std::array<double, 7> torque_limits = {85.0, 85.0, 85.0, 36.0, 36.0, 36.0, 36.0};
  std::array<double, 7> qdd_max = {100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0};  // 每关节期望加速度限幅 (rad/s^2)
  int print_interval = 100;     // 每 N 个控制周期打印一次，0 表示不打印
};

/** 简易解析 YAML：支持 "key: value" 与 "key: [v1,v2,...]"，忽略 # 注释 */
static Step4Config loadStep4Config(const std::string& config_path) {
  Step4Config c;
  std::ifstream f(config_path);
  if (!f.good()) return c;
  std::string line;
  while (std::getline(f, line)) {
    auto comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i >= line.size()) continue;
    size_t colon = line.find(':', i);
    if (colon == std::string::npos) continue;
    std::string key(line, i, colon - i);
    while (key.size() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    size_t val_start = colon + 1;
    while (val_start < line.size() && (line[val_start] == ' ' || line[val_start] == '\t')) val_start++;
    auto parse_array7 = [&line, val_start](std::array<double, 7>& arr) {
      std::string rest(line, val_start);
      size_t br = rest.find('[');
      size_t br2 = rest.find(']');
      if (br == std::string::npos || br2 == std::string::npos) return false;
      std::istringstream ss(rest.substr(br + 1, br2 - br - 1));
      double v;
      int idx = 0;
      while (ss >> v && idx < 7) { arr[idx++] = v; if (ss.peek() == ',') ss.ignore(); }
      return idx > 0;
    };
    if (key == "torque_limits") { parse_array7(c.torque_limits); continue; }
    if (key == "q_d_deg") {
      std::array<double, 7> deg;
      if (line.find('[', val_start) != std::string::npos && parse_array7(deg))
        for (int i = 0; i < 7; i++) c.q_d_rad[i] = deg[i] * M_PI / 180.0;
      continue;
    }
    if (key == "Kp") {
      if (line.find('[', val_start) != std::string::npos) parse_array7(c.Kp);
      else { double v = 0; if (std::istringstream(line.substr(val_start)) >> v) c.Kp.fill(v); }
      continue;
    }
    if (key == "Kv") {
      if (line.find('[', val_start) != std::string::npos) parse_array7(c.Kv);
      else { double v = 0; if (std::istringstream(line.substr(val_start)) >> v) c.Kv.fill(v); }
      continue;
    }
    if (key == "urdf") {
      std::string s(line, val_start);
      while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
      c.urdf_file = s;
      continue;
    }
    if (key == "pos_deadzone") {
      if (line.find('[', val_start) != std::string::npos) parse_array7(c.pos_deadzone);
      else { double v = 0; if (std::istringstream(line.substr(val_start)) >> v) c.pos_deadzone.fill(v); }
      continue;
    }
    if (key == "qdd_max") {
      if (line.find('[', val_start) != std::string::npos) parse_array7(c.qdd_max);
      else { double v = 0; if (std::istringstream(line.substr(val_start)) >> v) c.qdd_max.fill(v); }
      continue;
    }
    if (key == "print_interval") {
      int ival = 0;
      if (val_start < line.size() && (std::istringstream(line.substr(val_start)) >> ival)) c.print_interval = ival;
      continue;
    }
    double val = 0;
    if (val_start < line.size() && (std::istringstream(line.substr(val_start)) >> val)) {
      if (key == "run_seconds") c.run_seconds = val;
    }
  }
  return c;
}

/** 解析可执行文件所在目录（用于定位 config） */
static std::string dirnameOf(const std::string& path) {
  size_t p = path.find_last_of("/\\");
  return p == std::string::npos ? "" : path.substr(0, p);
}

/** 从 URDF 解析各关节的 dynamics damping(Fv)、friction(Fc)，顺序按 joint_1..joint_7 → index 0..6 */
static void parseUrdfFriction(const std::string& urdf_path,
                             std::array<double, 7>& f_viscous,
                             std::array<double, 7>& f_coulomb) {
  for (int i = 0; i < 7; i++) { f_viscous[i] = 0; f_coulomb[i] = 0; }
  std::ifstream f(urdf_path);
  if (!f.good()) return;
  std::string line;
  int current_joint_index = -1;
  auto extract_attr = [](const std::string& s, const char* attr, double& out) {
    std::string key = std::string(attr) + "=\"";
    size_t pos = s.find(key);
    if (pos == std::string::npos) return false;
    pos += key.size();
    size_t end = s.find('"', pos);
    if (end == std::string::npos) return false;
    return (std::istringstream(s.substr(pos, end - pos)) >> out).good();
  };
  while (std::getline(f, line)) {
    if (line.find("<joint ") != std::string::npos) {
      size_t j = line.find("joint_");
      if (j != std::string::npos) {
        int num = 0;
        if (std::sscanf(line.c_str() + j, "joint_%d", &num) == 1 && num >= 1 && num <= 7)
          current_joint_index = num - 1;
      }
    }
    if (line.find("<dynamics") != std::string::npos && current_joint_index >= 0) {
      double d = 0, fr = 0;
      if (extract_attr(line, "damping", d) && extract_attr(line, "friction", fr)) {
        f_viscous[current_joint_index] = d;
        f_coulomb[current_joint_index] = fr;
      }
      current_joint_index = -1;
    }
  }
}

int main(int argc, char* argv[]) {
#ifdef PINOCCHIO_AVAILABLE
  std::string config_path = "config/step4_torque_hold.yaml";
  if (argc > 0 && argv[0]) {
    std::string exe_dir = dirnameOf(argv[0]);
    if (!exe_dir.empty()) {
      std::ifstream check_exe(config_path);
      if (!check_exe.good()) config_path = exe_dir + "/config/step4_torque_hold.yaml";
    }
  }
  Step4Config cfg_early = loadStep4Config(config_path);

  std::string urdf_file = "AR5-5_07R-W4C4A2_identified_joint.urdf";
  if (!cfg_early.urdf_file.empty())
    urdf_file = cfg_early.urdf_file;
  else if (argc > 1)
    urdf_file = argv[1];

  auto file_exists = [](const std::string& path) {
    std::ifstream f(path);
    return f.good();
  };
  if (!file_exists(urdf_file)) {
    if (argc > 0 && argv[0]) {
      std::string exe_dir = dirnameOf(argv[0]);
      if (!exe_dir.empty() && urdf_file.find('/') != std::string::npos) {
        std::string alt = exe_dir + "/../../" + urdf_file;
        if (file_exists(alt)) urdf_file = alt;
      }
    }
  }
  if (!file_exists(urdf_file)) {
    std::cerr << "错误: URDF 文件不存在: " << urdf_file << std::endl;
    std::cerr << "  用法: " << argv[0] << " [URDF路径]，或在 config 中设置 urdf:" << std::endl;
    return 1;
  }

  pinocchio::Model model;
  pinocchio::Data data;
  pinocchio::urdf::buildModel(urdf_file, model);
  Eigen::Matrix3d R_y = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Matrix3d R_z = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  model.gravity.linear() = (R_z * R_y).transpose() * Eigen::Vector3d(0, 0, -9.81);
  data = pinocchio::Data(model);

  std::array<double, 7> f_viscous{}, f_coulomb{};
  parseUrdfFriction(urdf_file, f_viscous, f_coulomb);

  std::cout << "========================================" << std::endl;
  std::cout << "Step4 力矩保持：辨识URDF(Pinocchio) 关节力矩控制" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "URDF: " << urdf_file << std::endl;

  std::string robot_ip = "10.17.0.110";
  std::string local_ip = detectLocalIp(robot_ip);
  if (local_ip.empty()) {
    std::cerr << "无法自动检测本机IP，请与机器人同网段" << std::endl;
    return 1;
  }
  std::cout << "机器人IP: " << robot_ip << ", 本机IP: " << local_ip << std::endl;

  error_code ec;
  xMateErProRobot robot;
  try {
    robot.connectToRobot(robot_ip, local_ip);
  } catch (const std::exception& e) {
    std::cerr << "连接失败: " << e.what() << std::endl;
    return 1;
  }

  robot.setOperateMode(OperateMode::automatic, ec);
  if (ec) {
    print(std::cerr, "设置自动模式失败:", ec);
    return 1;
  }
  if (robot.operationState(ec) == OperationState::rtControlling) {
    robot.setMotionControlMode(MotionControlMode::Idle, ec);
    if (ec) {
      print(std::cerr, "退出实时模式失败:", ec);
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  // 与 torque_control.cpp 一致：先设网络丢包阈值，再切实时模式，再上电
  robot.setRtNetworkTolerance(20, ec);
  if (ec) {
    print(std::cerr, "设置网络丢包阈值失败:", ec);
    return 1;
  }
  robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
  if (ec) {
    print(std::cerr, "设置实时指令模式失败:", ec);
    return 1;
  }
  std::cout << "机器人上电..." << std::endl;
  robot.setPowerState(true, ec);
  if (ec) {
    print(std::cerr, "上电失败:", ec);
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  {
    auto power_state = robot.powerState(ec);
    if (ec) {
      print(std::cerr, "查询电源状态失败:", ec);
      return 1;
    }
    using PS = rokae::PowerState;
    if (power_state == PS::estop) {
      std::cerr << "当前为急停状态，请先解除急停。" << std::endl;
      return 1;
    }
    if (power_state == PS::gstop) {
      std::cerr << "安全门已打开，请关闭安全门。" << std::endl;
      return 1;
    }
    if (power_state != PS::on) {
      std::cout << "上电未就绪，等待 1s 后再次上电..." << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      robot.setPowerState(true, ec);
      if (ec) {
        print(std::cerr, "再次上电失败:", ec);
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      power_state = robot.powerState(ec);
      if (ec) {
        print(std::cerr, "查询电源状态失败:", ec);
        return 1;
      }
      if (power_state != PS::on) {
        std::cerr << "再次上电后仍异常，请确认示教器允许上电。" << std::endl;
        return 1;
      }
      std::cout << "再次上电成功。" << std::endl;
    } else {
      std::cout << "上电成功。" << std::endl;
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  using namespace RtSupportedFields;
  auto rtCon = robot.getRtMotionController().lock();
  robot.stopReceiveRobotState();
  robot.startReceiveRobotState(std::chrono::milliseconds(1),
                               {jointPos_m, jointVel_m});

  // 先运动到目标位置（与 torque_control.cpp 中 q_drag 一致）
  const std::array<double, 7> q_target_rad = Utils::degToRad(std::array<double, 7>({
    -2.458,   // 一轴
    -70.651,  // 二轴
    -2.044,   // 三轴
    118.492,  // 四轴
    4.682,    // 五轴
    -47.719,  // 六轴
    -5.433    // 七轴
  }));
  std::cout << "  运动到目标位置（与 torque_control 一致）..." << std::endl;
  auto current_pos = robot.jointPos(ec);
  if (ec) {
    print(std::cerr, "  获取当前位置失败:", ec);
    return 1;
  }
  std::cout << "  当前位置（度）: [";
  for (size_t i = 0; i < current_pos.size(); i++) {
    std::cout << (i ? ", " : "") << (current_pos[i] * 180.0 / M_PI);
  }
  std::cout << "]" << std::endl;
  std::cout << "  目标角度（度）: [";
  for (size_t i = 0; i < q_target_rad.size(); i++) {
    std::cout << (i ? ", " : "") << (q_target_rad[i] * 180.0 / M_PI);
  }
  std::cout << "]" << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  rtCon->MoveJ(0.2, robot.jointPos(ec), q_target_rad);
  std::cout << "  已到达目标位置" << std::endl;

  rtCon->setFilterFrequency(10.0, 10.0, 10.0, ec);
  if (ec) {
    print(std::cerr, "设置滤波失败:", ec);
    return 1;
  }

  try {
    rtCon->startMove(RtControllerMode::torque);
  } catch (const std::exception& e) {
    std::cerr << "启动力矩控制失败: " << e.what() << std::endl;
    std::cerr << "  提示: 确认电机已上电、实时指令模式 (RtCommand)，且示教器未占用控制权。" << std::endl;
    return 1;
  }

  // 使用程序开头已确定的 config_path 再次加载（供回调使用）
  Step4Config cfg = loadStep4Config(config_path);
  std::cout << "  配置: " << config_path << " 死区(rad)=[";
  for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << cfg.pos_deadzone[i];
  std::cout << "] "
            << " run_seconds=" << cfg.run_seconds << std::endl;
  std::cout << "  期望关节角 (度): [";
  for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << (cfg.q_d_rad[i] * 180.0 / M_PI);
  std::cout << "]" << std::endl;
  std::cout << "  Kp: [";
  for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << cfg.Kp[i];
  std::cout << "]" << std::endl;
  std::cout << "  Kv: [";
  for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << cfg.Kv[i];
  std::cout << "]" << std::endl;

  const double control_dt = 0.001;  // 控制周期 1ms，用于自算 dq
  std::function<Torque(void)> callback = [&, cfg, control_dt]() {
    static double time = 0;
    time += control_dt;
    if (time > cfg.run_seconds) {
      Torque cmd(7);
      cmd.setFinished();
      return cmd;
    }

    std::array<double, 7> q{}, dq_robot{};
    robot.getStateData(jointPos_m, q);
    robot.getStateData(jointVel_m, dq_robot);

    // 自算 dq：dq_self = (q - q_prev) / dt，控制中只使用 dq_self；dq_robot 仅用于打印对比
    static std::array<double, 7> q_prev = {};
    static bool first_call = true;
    std::array<double, 7> dq_self{};
    if (first_call) {
      q_prev = q;
      first_call = false;
      dq_self = {};
    } else {
      for (int i = 0; i < 7; i++)
        dq_self[i] = (q[i] - q_prev[i]) / control_dt;
      q_prev = q;
    }

    // WBC：任务层期望加速度 (PD)，q_d 来自配置；位置误差带死区以减轻抖动（使用自算 dq_self）
    std::array<double, 7> eq_rad{};       // 位置误差 (rad)，死区前
    std::array<double, 7> eq_after_dz{};   // 死区后，用于 PD
    Eigen::VectorXd qdd_des(7), qdd_des_raw(7);
    for (int i = 0; i < 7; i++) {
      double eq = cfg.q_d_rad[i] - q[i];
      eq_rad[i] = eq;
      // 软死区：|eq|<=dz 时置 0；|eq|>dz 时用 eq_eff = sign(eq)*(|eq|-dz)，在边界连续、无突变，减轻抖动
      const double dz = cfg.pos_deadzone[i];
      if (std::abs(eq) <= dz)
        eq_after_dz[i] = 0.0;
      else
        eq_after_dz[i] = (eq > 0 ? 1.0 : -1.0) * (std::abs(eq) - dz);
      qdd_des_raw(i) = cfg.Kp[i] * eq_after_dz[i] - cfg.Kv[i] * dq_self[i];
      qdd_des(i) = qdd_des_raw(i);
      if (qdd_des(i) > cfg.qdd_max[i]) qdd_des(i) = cfg.qdd_max[i];
      else if (qdd_des(i) < -cfg.qdd_max[i]) qdd_des(i) = -cfg.qdd_max[i];
    }

    Eigen::VectorXd q_e(7), v_e(7);
    for (int i = 0; i < 7; i++) {
      q_e(i) = q[i];
      v_e(i) = dq_self[i];  // 逆动力学与摩擦力均使用自算 dq_self
    }
    // 质量矩阵 M(q)
    pinocchio::crba(model, data, q_e);
    Eigen::MatrixXd M_full = data.M.triangularView<Eigen::Upper>();
    M_full += M_full.transpose();
    M_full.diagonal() *= 0.5;
    // 重力项 g(q) = rnea(q, 0, 0)
    Eigen::VectorXd zero_v = Eigen::VectorXd::Zero(7);
    Eigen::VectorXd zero_a = Eigen::VectorXd::Zero(7);
    pinocchio::rnea(model, data, q_e, zero_v, zero_a);
    Eigen::VectorXd g_vec = data.tau;
    // 逆动力学 tau_des = M*qdd_des + h = rnea(q, dq, qdd_des)
    pinocchio::rnea(model, data, q_e, v_e, qdd_des);
    const Eigen::VectorXd& tau_des = data.tau;
    Eigen::VectorXd M_qdd_des = M_full * qdd_des;  // 惯性项 M*qdd_des (Nm)
    Eigen::VectorXd h_vec = tau_des - M_qdd_des;
    Eigen::VectorXd c_vec = h_vec - g_vec;  // 纯科氏/离心 h - g(q)
    // 摩擦力项（辨识 URDF 中的 damping=Fv, friction=Fc）：tau_fric = Fv*dq + Fc*sign(dq)
    Eigen::VectorXd tau_fric(7);
    for (int i = 0; i < 7; i++) {
      double dqi = v_e(i);
      double sign_dq = (dqi > 1e-9) ? 1.0 : (dqi < -1e-9) ? -1.0 : 0.0;
      tau_fric(i) = f_viscous[i] * dqi + f_coulomb[i] * sign_dq;
    }

    // 期望力矩 = 刚体逆动力学 + 摩擦力补偿（辨识 URDF 的 Fv,Fc）
    // QP: min (1/2)||tau - (tau_des + tau_fric)||^2  s.t.  tau_min <= tau <= tau_max  → 解为限幅
    Torque cmd(7);
    for (int i = 0; i < 7; i++) {
      double t = tau_des(i) + tau_fric(i);
      if (t > cfg.torque_limits[i]) t = cfg.torque_limits[i];
      else if (t < -cfg.torque_limits[i]) t = -cfg.torque_limits[i];
      cmd.tau[i] = t;
    }
    // Debug：按 print_interval 周期打印（0 表示不打印）
    static int print_count = 0;
    if (cfg.print_interval > 0 && (print_count++ % cfg.print_interval) == 0) {
      std::cout << "\n==================" << std::endl;
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  Kp          = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << cfg.Kp[i];
      std::cout << "]" << std::endl;
      std::cout << "  Kv          = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << cfg.Kv[i];
      std::cout << "]" << std::endl;
      std::cout << "  eq(deg)     = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << (eq_after_dz[i] * 180.0 / M_PI);
      std::cout << "]  (PD用)" << std::endl;
      std::cout << "  dq_robot(deg/s) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << (dq_robot[i] * 180.0 / M_PI);
      std::cout << "]  (仅对比)" << std::endl;
      std::cout << "  dq_self(deg/s)  = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << (dq_self[i] * 180.0 / M_PI);
      std::cout << "]  (用于控制)" << std::endl;
      std::cout << "  q_e(deg)    = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << (q_e(i) * 180.0 / M_PI);
      std::cout << "]" << std::endl;
      std::cout << "  v_e(deg/s)  = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << (v_e(i) * 180.0 / M_PI);
      std::cout << "]" << std::endl;
      std::cout << "  qdd_des(限幅前,rad/s^2) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << qdd_des_raw(i);
      std::cout << "]" << std::endl;
      std::cout << "  qdd_des(限幅后,rad/s^2) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << qdd_des(i);
      std::cout << "]" << std::endl;
      std::cout << "  tau         = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << cmd.tau[i];
      std::cout << "]" << std::endl;
      std::cout << "  M (质量矩阵, kg·m^2):" << std::endl;
      for (int i = 0; i < 7; i++) {
        std::cout << "    [";
        for (int j = 0; j < 7; j++) std::cout << (j ? ", " : "") << M_full(i, j);
        std::cout << "]" << std::endl;
      }
      std::cout << "  M*qdd_des (惯性项, Nm) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << M_qdd_des(i);
      std::cout << "]" << std::endl;
      std::cout << "  科氏/离心项 h-g(q) (Nm) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << c_vec(i);
      std::cout << "]" << std::endl;
      std::cout << "  g(q) (重力项, Nm) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << g_vec(i);
      std::cout << "]" << std::endl;
      std::cout << "  摩擦力项 Fv*dq+Fc*sign(dq) (Nm) = [";
      for (int i = 0; i < 7; i++) std::cout << (i ? ", " : "") << tau_fric(i);
      std::cout << "]" << std::endl;
      std::cout << "==================" << std::endl;
      std::cout.flush();
    }
    return cmd;
  };

  std::cout << "  力矩控制已启动（WBC QP），保持 " << cfg.run_seconds << " 秒" << std::endl;
  rtCon->setControlLoop(callback, 0, true);
  rtCon->startLoop(true);
  std::cout << "  结束" << std::endl;
  return 0;
#else
  (void)argc;
  (void)argv;
  std::cerr << "错误: 未找到 Pinocchio" << std::endl;
  return 1;
#endif
}
