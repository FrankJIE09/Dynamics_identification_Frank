/**
 * @file step2.2_dynamics_parameter_estimation_joint.cpp
 * @brief 动力学参数联合辨识计算程序（惯性参数 + 摩擦参数）
 * 
 * 此程序用于从采集的数据中联合辨识机器人的动力学参数：
 * 1. 从CSV文件读取采集的数据
 * 2. 使用 Pinocchio 计算惯性回归矩阵
 * 3. 手动构造摩擦回归矩阵（粘性摩擦 + 库仑摩擦）
 * 4. 拼接成联合回归矩阵
 * 5. 通过最小二乘法求解联合参数（惯性 + 摩擦）
 * 6. 验证辨识结果
 * 7. 保存辨识结果（包括摩擦参数）
 * 
 * 此示例需要使用xMateModel模型库，请设置编译选项XCORE_USE_XMATE_MODEL=ON
 * 需要 Pinocchio 库支持
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <map>
#include "Eigen/Geometry"
#include "Eigen/Dense"

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
 * @brief 生成新的URDF文件（更新惯性参数和摩擦参数）
 */
void generateIdentifiedURDFJoint(const std::string& original_urdf_file, 
                                  const std::string& output_urdf_file,
                                  const pinocchio::Model& pinocchio_model,
                                  const Eigen::VectorXd& theta_inertial,
                                  const Eigen::VectorXd& f_viscous,
                                  const Eigen::VectorXd& f_coulomb) {
  std::ifstream in_file(original_urdf_file);
  if (!in_file.is_open()) {
    std::cerr << "  警告: 无法打开原始URDF文件: " << original_urdf_file << std::endl;
    return;
  }
  
  // 读取所有行
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in_file, line)) {
    lines.push_back(line);
  }
  in_file.close();
  
  // 与 step2 一致：存储每个 link 的新 inertial（7 个连杆，70 维 theta）
  const int n_inertial_links = std::min(7, static_cast<int>(theta_inertial.size() / 10));
  std::map<std::string, std::string> link_inertial_map;
  for (int j = 0; j < n_inertial_links; ++j) {
    const pinocchio::JointIndex jid = static_cast<pinocchio::JointIndex>(j + 1);
    const int base = 10 * j;
    const auto pi_id = theta_inertial.segment(base, 10);
    const auto inertia_id = pinocchio::Inertia::FromDynamicParameters(pi_id);

    const double m_id = inertia_id.mass();
    const Eigen::Vector3d c_id = inertia_id.lever();
    const Eigen::Matrix3d I_id = inertia_id.inertia();

    std::string joint_name = pinocchio_model.names[jid];  // Pinocchio 的 names[] 是关节名
    std::string link_name = joint_name;
    size_t jpos = joint_name.find("joint_");
    if (jpos != std::string::npos)
      link_name = joint_name.substr(0, jpos) + "link" + joint_name.substr(jpos + 6);  // joint_1 -> link1

    std::ostringstream inertial_oss;
    inertial_oss << "    <inertial>\n";
    inertial_oss << "      <mass value=\"" << std::fixed << std::setprecision(6) << m_id << "\" />\n";
    inertial_oss << "      <inertia ixx=\"" << std::scientific << std::setprecision(10) << I_id(0,0)
                 << "\" ixy=\"" << I_id(0,1) << "\" ixz=\"" << I_id(0,2)
                 << "\" iyy=\"" << I_id(1,1) << "\" iyz=\"" << I_id(1,2)
                 << "\" izz=\"" << I_id(2,2) << "\" />\n";
    inertial_oss << "      <origin rpy=\"0 0 0\" xyz=\"" << std::fixed << std::setprecision(6)
                 << c_id(0) << " " << c_id(1) << " " << c_id(2) << "\" />\n";
    inertial_oss << "    </inertial>\n";
    link_inertial_map[link_name] = inertial_oss.str();
    if (link_name.find("AR5-5") == std::string::npos && link_name.find("link") != std::string::npos) {
      link_inertial_map["AR5-5_07R-W4C4A2_" + link_name] = link_inertial_map[link_name];
    }
  }
  
  // 存储每个joint的新dynamics内容（摩擦参数）
  std::map<std::string, std::string> joint_dynamics_map;
  for (int i = 0; i < 7; i++) {
    std::string joint_name = "AR5-5_07R-W4C4A2_joint_" + std::to_string(i + 1);
    std::ostringstream dynamics_oss;
    dynamics_oss << "    <dynamics damping=\"" << std::fixed << std::setprecision(6) << f_viscous(i)
                 << "\" friction=\"" << f_coulomb(i) << "\" />";
    joint_dynamics_map[joint_name] = dynamics_oss.str();
  }
  
  // 处理每一行，找到并替换inertial和dynamics标签
  std::ofstream out_file(output_urdf_file);
  if (!out_file.is_open()) {
    std::cerr << "  警告: 无法创建输出URDF文件: " << output_urdf_file << std::endl;
    return;
  }
  
  std::string current_link_name;
  std::string current_joint_name;
  bool in_inertial = false;
  bool in_dynamics = false;
  bool inertial_replaced = false;
  bool dynamics_replaced = false;
  bool in_joint = false;
  int inertial_indent = 0;
  int dynamics_indent = 0;
  int joint_indent = 0;
  
  for (size_t i = 0; i < lines.size(); ++i) {
    line = lines[i];
    
    // 查找link开始标签
    size_t link_pos = line.find("<link name=\"");
    if (link_pos != std::string::npos) {
      size_t name_start = link_pos + 12;
      size_t name_end = line.find("\"", name_start);
      if (name_end != std::string::npos) {
        current_link_name = line.substr(name_start, name_end - name_start);
      }
      inertial_replaced = false;
    }
    
    // 查找joint开始标签
    size_t joint_pos = line.find("<joint name=\"");
    if (joint_pos != std::string::npos) {
      size_t name_start = joint_pos + 13;
      size_t name_end = line.find("\"", name_start);
      if (name_end != std::string::npos) {
        current_joint_name = line.substr(name_start, name_end - name_start);
      }
      dynamics_replaced = false;
      in_joint = true;
      joint_indent = 0;
      for (char c : line) {
        if (c == ' ') joint_indent++;
        else break;
      }
    }
    
    // 检查joint是否结束
    if (in_joint && line.find("</joint>") != std::string::npos) {
      // 如果joint结束前还没有添加dynamics，现在添加
      if (!dynamics_replaced && joint_dynamics_map.find(current_joint_name) != joint_dynamics_map.end()) {
        std::string new_dynamics = joint_dynamics_map[current_joint_name];
        std::string indent_str(joint_indent + 2, ' ');
        out_file << indent_str << new_dynamics << "\n";
        dynamics_replaced = true;
      }
      in_joint = false;
    }
    
    // 查找inertial开始标签
    if (line.find("<inertial>") != std::string::npos && !inertial_replaced) {
      inertial_indent = 0;
      for (char c : line) {
        if (c == ' ') inertial_indent++;
        else break;
      }

      auto it = link_inertial_map.find(current_link_name);
      if (it != link_inertial_map.end()) {
        while (i + 1 < lines.size() && lines[i + 1].find("</inertial>") == std::string::npos) {
          i++;
        }
        const std::string& new_inertial = it->second;
        std::string indent_str(inertial_indent, ' ');
        std::istringstream iss(new_inertial);
        std::string new_line;
        while (std::getline(iss, new_line)) {
          if (new_line.length() >= 4 && new_line.substr(0, 4) == "    ") {
            new_line = new_line.substr(4);
          }
          out_file << indent_str << new_line << "\n";
        }
        out_file.flush();
        i++;
        in_inertial = false;
        inertial_replaced = true;
        continue;
      }

      in_inertial = true;
      out_file << line << "\n";
      continue;
    }

    if (in_inertial) {
      out_file << line << "\n";
      if (line.find("</inertial>") != std::string::npos) {
        in_inertial = false;
      }
      continue;
    }

    if (inertial_replaced && line.find("</inertial>") != std::string::npos) {
      continue;
    }

    // 查找dynamics标签（可能在joint内）
    if (line.find("<dynamics") != std::string::npos && !dynamics_replaced) {
      in_dynamics = true;
      dynamics_indent = 0;
      for (char c : line) {
        if (c == ' ') dynamics_indent++;
        else break;
      }
      
      if (joint_dynamics_map.find(current_joint_name) != joint_dynamics_map.end()) {
        // 跳过旧的dynamics标签（可能是自闭合标签或需要找到结束标签）
        if (line.find("/>") == std::string::npos) {
          while (i < lines.size() && lines[i].find("</dynamics>") == std::string::npos) {
            i++;
          }
        }
        // 写入新的dynamics内容
        std::string new_dynamics = joint_dynamics_map[current_joint_name];
        std::string indent_str(dynamics_indent, ' ');
        out_file << indent_str << new_dynamics << "\n";
        in_dynamics = false;
        dynamics_replaced = true;
        continue;
      }
    }
    
    // 检查inertial和dynamics是否结束
    if (in_inertial && line.find("</inertial>") != std::string::npos) {
      in_inertial = false;
    }
    if (in_dynamics && (line.find("</dynamics>") != std::string::npos || 
        (line.find("/>") != std::string::npos && line.find("<dynamics") != std::string::npos))) {
      in_dynamics = false;
    }
    
    // 如果不在需要跳过的块内，写入该行
    if (!in_inertial && !in_dynamics) {
      out_file << line << "\n";
    } else if (in_dynamics && !dynamics_replaced && line.find("/>") != std::string::npos) {
      // 如果是自闭合的dynamics标签且未替换，跳过它
      continue;
    }
  }
  
  out_file.close();
  std::cout << "  已生成新的URDF文件（包含摩擦参数）: " << output_urdf_file << std::endl;
}

/**
 * @brief 平滑的sign函数（用于库仑摩擦回归）
 * @param v 速度值
 * @param epsilon 死区阈值（默认1e-3 rad/s）
 * @return 平滑后的sign值
 */
double sign_smooth(double v, double epsilon = 1e-3) {
    if (std::abs(v) < epsilon) {
        return 0.0;  // 死区
    } else {
        return std::tanh(v / epsilon);  // 平滑sign
    }
}

/**
 * @brief 从CSV文件读取数据
 */
std::vector<DataPoint> loadDataFromCSV(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << filename << std::endl;
        return data;
    }
    
    // 跳过标题行
    std::string line;
    std::getline(file, line);
    
    // 读取数据
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string token;
        DataPoint point;
        
        // 读取时间戳
        std::getline(iss, token, ',');
        point.timestamp = std::stod(token);
        
        // 读取关节位置
        for (int i = 0; i < 7; i++) {
            std::getline(iss, token, ',');
            point.q[i] = std::stod(token);
        }
        
        // 读取关节速度
        for (int i = 0; i < 7; i++) {
            std::getline(iss, token, ',');
            point.dq[i] = std::stod(token);
        }
        
        // 读取关节加速度
        for (int i = 0; i < 7; i++) {
            std::getline(iss, token, ',');
            point.ddq[i] = std::stod(token);
        }
        
        // 读取关节力矩
        for (int i = 0; i < 7; i++) {
            std::getline(iss, token, ',');
            point.tau[i] = std::stod(token);
        }
        
        data.push_back(point);
    }
    
    file.close();
    std::cout << "  从文件读取了 " << data.size() << " 个数据点" << std::endl;
    return data;
}

/**
 * @brief 动力学参数联合辨识主函数（惯性 + 摩擦）
 */
void estimateDynamicsParametersJoint(const std::string& data_file, const std::string& urdf_file) {
#ifdef PINOCCHIO_AVAILABLE
  // 初始化 Pinocchio 模型
  std::cout << "\n  初始化 Pinocchio 动力学模型..." << std::endl;
  
  pinocchio::Model pinocchio_model;
  pinocchio::Data pinocchio_data;
  
  try {
    std::string urdf_xml;
    {
      std::ifstream ifs(urdf_file);
      if (!ifs) throw std::runtime_error("无法打开 URDF 文件: " + urdf_file);
      urdf_xml.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    }
    pinocchio::urdf::buildModelFromXML(urdf_xml, pinocchio_model, false, false);
    std::cout << "  Pinocchio 模型加载成功: " << urdf_file << std::endl;
    std::cout << "  模型自由度: " << pinocchio_model.nq << std::endl;
    
    Eigen::Matrix3d R_y = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d R_z = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d R_base_to_world = R_z * R_y;
    Eigen::Vector3d gravity_world(0, 0, -9.81);
    Eigen::Vector3d gravity_base = R_base_to_world.transpose() * gravity_world;
    pinocchio_model.gravity.linear() = gravity_base;
    std::cout << "  重力方向已设置" << std::endl;
    
    pinocchio_data = pinocchio::Data(pinocchio_model);
  } catch (const std::exception& e) {
    std::cerr << "  错误: Pinocchio 模型加载失败: " << e.what() << std::endl;
    std::cerr << "  提示: 若为 \"body named world\"，可尝试使用 base 为根的 URDF，或从源码编译 Pinocchio。" << std::endl;
    return;
  }
#else
  std::cerr << "  错误: Pinocchio 库未找到，无法进行动力学辨识" << std::endl;
  return;
#endif

  // 从CSV文件加载数据
  std::cout << "\n  从文件加载数据: " << data_file << std::endl;
  std::vector<DataPoint> collected_data = loadDataFromCSV(data_file);
  
  if (collected_data.empty()) {
    std::cerr << "  错误: 没有加载到数据" << std::endl;
    return;
  }

  // 构造联合回归矩阵（惯性 + 摩擦）
  std::cout << "\n========================================" << std::endl;
  std::cout << "开始构造联合回归矩阵（惯性 + 摩擦）..." << std::endl;
  std::cout << "========================================" << std::endl;

  int n_joints = 7;
  int n_params_inertial = 0;  // 惯性参数数量（将在第一次计算回归矩阵时确定）
  int n_params_friction = 2 * n_joints;  // 摩擦参数数量：粘性摩擦(7) + 库仑摩擦(7) = 14
  Eigen::VectorXd theta_urdf_inertial;  // URDF先验（惯性参数）
  
#ifdef PINOCCHIO_AVAILABLE
  std::cout << "  使用Pinocchio计算惯性回归矩阵" << std::endl;
  // 先计算一次回归矩阵以确定惯性参数数量
  Eigen::VectorXd q_test(7), dq_test(7), ddq_test(7);
  q_test.setZero();
  dq_test.setZero();
  ddq_test.setZero();
  try {
    // Pinocchio的computeJointTorqueRegressor返回回归矩阵的引用
    auto& Y_test = pinocchio::computeJointTorqueRegressor(
        pinocchio_model, pinocchio_data, q_test, dq_test, ddq_test);
    n_params_inertial = Y_test.cols();
    std::cout << "  Pinocchio惯性回归矩阵参数数量: " << n_params_inertial << std::endl;

    // 由URDF模型构造对应的参数先验 theta_urdf_inertial（每个关节10个惯性参数）
    theta_urdf_inertial.resize(n_params_inertial);
    theta_urdf_inertial.setZero();
    const int expected_params = 10 * static_cast<int>(pinocchio_model.njoints - 1);
    if (n_params_inertial != expected_params) {
      std::cerr << "  警告: n_params_inertial(" << n_params_inertial << ") != 10*(njoints-1)(" << expected_params
                << ")，无法可靠构造URDF先验theta_urdf_inertial，将只做纯最小二乘。" << std::endl;
      theta_urdf_inertial.resize(0);
    } else {
      for (pinocchio::JointIndex jid = 1; jid < pinocchio_model.njoints; ++jid) {
        const auto pi = pinocchio_model.inertias[jid].toDynamicParameters(); // 10x1
        const int base = 10 * static_cast<int>(jid - 1);
        theta_urdf_inertial.segment(base, 10) = pi;
      }
      std::cout << "  已构造URDF先验theta_urdf_inertial（维度 " << theta_urdf_inertial.size() << "）" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "  警告: Pinocchio回归矩阵计算失败: " << e.what() << std::endl;
    std::cerr << "  回退到简化参数集" << std::endl;
    n_params_inertial = 3 * n_joints;  // 21个参数
  }
#else
  // 如果没有Pinocchio，使用简化参数集
  n_params_inertial = 3 * n_joints;  // 21个参数（每个关节3个：惯性、重力、科氏力）
  std::cout << "  警告: Pinocchio不可用，使用简化参数集" << std::endl;
#endif
  
  // 联合参数维度：惯性参数 + 摩擦参数
  int n_params_aug = n_params_inertial + n_params_friction;
  std::cout << "  联合参数维度: " << n_params_aug << " = " << n_params_inertial 
            << " (惯性) + " << n_params_friction << " (摩擦)" << std::endl;
  
  // 构造联合先验（惯性先验 + 摩擦先验，摩擦先验设为0）
  Eigen::VectorXd theta_urdf_aug(n_params_aug);
  theta_urdf_aug.setZero();
  if (theta_urdf_inertial.size() == n_params_inertial) {
    theta_urdf_aug.head(n_params_inertial) = theta_urdf_inertial;
    // 摩擦先验设为0（未知）
    std::cout << "  已构造联合先验theta_urdf_aug（惯性来自URDF，摩擦先验为0）" << std::endl;
  } else {
    theta_urdf_aug.resize(0);  // 如果无法构造惯性先验，则联合先验也为空
  }
  
  Eigen::MatrixXd Y_aug_all;
  Eigen::VectorXd tau_all;
  
  std::cout << "  处理 " << collected_data.size() << " 个数据点..." << std::endl;
  
  int processed = 0;
  const double epsilon_sign = 1e-3;  // sign函数平滑阈值
  
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
    
    // 1. 使用Pinocchio计算惯性回归矩阵
    Eigen::MatrixXd Y_inertial(n_joints, n_params_inertial);
    Y_inertial.setZero();
    
#ifdef PINOCCHIO_AVAILABLE
    try {
      // 使用Pinocchio的回归矩阵计算函数（返回回归矩阵的引用）
      auto& Y_ref = pinocchio::computeJointTorqueRegressor(
          pinocchio_model, pinocchio_data, q_eigen, dq_eigen, ddq_eigen);
      Y_inertial = Y_ref;  // 复制回归矩阵
    } catch (const std::exception& e) {
      std::cerr << "  警告: Pinocchio回归矩阵计算失败: " << e.what() << std::endl;
      std::cerr << "  跳过该数据点..." << std::endl;
      continue;
    }
#else
    std::cerr << "  错误: 需要Pinocchio库" << std::endl;
    return;
#endif
    
    // 2. 手动构造摩擦回归矩阵
    // 2.1 粘性摩擦回归矩阵：Y_viscous = diag(dq)
    Eigen::MatrixXd Y_viscous = dq_eigen.asDiagonal();  // 7×7
    
    // 2.2 库仑摩擦回归矩阵：Y_coulomb = diag(sign_smooth(dq))
    Eigen::VectorXd sign_dq(7);
    for (int i = 0; i < 7; i++) {
      sign_dq(i) = sign_smooth(dq_eigen(i), epsilon_sign);
    }
    Eigen::MatrixXd Y_coulomb = sign_dq.asDiagonal();  // 7×7
    
    // 3. 拼接成联合回归矩阵：Y_aug = [Y_inertial, Y_viscous, Y_coulomb]
    Eigen::MatrixXd Y_aug(n_joints, n_params_aug);
    Y_aug << Y_inertial, Y_viscous, Y_coulomb;
    
    // 转换为Eigen向量（力矩）
    Eigen::VectorXd tau(7);
    for (int i = 0; i < 7; i++) {
      tau(i) = tau_arr[i];
    }
    
    // 堆叠联合回归矩阵和力矩向量
    if (Y_aug_all.rows() == 0) {
      Y_aug_all = Y_aug;
      tau_all = tau;
    } else {
      // 检查参数数量是否一致
      if (Y_aug_all.cols() != Y_aug.cols()) {
        std::cerr << "  错误: 回归矩阵列数不一致！" << std::endl;
        std::cerr << "  之前: " << Y_aug_all.cols() << ", 现在: " << Y_aug.cols() << std::endl;
        break;
      }
      
      Eigen::MatrixXd Y_new(Y_aug_all.rows() + n_joints, n_params_aug);
      Y_new << Y_aug_all, Y_aug;
      Y_aug_all = Y_new;
      
      Eigen::VectorXd tau_new(tau_all.size() + n_joints);
      tau_new << tau_all, tau;
      tau_all = tau_new;
    }
    
    processed++;
    if (processed % 100 == 0) {
      std::cout << "  已处理: " << processed << " / " << collected_data.size() << std::endl;
    }
  }
  
  std::cout << "\n  联合回归矩阵构造完成" << std::endl;
  std::cout << "  回归矩阵维度: " << Y_aug_all.rows() << " x " << Y_aug_all.cols() << std::endl;
  std::cout << "  力矩向量维度: " << tau_all.size() << std::endl;

  // 求解联合参数
  std::cout << "\n========================================" << std::endl;
  std::cout << "开始求解联合动力学参数（惯性 + 摩擦）..." << std::endl;
  std::cout << "========================================" << std::endl;

  // 使用 SVD 求解（更稳定）
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      Y_aug_all, Eigen::ComputeThinU | Eigen::ComputeThinV);
  
  // 设置奇异值阈值
  double threshold = 1e-6;
  Eigen::VectorXd singular_values = svd.singularValues();
  int rank = 0;
  for (int i = 0; i < singular_values.size(); i++) {
    if (singular_values(i) > threshold * singular_values(0)) {
      rank++;
    }
  }
  std::cout << "  联合回归矩阵的秩: " << rank << " / " << n_params_aug << std::endl;
  
  if (rank < n_params_aug) {
    std::cerr << "  警告: 联合回归矩阵不满秩，某些参数可能无法辨识" << std::endl;
    std::cerr << "  建议: 增加激励轨迹的多样性（特别是速度激励）或延长采集时间" << std::endl;
  }

  // 求解参数（两种解都给出：纯最小二乘 + 带URDF先验的正则化最小二乘）
  Eigen::VectorXd theta_aug_ls = svd.solve(tau_all);  // 最小范数最小二乘解
  Eigen::VectorXd theta_aug_estimated = theta_aug_ls;

  double lambda_rel = 1e-3; // 相对正则强度（越大越贴近URDF）
  double lambda_friction_ratio = 0.1; // 摩擦参数的正则化强度相对于惯性参数的比例
  double lambda = 0.0; // 惯性参数的正则化强度
  double lambda_friction = 0.0; // 摩擦参数的正则化强度
  if (theta_urdf_aug.size() == n_params_aug) {
    Eigen::MatrixXd A = Y_aug_all.transpose() * Y_aug_all;
    Eigen::VectorXd b = Y_aug_all.transpose() * tau_all;
    const double traceA = A.trace();
    lambda = (traceA > 1e-12) ? (lambda_rel * traceA / static_cast<double>(n_params_aug)) : lambda_rel;
    lambda_friction = lambda * lambda_friction_ratio; // 摩擦参数使用更小的正则化强度
    
    // 对惯性参数和摩擦参数分别应用不同的正则化强度
    // 惯性参数部分（前n_params_inertial个）
    for (int i = 0; i < n_params_inertial; i++) {
      A(i, i) += lambda;
      b(i) += lambda * theta_urdf_aug(i);
    }
    // 摩擦参数部分（后n_params_friction个）
    for (int i = n_params_inertial; i < n_params_aug; i++) {
      A(i, i) += lambda_friction;
      b(i) += lambda_friction * theta_urdf_aug(i);  // 注意：theta_urdf_aug的摩擦部分为0
    }
    
    theta_aug_estimated = A.ldlt().solve(b);
    std::cout << "  使用URDF先验的Ridge最小二乘（摩擦参数使用更小的正则化强度）:" << std::endl;
    std::cout << "    惯性参数 lambda=" << std::scientific << lambda << std::fixed << std::endl;
    std::cout << "    摩擦参数 lambda_friction=" << std::scientific << lambda_friction 
              << " (ratio=" << lambda_friction_ratio << ")" << std::fixed << std::endl;
  } else {
    std::cout << "  未构造URDF先验theta_urdf_aug，使用纯最小二乘解（SVD）" << std::endl;
  }
  
  // 提取惯性参数和摩擦参数
  Eigen::VectorXd theta_inertial = theta_aug_estimated.head(n_params_inertial);
  Eigen::VectorXd f_viscous = theta_aug_estimated.segment(n_params_inertial, n_joints);
  Eigen::VectorXd f_coulomb = theta_aug_estimated.segment(n_params_inertial + n_joints, n_joints);
  
  std::cout << "\n  辨识的联合动力学参数:" << std::endl;
  std::cout << "  惯性参数数量: " << n_params_inertial << std::endl;
  std::cout << "  摩擦参数数量: " << n_params_friction << " (粘性摩擦: " << n_joints 
            << ", 库仑摩擦: " << n_joints << ")" << std::endl;
  
  std::cout << "\n  粘性摩擦系数 f_v (Nm·s/rad):" << std::endl;
  for (int i = 0; i < n_joints; i++) {
    std::cout << "    关节 " << (i+1) << ": " << std::fixed << std::setprecision(6) 
              << f_viscous(i) << std::endl;
  }
  
  std::cout << "\n  库仑摩擦系数 f_c (Nm):" << std::endl;
  for (int i = 0; i < n_joints; i++) {
    std::cout << "    关节 " << (i+1) << ": " << std::fixed << std::setprecision(6) 
              << f_coulomb(i) << std::endl;
  }
  
  std::cout << "\n  参数求解完成" << std::endl;
  std::cout << "  辨识的联合参数数量: " << theta_aug_estimated.size() << std::endl;

  // 计算拟合误差
  Eigen::VectorXd tau_predicted = Y_aug_all * theta_aug_estimated;
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
  std::ofstream result_file("dynamics_identification_joint_results.txt");
  result_file << "动力学参数联合辨识结果（惯性 + 摩擦）\n";
  result_file << "========================================\n\n";
  result_file << "辨识时间: " << std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
  result_file << "数据点数: " << collected_data.size() << "\n";
  result_file << "惯性参数数量: " << n_params_inertial << "\n";
  result_file << "摩擦参数数量: " << n_params_friction << "\n";
  result_file << "联合参数总数: " << n_params_aug << "\n\n";
  result_file << "误差统计:\n";
  result_file << "  RMSE: " << rmse << " Nm\n";
  result_file << "  最大误差: " << max_error << " Nm\n";
  result_file << "  平均误差: " << mean_error << " Nm\n\n";
  result_file << "求解说明:\n";
  if (theta_urdf_aug.size() == n_params_aug) {
    result_file << "  使用URDF先验的Ridge最小二乘（在不满秩/不可辨识方向上更稳定）\n";
    result_file << "  摩擦参数使用更小的正则化强度，以减少先验为0的影响\n";
    result_file << "  lambda_rel: " << std::scientific << lambda_rel << std::fixed << "\n";
    result_file << "  惯性参数 lambda: " << std::scientific << lambda << std::fixed << "\n";
    result_file << "  摩擦参数 lambda_friction: " << std::scientific << lambda_friction 
                << " (ratio=" << lambda_friction_ratio << ")" << std::fixed << "\n\n";
  } else {
    result_file << "  使用纯最小二乘（SVD最小范数解）\n\n";
  }
  
  result_file << "粘性摩擦系数 f_v (Nm·s/rad):\n";
  for (int i = 0; i < n_joints; i++) {
    result_file << "  关节 " << (i+1) << ": " << std::scientific << std::setprecision(6) 
                << f_viscous(i) << "\n";
  }
  result_file << "\n";
  
  result_file << "库仑摩擦系数 f_c (Nm):\n";
  for (int i = 0; i < n_joints; i++) {
    result_file << "  关节 " << (i+1) << ": " << std::scientific << std::setprecision(6) 
                << f_coulomb(i) << "\n";
  }
  result_file << "\n";
  
  result_file << "惯性参数（前20个）:\n";
  for (int i = 0; i < std::min(n_params_inertial, 20); i++) {
    result_file << "  theta_inertial[" << i << "] = " << std::scientific 
                << std::setprecision(6) << theta_inertial(i) << "\n";
  }
  if (n_params_inertial > 20) {
    result_file << "  ... (共 " << n_params_inertial << " 个惯性参数)\n";
  }
  result_file.close();
  std::cout << "  结果已保存到: dynamics_identification_joint_results.txt" << std::endl;

  // 保存参数到CSV（便于后续使用）
  std::ofstream param_file("dynamics_parameters_joint.csv");
  param_file << "parameter_type,parameter_index,value\n";
  // 惯性参数
  for (int i = 0; i < theta_inertial.size(); i++) {
    param_file << "inertial," << i << "," << std::scientific << std::setprecision(10) 
               << theta_inertial(i) << "\n";
  }
  // 粘性摩擦参数
  for (int i = 0; i < f_viscous.size(); i++) {
    param_file << "friction_viscous," << i << "," << std::scientific << std::setprecision(10) 
               << f_viscous(i) << "\n";
  }
  // 库仑摩擦参数
  for (int i = 0; i < f_coulomb.size(); i++) {
    param_file << "friction_coulomb," << i << "," << std::scientific << std::setprecision(10) 
               << f_coulomb(i) << "\n";
  }
  param_file.close();
  std::cout << "  参数已保存到: dynamics_parameters_joint.csv" << std::endl;

  // 额外保存：纯最小二乘解，便于对比
  Eigen::VectorXd theta_inertial_ls = theta_aug_ls.head(n_params_inertial);
  Eigen::VectorXd f_viscous_ls = theta_aug_ls.segment(n_params_inertial, n_joints);
  Eigen::VectorXd f_coulomb_ls = theta_aug_ls.segment(n_params_inertial + n_joints, n_joints);
  
  std::ofstream param_file_ls("dynamics_parameters_joint_ls.csv");
  param_file_ls << "parameter_type,parameter_index,value\n";
  for (int i = 0; i < theta_inertial_ls.size(); i++) {
    param_file_ls << "inertial," << i << "," << std::scientific << std::setprecision(10)
                  << theta_inertial_ls(i) << "\n";
  }
  for (int i = 0; i < f_viscous_ls.size(); i++) {
    param_file_ls << "friction_viscous," << i << "," << std::scientific << std::setprecision(10)
                  << f_viscous_ls(i) << "\n";
  }
  for (int i = 0; i < f_coulomb_ls.size(); i++) {
    param_file_ls << "friction_coulomb," << i << "," << std::scientific << std::setprecision(10)
                  << f_coulomb_ls(i) << "\n";
  }
  param_file_ls.close();
  std::cout << "  纯最小二乘参数已保存到: dynamics_parameters_joint_ls.csv" << std::endl;

#ifdef PINOCCHIO_AVAILABLE
  // 输出“辨识后的物理参数”（只有当n_params_inertial=10*(njoints-1)时才能从theta重建每个关节Inertia）
  if (theta_urdf_inertial.size() == n_params_inertial) {
    std::cout << "\n  输出辨识后的物理参数（与URDF对比）..." << std::endl;
    std::ofstream physical_param_file("dynamics_physical_parameters_joint_identified.txt");
    physical_param_file << "辨识后的物理参数（由theta重建，与URDF对比）\n";
    physical_param_file << "========================================\n\n";
    physical_param_file << "说明:\n";
    physical_param_file << "1) 本次Y_aug的秩可能小于参数数(n_params_aug)，因此存在不可辨识方向。\n";
    physical_param_file << "2) 这里采用URDF先验的Ridge最小二乘，在不可辨识方向上更贴近URDF。\n";
    physical_param_file << "3) theta_inertial块(每关节10维)用 pinocchio::Inertia::FromDynamicParameters 重建质量/质心/惯性。\n";
    physical_param_file << "4) 摩擦参数直接从theta_aug中提取。\n\n";

    for (pinocchio::JointIndex jid = 1; jid < pinocchio_model.njoints; ++jid) {
      const int base = 10 * static_cast<int>(jid - 1);
      const auto pi_urdf = pinocchio_model.inertias[jid].toDynamicParameters();
      const auto pi_id = theta_inertial.segment(base, 10);
      const auto inertia_id = pinocchio::Inertia::FromDynamicParameters(pi_id);

      const double m_urdf = pinocchio_model.inertias[jid].mass();
      const Eigen::Vector3d c_urdf = pinocchio_model.inertias[jid].lever();
      const Eigen::Matrix3d I_urdf = pinocchio_model.inertias[jid].inertia();

      const double m_id = inertia_id.mass();
      const Eigen::Vector3d c_id = inertia_id.lever();
      const Eigen::Matrix3d I_id = inertia_id.inertia();

      physical_param_file << "关节 " << jid << " (" << pinocchio_model.names[jid] << "):\n";
      physical_param_file << "  URDF 质量(kg): " << std::fixed << std::setprecision(6) << m_urdf
                          << " | 辨识质量(kg): " << m_id << "\n";
      physical_param_file << "  URDF 质心(m): [" << c_urdf(0) << ", " << c_urdf(1) << ", " << c_urdf(2) << "]\n";
      physical_param_file << "  辨识质心(m): [" << c_id(0) << ", " << c_id(1) << ", " << c_id(2) << "]\n";
      physical_param_file << "  URDF 惯性(kg·m²):\n";
      physical_param_file << "    [" << I_urdf(0,0) << ", " << I_urdf(0,1) << ", " << I_urdf(0,2) << "]\n";
      physical_param_file << "    [" << I_urdf(1,0) << ", " << I_urdf(1,1) << ", " << I_urdf(1,2) << "]\n";
      physical_param_file << "    [" << I_urdf(2,0) << ", " << I_urdf(2,1) << ", " << I_urdf(2,2) << "]\n";
      physical_param_file << "  辨识 惯性(kg·m²):\n";
      physical_param_file << "    [" << I_id(0,0) << ", " << I_id(0,1) << ", " << I_id(0,2) << "]\n";
      physical_param_file << "    [" << I_id(1,0) << ", " << I_id(1,1) << ", " << I_id(1,2) << "]\n";
      physical_param_file << "    [" << I_id(2,0) << ", " << I_id(2,1) << ", " << I_id(2,2) << "]\n\n";
    }
    
    // 输出摩擦参数
    physical_param_file << "========================================\n";
    physical_param_file << "摩擦参数:\n";
    physical_param_file << "========================================\n";
    physical_param_file << "粘性摩擦系数 f_v (Nm·s/rad):\n";
    for (int i = 0; i < n_joints; i++) {
      physical_param_file << "  关节 " << (i+1) << ": " << std::fixed << std::setprecision(6) 
                          << f_viscous(i) << "\n";
    }
    physical_param_file << "\n库仑摩擦系数 f_c (Nm):\n";
    for (int i = 0; i < n_joints; i++) {
      physical_param_file << "  关节 " << (i+1) << ": " << std::fixed << std::setprecision(6) 
                          << f_coulomb(i) << "\n";
    }

    // 简单验证：q=0时的预测力矩
    Eigen::VectorXd q0(7), dq0(7), ddq0(7);
    q0.setZero(); dq0.setZero(); ddq0.setZero();
    auto& Y0_inertial = pinocchio::computeJointTorqueRegressor(pinocchio_model, pinocchio_data, q0, dq0, ddq0);
    Eigen::VectorXd sign_dq0(7);
    sign_dq0.setZero();  // q=0, dq=0时，sign(dq)=0
    Eigen::MatrixXd Y0_viscous = dq0.asDiagonal();
    Eigen::MatrixXd Y0_coulomb = sign_dq0.asDiagonal();
    Eigen::MatrixXd Y0_aug(n_joints, n_params_aug);
    Y0_aug << Y0_inertial, Y0_viscous, Y0_coulomb;
    const Eigen::VectorXd tau0 = Y0_aug * theta_aug_estimated;
    physical_param_file << "\n========================================\n";
    physical_param_file << "验证：q=0,dq=0,ddq=0 时预测力矩(Nm):\n";
    physical_param_file << "  [" << std::fixed << std::setprecision(4)
                        << tau0(0) << ", " << tau0(1) << ", " << tau0(2) << ", "
                        << tau0(3) << ", " << tau0(4) << ", " << tau0(5) << ", "
                        << tau0(6) << "]\n";
    physical_param_file.close();
    std::cout << "  已保存: dynamics_physical_parameters_joint_identified.txt" << std::endl;
    
    // 生成新的URDF文件（包含惯性参数和摩擦参数）
    std::cout << "\n  生成新的URDF文件（包含摩擦参数）..." << std::endl;
    std::string output_urdf = "AR5-5_07R-W4C4A2_identified_joint.urdf";
    generateIdentifiedURDFJoint(urdf_file, output_urdf, pinocchio_model, 
                                 theta_inertial, f_viscous, f_coulomb);
  } else {
    std::cout << "\n  跳过物理参数重建：当前theta_inertial不是每关节10维惯性参数形式（n_params_inertial=" 
              << n_params_inertial << "）。" << std::endl;
  }
  
  // 同时更新结果文件，添加物理参数说明
  std::ofstream result_file_append("dynamics_identification_joint_results.txt", std::ios::app);
  result_file_append << "\n========================================\n";
  result_file_append << "物理参数说明:\n";
  result_file_append << "========================================\n";
  result_file_append << "1) 若联合回归矩阵不满秩(rank < n_params_aug)，则存在不可辨识方向，参数解不唯一。\n";
  result_file_append << "2) 本程序可用URDF先验的Ridge最小二乘在不可辨识方向上稳定解（更贴近URDF）。\n";
  result_file_append << "3) 摩擦参数的正则化强度设置为惯性参数的" << lambda_friction_ratio 
                      << "倍，以减少先验为0对摩擦参数辨识的影响。\n";
  if (theta_urdf_inertial.size() == n_params_inertial) {
    result_file_append << "4) 已输出辨识后的质量/质心/惯性到: dynamics_physical_parameters_joint_identified.txt\n";
    result_file_append << "5) 已输出摩擦参数（粘性摩擦 + 库仑摩擦）\n";
  } else {
    result_file_append << "4) 当前theta_inertial不是10维惯性参数堆叠形式，无法重建质量/质心/惯性\n";
  }
  result_file_append.close();
#endif

  // 验证：使用部分数据验证
  std::cout << "\n========================================" << std::endl;
  std::cout << "使用验证数据验证辨识结果..." << std::endl;
  std::cout << "========================================" << std::endl;
  
  // 使用后20%的数据作为验证集
  int validation_start = static_cast<int>(collected_data.size() * 0.8);
  int validation_count = collected_data.size() - validation_start;
  
  Eigen::MatrixXd Y_aug_val;
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
    
    // 使用Pinocchio计算惯性回归矩阵（与训练时相同的方法）
    Eigen::MatrixXd Y_inertial(n_joints, n_params_inertial);
    Y_inertial.setZero();
    
#ifdef PINOCCHIO_AVAILABLE
    try {
      auto& Y_ref = pinocchio::computeJointTorqueRegressor(
          pinocchio_model, pinocchio_data, q_eigen, dq_eigen, ddq_eigen);
      Y_inertial = Y_ref;
    } catch (const std::exception& e) {
      continue;  // 跳过失败的数据点
    }
#endif
    
    // 构造摩擦回归矩阵
    Eigen::MatrixXd Y_viscous = dq_eigen.asDiagonal();
    Eigen::VectorXd sign_dq(7);
    for (int j = 0; j < 7; j++) {
      sign_dq(j) = sign_smooth(dq_eigen(j), epsilon_sign);
    }
    Eigen::MatrixXd Y_coulomb = sign_dq.asDiagonal();
    
    // 拼接联合回归矩阵
    Eigen::MatrixXd Y_aug(n_joints, n_params_aug);
    Y_aug << Y_inertial, Y_viscous, Y_coulomb;
    
    // 转换为Eigen向量（力矩）
    Eigen::VectorXd tau(7);
    for (int j = 0; j < 7; j++) {
      tau(j) = tau_arr[j];
    }
    
    if (Y_aug_val.rows() == 0) {
      Y_aug_val = Y_aug;
      tau_val = tau;
    } else {
      Eigen::MatrixXd Y_new(Y_aug_val.rows() + n_joints, n_params_aug);
      Y_new << Y_aug_val, Y_aug;
      Y_aug_val = Y_new;
      
      Eigen::VectorXd tau_new(tau_val.size() + n_joints);
      tau_new << tau_val, tau;
      tau_val = tau_new;
    }
  }
  
  Eigen::VectorXd tau_val_predicted = Y_aug_val * theta_aug_estimated;
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
  std::cout << "========================================" << std::endl;
  std::cout << "xCore SDK 动力学参数联合辨识计算程序（惯性 + 摩擦）" << std::endl;
  std::cout << "========================================" << std::endl;
  
  // 解析命令行参数
  std::string data_file = "dynamics_identification_data.csv";  // 默认数据文件
  // 默认URDF文件：优先使用编译期注入的绝对路径（Manual_fix），避免运行目录影响
  std::string urdf_file =
#ifdef URDF_FILE_PATH
      URDF_FILE_PATH;
#else
      "AR5-5_07R-W4C4A2_ Manual_fix.urdf";
#endif
  
  if (argc > 1) {
    data_file = argv[1];
  }
  if (argc > 2) {
    urdf_file = argv[2];
  }
  
  std::cout << "数据文件: " << data_file << std::endl;
  std::cout << "URDF文件: " << urdf_file << std::endl;
  std::cout << "========================================" << std::endl;
  
  try {
    std::cout << "\n========================================" << std::endl;
    std::cout << "开始动力学参数联合辨识（惯性 + 摩擦）..." << std::endl;
    std::cout << "========================================" << std::endl;
    estimateDynamicsParametersJoint(data_file, urdf_file);
    std::cout << "\n========================================" << std::endl;
    std::cout << "动力学参数联合辨识结束" << std::endl;
    std::cout << "========================================" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "发生异常: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}

