/**
 * @file step2_dynamics_parameter_estimation.cpp
 * @brief 动力学参数辨识计算程序
 * 
 * 此程序用于从采集的数据中辨识机器人的动力学参数：
 * 1. 从CSV文件读取采集的数据
 * 2. 使用 Pinocchio 计算回归矩阵
 * 3. 通过最小二乘法求解动力学参数
 * 4. 验证辨识结果
 * 5. 保存辨识结果
 * 
 * 此示例需要使用xMateModel模型库，请设置编译选项XCORE_USE_XMATE_MODEL=ON
 * 需要 Pinocchio 库支持
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <fstream>
#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <map>
#include "Eigen/Geometry"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

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
    #include "pinocchio/algorithm/rnea.hpp"
    #define PINOCCHIO_AVAILABLE
  #endif
#endif

// OSQP (QP solver for mass constraints)
#ifdef __has_include
  #if __has_include("osqp.h")
    #include "osqp.h"
    #define OSQP_AVAILABLE
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

/** Step2 配置：从 config/step2_dynamics_parameter_estimation.yaml 加载 */
struct Step2Config {
  std::string data_file = "dynamics_identification_data.csv";
  std::string urdf_file = "AR5-5_07R-W4C4A2.urdf";
  double lambda_rel = 1e-3;
  double m_min = 1e-3;
  double I_eps = 1e-6;
  double I_trace_min = 1e-3;
  std::string result_file = "dynamics_identification_results.txt";
  std::string dynamics_parameters_csv = "dynamics_parameters.csv";
  std::string dynamics_parameters_ls_csv = "dynamics_parameters_ls.csv";
  std::string dynamics_parameters_urdf_csv = "dynamics_parameters_urdf.csv";
  std::string dynamics_physical_parameters_txt = "dynamics_physical_parameters_identified.txt";
  // 质心处惯量（标准 URDF 语义）
  std::string output_urdf = "AR5-5_07R-W4C4A2_identified_inertia_com.urdf";
  // 关节原点处惯量
  std::string output_urdf_joint = "AR5-5_07R-W4C4A2_identified_inertia_joint_origin.urdf";
};

/** 简易解析 YAML：key: value，忽略 # 注释 */
static Step2Config loadStep2Config(const std::string& config_path) {
  Step2Config c;
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
    std::string val(line, val_start);
    while (val.size() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    if (key == "data_file") { c.data_file = val; continue; }
    if (key == "urdf") { c.urdf_file = val; continue; }
    if (key == "result_file") { c.result_file = val; continue; }
    if (key == "dynamics_parameters_csv") { c.dynamics_parameters_csv = val; continue; }
    if (key == "dynamics_parameters_ls_csv") { c.dynamics_parameters_ls_csv = val; continue; }
    if (key == "dynamics_parameters_urdf_csv") { c.dynamics_parameters_urdf_csv = val; continue; }
    if (key == "dynamics_physical_parameters_txt") { c.dynamics_physical_parameters_txt = val; continue; }
    if (key == "output_urdf") { c.output_urdf = val; continue; }
    if (key == "output_urdf_joint") { c.output_urdf_joint = val; continue; }
    double v = 0;
    if (val_start < line.size() && (std::istringstream(val) >> v)) {
      if (key == "lambda_rel") c.lambda_rel = v;
      else if (key == "m_min") c.m_min = v;
      else if (key == "I_eps") c.I_eps = v;
      else if (key == "I_trace_min") c.I_trace_min = v;
    }
  }
  return c;
}

static std::string dirnameOf(const std::string& path) {
  size_t p = path.find_last_of("/\\");
  return p == std::string::npos ? "" : path.substr(0, p);
}

/**
 * @brief 生成新的URDF文件（更新惯性参数，惯量在质心处）
 *
 * 约定：URDF 与 Pinocchio 的惯量均为绕质心 I_C。见 docs/06-URDF与Pinocchio惯量绕质心约定.tex
 */
void generateIdentifiedURDF(const std::string& original_urdf_file, 
                             const std::string& output_urdf_file,
                             const pinocchio::Model& pinocchio_model,
                             const Eigen::VectorXd& theta_estimated) {
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
  
  // 存储每个 link 的新 inertial 内容（仅 7 个连杆，与 theta_estimated 的 70 维对应）
  const int n_inertial_links = std::min(7, static_cast<int>(theta_estimated.size() / 10));
  std::map<std::string, std::string> link_inertial_map;
  for (int j = 0; j < n_inertial_links; ++j) {
    const pinocchio::JointIndex jid = static_cast<pinocchio::JointIndex>(j + 1);
    const int base = 10 * j;
    const auto pi_id = theta_estimated.segment(base, 10);
    const auto inertia_id = pinocchio::Inertia::FromDynamicParameters(pi_id);

    const double m_id = inertia_id.mass();
    const Eigen::Vector3d c_id = inertia_id.lever();
    // 显式使用 Inertia 的 3x3 惯量矩阵（绕质心 I_C），与 URDF 约定一致，保证 θ 写/读一致
    const Eigen::Matrix3d I_com = inertia_id.inertia().matrix();
    const double ixx = I_com(0, 0), ixy = I_com(0, 1), ixz = I_com(0, 2);
    const double iyy = I_com(1, 1), iyz = I_com(1, 2), izz = I_com(2, 2);

    std::string joint_name = pinocchio_model.names[jid];  // Pinocchio 的 names[] 是关节名，如 joint_1
    // URDF 中 <link name="..."> 是连杆名，如 link1；关节 jid 的 child link 对应 linkN，需从 joint 名推导
    std::string link_name = joint_name;
    size_t jpos = joint_name.find("joint_");
    if (jpos != std::string::npos)
      link_name = joint_name.substr(0, jpos) + "link" + joint_name.substr(jpos + 6);  // joint_1 -> link1

    std::ostringstream inertial_oss;
    inertial_oss << "    <inertial>\n";
    inertial_oss << "      <mass value=\"" << std::fixed << std::setprecision(6) << m_id << "\" />\n";
    inertial_oss << "      <inertia ixx=\"" << std::scientific << std::setprecision(6) << ixx
                 << "\" ixy=\"" << ixy << "\" ixz=\"" << ixz
                 << "\" iyy=\"" << iyy << "\" iyz=\"" << iyz
                 << "\" izz=\"" << izz << "\" />\n";
    inertial_oss << "      <origin rpy=\"0 0 0\" xyz=\"" << std::fixed << std::setprecision(6)
                 << c_id(0) << " " << c_id(1) << " " << c_id(2) << "\" />\n";
    inertial_oss << "    </inertial>\n";
    link_inertial_map[link_name] = inertial_oss.str();
    std::cout << "  [DEBUG] 写入 URDF " << link_name << ": m=" << std::scientific << m_id
              << " c=[" << c_id.transpose() << "] I_com diag=" << ixx << "," << iyy << "," << izz
              << " trace(I_com)=" << (ixx + iyy + izz) << std::endl;
    // 同时用 "AR5-5_07R-W4C4A2_linkN" 形式匹配（若 Pinocchio 只返回 linkN）
    if (link_name.find("AR5-5") == std::string::npos && link_name.find("link") != std::string::npos) {
      link_inertial_map["AR5-5_07R-W4C4A2_" + link_name] = link_inertial_map[link_name];
    }
  }
  
  // 处理每一行，找到并替换inertial标签
  std::ofstream out_file(output_urdf_file);
  if (!out_file.is_open()) {
    std::cerr << "  警告: 无法创建输出URDF文件: " << output_urdf_file << std::endl;
    return;
  }
  
  std::string current_link_name;
  bool in_inertial = false;
  bool inertial_replaced = false;
  int inertial_indent = 0;
  
  for (size_t i = 0; i < lines.size(); ++i) {
    line = lines[i];

    size_t link_pos = line.find("<link name=\"");
    if (link_pos != std::string::npos) {
      size_t name_start = link_pos + 12;
      size_t name_end = line.find("\"", name_start);
      if (name_end != std::string::npos) {
        current_link_name = line.substr(name_start, name_end - name_start);
      }
      inertial_replaced = false;
    }

    if (line.find("<inertial>") != std::string::npos && !inertial_replaced) {
      inertial_indent = 0;
      for (char c : line) {
        if (c == ' ') inertial_indent++;
        else break;
      }

      auto it = link_inertial_map.find(current_link_name);
      if (it != link_inertial_map.end()) {
        // 跳过旧 inertial 块内容
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

    out_file << line << "\n";
  }
  
  out_file.close();
  std::cout << "  已生成新的URDF文件: " << output_urdf_file << std::endl;
}

/**
 * @brief 生成新的URDF文件（更新惯性参数，惯量在关节原点处）
 */
void generateIdentifiedURDFJointOrigin(const std::string& original_urdf_file,
                                       const std::string& output_urdf_file,
                                       const pinocchio::Model& pinocchio_model,
                                       const Eigen::VectorXd& theta_estimated) {
  std::ifstream in_file(original_urdf_file);
  if (!in_file.is_open()) {
    std::cerr << "  警告: 无法打开原始URDF文件: " << original_urdf_file << std::endl;
    return;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in_file, line)) {
    lines.push_back(line);
  }
  in_file.close();

  const int n_inertial_links = std::min(7, static_cast<int>(theta_estimated.size() / 10));
  std::map<std::string, std::string> link_inertial_map;
  for (int j = 0; j < n_inertial_links; ++j) {
    const pinocchio::JointIndex jid = static_cast<pinocchio::JointIndex>(j + 1);
    const int base = 10 * j;
    const auto pi_id = theta_estimated.segment(base, 10);
    const auto inertia_id = pinocchio::Inertia::FromDynamicParameters(pi_id);

    const double m_id = inertia_id.mass();
    const Eigen::Vector3d c_id = inertia_id.lever();
    const Eigen::Matrix3d I_origin = inertia_id.inertia().matrix();

    std::string joint_name = pinocchio_model.names[jid];
    std::string link_name = joint_name;
    size_t jpos = joint_name.find("joint_");
    if (jpos != std::string::npos)
      link_name = joint_name.substr(0, jpos) + "link" + joint_name.substr(jpos + 6);

    std::ostringstream inertial_oss;
    inertial_oss << "    <inertial>\n";
    inertial_oss << "      <mass value=\"" << std::fixed << std::setprecision(6) << m_id << "\" />\n";
    inertial_oss << "      <inertia ixx=\"" << std::scientific << std::setprecision(10) << I_origin(0,0)
                 << "\" ixy=\"" << I_origin(0,1) << "\" ixz=\"" << I_origin(0,2)
                 << "\" iyy=\"" << I_origin(1,1) << "\" iyz=\"" << I_origin(1,2)
                 << "\" izz=\"" << I_origin(2,2) << "\" />\n";
    // 关节原点惯量版本：<inertia> 使用关节原点处 I_origin，但 <origin> 仍给出质心位置，
    // 方便你查看/比对质心（虽然这与标准 URDF 语义不完全一致）
    inertial_oss << "      <origin rpy=\"0 0 0\" xyz=\"" << std::fixed << std::setprecision(6)
                 << c_id(0) << " " << c_id(1) << " " << c_id(2) << "\" />\n";
    inertial_oss << "    </inertial>\n";
    link_inertial_map[link_name] = inertial_oss.str();
    if (link_name.find("AR5-5") == std::string::npos && link_name.find("link") != std::string::npos) {
      link_inertial_map["AR5-5_07R-W4C4A2_" + link_name] = link_inertial_map[link_name];
    }
  }

  std::ofstream out_file(output_urdf_file);
  if (!out_file.is_open()) {
    std::cerr << "  警告: 无法创建输出URDF文件: " << output_urdf_file << std::endl;
    return;
  }

  std::string current_link_name;
  bool in_inertial = false;
  bool inertial_replaced = false;
  int inertial_indent = 0;

  for (size_t i = 0; i < lines.size(); ++i) {
    line = lines[i];

    size_t link_pos = line.find("<link name=\"");
    if (link_pos != std::string::npos) {
      size_t name_start = link_pos + 12;
      size_t name_end = line.find("\"", name_start);
      if (name_end != std::string::npos) {
        current_link_name = line.substr(name_start, name_end - name_start);
      }
      inertial_replaced = false;
    }

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

    out_file << line << "\n";
  }

  out_file.close();
  std::cout << "  已生成新的URDF文件(关节原点惯量): " << output_urdf_file << std::endl;
}

#ifdef PINOCCHIO_AVAILABLE
/**
 * @brief 将 theta 投影到物理可行域：m >= m_min，I 对称正定且满足惯性三角不等式。
 * 若提供 theta_urdf_fallback 且某连杆投影后 trace(I) < I_trace_min，则用 URDF 先验替代该连杆，避免写出 1e-6 级无效惯量。
 */
void projectThetaToPhysical(Eigen::VectorXd& theta, int n_params,
                            double m_min, double I_eps,
                            const Eigen::VectorXd* theta_urdf_fallback = nullptr,
                            double I_trace_min = 1e-3) {
  const int n_links = n_params / 10;
  if (n_links * 10 != n_params || n_links <= 0) return;
  const bool use_fallback = (theta_urdf_fallback != nullptr &&
                             theta_urdf_fallback->size() == n_params);

  std::cout << "\n  [DEBUG] 投影前各连杆 (m, c, trace(I)):" << std::endl;
  for (int j = 0; j < n_links; ++j) {
    const int base = 10 * j;
    Eigen::VectorXd pi = theta.segment(base, 10);
    pinocchio::Inertia inv = pinocchio::Inertia::FromDynamicParameters(pi);
    Eigen::Matrix3d I0 = inv.inertia();
    std::cout << "    link" << (j+1) << ": m=" << std::scientific << inv.mass()
              << " c=[" << inv.lever().transpose() << "] trace(I)=" << I0.trace() << std::endl;
  }

  for (int j = 0; j < n_links; ++j) {
    const int base = 10 * j;
    Eigen::VectorXd pi = theta.segment(base, 10);
    pinocchio::Inertia inv = pinocchio::Inertia::FromDynamicParameters(pi);
    double m = inv.mass();
    Eigen::Vector3d c = inv.lever();
    Eigen::Matrix3d I = inv.inertia();

    m = std::max(m, m_min);

    I = (I + I.transpose()) * 0.5;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(I);
    Eigen::Vector3d ev = es.eigenvalues();
    for (int i = 0; i < 3; ++i) ev(i) = std::max(ev(i), I_eps);
    std::sort(ev.data(), ev.data() + 3);
    if (ev(2) > ev(0) + ev(1) - I_eps)
      ev(2) = ev(0) + ev(1) - I_eps;
    ev(2) = std::max(ev(2), I_eps);
    I = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();

    const double trace_I = I.trace();
    if (use_fallback && trace_I < I_trace_min) {
      theta.segment(base, 10) = theta_urdf_fallback->segment(base, 10);
      std::cout << "  [DEBUG] link" << (j+1) << ": trace(I)=" << std::scientific << trace_I
                << " < I_trace_min=" << I_trace_min << " => 使用 URDF 先验" << std::endl;
      continue;
    }
    pinocchio::Inertia inv_proj(m, c, I);
    Eigen::VectorXd pi_new = inv_proj.toDynamicParameters();
    theta.segment(base, 10) = pi_new;
  }

  std::cout << "  [DEBUG] 投影后各连杆 (m, c, trace(I), I 特征值):" << std::endl;
  for (int j = 0; j < n_links; ++j) {
    const int base = 10 * j;
    Eigen::VectorXd pi = theta.segment(base, 10);
    pinocchio::Inertia inv = pinocchio::Inertia::FromDynamicParameters(pi);
    Eigen::Matrix3d I1 = inv.inertia();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(I1);
    Eigen::Vector3d ev = es.eigenvalues();
    std::cout << "    link" << (j+1) << ": m=" << std::scientific << inv.mass()
              << " c=[" << inv.lever().transpose() << "] trace(I)=" << I1.trace()
              << " ev=[" << ev.transpose() << "]" << std::endl;
  }
}
#endif

#ifdef OSQP_AVAILABLE
/**
 * @brief 使用 QP（OSQP）求解带质量约束的 Ridge 问题：
 *   min (1/2)||Y*theta - tau||^2 + (lambda/2)||theta - theta_urdf||^2
 *   s.t. theta[10*j] >= m_min, j = 0..n_links-1
 * 即 min (1/2) theta' H theta + g' theta,  s.t. -theta[10*j] <= -m_min.
 */
bool solveRidgeQP(const Eigen::MatrixXd& Y_all, const Eigen::VectorXd& tau_all,
                  double lambda, const Eigen::VectorXd& theta_urdf,
                  int n_params, int n_links, double m_min,
                  Eigen::VectorXd& theta_out) {
  const int n = n_params;
  const int m = n_links;
  if (n_links <= 0 || n != static_cast<int>(theta_urdf.size()) || 10 * n_links != n) {
    return false;
  }
  Eigen::MatrixXd H_dense = 2.0 * (Y_all.transpose() * Y_all);
  H_dense.diagonal().array() += 2.0 * lambda;
  Eigen::VectorXd g = -2.0 * (Y_all.transpose() * tau_all + lambda * theta_urdf);

  // P: upper triangular CSC, nnz = n*(n+1)/2
  const c_int P_nnz = static_cast<c_int>(n * (n + 1) / 2);
  std::vector<c_float> P_x(P_nnz);
  std::vector<c_int> P_i(P_nnz), P_p(static_cast<size_t>(n + 1));
  c_int idx = 0;
  for (int col = 0; col < n; ++col) {
    P_p[col] = idx;
    for (int row = 0; row <= col; ++row) {
      P_i[idx] = static_cast<c_int>(row);
      P_x[idx] = static_cast<c_float>(H_dense(row, col));
      ++idx;
    }
  }
  P_p[n] = idx;

  // A: m x n, row j has -1 at column 10*j. CSC: col 10*j has one entry (row j, -1)
  const c_int A_nnz = static_cast<c_int>(m);
  std::vector<c_float> A_x(static_cast<size_t>(A_nnz), -1.0f);
  std::vector<c_int> A_i(static_cast<size_t>(A_nnz));
  std::vector<c_int> A_p(static_cast<size_t>(n + 1));
  for (int j = 0; j < m; ++j) A_i[j] = static_cast<c_int>(j);
  A_p[0] = 0;
  for (int k = 1; k <= n; ++k)
    A_p[k] = static_cast<c_int>(std::min(k / 10, m));

  std::vector<c_float> q(n), l(m), u(m);
  for (int i = 0; i < n; ++i) q[i] = static_cast<c_float>(g(i));
  const c_float infty = static_cast<c_float>(1e30);
  for (int j = 0; j < m; ++j) {
    l[j] = -infty;
    u[j] = static_cast<c_float>(-m_min);
  }

  OSQPData* data = reinterpret_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));
  if (!data) return false;
  data->n = static_cast<c_int>(n);
  data->m = static_cast<c_int>(m);
  data->P = csc_matrix(data->n, data->n, P_nnz, P_x.data(), P_i.data(), P_p.data());
  data->A = csc_matrix(data->m, data->n, A_nnz, A_x.data(), A_i.data(), A_p.data());
  data->q = q.data();
  data->l = l.data();
  data->u = u.data();

  OSQPSettings* settings = reinterpret_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  if (!settings) {
    c_free(data->P);
    c_free(data->A);
    c_free(data);
    return false;
  }
  osqp_set_default_settings(settings);
  settings->verbose = 0;
  settings->polish = 1;

  OSQPWorkspace* work = nullptr;
  c_int exitflag = osqp_setup(&work, data, settings);
  if (exitflag != 0) {
    c_free(data->P);
    c_free(data->A);
    c_free(data);
    c_free(settings);
    return false;
  }
  exitflag = osqp_solve(work);
  c_int st = work->info->status_val;
  if (exitflag != 0 || (st != OSQP_SOLVED && st != OSQP_SOLVED_INACCURATE)) {
    osqp_cleanup(work);
    c_free(data->P);
    c_free(data->A);
    c_free(data);
    c_free(settings);
    return false;
  }
  theta_out.resize(n);
  for (int i = 0; i < n; ++i) theta_out(i) = static_cast<double>(work->solution->x[i]);
  osqp_cleanup(work);
  c_free(data->P);
  c_free(data->A);
  c_free(data);
  c_free(settings);
  return true;
}
#endif

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
 * @brief 动力学参数辨识主函数
 */
void estimateDynamicsParameters(const std::string& data_file, const std::string& urdf_file, const Step2Config& cfg) {
#ifdef PINOCCHIO_AVAILABLE
  // 初始化 Pinocchio 模型
  std::cout << "\n  初始化 Pinocchio 动力学模型..." << std::endl;
  
  pinocchio::Model pinocchio_model;
  pinocchio::Data pinocchio_data(pinocchio_model);
  
  try {
    pinocchio::urdf::buildModel(urdf_file, pinocchio_model);
    std::cout << "  Pinocchio 模型加载成功: " << urdf_file << std::endl;
    std::cout << "  模型自由度: " << pinocchio_model.nq << std::endl;
    
    // 设置重力方向
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

  // 构造回归矩阵
  std::cout << "\n========================================" << std::endl;
  std::cout << "开始构造回归矩阵..." << std::endl;
  std::cout << "========================================" << std::endl;

  int n_joints = 7;
  int n_params = 0;  // 将在第一次计算回归矩阵时确定
  Eigen::VectorXd theta_urdf;  // URDF先验（与回归矩阵列顺序一致的10维惯性参数堆叠）
  
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

    // 由URDF模型构造对应的参数先验 theta_urdf（每个关节10个惯性参数）
    // Pinocchio中回归矩阵列与 model.inertias[i].toDynamicParameters() 的堆叠一致。
    // 10维顺序（与URDF对应）：[m, m*cx, m*cy, m*cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]，详见 docs/dynamic_parameters_10d_conversion.md
    theta_urdf.resize(n_params);
    theta_urdf.setZero();
    const int expected_params = 10 * static_cast<int>(pinocchio_model.njoints - 1);
    if (n_params != expected_params) {
      std::cerr << "  警告: n_params(" << n_params << ") != 10*(njoints-1)(" << expected_params
                << ")，无法可靠构造URDF先验theta_urdf，将只做纯最小二乘。" << std::endl;
      theta_urdf.resize(0);
    } else {
      for (pinocchio::JointIndex jid = 1; jid < pinocchio_model.njoints; ++jid) {
        const auto pi = pinocchio_model.inertias[jid].toDynamicParameters(); // 10x1
        const int base = 10 * static_cast<int>(jid - 1);
        theta_urdf.segment(base, 10) = pi;
      }
      std::cout << "  已构造URDF先验theta_urdf（维度 " << theta_urdf.size() << "）" << std::endl;

      // 运行前：打印 URDF 转换后的 70 个数据，并与 URDF 各连杆的原始参数对比
      // I 的转换：Pinocchio 内部 inertia() 为连杆原点处惯性 I_origin；toDynamicParameters() 中 I6 为质心处惯性
      // I_C = I_origin - m*S(c)^T*S(c)（平行轴定理），故“URDF 原始 I6”与“转换后 I6”数值不同属正常
      std::cout << "\n  [URDF 先验] 转换后的 70 个参数与 URDF 原始参数对比:" << std::endl;
      for (pinocchio::JointIndex jid = 1; jid < pinocchio_model.njoints; ++jid) {
        const int base = 10 * static_cast<int>(jid - 1);
        const auto& inv = pinocchio_model.inertias[jid];
        double m_urdf = inv.mass();
        Eigen::Vector3d c_urdf = inv.lever();
        Eigen::Matrix3d I_origin_3x3 = inv.inertia().matrix();  // 连杆原点处惯性
        const Eigen::VectorXd pi = theta_urdf.segment(base, 10);
        // 质心处惯性 I_C = I_origin - m*S(c)'*S(c)，与 theta_urdf 的 I6 一致
        auto I_com_s3 = inv.inertia() - pinocchio::Symmetric3::AlphaSkewSquare(m_urdf, c_urdf);
        const Eigen::Matrix<double, 6, 1> I_com_6 = I_com_s3.data();
        // Pinocchio Symmetric3 与 10 维顺序一致：I6 = [Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        std::cout << "  --- 连杆 " << jid << " (" << pinocchio_model.names[jid] << ") ---" << std::endl;
        std::cout << "    I_origin (连杆原点): m=" << std::scientific << m_urdf
                  << " c=[" << c_urdf.transpose() << "]"
                  << " I6=[Ixx,Ixy,Iyy,Ixz,Iyz,Izz]=[" << I_origin_3x3(0,0) << "," << I_origin_3x3(1,0) << "," << I_origin_3x3(1,1)
                  << "," << I_origin_3x3(2,0) << "," << I_origin_3x3(2,1) << "," << I_origin_3x3(2,2) << "]" << std::endl;
        std::cout << "    I_at_COM (质心处, 转换后 theta_urdf 的 I6): I6=[" << I_com_6.transpose() << "]" << std::endl;
        std::cout << "    转换后 10 维 theta_urdf[" << base << ".." << (base+9) << "]: "
                  << "m=" << pi(0) << " m*c=[" << pi(1) << "," << pi(2) << "," << pi(3) << "] "
                  << "I6=[" << pi(4) << "," << pi(5) << "," << pi(6) << "," << pi(7) << "," << pi(8) << "," << pi(9) << "]" << std::endl;
        std::cout << "    校验: m一致=" << (std::abs(pi(0) - m_urdf) < 1e-10 ? "是" : "否")
                  << " I6=质心处=" << (I_com_6.isApprox(pi.segment<6>(4), 1e-8) ? "是" : "否")
                  << " c=pi[1:3]/m [" << (pi(1)/pi(0)) << "," << (pi(2)/pi(0)) << "," << (pi(3)/pi(0)) << "]" << std::endl;
      }
      std::cout << "  70 个参数完整列出 (parameter_index, value):" << std::endl;
      for (int i = 0; i < theta_urdf.size(); ++i)
        std::cout << "    " << i << " " << std::scientific << std::setprecision(10) << theta_urdf(i) << std::endl;
      std::cout << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "  警告: Pinocchio回归矩阵计算失败: " << e.what() << std::endl;
    std::cerr << "  回退到简化参数集" << std::endl;
    n_params = 3 * n_joints;  // 21个参数
  }
#else
  // 如果没有Pinocchio，使用简化参数集
  n_params = 3 * n_joints;  // 21个参数（每个关节3个：惯性、重力、科氏力）
  std::cout << "  警告: Pinocchio不可用，使用简化参数集" << std::endl;
#endif
  
  // 预分配，避免每次堆叠时复制整块矩阵（原实现为 O(N^2) 拷贝，导致比 Python 慢很多）
  const size_t max_rows = collected_data.size() * static_cast<size_t>(n_joints);
  Eigen::MatrixXd Y_all(max_rows, n_params);
  Eigen::VectorXd tau_all(max_rows);
  
  std::cout << "  处理 " << collected_data.size() << " 个数据点..." << std::endl;
  
  int processed = 0;
  std::vector<std::array<double, 21>> Y_input_used;  // 每样本: q1..q7, dq1..dq7, ddq1..ddq7
  Y_input_used.reserve(collected_data.size());
  
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
      std::cerr << "  跳过该数据点..." << std::endl;
      continue;
    }
#else
    // 如果没有Pinocchio，使用简化方法（这里需要xMateModel，但为了简化，跳过）
    std::cerr << "  错误: 需要Pinocchio库" << std::endl;
    return;
#endif
    
    // 转换为Eigen向量（力矩）
    Eigen::VectorXd tau(7);
    for (int i = 0; i < 7; i++) {
      tau(i) = tau_arr[i];
    }
    
    // 写入当前块，避免逐次复制整块矩阵
    Y_all.block(processed * n_joints, 0, n_joints, n_params) = Y;
    tau_all.segment(processed * n_joints, n_joints) = tau;
    {
      std::array<double, 21> row;
      for (int i = 0; i < 7; i++) {
        row[i] = q_arr[i];
        row[7 + i] = dq_arr[i];
        row[14 + i] = ddq_arr[i];
      }
      Y_input_used.push_back(row);
    }
    processed++;
    if (processed % 100 == 0) {
      std::cout << "  已处理: " << processed << " / " << collected_data.size() << std::endl;
    }
  }
  
  Y_all.conservativeResize(processed * n_joints, n_params);
  tau_all.conservativeResize(processed * n_joints);
  
  std::cout << "\n  回归矩阵构造完成" << std::endl;
  std::cout << "  回归矩阵维度: " << Y_all.rows() << " x " << Y_all.cols() << std::endl;
  std::cout << "  力矩向量维度: " << tau_all.size() << std::endl;

  // 保存 Y_all 到二进制文件，便于与 Python 对比（格式: int32 rows, int32 cols, row-major double）
  {
    const std::string y_all_bin = "Y_all_cpp.bin";
    std::ofstream fy(y_all_bin, std::ios::binary);
    if (fy.good()) {
      const int32_t rows = static_cast<int32_t>(Y_all.rows());
      const int32_t cols = static_cast<int32_t>(Y_all.cols());
      fy.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
      fy.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
      for (Eigen::Index i = 0; i < Y_all.rows(); ++i)
        for (Eigen::Index j = 0; j < Y_all.cols(); ++j) {
          const double v = Y_all(i, j);
          fy.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
      fy.close();
      std::cout << "  Y_all 已保存到: " << y_all_bin << std::endl;
    }
  }

  // 保存 Y_all 使用的输入数据 (q,dq,ddq)，便于与 Python 对比
  {
    const std::string input_csv = "Y_input_cpp.csv";
    std::ofstream fc(input_csv);
    if (fc.good()) {
      fc << "sample_idx,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7,ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,ddq7\n";
      for (size_t i = 0; i < Y_input_used.size(); i++) {
        const auto& r = Y_input_used[i];
        fc << static_cast<int>(i);
        for (int j = 0; j < 21; j++)
          fc << "," << std::setprecision(17) << r[j];
        fc << "\n";
      }
      fc.close();
      std::cout << "  Y_all 输入数据已保存到: " << input_csv << std::endl;
    }
  }

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

  // 求解参数：优先 QP（质量约束），否则 Ridge 最小二乘
  Eigen::VectorXd theta_ls = svd.solve(tau_all);
  Eigen::VectorXd theta_estimated = theta_ls;

  const double lambda_rel = cfg.lambda_rel;
  const double lambda = (Y_all.rows() > 0 && n_params > 0)
      ? (lambda_rel * (Y_all.transpose() * Y_all).trace() / static_cast<double>(n_params))
      : lambda_rel;
  const double m_min = cfg.m_min;

  // 带 URDF 偏离惩罚的最小二乘（Ridge）：min ||Y*theta-tau||^2 + lambda*||theta-theta_urdf||^2，用于保存到 dynamics_parameters_ls.csv
  Eigen::VectorXd theta_ls_ridge = theta_ls;
  if (theta_urdf.size() == n_params) {
    Eigen::MatrixXd A_ridge = Y_all.transpose() * Y_all;
    Eigen::VectorXd b_ridge = Y_all.transpose() * tau_all;
    A_ridge.diagonal().array() += lambda;
    b_ridge += lambda * theta_urdf;
    theta_ls_ridge = A_ridge.ldlt().solve(b_ridge);
  }

#ifdef OSQP_AVAILABLE
  if (theta_urdf.size() == n_params && n_params == 70) {
    const int n_links = 7;
    if (solveRidgeQP(Y_all, tau_all, lambda, theta_urdf, n_params, n_links, m_min, theta_estimated)) {
      std::cout << "  使用 QP（OSQP）求解: 质量约束 theta[10*j] >= " << m_min
                << ", lambda=" << std::scientific << lambda << std::fixed << std::endl;
    } else {
      std::cout << "  QP 求解失败，回退到 Ridge 最小二乘" << std::endl;
      Eigen::MatrixXd A = Y_all.transpose() * Y_all;
      Eigen::VectorXd b = Y_all.transpose() * tau_all;
      A.diagonal().array() += lambda;
      b += lambda * theta_urdf;
      theta_estimated = A.ldlt().solve(b);
    }
  } else
#endif
  if (theta_urdf.size() == n_params) {
    Eigen::MatrixXd A = Y_all.transpose() * Y_all;
    Eigen::VectorXd b = Y_all.transpose() * tau_all;
    A.diagonal().array() += lambda;
    b += lambda * theta_urdf;
    theta_estimated = A.ldlt().solve(b);
    std::cout << "  使用URDF先验的Ridge最小二乘: lambda_rel=" << lambda_rel
              << ", lambda=" << std::scientific << lambda << std::fixed << std::endl;
  } else {
    std::cout << "  未构造URDF先验theta_urdf，使用纯最小二乘解（SVD）" << std::endl;
  }

#ifdef PINOCCHIO_AVAILABLE
  if (n_params == 70) {
    std::cout << "\n  [DEBUG] 投影前 theta_estimated 范数: " << std::scientific << theta_estimated.norm()
              << ", 与 theta_urdf 差范数: " << (theta_urdf.size() == n_params ? (theta_estimated - theta_urdf).norm() : 0.0) << std::endl;
    const Eigen::VectorXd* urdf_fallback = (theta_urdf.size() == n_params) ? &theta_urdf : nullptr;
    const double I_trace_min = cfg.I_trace_min;
    projectThetaToPhysical(theta_estimated, n_params, cfg.m_min, cfg.I_eps, urdf_fallback, I_trace_min);
    std::cout << "  已应用物理约束投影: 质量 m>=m_min，惯性 I 对称正定+三角不等式；trace(I)<" << I_trace_min << " 时用 URDF 先验" << std::endl;
  }
#endif

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
  std::ofstream result_file(cfg.result_file);
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
  result_file << "求解说明:\n";
  if (theta_urdf.size() == n_params) {
    result_file << "  使用URDF先验的Ridge最小二乘（在不满秩/不可辨识方向上更稳定）\n";
    result_file << "  lambda_rel: " << std::scientific << lambda_rel << std::fixed << "\n\n";
  } else {
    result_file << "  使用纯最小二乘（SVD最小范数解）\n\n";
  }
  result_file << "辨识的参数值:\n";
  for (int i = 0; i < theta_estimated.size(); i++) {
    result_file << "  theta[" << i << "] = " << std::scientific 
                << std::setprecision(6) << theta_estimated(i) << "\n";
  }
  result_file.close();
  std::cout << "  结果已保存到: " << cfg.result_file << std::endl;

  // 保存参数到CSV（便于后续使用）
  std::ofstream param_file(cfg.dynamics_parameters_csv);
  param_file << "parameter_index,value\n";
  for (int i = 0; i < theta_estimated.size(); i++) {
    param_file << i << "," << std::scientific << std::setprecision(10) 
               << theta_estimated(i) << "\n";
  }
  param_file.close();
  std::cout << "  参数已保存到: " << cfg.dynamics_parameters_csv << std::endl;

  // 额外保存：最小二乘 + 偏离 URDF 的惩罚项（Ridge），使解更接近 theta_urdf；无先验时为纯 LS
  std::ofstream param_file_ls(cfg.dynamics_parameters_ls_csv);
  param_file_ls << "parameter_index,value\n";
  for (int i = 0; i < theta_ls_ridge.size(); i++) {
    param_file_ls << i << "," << std::scientific << std::setprecision(10)
                  << theta_ls_ridge(i) << "\n";
  }
  param_file_ls.close();
  std::cout << "  带URDF偏离惩罚的最小二乘（Ridge）参数已保存到: " << cfg.dynamics_parameters_ls_csv << std::endl;

  // 额外保存：URDF先验参数，便于对比
  if (theta_urdf.size() == n_params) {
    std::ofstream param_file_urdf(cfg.dynamics_parameters_urdf_csv);
    param_file_urdf << "parameter_index,value\n";
    for (int i = 0; i < theta_urdf.size(); i++) {
      param_file_urdf << i << "," << std::scientific << std::setprecision(10)
                      << theta_urdf(i) << "\n";
    }
    param_file_urdf.close();
    std::cout << "  URDF先验参数已保存到: " << cfg.dynamics_parameters_urdf_csv << std::endl;
  }

#ifdef PINOCCHIO_AVAILABLE
  // 输出“辨识后的物理参数”（只有当n_params=10*(njoints-1)时才能从theta直接重建每个关节Inertia）
  if (theta_urdf.size() == n_params) {
    std::cout << "\n  输出辨识后的物理参数（与URDF对比）..." << std::endl;
    std::ofstream physical_param_file(cfg.dynamics_physical_parameters_txt);
    physical_param_file << "辨识后的物理参数（由theta重建，与URDF对比）\n";
    physical_param_file << "========================================\n\n";
    physical_param_file << "说明:\n";
    physical_param_file << "1) 本次Y的秩可能小于参数数(n_params)，因此存在不可辨识方向。\n";
    physical_param_file << "2) 这里采用URDF先验的Ridge最小二乘，在不可辨识方向上更贴近URDF。\n";
    physical_param_file << "3) theta块(每关节10维)用 pinocchio::Inertia::FromDynamicParameters 重建质量/质心/惯性。\n";
    physical_param_file << "4) 已对 theta 做物理约束投影：m>=m_min，I 正定+三角不等式；若某连杆投影后 trace(I)<阈值则回退 URDF 先验。\n";
    physical_param_file << "5) 以下同时给出关节原点处惯性和质心处惯性（URDF <inertia> 与 10 维参数对应质心处惯性）。\n\n";

    for (pinocchio::JointIndex jid = 1; jid < pinocchio_model.njoints; ++jid) {
      const int base = 10 * static_cast<int>(jid - 1);
      const auto pi_id = theta_estimated.segment(base, 10);

      const double m_urdf = pinocchio_model.inertias[jid].mass();
      const Eigen::Vector3d c_urdf = pinocchio_model.inertias[jid].lever();
      const double m_id = pi_id(0);
      const Eigen::Vector3d c_id = pi_id.segment<3>(1) / pi_id(0);

      // 质心处惯性：URDF 对应 I_com = I_origin - m*S(c)'*S(c)；辨识值直接取 10 维 [4:9]
      const Eigen::Matrix3d I_origin_urdf = pinocchio_model.inertias[jid].inertia().matrix();
      pinocchio::Inertia inv_id = pinocchio::Inertia::FromDynamicParameters(pi_id);
      const Eigen::Matrix3d I_origin_id = inv_id.inertia().matrix();
      auto I_com_urdf_s3 = pinocchio_model.inertias[jid].inertia() - pinocchio::Symmetric3::AlphaSkewSquare(m_urdf, c_urdf);
      const Eigen::Matrix3d I_com_urdf = I_com_urdf_s3.matrix();
      Eigen::Matrix3d I_com_id;
      I_com_id << pi_id(4), pi_id(5), pi_id(7),
                  pi_id(5), pi_id(6), pi_id(8),
                  pi_id(7), pi_id(8), pi_id(9);

      physical_param_file << "关节 " << jid << " (" << pinocchio_model.names[jid] << "):\n";
      physical_param_file << "  URDF 质量(kg): " << std::fixed << std::setprecision(6) << m_urdf
                          << " | 辨识质量(kg): " << m_id << "\n";
      physical_param_file << "  URDF 质心(m): [" << c_urdf(0) << ", " << c_urdf(1) << ", " << c_urdf(2) << "]\n";
      physical_param_file << "  辨识质心(m): [" << c_id(0) << ", " << c_id(1) << ", " << c_id(2) << "]\n";
      physical_param_file << "  URDF 惯性(关节原点处, kg·m²):\n";
      physical_param_file << "    [" << I_origin_urdf(0,0) << ", " << I_origin_urdf(0,1) << ", " << I_origin_urdf(0,2) << "]\n";
      physical_param_file << "    [" << I_origin_urdf(1,0) << ", " << I_origin_urdf(1,1) << ", " << I_origin_urdf(1,2) << "]\n";
      physical_param_file << "    [" << I_origin_urdf(2,0) << ", " << I_origin_urdf(2,1) << ", " << I_origin_urdf(2,2) << "]\n";
      physical_param_file << "  URDF 惯性(质心处, kg·m²):\n";
      physical_param_file << "    [" << I_com_urdf(0,0) << ", " << I_com_urdf(0,1) << ", " << I_com_urdf(0,2) << "]\n";
      physical_param_file << "    [" << I_com_urdf(1,0) << ", " << I_com_urdf(1,1) << ", " << I_com_urdf(1,2) << "]\n";
      physical_param_file << "    [" << I_com_urdf(2,0) << ", " << I_com_urdf(2,1) << ", " << I_com_urdf(2,2) << "]\n";
      physical_param_file << "  辨识 惯性(关节原点处, kg·m²):\n";
      physical_param_file << "    [" << I_origin_id(0,0) << ", " << I_origin_id(0,1) << ", " << I_origin_id(0,2) << "]\n";
      physical_param_file << "    [" << I_origin_id(1,0) << ", " << I_origin_id(1,1) << ", " << I_origin_id(1,2) << "]\n";
      physical_param_file << "    [" << I_origin_id(2,0) << ", " << I_origin_id(2,1) << ", " << I_origin_id(2,2) << "]\n";
      physical_param_file << "  辨识 惯性(质心处, kg·m²):\n";
      physical_param_file << "    [" << I_com_id(0,0) << ", " << I_com_id(0,1) << ", " << I_com_id(0,2) << "]\n";
      physical_param_file << "    [" << I_com_id(1,0) << ", " << I_com_id(1,1) << ", " << I_com_id(1,2) << "]\n";
      physical_param_file << "    [" << I_com_id(2,0) << ", " << I_com_id(2,1) << ", " << I_com_id(2,2) << "]\n\n";
    }

    // 简单验证：q=0时的预测力矩
    Eigen::VectorXd q0(7), dq0(7), ddq0(7);
    q0.setZero(); dq0.setZero(); ddq0.setZero();
    auto& Y0 = pinocchio::computeJointTorqueRegressor(pinocchio_model, pinocchio_data, q0, dq0, ddq0);
    const Eigen::VectorXd tau0 = Y0 * theta_estimated;
    physical_param_file << "========================================\n";
    physical_param_file << "验证：q=0,dq=0,ddq=0 时预测力矩(Nm):\n";
    physical_param_file << "  [" << std::fixed << std::setprecision(4)
                        << tau0(0) << ", " << tau0(1) << ", " << tau0(2) << ", "
                        << tau0(3) << ", " << tau0(4) << ", " << tau0(5) << ", "
                        << tau0(6) << "]\n";
    physical_param_file.close();
    std::cout << "  已保存: " << cfg.dynamics_physical_parameters_txt << std::endl;
    
    // 生成新的URDF文件
    std::cout << "\n  生成新的URDF文件(质心处惯量)..." << std::endl;
    generateIdentifiedURDF(urdf_file, cfg.output_urdf, pinocchio_model, theta_estimated);
    std::cout << "  生成新的URDF文件(关节原点惯量)..." << std::endl;
    generateIdentifiedURDFJointOrigin(urdf_file, cfg.output_urdf_joint, pinocchio_model, theta_estimated);
  } else {
    std::cout << "\n  跳过物理参数重建：当前theta不是每关节10维惯性参数形式（n_params=" << n_params << "）。" << std::endl;
  }
  
  // 同时更新结果文件，添加物理参数说明
  std::ofstream result_file_append(cfg.result_file, std::ios::app);
  result_file_append << "\n========================================\n";
  result_file_append << "物理参数说明:\n";
  result_file_append << "========================================\n";
  result_file_append << "1) 若回归矩阵不满秩(rank < n_params)，则存在不可辨识方向，参数解不唯一。\n";
  result_file_append << "2) 本程序可用URDF先验的Ridge最小二乘在不可辨识方向上稳定解（更贴近URDF）。\n";
  result_file_append << "3) 已对 70 维 theta 做物理约束投影（m>=m_min, I 正定+三角不等式）；投影过小则回退 URDF 先验，避免 1e-6 级无效惯性。\n";
  if (theta_urdf.size() == n_params) {
    result_file_append << "4) 已输出辨识后的质量/质心/惯性到: " << cfg.dynamics_physical_parameters_txt << "\n";
  } else {
    result_file_append << "5) 当前theta不是10维惯性参数堆叠形式，无法重建质量/质心/惯性\n";
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
      continue;  // 跳过失败的数据点
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
  std::cout << "  验证集 RMSE (基于原URDF+Y*theta): " << std::fixed << std::setprecision(4) 
            << rmse_val << " Nm" << std::endl;
  
  if (rmse_val < rmse * 1.5) {
    std::cout << "  ✓ 验证通过：验证误差在合理范围内" << std::endl;
  } else {
    std::cout << "  ⚠ 警告：验证误差较大，可能需要更多数据或改进激励轨迹" << std::endl;
  }

#ifdef PINOCCHIO_AVAILABLE
  // 额外验证：加载原始 URDF，用逆动力学 rnea 在验证集上计算力矩并求 RMSE
  try {
    std::cout << "\n  使用原始URDF(" << urdf_file
              << ") 在验证集上验证（rnea 逆动力学）..." << std::endl;
    pinocchio::Model pinocchio_model_orig;
    pinocchio::urdf::buildModel(urdf_file, pinocchio_model_orig);
    pinocchio::Data pinocchio_data_orig(pinocchio_model_orig);

    Eigen::Matrix3d R_y_o = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d R_z_o = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d R_base_o = R_z_o * R_y_o;
    Eigen::Vector3d gravity_world_o(0, 0, -9.81);
    pinocchio_model_orig.gravity.linear() = R_base_o.transpose() * gravity_world_o;

    std::vector<double> tau_pred_orig_list, tau_meas_orig_list;
    for (int i = validation_start; i < collected_data.size(); i++) {
      const auto& point = collected_data[i];
      Eigen::VectorXd q_e(7), dq_e(7), ddq_e(7);
      for (int j = 0; j < 7; j++) {
        q_e(j) = point.q[j];
        dq_e(j) = point.dq[j];
        ddq_e(j) = point.ddq[j];
      }
      Eigen::VectorXd tau_orig = pinocchio::rnea(pinocchio_model_orig, pinocchio_data_orig, q_e, dq_e, ddq_e);
      for (int j = 0; j < 7; j++) {
        tau_pred_orig_list.push_back(tau_orig(j));
        tau_meas_orig_list.push_back(point.tau[j]);
      }
    }

    if (!tau_pred_orig_list.empty() && tau_pred_orig_list.size() == tau_meas_orig_list.size()) {
      Eigen::Map<Eigen::VectorXd> tau_po(tau_pred_orig_list.data(), tau_pred_orig_list.size());
      Eigen::Map<Eigen::VectorXd> tau_mo(tau_meas_orig_list.data(), tau_meas_orig_list.size());
      Eigen::VectorXd err_o = tau_mo - tau_po;
      double rmse_orig = std::sqrt(err_o.squaredNorm() / err_o.size());
      double max_err_orig = err_o.cwiseAbs().maxCoeff();
      std::cout << "  验证集 RMSE (原URDF + rnea): " << std::fixed << std::setprecision(4)
                << rmse_orig << " Nm" << std::endl;
      std::cout << "  验证集 最大绝对误差 (原URDF + rnea): " << std::fixed << std::setprecision(4)
                << max_err_orig << " Nm" << std::endl;

      std::ofstream res_append_orig(cfg.result_file, std::ios::app);
      res_append_orig << "\n验证集-原URDF(rnea): RMSE=" << rmse_orig << " Nm, 最大绝对误差=" << max_err_orig << " Nm\n";
      res_append_orig.close();
    } else {
      std::cout << "  (提示) 验证集数据为空，跳过原URDF的RMSE计算。" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cout << "  (提示) 使用原URDF进行验证时出错: " << e.what() << std::endl;
  }

  // 额外验证：加载辨识后的 URDF（质心处惯量），用逆动力学 rnea 在验证集上计算力矩并求 RMSE
  try {
    std::cout << "\n  使用辨识后的URDF(" << cfg.output_urdf
              << ") 在验证集上再次验证（rnea 逆动力学，质心处惯量）..." << std::endl;
    std::string id_urdf_path = cfg.output_urdf;
    {
      std::ifstream f(id_urdf_path);
      if (!f.good()) {
        std::string alt = dirnameOf(urdf_file);
        if (!alt.empty()) id_urdf_path = alt + "/" + cfg.output_urdf;
      }
    }
    pinocchio::Model pinocchio_model_id;
    pinocchio::urdf::buildModel(id_urdf_path, pinocchio_model_id);
    pinocchio::Data pinocchio_data_id(pinocchio_model_id);

    // 与主流程一致的重力方向
    Eigen::Matrix3d R_y = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d R_z = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d R_base_to_world = R_z * R_y;
    Eigen::Vector3d gravity_world(0, 0, -9.81);
    pinocchio_model_id.gravity.linear() = R_base_to_world.transpose() * gravity_world;

    std::vector<double> tau_pred_list, tau_meas_list;
    for (int i = validation_start; i < collected_data.size(); i++) {
      const auto& point = collected_data[i];
      Eigen::VectorXd q_e(7), dq_e(7), ddq_e(7);
      for (int j = 0; j < 7; j++) {
        q_e(j) = point.q[j];
        dq_e(j) = point.dq[j];
        ddq_e(j) = point.ddq[j];
      }
      Eigen::VectorXd tau_pred = pinocchio::rnea(pinocchio_model_id, pinocchio_data_id, q_e, dq_e, ddq_e);
      for (int j = 0; j < 7; j++) {
        tau_pred_list.push_back(tau_pred(j));
        tau_meas_list.push_back(point.tau[j]);
      }
    }

    if (!tau_pred_list.empty() && tau_pred_list.size() == tau_meas_list.size()) {
      Eigen::Map<Eigen::VectorXd> tau_p(tau_pred_list.data(), tau_pred_list.size());
      Eigen::Map<Eigen::VectorXd> tau_m(tau_meas_list.data(), tau_meas_list.size());
      Eigen::VectorXd err = tau_m - tau_p;
      double rmse_urdf = std::sqrt(err.squaredNorm() / err.size());
      double max_err_urdf = err.cwiseAbs().maxCoeff();
      std::cout << "  验证集 RMSE (辨识URDF[COM] + rnea): " << std::fixed << std::setprecision(4)
                << rmse_urdf << " Nm" << std::endl;
      std::cout << "  验证集 最大绝对误差 (辨识URDF[COM] + rnea): " << std::fixed << std::setprecision(4)
                << max_err_urdf << " Nm" << std::endl;

      std::ofstream res_append(cfg.result_file, std::ios::app);
      res_append << "\n验证集-辨识URDF[COM](rnea): RMSE=" << rmse_urdf << " Nm, 最大绝对误差=" << max_err_urdf << " Nm\n";
      res_append.close();
    } else {
      std::cout << "  (提示) 验证集数据为空，跳过辨识URDF[COM]的RMSE计算。" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cout << "  (提示) 使用辨识URDF[COM]进行验证时出错: " << e.what() << std::endl;
  }

  // 额外验证：加载关节原点惯量版 URDF，用逆动力学 rnea 在验证集上计算力矩并求 RMSE
  try {
    std::cout << "\n  使用辨识后的URDF(" << cfg.output_urdf_joint
              << ") 在验证集上再次验证（rnea 逆动力学，关节原点惯量）..." << std::endl;
    std::string id_joint_path = cfg.output_urdf_joint;
    {
      std::ifstream f(id_joint_path);
      if (!f.good()) {
        std::string alt = dirnameOf(urdf_file);
        if (!alt.empty()) id_joint_path = alt + "/" + cfg.output_urdf_joint;
      }
    }
    pinocchio::Model pinocchio_model_joint;
    pinocchio::urdf::buildModel(id_joint_path, pinocchio_model_joint);
    pinocchio::Data pinocchio_data_joint(pinocchio_model_joint);

    Eigen::Matrix3d R_y_j = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d R_z_j = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d R_base_to_world_j = R_z_j * R_y_j;
    Eigen::Vector3d gravity_world_j(0, 0, -9.81);
    pinocchio_model_joint.gravity.linear() = R_base_to_world_j.transpose() * gravity_world_j;

    std::vector<double> tau_pred_joint_list, tau_meas_joint_list;
    for (int i = validation_start; i < collected_data.size(); i++) {
      const auto& point = collected_data[i];
      Eigen::VectorXd q_e(7), dq_e(7), ddq_e(7);
      for (int j = 0; j < 7; j++) {
        q_e(j) = point.q[j];
        dq_e(j) = point.dq[j];
        ddq_e(j) = point.ddq[j];
      }
      Eigen::VectorXd tau_pred_j = pinocchio::rnea(pinocchio_model_joint, pinocchio_data_joint, q_e, dq_e, ddq_e);
      for (int j = 0; j < 7; j++) {
        tau_pred_joint_list.push_back(tau_pred_j(j));
        tau_meas_joint_list.push_back(point.tau[j]);
      }
    }

    if (!tau_pred_joint_list.empty() && tau_pred_joint_list.size() == tau_meas_joint_list.size()) {
      Eigen::Map<Eigen::VectorXd> tau_pj(tau_pred_joint_list.data(), tau_pred_joint_list.size());
      Eigen::Map<Eigen::VectorXd> tau_mj(tau_meas_joint_list.data(), tau_meas_joint_list.size());
      Eigen::VectorXd err_j = tau_mj - tau_pj;
      double rmse_joint = std::sqrt(err_j.squaredNorm() / err_j.size());
      double max_err_joint = err_j.cwiseAbs().maxCoeff();
      std::cout << "  验证集 RMSE (辨识URDF[JOINT] + rnea): " << std::fixed << std::setprecision(4)
                << rmse_joint << " Nm" << std::endl;
      std::cout << "  验证集 最大绝对误差 (辨识URDF[JOINT] + rnea): " << std::fixed << std::setprecision(4)
                << max_err_joint << " Nm" << std::endl;

      std::ofstream res_append_joint(cfg.result_file, std::ios::app);
      res_append_joint << "验证集-辨识URDF[JOINT](rnea): RMSE=" << rmse_joint << " Nm, 最大绝对误差=" << max_err_joint << " Nm\n";
      res_append_joint.close();
    } else {
      std::cout << "  (提示) 验证集数据为空，跳过辨识URDF[JOINT]的RMSE计算。" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cout << "  (提示) 使用辨识URDF[JOINT]进行验证时出错: " << e.what() << std::endl;
  }
#endif
}

/**
 * @brief main program
 */
int main(int argc, char * argv[])
{
  std::cout << "========================================" << std::endl;
  std::cout << "xCore SDK 动力学参数辨识计算程序" << std::endl;
  std::cout << "========================================" << std::endl;

  // 加载配置：依次尝试 config/、src/config/、可执行文件目录/config/
  std::string config_path = "config/step2_dynamics_parameter_estimation.yaml";
  {
    std::ifstream check(config_path);
    if (!check.good()) {
      config_path = "src/config/step2_dynamics_parameter_estimation.yaml";
      check.open(config_path);
    }
    if (!check.good()) {
      std::string exe_dir = dirnameOf(argv[0]);
      if (!exe_dir.empty()) {
        config_path = exe_dir + "/config/step2_dynamics_parameter_estimation.yaml";
      }
    }
  }
  Step2Config cfg = loadStep2Config(config_path);
  std::cout << "  配置: " << config_path << std::endl;

  // 命令行覆盖：argv[1]=数据文件, argv[2]=URDF
  std::string data_file = cfg.data_file;
  std::string urdf_file = cfg.urdf_file;
  if (argc > 1) data_file = argv[1];
  if (argc > 2) urdf_file = argv[2];

  std::cout << "数据文件: " << data_file << std::endl;
  std::cout << "URDF文件: " << urdf_file << std::endl;
  std::cout << "========================================" << std::endl;

  try {
    std::cout << "\n========================================" << std::endl;
    std::cout << "开始动力学参数辨识..." << std::endl;
    std::cout << "========================================" << std::endl;
    estimateDynamicsParameters(data_file, urdf_file, cfg);
    std::cout << "\n========================================" << std::endl;
    std::cout << "动力学参数辨识结束" << std::endl;
    std::cout << "========================================" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "发生异常: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}

