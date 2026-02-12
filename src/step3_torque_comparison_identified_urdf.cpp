/**
 * @file step3_torque_comparison_identified_urdf.cpp
 * @brief Step3：对比运动过程中由辨识 URDF + Pinocchio 计算的关节力矩 与 力/力矩传感器（关节力矩）实测值的差异
 *
 * 用法:
 *   ./step3_torque_comparison_identified_urdf <数据CSV> [URDF路径]
 * 数据CSV 格式与 step1 采集一致：timestamp, q1..q7, dq1..dq7, ddq1..ddq7, tau1..tau7（tau 为传感器测量）
 * 默认 URDF: AR5-5_07R-W4C4A2_identified_joint.urdf
 *
 * 输出: 各关节及整体的 RMSE、最大绝对误差，并可选输出 torque_comparison.csv
 *
 * @copyright Copyright (C) 2024 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
 
 #ifdef __has_include
   #if __has_include("pinocchio/multibody/model.hpp")
     #include "pinocchio/multibody/model.hpp"
     #include "pinocchio/multibody/data.hpp"
     #include "pinocchio/parsers/urdf.hpp"
     #include "pinocchio/algorithm/rnea.hpp"
     #define PINOCCHIO_AVAILABLE
   #endif
 #endif
 
 #include "Eigen/Dense"
 
 struct DataPoint {
   std::array<double, 7> q;
   std::array<double, 7> dq;
   std::array<double, 7> ddq;
   std::array<double, 7> tau;  // 传感器测量关节力矩 (Nm)
   double timestamp;
 };
 
 static std::vector<DataPoint> loadDataFromCSV(const std::string& filename) {
   std::vector<DataPoint> data;
   std::ifstream file(filename);
   if (!file.is_open()) {
     std::cerr << "错误: 无法打开文件 " << filename << std::endl;
     return data;
   }
   std::string line;
   std::getline(file, line);  // 标题行
   while (std::getline(file, line)) {
     if (line.empty()) continue;
     std::istringstream iss(line);
     std::string token;
     DataPoint point;
     std::getline(iss, token, ',');
     point.timestamp = std::stod(token);
     for (int i = 0; i < 7; i++) {
       std::getline(iss, token, ',');
       point.q[i] = std::stod(token);
     }
     for (int i = 0; i < 7; i++) {
       std::getline(iss, token, ',');
       point.dq[i] = std::stod(token);
     }
     for (int i = 0; i < 7; i++) {
       std::getline(iss, token, ',');
       point.ddq[i] = std::stod(token);
     }
     for (int i = 0; i < 7; i++) {
       std::getline(iss, token, ',');
       point.tau[i] = std::stod(token);
     }
     data.push_back(point);
   }
   file.close();
   return data;
 }
 
 int main(int argc, char* argv[]) {
 #ifdef PINOCCHIO_AVAILABLE
   std::string data_file = "dynamics_identification_data.csv";
   std::string urdf_file = "AR5-5_07R-W4C4A2_identified_joint.urdf";
   if (argc > 1) data_file = argv[1];
   if (argc > 2) urdf_file = argv[2];
 
   // 若默认 URDF 路径下文件不存在，尝试数据文件所在目录下的同名 URDF（便于从 build/ 运行）
   auto file_exists = [](const std::string& path) {
     std::ifstream f(path);
     return f.good();
   };
   if (!file_exists(urdf_file)) {
     std::string dir = data_file;
     size_t sep = dir.find_last_of("/\\");
     if (sep != std::string::npos) {
       std::string candidate = dir.substr(0, sep + 1) + "AR5-5_07R-W4C4A2_identified_joint.urdf";
       if (file_exists(candidate)) urdf_file = candidate;
     }
   }
 
   std::cout << "========================================" << std::endl;
   std::cout << "Step3 关节力矩对比：辨识URDF(Pinocchio) vs 传感器" << std::endl;
   std::cout << "========================================" << std::endl;
   std::cout << "数据文件: " << data_file << std::endl;
   std::cout << "URDF文件: " << urdf_file << std::endl;
   std::cout << "========================================" << std::endl;
 
   pinocchio::Model model;
   pinocchio::Data data;
   try {
     if (!file_exists(urdf_file)) {
       std::cerr << "错误: URDF 文件不存在: " << urdf_file << std::endl;
       std::cerr << "  请指定数据与 URDF 路径，例如: " << argv[0] << " <数据.csv> <URDF.urdf>" << std::endl;
       return 1;
     }
     pinocchio::urdf::buildModel(urdf_file, model);
   } catch (const std::exception& e) {
     std::cerr << "错误: 无法加载 URDF: " << e.what() << std::endl;
     return 1;
   }
   std::cout << "  Pinocchio 模型加载成功, nq=" << model.nq << ", nv=" << model.nv << std::endl;
 
   // 重力方向（与 step2.2 一致：基座坐标系下）
   Eigen::Matrix3d R_y = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
   Eigen::Matrix3d R_z = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
   Eigen::Matrix3d R_base_to_world = R_z * R_y;
   Eigen::Vector3d gravity_world(0, 0, -9.81);
   model.gravity.linear() = R_base_to_world.transpose() * gravity_world;
   data = pinocchio::Data(model);
 
   std::vector<DataPoint> points = loadDataFromCSV(data_file);
   if (points.empty()) {
     std::cerr << "错误: 未读取到数据点" << std::endl;
     return 1;
   }
   std::cout << "  读取数据点: " << points.size() << std::endl;
 
   const int nv = model.nv;
   Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
   Eigen::VectorXd v = Eigen::VectorXd::Zero(nv);
   Eigen::VectorXd a = Eigen::VectorXd::Zero(nv);
 
  std::vector<double> sum_sq_err(7, 0.0);
  std::vector<double> max_abs_err(7, 0.0);
  std::vector<int> count(7, 0);
  std::vector<std::array<double, 7>> tau_pred_list;

  std::ofstream out_csv("torque_comparison.csv");
  if (out_csv.is_open()) {
    out_csv << "time,tau_meas_1,tau_meas_2,tau_meas_3,tau_meas_4,tau_meas_5,tau_meas_6,tau_meas_7"
            << ",tau_pred_1,tau_pred_2,tau_pred_3,tau_pred_4,tau_pred_5,tau_pred_6,tau_pred_7"
            << ",err_1,err_2,err_3,err_4,err_5,err_6,err_7\n";
  }

  for (const auto& pt : points) {
    for (int i = 0; i < 7; i++) {
      q(i) = pt.q[i];
      v(i) = pt.dq[i];
      a(i) = pt.ddq[i];
    }
    pinocchio::rnea(model, data, q, v, a);
    const Eigen::VectorXd& tau_pred = data.tau;

    std::array<double, 7> tp;
    for (int i = 0; i < 7; i++) tp[i] = tau_pred(i);
    tau_pred_list.push_back(tp);

    for (int i = 0; i < 7; i++) {
      double e = pt.tau[i] - tau_pred(i);
      sum_sq_err[i] += e * e;
      max_abs_err[i] = std::max(max_abs_err[i], std::abs(e));
      count[i]++;
    }

    if (out_csv.is_open()) {
      out_csv << std::fixed << std::setprecision(6) << pt.timestamp;
      for (int i = 0; i < 7; i++) out_csv << "," << pt.tau[i];
      for (int i = 0; i < 7; i++) out_csv << "," << tau_pred(i);
      for (int i = 0; i < 7; i++) out_csv << "," << (pt.tau[i] - tau_pred(i));
      out_csv << "\n";
    }
  }
   if (out_csv.is_open()) {
     out_csv.close();
     std::cout << "  已写入: torque_comparison.csv" << std::endl;
   }
 
   std::cout << "\n========================================" << std::endl;
   std::cout << "关节力矩对比结果 (URDF+Pinocchio vs 传感器)" << std::endl;
   std::cout << "========================================" << std::endl;
   std::cout << std::fixed << std::setprecision(6);
   double total_sq = 0.0;
   int total_n = 0;
   for (int i = 0; i < 7; i++) {
     int n = count[i];
     double rmse = (n > 0) ? std::sqrt(sum_sq_err[i] / n) : 0.0;
     total_sq += sum_sq_err[i];
     total_n += n;
     std::cout << "  关节 " << (i + 1)
               << "  RMSE(Nm): " << std::setw(10) << rmse
               << "  最大绝对误差(Nm): " << std::setw(10) << max_abs_err[i] << std::endl;
   }
  double overall_rmse = (total_n > 0) ? std::sqrt(total_sq / total_n) : 0.0;
  std::cout << "  ---------------" << std::endl;
  std::cout << "  整体 RMSE(Nm): " << overall_rmse << std::endl;
  std::cout << "========================================" << std::endl;

  // 随机抽取 10 个数据点做逐点对比
  const int N = static_cast<int>(points.size());
  const int sample_count = std::min(10, N);
  if (sample_count > 0 && tau_pred_list.size() == static_cast<size_t>(N)) {
    std::vector<int> indices(N);
    for (int i = 0; i < N; i++) indices[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    std::sort(indices.begin(), indices.begin() + sample_count);

    std::cout << "\n随机 " << sample_count << " 个数据点对比 (传感器 vs Pinocchio):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (int s = 0; s < sample_count; s++) {
      int idx = indices[s];
      const auto& pt = points[idx];
      const auto& tp = tau_pred_list[idx];
      std::cout << "  [#" << (idx + 1) << " t=" << pt.timestamp << "s]" << std::endl;
      std::cout << "    关节  传感器(Nm)  Pinocchio(Nm)  误差(Nm)" << std::endl;
      for (int j = 0; j < 7; j++) {
        double err = pt.tau[j] - tp[j];
        std::cout << "    " << (j + 1) << "     " << std::setw(9) << pt.tau[j]
                  << "   " << std::setw(12) << tp[j] << "   " << std::setw(9) << err << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "========================================" << std::endl;
  }

  return 0;
 #else
   (void)argc;
   (void)argv;
   std::cerr << "错误: 未找到 Pinocchio，无法编译本程序" << std::endl;
   return 1;
 #endif
 }
 