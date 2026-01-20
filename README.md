# 动力学辨识归档包（xCoreSDK_cpp-v0.5.1）

本文件夹用于**长期保存**本次“动力学数据采集 + Pinocchio 回归辨识 + 结果导出 + URDF 更新”相关代码与产物（仅归档，不影响原工程）。

---

## 目录结构

- `src/`
  - `dynamics_data_collection.cpp`：上电、进入RT、执行激励轨迹、采集 `q/dq/ddq/tau` 并保存 CSV
  - `dynamics_parameter_estimation.cpp`：离线读取 CSV，用 Pinocchio 构建回归矩阵并辨识参数，输出结果文件
  - `dynamics_identification.cpp`：历史/合并版（保留参考）
  - `CMakeLists_rt.txt`：当时 `example/rt/CMakeLists.txt` 的快照
- `build_outputs/`（运行产物）
  - `dynamics_identification_data.csv`：采集数据（大文件）
  - `dynamics_identification_results.txt`：辨识摘要与误差评估
  - `dynamics_parameters.csv`：带 URDF 先验的 Ridge 结果
  - `dynamics_parameters_ls.csv`：纯最小二乘（SVD）结果
  - `dynamics_parameters_urdf.csv`：URDF 对应参数（先验）
  - `dynamics_physical_parameters.txt`：URDF 物理参数导出（参考）
  - `dynamics_physical_parameters_identified.txt`：由辨识参数重建的物理参数（对比用）
- `urdf/`
  - `AR5-5_07R-W4C4A2.urdf`：原始URDF
  - `AR5-5_07R-W4C4A2_identified.urdf`：**带“换算/正定化”处理**后的 identified URDF（更稳）
  - `AR5-5_07R-W4C4A2_identified_direct_from_txt.urdf`：**严格按 txt 直接覆盖**（不换算/不正定化，可能包含负惯性）
  - `generate_identified_urdf.py`：从 `dynamics_physical_parameters_identified.txt` 生成 identified URDF 的脚本
- `docs/`
  - `pinocchio_dynamics*.tex` 等：Pinocchio 动力学/辨识/双臂/运动学辨识说明文档
- `MANIFEST.txt`
  - 本归档的文件清单（含大小、时间）

---

## 推荐工作流（最常用）

## 如何编译（从源码到可执行文件）

> 说明：归档包里保存的是源码快照；**实际编译请在工程目录** `03_control_system/xCoreSDK_cpp-v0.5.1/` 下进行。

### 依赖前提

- **必须**：开启 xMateModel（否则相关 target 不会生成）
  - CMake 选项：`-DXCORE_USE_XMATE_MODEL=ON`
- **需要 Pinocchio 的程序**：
  - `dynamics_parameter_estimation`
  - `dynamics_identification`
  -（如果系统已安装到 `/opt/ros/humble` 或系统路径，一般可直接找到）

### 编译命令（推荐）

在工程目录执行：

```bash
cd /home/lenovo/Frank/doc/rokae_arm_source/03_control_system/xCoreSDK_cpp-v0.5.1
mkdir -p build
cmake -S . -B build -DXCORE_USE_XMATE_MODEL=ON
cmake --build build -j"$(nproc)" --target dynamics_data_collection dynamics_parameter_estimation dynamics_identification
```

编译成功后，可执行文件通常在：

- `build/bin/dynamics_data_collection`
- `build/bin/dynamics_parameter_estimation`
- `build/bin/dynamics_identification`

### 1) 采集数据（在线，接机器人）

使用工程里编译出来的 `dynamics_data_collection` 采集，得到：

- `build_outputs/dynamics_identification_data.csv`

> 归档包里只保存源码与产物；实际可执行文件在工程的 `build/bin` 下。

#### 运行方法（推荐用这个程序采集）

```bash
cd /home/lenovo/Frank/doc/rokae_arm_source/03_control_system/xCoreSDK_cpp-v0.5.1/build/bin
./dynamics_data_collection <robot_ip> <output_csv>
```

- **robot_ip**：机器人控制器IP（默认示例里是 `192.168.110.15`）
- **output_csv**：输出CSV文件名（默认 `dynamics_identification_data.csv`）
- 程序会自动检测**本机IP**（要求本机网卡与机器人同网段）

输出文件（默认）：

- `dynamics_identification_data.csv`（采集到的 `time,q,dq,ddq,tau`）

### 2) 参数辨识（离线）

用 `dynamics_parameter_estimation` 读取 CSV 并输出：

- `dynamics_identification_results.txt`
- `dynamics_parameters*.csv`
- `dynamics_physical_parameters_identified.txt`

#### 运行方法（离线辨识）

```bash
cd /home/lenovo/Frank/doc/rokae_arm_source/03_control_system/xCoreSDK_cpp-v0.5.1/build/bin
./dynamics_parameter_estimation <data_csv> [urdf_path]
```

- **data_csv**：采集得到的CSV（例如 `dynamics_identification_data.csv`）
- **urdf_path**：可选；不传时默认使用编译期注入的 `URDF_FILE_PATH`（避免相对路径坑）

典型输出（在运行目录下生成/更新）：

- `dynamics_identification_results.txt`：整体误差、每关节RMSE等
- `dynamics_parameters.csv`：带URDF先验Ridge的参数
- `dynamics_parameters_ls.csv`：纯最小二乘（SVD）参数
- `dynamics_parameters_urdf.csv`：URDF先验参数向量
- `dynamics_physical_parameters_identified.txt`：把参数重建为质量/质心/惯性用于对比

### 3) 生成新的 URDF

#### 方式A（推荐更稳）：换算到质心 + 正定化

使用 `urdf/generate_identified_urdf.py` 生成 `AR5-5_07R-W4C4A2_identified.urdf` 风格的URDF。

#### 方式B（严格按 txt 直接写入）：不换算/不正定化

`urdf/AR5-5_07R-W4C4A2_identified_direct_from_txt.urdf` 就是这种方式的结果。

---

## 重要注意事项（强烈建议读）

- **惯性参考点问题**：`dynamics_physical_parameters_identified.txt` 里打印的“辨识惯性矩阵”与 URDF `<inertia>` 期望的参考点可能不同（是否为质心）。如果你“直接覆盖”，理论上可能把参考点写错。
- **物理可行性问题**：辨识得到的惯性矩阵可能出现**负惯性/非正定**，直接写入 URDF 可能导致后续动力学/仿真数值不稳定甚至崩溃。
- **先验与不可辨识方向**：当回归矩阵秩不足时，部分参数不可辨识，Ridge/URDF先验会影响这些方向上的结果，这属于预期行为。

---

## `dynamics_identification.cpp`（历史/合并版）怎么用？

该程序把“在线采集 + 离线计算”揉在一起，主要用于早期调试验证，当前仍保留用于参考。

运行方式：

```bash
cd /home/lenovo/Frank/doc/rokae_arm_source/03_control_system/xCoreSDK_cpp-v0.5.1/build/bin
./dynamics_identification
```

注意：

- 它的 `robot_ip` 在源码里是**写死的**（默认 `192.168.110.15`），如需修改请改源码后重新编译。
- 它会在运行目录下写出 `dynamics_identification_data.csv` 等文件（具体以程序输出为准）。

---

## 版本信息

- 归档创建时间：2026-01-20
- 对应工程：`03_control_system/xCoreSDK_cpp-v0.5.1/`


