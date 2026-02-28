# Step2 动力学参数辨识 (Python 版)

与 C++ 版 `src/step2_dynamics_parameter_estimation.cpp` 等价的 Python 实现，用于从采集数据辨识机械臂动力学参数。

## 依赖

**方式一：直接使用 ROS 自带的 pinocchio（推荐在已有 ROS 环境下）**

```bash
source /opt/ros/humble/setup.bash   # 或你的 ROS 发行版
pip install -r requirements.txt     # 安装 numpy、PyYAML、osqp 等；若用 ROS pinocchio 则需 numpy<2
```

ROS 自带的 pinocchio 多为 NumPy 1.x 编译，请使用 **numpy<2**（例如 `pip install "numpy<2"`），否则可能报错。

**方式二：使用 conda / pip 的 Pinocchio（PyPI 包名 `pin`）**

```bash
conda activate pin27
pip install pin --upgrade
pip install -r requirements.txt
```

- **numpy**：数值计算；用 ROS pinocchio 时需 numpy<2  
- **PyYAML**：读取 YAML 配置（可选）  
- **pinocchio**：ROS 自带或 pip/conda 安装（`import pinocchio as pin`）  
- **osqp**（可选）：带质量约束的 QP 求解

## 配置

与 C++ 共用同一配置文件，默认从以下路径之一加载：

- `config/step2_dynamics_parameter_estimation.yaml`
- `src/config/step2_dynamics_parameter_estimation.yaml`

配置项：`data_file`、`urdf`、`lambda_rel`、`m_min`、`I_eps`、`I_trace_min`、**`output_dir`**（输出目录，默认 `build_outputs`）及各输出文件名（`result_file`、`dynamics_parameters_csv`、`output_urdf`、`output_urdf_joint` 等）。所有结果文件统一写入 `output_dir`（若配置为非空）。

## 运行

在项目根目录下（或保证数据/URDF 路径正确）：

```bash
# 使用配置文件中的 data_file 和 urdf
python scripts/step2_dynamics_parameter_estimation.py

# 指定配置与数据、URDF
python scripts/step2_dynamics_parameter_estimation.py --config src/config/step2_dynamics_parameter_estimation.yaml
python scripts/step2_dynamics_parameter_estimation.py config/dynamics_identification_data.csv urdf/AR5-5_07R-W4C4A2_Manual_fix.urdf

# 与 C++ 结果一致：调用 C++ 可执行文件做辨识（需先编译 C++ 工程）
python scripts/step2_dynamics_parameter_estimation.py --use-cpp src/config/dynamics_identification_data.csv urdf/AR5-5_07R-W4C4A2_Manual_fix.urdf
```

## 输入

- **CSV**：表头一行；每行 `timestamp, q1..q7, dq1..dq7, ddq1..ddq7, tau1..tau7`。  
- **URDF**：与 C++ 相同的 7 自由度机械臂 URDF。

## 输出

与 C++ 一致；**默认统一写入目录 `build_outputs/`**（可在配置中修改 `output_dir`，空则写当前目录）：

- `build_outputs/dynamics_identification_results.txt`：辨识结果与验证集 RMSE  
- `build_outputs/dynamics_parameters.csv` / `dynamics_parameters_ls.csv` / `dynamics_parameters_urdf.csv`  
- `build_outputs/dynamics_physical_parameters_identified.txt`：物理参数对比  
- `build_outputs/*_identified_inertia_com.urdf`：质心处惯量  
- `build_outputs/*_identified_inertia_joint_origin.urdf`：关节原点处惯量  

验证：后 20% 数据作为验证集，汇报「原 URDF + Y*θ」与「原 URDF / 辨识 COM / 辨识 JOINT + rnea」的 RMSE 与最大绝对误差。

## 与 C++ 结果差异

若 Python 的**回归矩阵秩**明显低于 C++（例如 Python 约 7、C++ 约 45），则辨识结果会偏重 URDF 先验，RMSE 会高于 C++。原因多为 **Pinocchio 实现差异**（C++ 通常链接系统 Pinocchio 2.x，Python 常用 pip 的 0.4 或 conda 的 3.x，回归矩阵数值秩不同）。

建议：

1. **同一数据下对比**：用相同 CSV 与 URDF 分别运行 C++ step2 与 Python 脚本，看终端打印的「回归矩阵的秩」是否一致。  
2. **C++ 秩高、Python 秩低**：属库差异，可接受 Python 结果更偏 URDF，或在本机用 conda 安装与 C++ 同源的 Pinocchio 再试：`conda install -c conda-forge pinocchio`。  
3. **两边秩都较低**：多为激励不足，可增加轨迹多样性或采集时长。
