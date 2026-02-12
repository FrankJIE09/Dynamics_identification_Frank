# 动力学辨识（Dynamics Identification）

基于 Pinocchio 的机械臂动力学参数辨识：从 DH/URDF 到数据采集、回归辨识、结果导出与 URDF 更新。

---

## 目录结构

```
├── CMakeLists.txt          # 本仓库独立编译（step2 等）
├── src/
│   ├── step0_dh_to_urdf.py           # Step0: DH → URDF（标准 DH）
│   ├── step0_mdh_to_urdf.py           # Step0: MDH → URDF（修正 DH）
│   ├── step1_dynamics_data_collection*.cpp   # Step1: 在线数据采集（需 Rokae SDK）
│   ├── step2_dynamics_parameter_estimation.cpp      # Step2: 离线参数辨识
│   ├── step2.2_dynamics_parameter_estimation_joint.cpp  # Step2.2: 惯性+摩擦联合辨识
│   ├── dynamics_parameter_estimation.cpp   # 早期版辨识程序（保留）
│   ├── dynamics_data_collection.cpp
│   ├── dynamics_identification.cpp
│   └── CMakeLists_rt.txt   # xCoreSDK 工程内使用的 CMake 快照
├── urdf/
│   ├── AR5-5_07R-W4C4A2.urdf              # 原始/标称 URDF
│   ├── AR5-5_07R-W4C4A2_ Manual_fix.urdf   # 手动修正 URDF（可选）
│   ├── AR5-5_07R-W4C4A2_identified.urdf    # 辨识后 URDF（推荐）
│   ├── AR5-5_07R-W4C4A2_identified_direct_from_txt.urdf
│   └── generate_identified_urdf.py        # 从辨识结果生成 identified URDF
├── build_outputs/         # 推荐放置采集数据与辨识产物的目录
│   ├── dynamics_identification_data.csv   # 采集数据 (q,dq,ddq,tau)
│   ├── dynamics_identification_results.txt
│   ├── dynamics_parameters*.csv
│   └── dynamics_physical_parameters_identified.txt
└── docs/                  # Pinocchio 动力学/辨识说明文档 (.tex)
```

---

## 依赖

- **Eigen3**、**Pinocchio**：本仓库内编译 step2 所需（如通过 ROS Humble 或系统安装）。
- **Rokae xCore SDK**（仅 step1 数据采集）：需在 xCoreSDK 工程内编译 step1 可执行文件。

---

## 编译（本仓库）

在项目根目录执行：

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j$(nproc)
```

生成的可执行文件在 `build/` 下：

| 可执行文件 | 说明 |
|------------|------|
| `dynamics_parameter_estimation` | 离线辨识（早期版） |
| `step2_dynamics_parameter_estimation` | 离线辨识，支持生成 identified URDF |
| `step2_2_dynamics_parameter_estimation_joint` | 惯性 + 摩擦联合辨识 |

**说明**：`step1_dynamics_data_collection` 与 `step1_dynamics_data_collection_joint` 依赖 Rokae SDK（`rokae/robot.h`、`print_helper.hpp`），本仓库不包含这些依赖，需在 **xCoreSDK 工程**（如 `xCoreSDK_cpp-v0.5.1`）下打开对应 target 后编译。

---

## 使用流程

### Step0：DH → URDF（可选）

根据 DH/MDH 参数生成或修正 URDF，无需编译，直接运行：

```bash
python3 src/step0_dh_to_urdf.py    # 标准 DH
python3 src/step0_mdh_to_urdf.py   # 修正 DH
```

按脚本内注释修改 DH 表与输入/输出路径即可。

### Step1：数据采集（需接机与 xCoreSDK）

在 **xCoreSDK 工程** 下编译并运行数据采集程序，得到 `dynamics_identification_data.csv`（包含 `time, q, dq, ddq, tau`）。将生成的 CSV 放到本仓库的 `build_outputs/` 或任意路径备用。

### Step2：参数辨识（本仓库编译的程序）

在 `build` 目录下运行，**参数顺序**：`<数据 CSV> [URDF 路径]`。

```bash
cd build

# 使用默认 URDF（CMake 中设置的默认路径）
./dynamics_parameter_estimation ../build_outputs/dynamics_identification_data.csv
./step2_dynamics_parameter_estimation ../build_outputs/dynamics_identification_data.csv
./step2_2_dynamics_parameter_estimation_joint ../build_outputs/dynamics_identification_data.csv

# 指定不同 URDF（例如手动修正的 URDF，路径含空格时请加引号）
./dynamics_parameter_estimation ../build_outputs/dynamics_identification_data.csv "../urdf/AR5-5_07R-W4C4A2_ Manual_fix.urdf"
./step2_dynamics_parameter_estimation ../build_outputs/dynamics_identification_data.csv "../urdf/AR5-5_07R-W4C4A2_ Manual_fix.urdf"
./step2_2_dynamics_parameter_estimation_joint ../build_outputs/dynamics_identification_data.csv "../urdf/AR5-5_07R-W4C4A2_ Manual_fix.urdf"
```

辨识结果会写在**当前工作目录**（即 `build/`）下，例如：

- `dynamics_identification_results.txt` — 整体 RMSE、各关节误差
- `dynamics_parameters.csv` / `dynamics_parameters_ls.csv` / `dynamics_parameters_urdf.csv`
- `dynamics_physical_parameters_identified.txt` — 辨识得到的质量/质心/惯性（与 URDF 对比）

### 指定不同 URDF

- **运行时**：如上，第二个命令行参数传入 URDF 路径（相对或绝对均可）。
- **编译默认值**：在 `CMakeLists.txt` 中修改 `URDF_FILE_PATH`，不传第二个参数时生效。

### Step3：生成辨识后的 URDF

使用 `urdf/generate_identified_urdf.py`，根据 `dynamics_physical_parameters_identified.txt` 和原始 URDF 生成带“换算/正定化”的 identified URDF（推荐）。用法见脚本内说明或：

```bash
python3 urdf/generate_identified_urdf.py
```

---

## 注意事项

- **惯性参考点**：辨识得到的惯性矩阵与 URDF `<inertial>` 的参考点（是否在质心）可能不一致，生成 URDF 时脚本会做换算与正定化。
- **回归矩阵秩**：若不满秩，部分参数不可辨识，程序会使用 URDF 先验（Ridge）填补；可增加激励多样性或采集时长以改善。
- **step1**：仅能在配备 Rokae SDK 的 xCoreSDK 工程中编译与运行，本仓库仅提供 step2 的独立编译与运行说明。

---

## 版本与参考

- 本仓库可独立完成：Step0（Python）、Step2 编译与运行、URDF 生成。
- 数据采集（Step1）依赖 xCoreSDK 与实机/仿真环境。
