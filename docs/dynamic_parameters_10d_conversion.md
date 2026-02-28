# 10 维动力学参数与 URDF 惯性参数的转换关系

本项目中 `dynamics_parameters_urdf.csv`、`dynamics_parameters.csv` 中每连杆的 10 个数，与 Pinocchio 的 `Inertia::toDynamicParameters()` / `Inertia::FromDynamicParameters()` 一致，和 URDF 的 `<inertial>` 存在如下对应关系。

## 1. 代码中的转换位置

转换**完全由 Pinocchio 完成**，本仓库只调用其接口：

- **URDF → 10 维**：加载 URDF 后，`model.inertias[jid].toDynamicParameters()` 得到 10 维向量，写入 `theta_urdf` 并保存到 `dynamics_parameters_urdf.csv`。
- **10 维 → 质量/质心/惯性**：`pinocchio::Inertia::FromDynamicParameters(pi)` 将 10 维向量还原为质量、质心、惯性张量，用于写 URDF 和打印 `dynamics_physical_parameters_identified.txt`。

相关代码见：

- `src/step2_dynamics_parameter_estimation.cpp`：约 509–521 行（构造 `theta_urdf`）、94 行与 224/233/258/266 行（`FromDynamicParameters` / `toDynamicParameters`）。

## 2. Pinocchio 的 10 维定义（与源码一致）

Pinocchio 头文件 `pinocchio/spatial/inertia.hpp` 中注释为：

```text
v = [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]^T
```

其中：

- `c = (c_x, c_y, c_z)` 为**质心在连杆坐标系下的位置**（即 URDF 的 `<origin xyz="cx cy cz"/>`）。
- 惯性 6 元为**质心处**惯性张量 \(I_C\) 的 6 个独立分量；连杆原点处惯性 \(I = I_C + m S^T(c)S(c)\)（\(S(c)\) 为叉积反对称矩阵）。

即：

| 下标 | 含义 | 与 URDF 的对应 |
|------|------|----------------|
| 0 | \(m\) 质量 | `<mass value="m"/>` |
| 1 | \(m c_x\) | 质心 x：`origin xyz` 第一项，= param[1]/param[0] |
| 2 | \(m c_y\) | 质心 y：= param[2]/param[0] |
| 3 | \(m c_z\) | 质心 z：= param[3]/param[0] |
| 4 | \(I_{xx}\)（质心处） | `<inertia ixx="..."/>` |
| 5 | \(I_{xy}\) | `ixy` |
| 6 | \(I_{yy}\) | `iyy` |
| 7 | \(I_{xz}\) | `ixz` |
| 8 | \(I_{yz}\) | `iyz` |
| 9 | \(I_{zz}\) | `izz` |

惯性 6 元顺序与 Pinocchio 的 `Symmetric3` 一致：Ixx, Ixy, Iyy, Ixz, Iyz, Izz（见 `pinocchio/spatial/symmetric3.hpp`）。

## 3. 公式小结

- **URDF → 10 维（toDynamicParameters）**  
  - \(v_0 = m\)  
  - \(v_{1:3} = m \cdot (c_x, c_y, c_z)\)  
  - \(v_{4:9}\) = 质心处惯性 \(I_C\) 的 6 元 (Ixx, Ixy, Iyy, Ixz, Iyz, Izz)。  
  源码中 \(v_{4:9}\) 来自 `(inertia() - AlphaSkewSquare(mass(), lever())).data()`，即从连杆原点惯性减去 \(m S^T(c)S(c)\) 得到 \(I_C\)。

- **10 维 → URDF（FromDynamicParameters）**  
  - \(m = \texttt{params}[0]\)  
  - \((c_x, c_y, c_z) = \texttt{params}[1:3] / m\)  
  - 质心处惯性 6 元 = \(\texttt{params}[4:9]\)；连杆原点惯性 = 该 6 元构成的 \(I_C + m S^T(c)S(c)\)。

因此：**CSV 中每连杆的 10 个数与 URDF 的 mass、origin xyz、inertia 6 元是一一对应的**，仅顺序和“质量×质心”的写法与 URDF 标签不同；用上表即可在 CSV 与 URDF 之间互推。
