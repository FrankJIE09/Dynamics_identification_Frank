# MuJoCo `mj_inverse` 行为测试报告

## 1. 目的与背景

在动力学对比中需要从 MuJoCo 得到逆动力学 τ = M(q)q̈ + C(q,q̇)q̇ + g(q)。文档说明 `mj_inverse` 可根据当前 `qpos`、`qvel`、`qacc` 计算所需广义力并写入 `qfrc_inverse`。实际使用中发现：

- **不先调用 `mj_forward`、直接调用 `mj_inverse`**：`qfrc_inverse` 出现约 6×10⁴ 量级的异常大力矩；
- **先 `mj_forward` 再 `mj_inverse`**：`qfrc_inverse` 几乎为 0。

本报告通过可复现测试，记录上述行为及与“正确”逆动力学（用 `qfrc_bias` + M 手算 τ）的对比，便于排查与文档化。

---

## 2. 测试环境

| 项目 | 说明 |
|------|------|
| MuJoCo 版本 | 3.5.0（Python 包） |
| 模型 | `scripts/AR5-5_07R-W4C4A2/AR5-5_07R-W4C4A2.xml`，7 自由度 |
| 测试脚本 | `scripts/test_mj_inverse_behavior.py` |
| 测试状态 | 单组 (q, q̇, q̈)，与 `debug_one_sample_M_matrix.py` 一致 |

**参考“正确”逆动力学**：由 Pinocchio 及 MuJoCo 的 τ = M@q̈ + qfrc_bias 得到，作为期望值 TAU_EXPECTED。

---

## 3. 测试用例与结果

### 3.1 输入与期望值

- **q[0:7]**（rad）：[-0.4481, 0.2231, -3.0302, -0.324, 0.3123, -0.6217, -0.1062]
- **q̇[0:7]**（rad/s）：[0.863, 0.0596, -0.4948, 0.0404, 0.5824, -0.0126, 0.5475]
- **q̈[0:7]**（rad/s²）：[-2.0738, -1.0573, 4.659, -3.2083, 4.7876, -3.3108, -4.1833]
- **期望 τ**（Nm）：[-3.074, -22.6049, -0.6746, 5.274, 0.0242, 0.0917, -0.0281]

---

### 3.2 Test1：仅 `mj_inverse`（不调用 `mj_forward`）

**步骤**：`set_state(q, q̇, q̈)` → `mj_inverse(model, data)`

**结果**：

| 量 | 值 |
|----|-----|
| qfrc_inverse | [60251.50, -22.60, -0.67, 5.27, 0.024, -18.62, -0.98] |
| \|qfrc_inverse\|∞ | 60251.50 |
| 与期望 τ 的 L∞ 差 | 60254.57 |

**结论**：不先做 `mj_forward` 时，`mj_inverse` 写出的 `qfrc_inverse` 在第 0 轴约 6×10⁴ Nm，与期望约 -3 Nm 完全不符，其余轴量级合理。说明**未先更新正运动学/动力学内部状态时，`mj_inverse` 的输出不可信**。

---

### 3.3 Test2：先 `mj_forward`，再 `mj_inverse`

**步骤**：`set_state(q, q̇, q̈)` → `mj_forward(model, data)` → `mj_inverse(model, data)`

**结果**：

| 量 | 值 |
|----|-----|
| qfrc_inverse | [0, 0, 0, 0, 0, 0, 0]（数量级约 1e-10） |
| qfrc_bias（inverse 前） | [-3.07, -22.57, -0.71, 5.43, 0.0008, 0.11, -0.028] |
| qfrc_bias（inverse 后） | 与 inverse 前**完全一致**（L∞ 差 0） |
| 与期望 τ 的 L∞ 差 | 22.60（因 qfrc_inverse≈0） |

**结论**：先 `mj_forward` 再 `mj_inverse` 时，**`mj_inverse` 将 `qfrc_inverse` 写为近似 0**，而不是 τ = Mq̈ + qfrc_bias。`mj_inverse` **不修改** `qfrc_bias`。

---

### 3.4 Test3：`mj_forward` + `mj_makeM` 后 `mj_inverse`

**步骤**：`set_state` → `mj_forward` → `mj_makeM` → `mj_inverse`

**结果**：

| 量 | 值 |
|----|-----|
| qfrc_inverse | 仍为近似 0 |
| M@q̈ + qfrc_bias（手算） | [-3.074, -22.6049, -0.6746, 5.2741, 0.0242, 0.0918, -0.0281] |

**结论**：即使先形成质量矩阵 M，再调用 `mj_inverse`，`qfrc_inverse` 仍为 0；而用同一时刻的 `qfrc_bias` 与 M 手算的 τ 与期望一致。进一步说明**在“先 forward 再 inverse”的调用顺序下，`mj_inverse` 的输出不是我们通常理解的逆动力学力矩**。

---

### 3.5 Test4：基准——仅用 `qfrc_bias` 与 M，不调用 `mj_inverse`

**步骤**：`set_state` → `mj_forward` → `mj_makeM` → **τ = M@q̈ + qfrc_bias**

**结果**：

| 量 | 值 |
|----|-----|
| τ（手算） | [-3.074, -22.6049, -0.6746, 5.2741, 0.0242, 0.0918, -0.0281] |
| 与期望 τ 的 L∞ 差 | 0.0001 |

**结论**：不依赖 `mj_inverse`，仅用 `mj_forward` 得到的 `qfrc_bias` 与 `mj_makeM` 得到的 M 计算 τ，与 Pinocchio 及期望值一致，可作为**正确逆动力学的实现方式**。

---

## 4. 行为归纳与可能原因

### 4.1 行为归纳

| 调用顺序 | qfrc_inverse 结果 | 是否适合用作 τ |
|----------|-------------------|----------------|
| 仅 mj_inverse | 第 0 轴约 6×10⁴，其余量级正常 | 否（严重错误） |
| mj_forward → mj_inverse | 近似全 0 | 否（非逆动力学力矩） |
| 不调用 mj_inverse，用 qfrc_bias + M 算 τ | 与期望一致 | **是** |

### 4.2 可能原因（基于现象的推测）

1. **未先 `mj_forward` 就 `mj_inverse`**  
   内部依赖的正运动学、速度、偏差力等未更新，`mj_inverse` 可能使用了未初始化或过期的缓冲区，导致第 0 轴出现错误的大数值。

2. **先 `mj_forward` 再 `mj_inverse` 时 `qfrc_inverse` 为 0**  
   文档中 “inverse maps acceleration to force” 可能在此实现下被理解为：在**已经做过 forward** 的前提下，inverse 只计算“为达到给定 qacc 还需要**额外**施加的力”。若当前状态与 qacc 在数值上已被视为一致（或内部已用 qacc 更新过状态），则“额外力”为 0，从而 `qfrc_inverse` 被写为 0。  
   上述仅为对现象的合理解释，未查阅 MuJoCo 源码，不作为官方语义结论。

3. **`qfrc_bias` 不受 `mj_inverse` 影响**  
   测试中 inverse 前后 `qfrc_bias` 完全一致，说明 `mj_inverse` 不覆盖 `qfrc_bias`，用 `qfrc_bias` 做分解是安全的。

---

## 5. 建议用法（与现有文档一致）

- **不要**依赖 `mj_inverse` 的输出来得到“逆动力学 τ”或做 τ = g + Mq̈ + Cq̇ 的分解。
- **应**采用：
  1. `mj_forward` 在相应 (q, q̇) 下得到 `qfrc_bias`（用于 g 与 Cq̇）；
  2. `mj_makeM`（在 `mj_forward` 之后）得到 M；
  3. 自行计算 **τ = M@q̈ + qfrc_bias**，并据此分解 Mq̈、Cq̇、g。

实现参考：`scripts/debug_one_sample_M_matrix.py`、`scripts/test_mj_inverse_behavior.py`。

---

## 6. 原始测试输出（便于复现）

```
============================================================
mj_inverse 行为测试报告（原始数据）
============================================================
模型: AR5-5_07R-W4C4A2.xml, nv = 7
输入: q[0]=-0.4481, dq[0]=0.8630, ddq[0]=-2.0738
期望 τ (Pinocchio / M@ddq+qfrc_bias): [ -3.074  -22.6049  -0.6746   5.274    0.0242   0.0917  -0.0281]

------------------------------------------------------------
Test1: 仅 mj_inverse（不调用 mj_forward）
步骤: set_state(q,dq,ddq) → mj_inverse(model, data)
  qfrc_inverse: [60251.4959   -22.6049    -0.6746     5.2741     0.0242   -18.6191  -0.9848]
  |qfrc_inverse|_∞: 60251.4959
  与期望 τ 的差 (L∞): 60254.5699

------------------------------------------------------------
Test2: mj_forward 后 mj_inverse
步骤: set_state(q,dq,ddq) → mj_forward → mj_inverse
  qfrc_inverse: [ 0. -0. -0.  0. -0. -0. -0.]
  |qfrc_inverse|_∞: 0.0000
  与期望 τ 的差 (L∞): 22.6049
  qfrc_bias (inverse 前): [ -3.0723 -22.5733  -0.7126   5.4266   0.0008   0.1126  -0.0278]
  qfrc_bias (inverse 后): [ -3.0723 -22.5733  -0.7126   5.4266   0.0008   0.1126  -0.0278]
  inverse 是否改写 qfrc_bias (L∞): 0.000000

------------------------------------------------------------
Test3: mj_forward + mj_makeM 后 mj_inverse
步骤: set_state → mj_forward → mj_makeM → mj_inverse
  qfrc_inverse: [ 0. -0. -0.  0. -0. -0. -0.]
  M@ddq + qfrc_bias (应=τ): [ -3.074  -22.6049  -0.6746   5.2741   0.0242   0.0918  -0.0281]

------------------------------------------------------------
Test4: 基准 τ = M@ddq + qfrc_bias（不调用 mj_inverse）
  τ = M@ddq + qfrc_bias: [ -3.074  -22.6049  -0.6746   5.2741   0.0242   0.0918  -0.0281]
  与期望 τ 的差 (L∞): 0.0001

============================================================
结论摘要
============================================================
Test1 仅 inverse: qfrc_inverse[0] = 60251.4959 (期望约 -3.0740)
Test2 forward+inverse: qfrc_inverse[0] = 1.030860e-10 (期望约 -3.0740)
Test4 基准 τ: 与期望 L∞ 差 = 0.000055
```

---

## 7. step1 数据采集中记录的 tau 是否错误？

### 7.1 step1 实际记录的是什么

`step1_mujoco_dynamics_data_collection.py` 在采样时刻（每 `sample_interval` 步）执行：

1. `mj_forward(model, data)`
2. 记录 `q = qpos[:7]`, `dq = qvel[:7]`, `ddq = qacc[:7]`, **`tau = qfrc_actuator[:7]`**

因此写入 CSV 的 **tau 是执行器输出 `qfrc_actuator`**，不是逆动力学 τ = Mq̈ + qfrc_bias。**step1 没有使用 `mj_inverse`**，所以不存在“mj_inverse 写错”的问题；但若 step2 动力学辨识把 CSV 中的 tau 当作“逆动力学目标”，则拟合的是**执行器力矩**而非**刚体逆动力学力矩**。

### 7.2 测试：同一时刻 qfrc_actuator vs M@ddq + qfrc_bias

脚本 `scripts/test_step1_tau_vs_inverse_dynamics.py` 复现 step1 的仿真与采样逻辑，在每一采样时刻同时记录：

- **tau_actuator**：`qfrc_actuator`（即 step1 写入 CSV 的 “tau”）
- **tau_inverse**：M@ddq + qfrc_bias（正确的逆动力学）

**运行**（2 s 仿真，约 200 个采样点）：

```bash
python3 scripts/test_step1_tau_vs_inverse_dynamics.py
```

**结果摘要**（MuJoCo 3.5.0，同一模型）：

| 关节 | mean(τ_act − τ_inv) | max\|τ_act − τ_inv\| |
|------|----------------------|-----------------------|
| j0   | −0.46                | **126.4**             |
| j1   | +1.15                | 74.8                  |
| j2   | +0.27                | 20.1                  |
| j3   | −0.36                | 89.5                  |
| j4   | +0.27                | 11.3                  |
| j5   | +0.03                | 8.1                   |
| j6   | −0.07                | 2.1                   |

结论：**step1 记录的 tau 与逆动力学 τ 存在显著差异**（最大约 126 Nm，j0）。原因包括：位置控制下执行器输出包含跟踪误差引起的额外力矩，与“仅刚体动力学”的 τ 不同。若用 CSV 中 tau 做辨识目标，相当于用执行器输出拟合，可能带来偏差；若希望用逆动力学做辨识，应在采样时刻用 τ = M@ddq + qfrc_bias 覆盖或替代 CSV 的 tau 列（或单独生成“逆动力学 CSV”供 step2 使用）。

---

## 8. 复现方法

- **mj_inverse 行为**（第 3–5 节）：
  ```bash
  python3 scripts/test_mj_inverse_behavior.py
  ```
- **step1 记录的 tau 与逆动力学对比**（第 7 节）：
  ```bash
  python3 scripts/test_step1_tau_vs_inverse_dynamics.py
  ```

依赖：`mujoco`（本报告在 3.5.0 下测试）、`numpy`、`scipy`。
