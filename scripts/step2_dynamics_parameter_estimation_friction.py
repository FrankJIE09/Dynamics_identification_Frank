# -*- coding: utf-8 -*-
"""
Step2 动力学参数辨识（含摩擦）- Python 版
模型: τ = Y(q,dq,ddq)*θ + Fv*dq + Fc*sign(dq)，其中 Fv/Fc 为 7 维（每关节粘滞/库伦系数）。
流程：读配置 → 加载 URDF 与 CSV → 构造扩展回归矩阵 W=[Y,D_visc,D_coul]、tau → Ridge 求解 φ=[θ,Fv,Fc] → 物理投影 θ → 输出结果、摩擦系数与 URDF → 验证集验证。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- 避免被 PYTHONPATH/ROS 污染导致导入到 /opt/ros 下的 pinocchio ---
# 说明：ROS 自带的 pinocchio 常与当前环境的 NumPy/ABI 不匹配（例如 NumPy2 vs NumPy1 编译），会报 _ARRAY_API not found 或段错误。
# 这里等价于“unset PYTHONPATH”，并从 sys.path 中移除常见 ROS 路径，确保使用当前 Python 环境（conda/pip）里的 pinocchio。
os.environ.pop("PYTHONPATH", None)
sys.path[:] = [p for p in sys.path if p and ("/opt/ros/" not in p)]

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
    PINOCCHIO_VERSION = getattr(pin, "__version__", "0.0")
except ImportError:
    PINOCCHIO_AVAILABLE = False
    pin = None
    PINOCCHIO_VERSION = "0.0"

try:
    import yaml
except ImportError:
    yaml = None

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False


# ---------- 配置 ----------
def _find_config_path():
    candidates = [
        Path("config/step2_dynamics_parameter_estimation.yaml"),
        Path("src/config/step2_dynamics_parameter_estimation.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return candidates[0]


def load_config(config_path: str | None) -> dict:
    cfg = {
        "data_file": "../src/config/dynamics_identification_data.csv",
        "urdf_file": "../urdf/AR5-5_07R-W4C4A2_Manual_fix.urdf",
        "lambda_rel": 1e-2,
        "lam_friction": 1e-8,
        "m_min": 1e-4,
        "I_eps": 1e-6,
        "I_trace_min": 1e-3,
        "output_dir": "data_output",
        "result_file": "dynamics_identification_results.txt",
        "dynamics_parameters_csv": "dynamics_parameters.csv",
        "dynamics_parameters_ls_csv": "dynamics_parameters_ls.csv",
        "dynamics_parameters_urdf_csv": "dynamics_parameters_urdf.csv",
        "dynamics_parameters_friction_csv": "dynamics_parameters_friction.csv",
        "dynamics_physical_parameters_txt": "dynamics_physical_parameters_identified.txt",
        "output_urdf": "AR5-5_07R-W4C4A2_identified_inertia_com_friction.urdf",
    }
    path = config_path or _find_config_path()
    if yaml and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            for k, v in data.items():
                if k in cfg:
                    cfg[k] = v
            if "urdf" in data:
                cfg["urdf_file"] = data["urdf"]
    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key in cfg:
                    try:
                        if key in ("lambda_rel", "lam_friction", "m_min", "I_eps", "I_trace_min"):
                            cfg[key] = float(val)
                        else:
                            cfg[key] = val
                    except ValueError:
                        cfg[key] = val
    if "output_dir" not in cfg:
        cfg["output_dir"] = "build_outputs"
    return cfg


def dirname_of(path: str) -> str:
    return os.path.dirname(path) or ""


def _output_path(cfg: dict, key: str) -> str:
    """输出路径规范化：若配置了 output_dir 且路径非绝对，则写入 output_dir 下。"""
    path = cfg.get(key, "")
    output_dir = cfg.get("output_dir", "").strip()
    if output_dir and not os.path.isabs(path):
        return os.path.join(output_dir, os.path.basename(path))
    return path


def _ensure_output_dir(cfg: dict) -> dict:
    """创建 output_dir（若存在），并返回各输出项规范化后的路径字典。"""
    out_keys = (
        "result_file",
        "dynamics_parameters_csv",
        "dynamics_parameters_ls_csv",
        "dynamics_parameters_urdf_csv",
        "dynamics_parameters_friction_csv",
        "dynamics_physical_parameters_txt",
        "output_urdf",
    )
    out = {k: _output_path(cfg, k) for k in out_keys}
    output_dir = cfg.get("output_dir", "").strip()
    if output_dir and not os.path.isabs(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return out


# ---------- 数据 ----------
def load_data_from_csv(filename: str) -> list[dict]:
    """CSV 格式: timestamp, q1..q7, dq1..dq7, ddq1..ddq7, tau1..tau7"""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 1 + 7 * 4:
                continue
            try:
                point = {
                    "timestamp": float(row[0]),
                    "q": [float(row[1 + i]) for i in range(7)],
                    "dq": [float(row[8 + i]) for i in range(7)],
                    "ddq": [float(row[15 + i]) for i in range(7)],
                    "tau": [float(row[22 + i]) for i in range(7)],
                }
                data.append(point)
            except (ValueError, IndexError):
                continue
    print(f"  从文件读取了 {len(data)} 个数据点")
    return data


# ---------- Pinocchio 模型与重力 ----------
def build_model_and_data(urdf_file: str):
    if not PINOCCHIO_AVAILABLE:
        raise RuntimeError("需要 Pinocchio 库")
    model = pin.buildModelFromUrdf(urdf_file)
    data = model.createData()
    # 重力方向（与 C++ 一致）：世界系重力 [0,0,-9.81]，经基座旋转转到基座系；用 scipy 构造 R_base
    # 先绕 Y 轴 pi/2，再绕 Z 轴 -pi/2，即 R_base = R_z(-pi/2) @ R_y(pi/2)
    R_base = Rotation.from_euler("yz", [np.pi / 2, -np.pi / 2]).as_matrix()
    gravity_world = np.array([0.0, 0.0, -9.81])
    model.gravity.linear[:] = R_base.T @ gravity_world
    return model, data


# ---------- 物理投影 ----------
def project_theta_to_physical(
    theta: np.ndarray,
    n_params: int,
    m_min: float,
    I_eps: float,
    theta_urdf_fallback: np.ndarray | None = None,
    I_trace_min: float = 1e-3,
) -> None:
    """将 theta 投影到物理可行域；trace(I)<I_trace_min 时用 URDF 先验替代该连杆。"""
    if not PINOCCHIO_AVAILABLE:
        return
    n_links = n_params // 10
    if n_links * 10 != n_params or n_links <= 0:
        return
    use_fallback = theta_urdf_fallback is not None and theta_urdf_fallback.size == n_params

    for j in range(n_links):
        base = 10 * j
        pi = theta[base : base + 10].copy()
        inv = pin.Inertia.FromDynamicParameters(pi)
        m = max(inv.mass, m_min)
        c = np.array(inv.lever).ravel()
        I = np.array(inv.inertia).copy()
        I = (I + I.T) * 0.5
        ev, evec = np.linalg.eigh(I)
        ev = np.maximum(ev, I_eps)
        ev = np.sort(ev)
        if ev[2] > ev[0] + ev[1] - I_eps:
            ev[2] = ev[0] + ev[1] - I_eps
        ev[2] = max(ev[2], I_eps)
        I = (evec * ev) @ evec.T
        trace_I = np.trace(I)
        if use_fallback and trace_I < I_trace_min:
            theta[base : base + 10] = theta_urdf_fallback[base : base + 10]
            continue
        inv_proj = pin.Inertia(m, c, I)
        theta[base : base + 10] = np.array(inv_proj.toDynamicParameters()).ravel()


# ---------- QP 求解（OSQP）----------
def solve_ridge_qp(
    Y_all: np.ndarray,
    tau_all: np.ndarray,
    lam: float,
    theta_urdf: np.ndarray,
    n_params: int,
    n_links: int,
    m_min: float,
) -> np.ndarray | None:
    if not OSQP_AVAILABLE or n_links <= 0 or n_params != theta_urdf.size or 10 * n_links != n_params:
        return None
    n = n_params
    H = 2.0 * (Y_all.T @ Y_all)
    H.flat[:: n + 1] += 2.0 * lam
    g = -2.0 * (Y_all.T @ tau_all + lam * theta_urdf)
    # 约束: -theta[10*j] <= -m_min => theta[10*j] >= m_min. A (m x n): row j has -1 at col 10*j
    A = np.zeros((n_links, n))
    for j in range(n_links):
        A[j, 10 * j] = -1.0
    l = np.full(n_links, -1e30)
    u = np.full(n_links, -m_min)
    P = sparse.csc_matrix(H)
    A_sp = sparse.csc_matrix(A)
    m = osqp.OSQP()
    m.setup(P=P, q=g, A=A_sp, l=l, u=u, verbose=False, polish=True)
    r = m.solve()
    if r.info.status not in ("solved", "solved inaccurate"):
        return None
    return r.x


# ---------- URDF 生成 ----------
def _joint_name_to_link_name(joint_name: str) -> str:
    if "joint_" in joint_name:
        return joint_name.replace("joint_", "link", 1)
    return joint_name


def _inject_joint_dynamics(
    out_lines: list,
    joint_friction: dict,
    indent: str = "    ",
) -> list:
    """在 out_lines 中注入关节摩擦：遇到 <joint type="revolute"> 时替换或插入 <dynamics damping="Fv" friction="Fc"/>。"""
    result = []
    i = 0
    current_joint = ""
    dynamics_done = False
    while i < len(out_lines):
        line = out_lines[i]
        if '<joint name="' in line and 'type="revolute"' in line:
            start = line.find('<joint name="') + 13
            end = line.find('"', start)
            if end != -1:
                current_joint = line[start:end]
            dynamics_done = False
        if current_joint and current_joint in joint_friction and "<dynamics" in line:
            fv, fc = joint_friction[current_joint]
            result.append(f'{indent}<dynamics damping="{fv:.10e}" friction="{fc:.10e}"/>\n')
            while i < len(out_lines) and "/>" not in line and "</dynamics>" not in line:
                i += 1
                if i < len(out_lines):
                    line = out_lines[i]
            dynamics_done = True
            i += 1
            continue
        if current_joint and current_joint in joint_friction and "</joint>" in line and not dynamics_done:
            fv, fc = joint_friction[current_joint]
            result.append(f'{indent}<dynamics damping="{fv:.10e}" friction="{fc:.10e}"/>\n')
            dynamics_done = True
        result.append(line)
        if "</joint>" in line:
            current_joint = ""
        i += 1
    return result


def generate_identified_urdf_com(
    original_urdf_file: str,
    output_urdf_file: str,
    model: "pin.Model",
    theta_estimated: np.ndarray,
    Fv: np.ndarray | None = None,
    Fc: np.ndarray | None = None,
) -> None:
    """质心处惯量：<inertia> 为 I_com，<origin xyz> 为质心；若提供 Fv/Fc 则写入关节 <dynamics>。
    约定：URDF 与 Pinocchio 惯量均为绕质心 I_C，见 docs/06-URDF与Pinocchio惯量绕质心约定.tex"""
    with open(original_urdf_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n_inertial = min(7, theta_estimated.size // 10)
    link_inertial = {}
    for j in range(n_inertial):
        jid = j + 1
        base = 10 * j
        pi_id = theta_estimated[base : base + 10]
        inv_id = pin.Inertia.FromDynamicParameters(pi_id)
        m_id = inv_id.mass
        c_id = np.array(inv_id.lever).ravel()
        # 显式使用 Inertia 的 3x3 惯量矩阵（绕质心 I_C），与 URDF 约定一致，保证 θ 写/读一致
        I_3x3 = np.array(inv_id.inertia)
        ixx, ixy, ixz = I_3x3[0, 0], I_3x3[0, 1], I_3x3[0, 2]
        iyy, iyz, izz = I_3x3[1, 1], I_3x3[1, 2], I_3x3[2, 2]
        joint_name = model.names[jid]
        link_name = _joint_name_to_link_name(joint_name)
        blk = (
            f'    <inertial>\n'
            f'      <mass value="{m_id:.6f}" />\n'
            f'      <inertia ixx="{ixx:.6e}" ixy="{ixy:.6e}" ixz="{ixz:.6e}" iyy="{iyy:.6e}" iyz="{iyz:.6e}" izz="{izz:.6e}" />\n'
            f'      <origin rpy="0 0 0" xyz="{c_id[0]:.6f} {c_id[1]:.6f} {c_id[2]:.6f}" />\n'
            f'    </inertial>\n'
        )
        link_inertial[link_name] = blk
        if "AR5-5" not in link_name and "link" in link_name:
            link_inertial["AR5-5_07R-W4C4A2_" + link_name] = blk

    joint_friction = {}
    if Fv is not None and Fc is not None and Fv.size >= 7 and Fc.size >= 7:
        for j in range(7):
            if j + 1 < model.njoints:
                joint_friction[model.names[j + 1]] = (float(Fv[j]), float(Fc[j]))

    current_link = ""
    in_inertial = False
    inertial_replaced = False
    indent_len = 0
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '<link name="' in line:
            start = line.find('<link name="') + 12
            end = line.find('"', start)
            if end != -1:
                current_link = line[start:end]
            inertial_replaced = False
        if "<inertial>" in line and not inertial_replaced:
            indent_len = len(line) - len(line.lstrip())
            indent = " " * indent_len
            if current_link in link_inertial:
                new_block = link_inertial[current_link]
                for raw in new_block.strip().split("\n"):
                    out_lines.append(indent + raw.strip() + "\n")
                while i + 1 < len(lines) and "</inertial>" not in lines[i + 1]:
                    i += 1
                i += 1
                inertial_replaced = True
                i += 1
                continue
            in_inertial = True
            out_lines.append(line)
            i += 1
            continue
        if in_inertial:
            out_lines.append(line)
            if "</inertial>" in line:
                in_inertial = False
            i += 1
            continue
        if inertial_replaced and "</inertial>" in line:
            i += 1
            continue
        out_lines.append(line)
        i += 1

    if joint_friction:
        out_lines = _inject_joint_dynamics(out_lines, joint_friction)

    with open(output_urdf_file, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"  已生成新的URDF文件(质心处惯量+摩擦): {output_urdf_file}")


# ---------- 主流程 ----------
def estimate_dynamics_parameters(data_file: str, urdf_file: str, cfg: dict) -> None:
    if not PINOCCHIO_AVAILABLE:
        print("  错误: 需要 Pinocchio 库", file=sys.stderr)
        return

    out = _ensure_output_dir(cfg)
    if cfg.get("output_dir"):
        print(f"  输出目录: {os.path.abspath(cfg['output_dir'])}")

    print("\n  初始化 Pinocchio 动力学模型...")
    if PINOCCHIO_VERSION != "0.0":
        print(f"  Pinocchio 版本: {PINOCCHIO_VERSION}")
    model, data = build_model_and_data(urdf_file)
    print(f"  Pinocchio 模型加载成功: {urdf_file}")
    print(f"  模型自由度: {model.nq}")

    print("\n  从文件加载数据:", data_file)
    collected = load_data_from_csv(data_file)
    if not collected:
        print("  错误: 没有加载到数据", file=sys.stderr)
        return

    # 回归矩阵参数数量与 theta_urdf（用法符合 Pinocchio 官方 API：include/pinocchio/algorithm/regressor.hxx）
    q_test = np.zeros(7)
    v_test = np.zeros(7)
    a_test = np.zeros(7)
    pin.computeJointTorqueRegressor(model, data, q_test, v_test, a_test)
    # 结果在 data.jointTorqueRegressor，满足 τ=Y(q,v,a)*π，π_i=model.inertias[i].toDynamicParameters()
    Y_ref = data.jointTorqueRegressor
    
    Y_ref = np.asarray(Y_ref)
    n_params = Y_ref.shape[1]
    print(f"  Pinocchio回归矩阵参数数量: {n_params}")

    theta_urdf = np.zeros(n_params)
    expected_params = 10 * (model.njoints - 1)
    if n_params == expected_params:
        for jid in range(1, model.njoints):
            pi = np.array(model.inertias[jid].toDynamicParameters()).ravel()
            base = 10 * (jid - 1)
            theta_urdf[base : base + 10] = pi
        print(f"  已构造URDF先验theta_urdf（维度 {theta_urdf.size}）")
    else:
        theta_urdf = np.array([])
        print("  警告: n_params != 10*(njoints-1)，无法构造URDF先验")

    # 堆叠 Y_all, W_all（扩展摩擦）, tau_all，并记录每样本的 (q,dq,ddq)
    # 摩擦模型: τ = Y*θ + Fv*dq + Fc*sign(dq)，W = [Y, diag(dq), diag(sign(dq))]
    n_joints = 7
    n_friction = n_joints * 2  # Fv(7) + Fc(7)
    Y_list = []
    W_list = []
    tau_list = []
    Y_input_list = []  # 每样本: [q1..q7, dq1..dq7, ddq1..ddq7]
    for idx, point in enumerate(collected):
        q = np.array(point["q"], dtype=np.float64)
        v = np.array(point["dq"], dtype=np.float64)
        a = np.array(point["ddq"], dtype=np.float64)
        try:
            pin.computeJointTorqueRegressor(model, data, q, v, a)
            Y_ref = data.jointTorqueRegressor
            Y = np.asarray(Y_ref).copy()
            if Y.shape[1] != n_params:
                continue
        except Exception:
            continue
        tau = np.array(point["tau"], dtype=np.float64)
        # 扩展回归: W_block 每行对应一关节，列为 [Y_row, dq_j at col j, sign(dq_j) at col j]
        dq_diag = np.diag(v.ravel())
        sign_dq = np.sign(v)
        sign_dq[sign_dq == 0] = 0  # sign(0)=0
        sign_diag = np.diag(sign_dq.ravel())
        W_block = np.hstack([Y, dq_diag, sign_diag])  # (n_joints, n_params + 14)
        Y_list.append(Y)
        W_list.append(W_block)
        tau_list.append(tau)
        Y_input_list.append(np.concatenate([q, v, a]))
        if (idx + 1) % 100 == 0:
            print(f"  已处理: {idx + 1} / {len(collected)}")
    Y_all = np.vstack(Y_list)
    W_all = np.vstack(W_list)
    tau_all = np.concatenate(tau_list)
    n_phi = n_params + n_friction
    print(f"\n  回归矩阵构造完成 维度: Y {Y_all.shape[0]} x {Y_all.shape[1]}, W(含摩擦) {W_all.shape[0]} x {W_all.shape[1]}, tau: {tau_all.size}")

    # 保存 Y_all 到二进制文件，便于与 C++ 对比（格式: int32 rows, int32 cols, row-major float64）
    _y_all_bin = os.path.join(os.path.dirname(out["result_file"]), "Y_all_py.bin")
    with open(_y_all_bin, "wb") as f:
        np.array([Y_all.shape[0], Y_all.shape[1]], dtype=np.int32).tofile(f)
        Y_all.astype(np.float64).flatten(order="C").tofile(f)
    print(f"  Y_all 已保存到: {_y_all_bin}")

    # 保存 Y_all 使用的输入数据 (q,dq,ddq)，便于与 C++ 对比
    out_dir = os.path.dirname(out["result_file"])
    _y_input_csv = os.path.join(out_dir, "Y_input_py.csv")
    header = "sample_idx,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7,ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,ddq7"
    with open(_y_input_csv, "w") as f:
        f.write(header + "\n")
        for i, row in enumerate(Y_input_list):
            f.write(str(i) + "," + ",".join(f"{x:.17g}" for x in row) + "\n")
    print(f"  Y_all 输入数据已保存到: {_y_input_csv}")

    # 求解 φ = [θ, Fv, Fc]，τ = W*φ
    print("\n========================================")
    print("开始求解动力学参数（含摩擦）...")
    print("========================================")
    U, s, Vt = np.linalg.svd(W_all, full_matrices=False)
    threshold = 1e-6 * (s[0] if s.size else 0)
    rank = int(np.sum(s > threshold))
    print(f"  扩展回归矩阵 W 的秩: {rank} / {n_phi}")

    lambda_rel = cfg["lambda_rel"]
    lam_friction = float(cfg.get("lam_friction", 1e-8))
    lam = lambda_rel * (np.trace(W_all.T @ W_all) / n_phi) if W_all.size and n_phi else lambda_rel
    m_min = cfg["m_min"]

    phi_prior = np.zeros(n_phi)
    if theta_urdf.size == n_params:
        phi_prior[:n_params] = theta_urdf
    # 正则: θ 部分用 lam，摩擦部分用 lam_friction（避免摩擦系数过大）
    L_diag = np.ones(n_phi) * lam_friction
    L_diag[:n_params] = lam

    A_reg = W_all.T @ W_all + np.diag(L_diag)
    b_reg = W_all.T @ tau_all + np.diag(L_diag) @ phi_prior
    phi = np.linalg.solve(A_reg, b_reg)
    print(f"  使用 Ridge 求解 φ=[θ,Fv,Fc]: lambda_rel={lambda_rel}, lam_friction={lam_friction:.2e}")

    theta_estimated = np.asarray(phi[:n_params], dtype=np.float64).copy()
    Fv = np.asarray(phi[n_params : n_params + n_joints], dtype=np.float64)
    Fc = np.asarray(phi[n_params + n_joints : n_phi], dtype=np.float64)
    theta_ls_ridge = theta_estimated.copy()

    if n_params == 70 and theta_urdf.size == n_params:
        fallback = theta_urdf
        project_theta_to_physical(
            theta_estimated, n_params, cfg["m_min"], cfg["I_eps"],
            theta_urdf_fallback=fallback, I_trace_min=cfg["I_trace_min"],
        )
        print("  已应用物理约束投影")

    # 训练集误差（Y*θ + 摩擦）
    tau_pred = W_all @ phi
    error = tau_all - tau_pred
    rmse_train = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))
    print("\n========================================")
    print("辨识结果统计:")
    print("========================================")
    print(f"  训练集 RMSE (Y*θ+摩擦): {rmse_train:.4f} Nm")
    print(f"  最大误差: {max_error:.4f} Nm")
    print(f"  平均误差: {mean_error:.4f} Nm")
    print("  摩擦系数 Fv (粘滞), Fc (库伦):")
    for j in range(n_joints):
        print(f"    关节{j}: Fv={Fv[j]:.6e}, Fc={Fc[j]:.6e}")

    # 保存结果（写入规范化输出目录）
    result_path = out["result_file"]
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("动力学参数辨识结果（含摩擦）\n============================\n\n")
        f.write(f"辨识时间: {int(time.time())}\n")
        f.write(f"数据点数: {len(collected)}\n")
        f.write(f"参数数量: θ={n_params}, Fv/Fc 各 {n_joints}\n\n")
        f.write("误差统计:\n")
        f.write(f"  训练集 RMSE (Y*θ+摩擦): {rmse_train:.6f} Nm\n")
        f.write(f"  最大误差: {max_error} Nm\n")
        f.write(f"  平均误差: {mean_error} Nm\n\n")
        f.write("摩擦系数:\n")
        for j in range(n_joints):
            f.write(f"  关节{j}: Fv={Fv[j]:.6e}, Fc={Fc[j]:.6e}\n")
        f.write("\n辨识的 θ 值:\n")
        for i in range(len(theta_estimated)):
            f.write(f"  theta[{i}] = {theta_estimated[i]:.6e}\n")
    print(f"  结果已保存到: {result_path}")

    for name, arr in [
        (out["dynamics_parameters_csv"], theta_estimated),
        (out["dynamics_parameters_ls_csv"], theta_ls_ridge),
    ]:
        with open(name, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter_index", "value"])
            for i, v in enumerate(arr):
                w.writerow([i, f"{v:.10e}"])
        print(f"  参数已保存到: {name}")
    if theta_urdf.size == n_params:
        with open(out["dynamics_parameters_urdf_csv"], "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter_index", "value"])
            for i, v in enumerate(theta_urdf):
                w.writerow([i, f"{v:.10e}"])
        print(f"  URDF先验参数已保存到: {out['dynamics_parameters_urdf_csv']}")
    with open(out["dynamics_parameters_friction_csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "Fv", "Fc"])
        for j in range(n_joints):
            w.writerow([j, f"{Fv[j]:.10e}", f"{Fc[j]:.10e}"])
    print(f"  摩擦系数已保存到: {out['dynamics_parameters_friction_csv']}")

    # 物理参数 txt 与 URDF
    if theta_urdf.size == n_params and n_params == 70:
        phys_path = out["dynamics_physical_parameters_txt"]
        with open(phys_path, "w", encoding="utf-8") as pf:
            pf.write("辨识后的物理参数（由theta重建，与URDF对比）\n========================================\n\n")
            for jid in range(1, model.njoints):
                base = 10 * (jid - 1)
                pi_id = theta_estimated[base : base + 10]
                inv_id = pin.Inertia.FromDynamicParameters(pi_id)
                m_urdf = model.inertias[jid].mass
                c_urdf = np.array(model.inertias[jid].lever).ravel()
                m_id = inv_id.mass
                c_id = pi_id[1:4] / pi_id[0]
                I_origin_id = np.array(inv_id.inertia)
                I_com_id = np.array([[pi_id[4], pi_id[5], pi_id[7]], [pi_id[5], pi_id[6], pi_id[8]], [pi_id[7], pi_id[8], pi_id[9]]])
                pf.write(f"关节 {jid} ({model.names[jid]}):\n")
                pf.write(f"  URDF 质量(kg): {m_urdf:.6f} | 辨识质量(kg): {m_id:.6f}\n")
                pf.write(f"  URDF 质心(m): [{c_urdf[0]}, {c_urdf[1]}, {c_urdf[2]}]\n")
                pf.write(f"  辨识质心(m): [{c_id[0]}, {c_id[1]}, {c_id[2]}]\n")
                pf.write(f"  辨识 惯性(质心处):\n")
                pf.write(f"    [{I_com_id[0,0]}, {I_com_id[0,1]}, {I_com_id[0,2]}]\n")
                pf.write(f"    [{I_com_id[1,0]}, {I_com_id[1,1]}, {I_com_id[1,2]}]\n")
                pf.write(f"    [{I_com_id[2,0]}, {I_com_id[2,1]}, {I_com_id[2,2]}]\n\n")
            q0 = np.zeros(7)
            pin.computeJointTorqueRegressor(model, data, q0, q0, q0)
            Y0 = np.asarray(data.jointTorqueRegressor)
            tau0 = Y0 @ theta_estimated
            pf.write("验证：q=0,dq=0,ddq=0 时预测力矩(Nm):\n")
            pf.write(f"  [{tau0[0]:.4f}, {tau0[1]:.4f}, {tau0[2]:.4f}, {tau0[3]:.4f}, {tau0[4]:.4f}, {tau0[5]:.4f}, {tau0[6]:.4f}]\n")
        print(f"  已保存: {phys_path}")
        print("\n  生成新的URDF文件(质心处惯量+摩擦)...")
        generate_identified_urdf_com(urdf_file, out["output_urdf"], model, theta_estimated, Fv=Fv, Fc=Fc)

    # 验证集用「辨识 URDF」(COM) 的 model 算 Y，使 验证集 RMSE (Y*θ+摩擦) 与 辨识URDF+rnea+摩擦 一致
    model_val = model
    data_val = data
    id_urdf_for_val = out["output_urdf"]
    if not os.path.isfile(id_urdf_for_val):
        id_urdf_for_val = os.path.join(os.path.dirname(out["result_file"]), os.path.basename(out["output_urdf"]))
    if not os.path.isfile(id_urdf_for_val) and not os.path.isabs(out["output_urdf"]):
        id_urdf_for_val = os.path.join(dirname_of(urdf_file), os.path.basename(out["output_urdf"]))
    if os.path.isfile(id_urdf_for_val):
        model_val, data_val = build_model_and_data(id_urdf_for_val)
        print("  验证集 Y 使用辨识URDF(COM) 的 model 计算")
    else:
        print("  验证集 Y 使用原 URDF 的 model 计算（未找到辨识URDF）")

    # 验证集：后 20%，使用 W_val @ φ（Y*θ+摩擦），Y 来自辨识完成后的 model
    validation_start = int(len(collected) * 0.8)
    validation_count = len(collected) - validation_start
    W_val_list = []
    tau_val_list = []
    for i in range(validation_start, len(collected)):
        point = collected[i]
        q = np.array(point["q"], dtype=np.float64)
        v = np.array(point["dq"], dtype=np.float64)
        a = np.array(point["ddq"], dtype=np.float64)
        try:
            pin.computeJointTorqueRegressor(model_val, data_val, q, v, a)
            Y_ref = data_val.jointTorqueRegressor
            Y = np.asarray(Y_ref).copy()
            if Y.shape[1] != n_params:
                continue
        except Exception:
            continue
        tau = np.array(point["tau"], dtype=np.float64)
        dq_diag = np.diag(v.ravel())
        sign_dq = np.sign(v)
        sign_dq[sign_dq == 0] = 0
        sign_diag = np.diag(sign_dq.ravel())
        W_block = np.hstack([Y, dq_diag, sign_diag])
        W_val_list.append(W_block)
        tau_val_list.append(tau)
    if W_val_list:
        W_val = np.vstack(W_val_list)
        tau_val = np.concatenate(tau_val_list)
        tau_val_pred = W_val @ phi
        err_val = tau_val - tau_val_pred
        rmse_val = np.sqrt(np.mean(err_val ** 2))
        Y_val = W_val[:, :n_params]
        tau_val_pred_no_fric = Y_val @ theta_estimated
        err_val_no_fric = tau_val - tau_val_pred_no_fric
        rmse_val_no_fric = np.sqrt(np.mean(err_val_no_fric ** 2))
        print("\n========================================")
        print("使用验证数据验证辨识结果...")
        print("========================================")
        print(f"  验证集数据点数: {validation_count}")
        print(f"  验证集 RMSE (Y*θ): {rmse_val_no_fric:.4f} Nm")
        print(f"  验证集 RMSE (Y*θ+摩擦，Y=辨识URDF): {rmse_val:.4f} Nm")
        print("  说明: 上列 Y 由辨识完成后的 URDF(COM) 的 model 计算，与 辨识URDF+rnea+摩擦 应一致。")
        with open(out["result_file"], "a", encoding="utf-8") as f:
            f.write("\n验证集(Y*θ): RMSE=" + f"{rmse_val_no_fric:.6f}" + " Nm\n")
            f.write("验证集(Y*θ+摩擦，Y=辨识URDF): RMSE=" + f"{rmse_val:.6f}" + " Nm\n")
            f.write("说明: 上列 Y 由辨识完成后的 URDF(COM) 的 model 计算，与 辨识URDF+rnea+摩擦 应一致。\n")
    else:
        rmse_val = None
        rmse_val_no_fric = None

    # 诊断: 从辨识 URDF(COM) 读回 θ，与 theta_estimated 对比（写/读一致性）
    try:
        id_urdf = out["output_urdf"]
        if not os.path.isfile(id_urdf):
            id_urdf = os.path.join(os.path.dirname(out["result_file"]), os.path.basename(out["output_urdf"]))
        if not os.path.isfile(id_urdf) and not os.path.isabs(out["output_urdf"]):
            id_urdf = os.path.join(dirname_of(urdf_file) or ".", os.path.basename(out["output_urdf"]))
        if os.path.isfile(id_urdf):
            model_chk, _ = build_model_and_data(id_urdf)
            theta_from_urdf = np.zeros(n_params)
            for jid in range(1, min(model_chk.njoints, 1 + n_params // 10)):
                base = 10 * (jid - 1)
                theta_from_urdf[base : base + 10] = np.array(
                    model_chk.inertias[jid].toDynamicParameters()
                ).ravel()
            diff_theta = np.abs(theta_estimated - theta_from_urdf)
            max_diff = np.max(diff_theta)
            mean_diff = np.mean(diff_theta)
            print("\n  [诊断] θ 写/读一致性: 内存θ vs 保存URDF再读回θ 差异 最大={:.6e}, 平均={:.6e} (应为0)".format(max_diff, mean_diff))
            # 逐元素一一对应打印: 索引 关节 参数名 内存θ URDF读回θ |差|
            param_names = ["m", "mc_x", "mc_y", "mc_z", "I_xx", "I_xy", "I_xz", "I_yy", "I_yz", "I_zz"]
            print("  [诊断] θ 逐元素对比 (idx 关节 参数    内存θ            URDF读回θ        |差|):")
            for i in range(n_params):
                jid = i // 10
                pname = param_names[i % 10]
                te = theta_estimated[i]
                tu = theta_from_urdf[i]
                d = diff_theta[i]
                print("    {:3d}  J{}  {:6s}  {: .10e}  {: .10e}  {:.6e}".format(i, jid, pname, te, tu, d))
            with open(out["result_file"], "a", encoding="utf-8") as f:
                f.write("\n[诊断] θ写/读一致性(内存θ vs 保存URDF再读回θ): 最大={:.6e}, 平均={:.6e} (应为0)\n".format(max_diff, mean_diff))
                f.write("[诊断] θ 逐元素对比:\n")
                for i in range(n_params):
                    jid = i // 10
                    pname = param_names[i % 10]
                    te = theta_estimated[i]
                    tu = theta_from_urdf[i]
                    d = diff_theta[i]
                    f.write("  {:3d}  J{}  {}  内存={:.10e}  URDF={:.10e}  diff={:.6e}\n".format(i, jid, pname, te, tu, d))
            # 单点检查: 用辨识 model 在同一 (q,v,a) 上算 Y@θ 与 rnea，理论上应一致
            if validation_start < len(collected):
                pt = collected[validation_start]
                q0 = np.array(pt["q"], dtype=np.float64)
                v0 = np.array(pt["dq"], dtype=np.float64)
                a0 = np.array(pt["ddq"], dtype=np.float64)
                data_chk = model_chk.createData()
                pin.computeJointTorqueRegressor(model_chk, data_chk, q0, v0, a0)
                Y_id = np.asarray(data_chk.jointTorqueRegressor).copy()
                tau_Y = Y_id @ theta_estimated
                tau_rnea = pin.rnea(model_chk, data_chk, q0, v0, a0)
                diff_one = np.abs(tau_Y - np.array(tau_rnea))
                print("  [诊断] 单点(首验证样本): 辨识model 下 Y@θ 与 rnea 差 最大={:.6e}, 均值={:.6e}".format(np.max(diff_one), np.mean(diff_one)))
                if np.max(diff_one) > 1e-5:
                    print("        Y@θ =", np.array(tau_Y))
                    print("        rnea =", np.array(tau_rnea))
                with open(out["result_file"], "a", encoding="utf-8") as f:
                    f.write("[诊断] 单点 辨识model Y@θ与rnea差 最大={:.6e}\n".format(np.max(diff_one)))
            # 用辨识 URDF 的 model 重算 Y_all，与主流程 Y_all（原 URDF）对比，排查 copy 或 model 差异
            data_chk = model_chk.createData()
            Y_id_list = []
            for idx, point in enumerate(collected):
                q = np.array(point["q"], dtype=np.float64)
                v = np.array(point["dq"], dtype=np.float64)
                a = np.array(point["ddq"], dtype=np.float64)
                try:
                    pin.computeJointTorqueRegressor(model_chk, data_chk, q, v, a)
                    Y_ref_chk = data_chk.jointTorqueRegressor
                    Y_row = np.asarray(Y_ref_chk).copy()
                    if Y_row.shape[1] != n_params:
                        continue
                except Exception:
                    continue
                Y_id_list.append(Y_row)
            if Y_id_list:
                Y_all_identified = np.vstack(Y_id_list)
                if Y_all_identified.shape == Y_all.shape:
                    diff_Y = np.abs(Y_all - Y_all_identified)
                    max_diff_Y = np.max(diff_Y)
                    mean_diff_Y = np.mean(diff_Y)
                    print("  [诊断] Y_all(原URDF) vs Y_all(辨识URDF) 回归矩阵差异: 最大={:.6e}, 平均={:.6e}".format(max_diff_Y, mean_diff_Y))
                    with open(out["result_file"], "a", encoding="utf-8") as f:
                        f.write("[诊断] Y_all原URDF vs Y_all辨识URDF 最大={:.6e}, 平均={:.6e}\n".format(max_diff_Y, mean_diff_Y))
                    out_dir = os.path.dirname(out["result_file"])
                    y_id_bin = os.path.join(out_dir, "Y_all_identified_py.bin")
                    with open(y_id_bin, "wb") as f:
                        np.array([Y_all_identified.shape[0], Y_all_identified.shape[1]], dtype=np.int32).tofile(f)
                        Y_all_identified.astype(np.float64).flatten(order="C").tofile(f)
                    print("  [诊断] 辨识URDF 的 Y_all 已保存: {}".format(y_id_bin))
                else:
                    print("  [诊断] 形状不一致 原Y_all={} 辨识Y_all={}，跳过对比".format(Y_all.shape, Y_all_identified.shape))
    except Exception as e:
        print("  [诊断] 读回辨识URDF θ 时出错:", e)

    # 原 URDF + rnea
    try:
        model_orig, data_orig = build_model_and_data(urdf_file)
        tau_meas_orig = []
        tau_pred_orig = []
        for i in range(validation_start, len(collected)):
            point = collected[i]
            q = np.array(point["q"])
            v = np.array(point["dq"])
            a = np.array(point["ddq"])
            tau_p = pin.rnea(model_orig, data_orig, q, v, a)
            for j in range(7):
                tau_pred_orig.append(tau_p[j])
                tau_meas_orig.append(point["tau"][j])
        if tau_pred_orig:
            err_o = np.array(tau_meas_orig) - np.array(tau_pred_orig)
            rmse_orig = np.sqrt(np.mean(err_o ** 2))
            max_o = np.max(np.abs(err_o))
            print(f"\n  验证集 RMSE (原URDF + rnea): {rmse_orig:.4f} Nm")
            print(f"  验证集 最大绝对误差 (原URDF + rnea): {max_o:.4f} Nm")
            with open(out["result_file"], "a", encoding="utf-8") as f:
                f.write(f"\n验证集-原URDF(rnea): RMSE={rmse_orig:.6f} Nm, 最大绝对误差={max_o:.6f} Nm\n")
    except Exception as e:
        print(f"  (提示) 使用原URDF进行验证时出错: {e}")

    # 辨识 URDF[COM] + rnea，以及 + rnea + 摩擦
    try:
        id_urdf = out["output_urdf"]
        if not os.path.isfile(id_urdf):
            id_urdf = os.path.join(dirname_of(urdf_file), os.path.basename(out["output_urdf"]))
        model_id, data_id = build_model_and_data(id_urdf)
        tau_pred_list = []
        tau_pred_fric_list = []
        tau_meas_list = []
        for i in range(validation_start, len(collected)):
            point = collected[i]
            q = np.array(point["q"])
            v = np.array(point["dq"])
            a = np.array(point["ddq"])
            tau_p = pin.rnea(model_id, data_id, q, v, a)
            tau_p_fric = tau_p + Fv * v + Fc * np.sign(v)
            for j in range(7):
                tau_pred_list.append(tau_p[j])
                tau_pred_fric_list.append(tau_p_fric[j])
                tau_meas_list.append(point["tau"][j])
        if tau_pred_list:
            err = np.array(tau_meas_list) - np.array(tau_pred_list)
            rmse_com = np.sqrt(np.mean(err ** 2))
            max_com = np.max(np.abs(err))
            print(f"\n  验证集 RMSE (辨识URDF[COM] + rnea): {rmse_com:.4f} Nm")
            print(f"  验证集 最大绝对误差 (辨识URDF[COM] + rnea): {max_com:.4f} Nm")
            with open(out["result_file"], "a", encoding="utf-8") as f:
                f.write(f"\n验证集-辨识URDF[COM](rnea): RMSE={rmse_com:.6f} Nm, 最大绝对误差={max_com:.6f} Nm\n")
        if tau_pred_fric_list:
            err_fric = np.array(tau_meas_list) - np.array(tau_pred_fric_list)
            rmse_com_fric = np.sqrt(np.mean(err_fric ** 2))
            max_com_fric = np.max(np.abs(err_fric))
            print(f"  验证集 RMSE (辨识URDF[COM] + rnea + 摩擦): {rmse_com_fric:.4f} Nm")
            print(f"  验证集 最大绝对误差 (辨识URDF[COM] + rnea + 摩擦): {max_com_fric:.4f} Nm")
            with open(out["result_file"], "a", encoding="utf-8") as f:
                f.write(f"验证集-辨识URDF[COM](rnea+摩擦): RMSE={rmse_com_fric:.6f} Nm, 最大绝对误差={max_com_fric:.6f} Nm\n")

        # 调试: Y*θ+摩擦 vs 辨识URDF[COM]+rnea+摩擦 逐关节与逐样本对比
        if W_val_list and tau_pred_fric_list:
            n_val_y = len(tau_val) // 7
            n_val_r = len(tau_pred_fric_list) // 7
            n_val = min(n_val_y, n_val_r)
            if n_val > 0:
                tau_meas_2d = tau_val[: n_val * 7].reshape(n_val, 7)
                tau_Yfric_2d = tau_val_pred[: n_val * 7].reshape(n_val, 7)
                tau_rnea_fric_2d = np.array(tau_pred_fric_list[: n_val * 7]).reshape(n_val, 7)
                rmse_Yfric_per_j = np.sqrt(np.mean((tau_meas_2d - tau_Yfric_2d) ** 2, axis=0))
                rmse_rnea_fric_per_j = np.sqrt(np.mean((tau_meas_2d - tau_rnea_fric_2d) ** 2, axis=0))
                diff_pred = tau_Yfric_2d - tau_rnea_fric_2d
                mean_abs_diff_per_j = np.mean(np.abs(diff_pred), axis=0)
                max_abs_diff_per_j = np.max(np.abs(diff_pred), axis=0)
                print("\n  ---------- 调试: Y*θ+摩擦 vs 辨识URDF[COM]+rnea+摩擦 ----------")
                print("  按关节 RMSE (Y*θ+摩擦):     " + " ".join(f"J{j}={rmse_Yfric_per_j[j]:.4f}" for j in range(7)))
                print("  按关节 RMSE (rnea+摩擦):    " + " ".join(f"J{j}={rmse_rnea_fric_per_j[j]:.4f}" for j in range(7)))
                print("  两预测差 |Y-rnea| 均值/关节: " + " ".join(f"J{j}={mean_abs_diff_per_j[j]:.4f}" for j in range(7)))
                print("  两预测差 |Y-rnea| 最大/关节: " + " ".join(f"J{j}={max_abs_diff_per_j[j]:.4f}" for j in range(7)))
                print("  前 3 个验证样本 τ (实测 | Y*θ+摩擦 | rnea+摩擦) [Nm]:")
                for s in range(min(3, n_val)):
                    print(f"    样本{s}:")
                    for j in range(7):
                        print(f"      关节{j}: {tau_meas_2d[s,j]:7.3f}  |  {tau_Yfric_2d[s,j]:7.3f}  |  {tau_rnea_fric_2d[s,j]:7.3f}")
                with open(out["result_file"], "a", encoding="utf-8") as f:
                    f.write("\n[调试] Y*θ+摩擦 vs 辨识URDF[COM]+rnea+摩擦\n")
                    f.write("  按关节 RMSE (Y*θ+摩擦): " + " ".join(f"J{j}={rmse_Yfric_per_j[j]:.6f}" for j in range(7)) + "\n")
                    f.write("  按关节 RMSE (rnea+摩擦): " + " ".join(f"J{j}={rmse_rnea_fric_per_j[j]:.6f}" for j in range(7)) + "\n")
                    f.write("  两预测差 均值/关节: " + " ".join(f"J{j}={mean_abs_diff_per_j[j]:.6f}" for j in range(7)) + "\n")
                    f.write("  两预测差 最大/关节: " + " ".join(f"J{j}={max_abs_diff_per_j[j]:.6f}" for j in range(7)) + "\n")
    except Exception as e:
        print(f"  (提示) 使用辨识URDF[COM]进行验证时出错: {e}")

    # 若存在 C++ 保存的 Y_all，则对比
    project_root = Path(__file__).resolve().parent.parent
    cpp_bin = project_root / "build" / "Y_all_cpp.bin"
    if cpp_bin.exists():
        with open(cpp_bin, "rb") as f:
            rows = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            cols = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            data = np.fromfile(f, dtype=np.float64)
        Y_all_cpp = data.reshape(rows, cols)
        if Y_all_cpp.shape == Y_all.shape:
            diff = np.abs(Y_all - Y_all_cpp)
            print("\n========================================")
            print("Y_all 与 C++ 对比:")
            print("========================================")
            print(f"  形状: Python {Y_all.shape}, C++ {Y_all_cpp.shape}")
            print(f"  最大绝对差: {np.max(diff):.6e}")
            print(f"  平均绝对差: {np.mean(diff):.6e}")
            nf = np.linalg.norm(Y_all_cpp, "fro")
            print(f"  相对差 (Frobenius): {np.linalg.norm(diff, 'fro') / (nf + 1e-20):.6e}")
        else:
            print(f"\n  Y_all 形状不一致，无法对比 (Python {Y_all.shape} vs C++ {Y_all_cpp.shape})")


def _find_cpp_executable(project_root: Path) -> Path | None:
    """在项目根下查找 C++ step2 可执行文件。"""
    for d in ("build", "cmake-build-release"):
        exe = project_root / d / "step2_dynamics_parameter_estimation"
        if exe.exists():
            return exe
    return None


def run_cpp_and_use_results(data_file: str, urdf_file: str, cfg: dict) -> None:
    """
    调用 C++ step2 可执行文件，得到与 C++ 一致的秩与 RMSE，并将输出复制到配置指定路径。
    """
    project_root = Path(__file__).resolve().parent.parent
    exe = _find_cpp_executable(project_root)
    if not exe:
        print("  未找到 C++ 可执行文件 build/step2_dynamics_parameter_estimation，请先编译 C++ 工程。", file=sys.stderr)
        sys.exit(1)
    build_dir = exe.parent
    # C++ 从 build 目录运行，传入相对 project_root 的路径，在 build 里用 .. 访问
    data_p = Path(data_file)
    urdf_p = Path(urdf_file)
    if not data_p.is_absolute():
        data_p = Path.cwd() / data_p
    if not urdf_p.is_absolute():
        urdf_p = Path.cwd() / urdf_p
    try:
        data_rel = data_p.resolve().relative_to(project_root)
        urdf_rel = urdf_p.resolve().relative_to(project_root)
        data_for_cpp = str(Path("..") / data_rel)
        urdf_for_cpp = str(Path("..") / urdf_rel)
    except ValueError:
        data_for_cpp = str(data_p.resolve())
        urdf_for_cpp = str(urdf_p.resolve())
    cmd = [str(exe), data_for_cpp, urdf_for_cpp]
    print("  调用 C++ step2:", " ".join(cmd))
    print("  工作目录:", build_dir)
    ret = subprocess.run(cmd, cwd=str(build_dir), timeout=300)
    if ret.returncode != 0:
        print(f"  C++ 程序退出码: {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)
    # C++ 结果在 build_dir，复制到规范化输出目录（与 Python 自跑一致）
    out = _ensure_output_dir(cfg)
    if cfg.get("output_dir"):
        print(f"  输出目录: {os.path.abspath(cfg['output_dir'])}")
    out_keys = (
        "result_file",
        "dynamics_parameters_csv",
        "dynamics_parameters_ls_csv",
        "dynamics_parameters_urdf_csv",
        "dynamics_parameters_friction_csv",
        "dynamics_physical_parameters_txt",
        "output_urdf",
    )
    for key in out_keys:
        name = out[key]
        src = build_dir / os.path.basename(cfg.get(key, name))
        if not src.exists():
            src = build_dir / os.path.basename(name)
        if src.exists():
            dst = (project_root / name) if not os.path.isabs(name) else Path(name)
            if src.resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  已复制: {os.path.basename(name)} -> {dst}")
    # 打印结果摘要
    result_txt = Path(out["result_file"])
    if result_txt.exists():
        with open(result_txt, "r", encoding="utf-8") as f:
            for line in f:
                if "回归矩阵的秩" in line or "RMSE" in line or "最大误差" in line:
                    print(" ", line.rstrip())
    print("\n  已使用 C++ 辨识结果（秩与 RMSE 与 C++ 一致），结果已写入输出目录。")


def main():
    parser = argparse.ArgumentParser(description="Step2 动力学参数辨识 (Python)")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument(
        "--use-cpp",
        action="store_true",
        help="调用 C++ 可执行文件做辨识，得到与 C++ 一致的秩与 RMSE，结果写回当前/配置路径",
    )
    parser.add_argument("data_file", nargs="?", default=None, help="数据 CSV 路径（覆盖配置）")
    parser.add_argument("urdf_file", nargs="?", default=None, help="URDF 路径（覆盖配置）")
    args = parser.parse_args()
    config_path = args.config or _find_config_path()
    cfg = load_config(config_path)
    print("  配置:", config_path)
    data_file = args.data_file or cfg["data_file"]
    urdf_file = args.urdf_file or cfg["urdf_file"]
    print("数据文件:", data_file)
    print("URDF文件:", urdf_file)
    print("========================================")
    try:
        if args.use_cpp:
            print("\n========================================")
            print("使用 C++ step2 进行辨识...")
            print("========================================")
            run_cpp_and_use_results(data_file, urdf_file, cfg)
        else:
            print("\n========================================")
            print("开始动力学参数辨识...")
            print("========================================")
            estimate_dynamics_parameters(data_file, urdf_file, cfg)
        print("\n========================================")
        print("动力学参数辨识结束")
        print("========================================")
    except Exception as e:
        print(f"发生异常: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
