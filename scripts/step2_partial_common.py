# -*- coding: utf-8 -*-
"""
部分辨识公共逻辑：加载数据、构建 Y/W/tau、写 URDF 与 CSV。
供 step2_identify_friction_only / mass_com_only / partial_links / base_params 使用。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.pop("PYTHONPATH", None)
sys.path[:] = [p for p in sys.path if p and ("/opt/ros/" not in p)]

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None

# 从主辨识脚本复用
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from step2_dynamics_parameter_estimation_friction_sdp import (
    load_config,
    _find_config_path,
    load_data_from_csv,
    build_model_and_data,
    generate_identified_urdf_com,
    _ensure_output_dir,
    _inertia_origin_to_com,
    dirname_of,
)


def setup_for_partial_identification(
    data_file: str,
    urdf_file: str,
    cfg: dict,
    output_urdf_basename: str,
):
    """加载数据、构建 Y_all, W_all, tau_all 与 theta_urdf。覆盖 cfg['output_urdf']。
    返回: out, model, data, collected, Y_all, W_all, tau_all, theta_urdf, n_params, n_links, n_joints
    """
    if pin is None:
        raise RuntimeError("需要 Pinocchio 库")
    cfg["output_urdf"] = output_urdf_basename
    out = _ensure_output_dir(cfg)
    if cfg.get("output_dir"):
        print(f"  输出目录: {os.path.abspath(cfg['output_dir'])}")

    print("\n  初始化 Pinocchio 动力学模型...")
    model, data = build_model_and_data(urdf_file)
    print(f"  模型加载: {urdf_file}, 自由度: {model.nq}")

    print("\n  从文件加载数据:", data_file)
    collected = load_data_from_csv(data_file)
    if not collected:
        raise RuntimeError("没有加载到数据")

    q_test = np.zeros(7)
    v_test = np.zeros(7)
    a_test = np.zeros(7)
    pin.computeJointTorqueRegressor(model, data, q_test, v_test, a_test)
    Y_ref = np.asarray(data.jointTorqueRegressor)
    n_params = Y_ref.shape[1]
    n_links = n_params // 10
    n_joints = 7
    n_friction = n_joints * 2
    print(f"  回归矩阵参数数量: {n_params}")

    theta_urdf = np.zeros(n_params)
    expected_params = 10 * (model.njoints - 1)
    if n_params == expected_params:
        for jid in range(1, model.njoints):
            pi = np.array(model.inertias[jid].toDynamicParameters()).ravel()
            base = 10 * (jid - 1)
            theta_urdf[base : base + 10] = pi
        print(f"  已构造 URDF 先验 theta_urdf (dim {theta_urdf.size})")
    else:
        theta_urdf = np.array([])
        print("  警告: n_params != 10*(njoints-1)")

    Y_list = []
    W_list = []
    tau_list = []
    for idx, point in enumerate(collected):
        q = np.array(point["q"], dtype=np.float64)
        v = np.array(point["dq"], dtype=np.float64)
        a = np.array(point["ddq"], dtype=np.float64)
        try:
            pin.computeJointTorqueRegressor(model, data, q, v, a)
            Y = np.asarray(data.jointTorqueRegressor).copy()
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
        Y_list.append(Y)
        W_list.append(W_block)
        tau_list.append(tau)
        if (idx + 1) % 500 == 0:
            print(f"  已处理: {idx + 1} / {len(collected)}")

    Y_all = np.vstack(Y_list)
    W_all = np.vstack(W_list)
    tau_all = np.concatenate(tau_list)
    print(f"  回归矩阵: Y {Y_all.shape}, W {W_all.shape}, tau {tau_all.size}")

    return (
        out,
        model,
        data,
        collected,
        Y_all,
        W_all,
        tau_all,
        theta_urdf,
        n_params,
        n_links,
        n_joints,
    )


def write_result_csvs_and_urdf(
    out: dict,
    theta_estimated: np.ndarray,
    Fv: np.ndarray,
    Fc: np.ndarray,
    theta_urdf: np.ndarray,
    n_params: int,
    n_joints: int,
    urdf_file: str,
    model,
    cfg: dict,
    result_summary: str = "",
) -> None:
    """写 dynamics_parameters_csv, dynamics_parameters_friction_csv, dynamics_parameters_urdf_csv, physical_parameters_txt, result_file, 并生成 URDF。"""
    with open(out["dynamics_parameters_csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter_index", "value"])
        for i, v in enumerate(theta_estimated):
            w.writerow([i, f"{v:.10e}"])
    print(f"  参数已保存: {out['dynamics_parameters_csv']}")

    with open(out["dynamics_parameters_friction_csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "Fv", "Fc"])
        for j in range(n_joints):
            w.writerow([j, f"{Fv[j]:.10e}", f"{Fc[j]:.10e}"])
    print(f"  摩擦已保存: {out['dynamics_parameters_friction_csv']}")

    if theta_urdf.size == n_params:
        with open(out["dynamics_parameters_urdf_csv"], "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter_index", "value"])
            for i, v in enumerate(theta_urdf):
                w.writerow([i, f"{v:.10e}"])
        print(f"  URDF 先验已保存: {out['dynamics_parameters_urdf_csv']}")

    with open(out["result_file"], "w", encoding="utf-8") as f:
        f.write(result_summary)
    print(f"  结果摘要已保存: {out['result_file']}")

    if theta_estimated.size == n_params and n_params == 70:
        generate_identified_urdf_com(
            urdf_file,
            out["output_urdf"],
            model,
            theta_estimated,
            Fv=Fv,
            Fc=Fc,
            I_eps=cfg.get("I_eps", 1e-6),
        )


def run_validation(
    out: dict,
    collected: list,
    theta_estimated: np.ndarray,
    Fv: np.ndarray,
    Fc: np.ndarray,
    n_params: int,
    n_joints: int,
    urdf_file: str,
) -> tuple[float | None, float | None]:
    """用后 20% 数据做验证：用辨识 URDF 的 model 算 Y，报告验证集 RMSE(Y*θ) 与 RMSE(Y*θ+摩擦)。
    返回 (rmse_val_no_fric, rmse_val)，无验证数据时返回 (None, None)。"""
    if pin is None or not collected or theta_estimated.size != n_params:
        return (None, None)
    id_urdf = out["output_urdf"]
    if not os.path.isabs(id_urdf):
        id_urdf = os.path.join(os.path.dirname(out["result_file"]), os.path.basename(out["output_urdf"]))
    if not os.path.isfile(id_urdf):
        id_urdf = os.path.join(dirname_of(urdf_file), os.path.basename(out["output_urdf"]))
    if not os.path.isfile(id_urdf):
        print("  [验证] 未找到辨识 URDF，跳过验证")
        return (None, None)
    model_val, data_val = build_model_and_data(id_urdf)
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
            Y = np.asarray(data_val.jointTorqueRegressor).copy()
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
    if not W_val_list:
        print("  [验证] 验证集无有效样本")
        return (None, None)
    W_val = np.vstack(W_val_list)
    tau_val = np.concatenate(tau_val_list)
    phi = np.concatenate([theta_estimated, Fv, Fc])
    tau_val_pred = W_val @ phi
    err_val = tau_val - tau_val_pred
    rmse_val = float(np.sqrt(np.mean(err_val ** 2)))
    Y_val = W_val[:, :n_params]
    tau_val_pred_no_fric = Y_val @ theta_estimated
    rmse_val_no_fric = float(np.sqrt(np.mean((tau_val - tau_val_pred_no_fric) ** 2)))
    max_err = float(np.max(np.abs(err_val)))
    mean_err = float(np.mean(np.abs(err_val)))
    print("\n========================================")
    print("验证集结果（后 20% 数据）")
    print("========================================")
    print(f"  验证集数据点数: {validation_count}")
    print(f"  验证集 RMSE (Y*θ，仅刚性): {rmse_val_no_fric:.4f} Nm")
    print(f"  验证集 RMSE (Y*θ+摩擦): {rmse_val:.4f} Nm")
    print(f"  验证集 最大绝对误差: {max_err:.4f} Nm")
    print(f"  验证集 平均绝对误差: {mean_err:.4f} Nm")
    print("  说明: Y 由辨识得到的 URDF 的 model 计算。")
    with open(out["result_file"], "a", encoding="utf-8") as f:
        f.write("\n========================================\n")
        f.write("验证集结果（后 20% 数据）\n")
        f.write("========================================\n")
        f.write(f"验证集数据点数: {validation_count}\n")
        f.write(f"验证集 RMSE (Y*θ，仅刚性): {rmse_val_no_fric:.6f} Nm\n")
        f.write(f"验证集 RMSE (Y*θ+摩擦): {rmse_val:.6f} Nm\n")
        f.write(f"验证集 最大绝对误差: {max_err:.6f} Nm\n")
        f.write(f"验证集 平均绝对误差: {mean_err:.6f} Nm\n")
        f.write("说明: Y 由辨识得到的 URDF 的 model 计算。\n")
    return (rmse_val_no_fric, rmse_val)


def run_validation_initial_urdf(
    out: dict,
    collected: list,
    theta_urdf: np.ndarray,
    n_params: int,
    n_joints: int,
    urdf_file: str,
) -> float | None:
    """用后 20% 数据验证初始 URDF：Y 由初始 URDF 的 model 计算，θ=theta_urdf，摩擦=0。
    返回验证集 RMSE(Y*θ)，无验证数据时返回 None。"""
    if pin is None or not collected or theta_urdf.size != n_params:
        return None
    if not os.path.isfile(urdf_file):
        print("  [验证-初始URDF] 未找到初始 URDF，跳过")
        return None
    model_val, data_val = build_model_and_data(urdf_file)
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
            Y = np.asarray(data_val.jointTorqueRegressor).copy()
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
    if not W_val_list:
        print("  [验证-初始URDF] 验证集无有效样本")
        return None
    W_val = np.vstack(W_val_list)
    tau_val = np.concatenate(tau_val_list)
    Fv_zero = np.zeros(n_joints)
    Fc_zero = np.zeros(n_joints)
    phi = np.concatenate([theta_urdf, Fv_zero, Fc_zero])
    tau_val_pred = W_val @ phi
    err_val = tau_val - tau_val_pred
    rmse_val = float(np.sqrt(np.mean(err_val ** 2)))
    Y_val = W_val[:, :n_params]
    tau_val_pred_rigid = Y_val @ theta_urdf
    rmse_val_rigid = float(np.sqrt(np.mean((tau_val - tau_val_pred_rigid) ** 2)))
    max_err = float(np.max(np.abs(err_val)))
    mean_err = float(np.mean(np.abs(err_val)))
    print("\n========================================")
    print("验证集结果（初始 URDF，无摩擦）")
    print("========================================")
    print(f"  验证集数据点数: {validation_count}")
    print(f"  验证集 RMSE (Y*θ，仅刚性): {rmse_val_rigid:.4f} Nm")
    print(f"  验证集 RMSE (Y*θ+摩擦=0): {rmse_val:.4f} Nm")
    print(f"  验证集 最大绝对误差: {max_err:.4f} Nm")
    print(f"  验证集 平均绝对误差: {mean_err:.4f} Nm")
    print("  说明: Y 由初始 URDF 的 model 计算，θ=URDF 先验，摩擦=0。")
    with open(out["result_file"], "a", encoding="utf-8") as f:
        f.write("\n========================================\n")
        f.write("验证集结果（初始 URDF，无摩擦）\n")
        f.write("========================================\n")
        f.write(f"验证集数据点数: {validation_count}\n")
        f.write(f"验证集 RMSE (Y*θ，仅刚性): {rmse_val_rigid:.6f} Nm\n")
        f.write(f"验证集 RMSE (Y*θ+摩擦=0): {rmse_val:.6f} Nm\n")
        f.write(f"验证集 最大绝对误差: {max_err:.6f} Nm\n")
        f.write(f"验证集 平均绝对误差: {mean_err:.6f} Nm\n")
        f.write("说明: Y 由初始 URDF 的 model 计算，θ=URDF 先验，摩擦=0。\n")
    return rmse_val
