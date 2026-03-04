# -*- coding: utf-8 -*-
"""
部分辨识（2）：只辨识质量与质心 (m, mc) 每连杆 4 维，惯量 I 固定（来自 URDF/CAD）。
模型: τ = Y*θ，θ 中仅 [m, mc_x, mc_y, mc_z] 每连杆为未知，其余取 theta_urdf。
即 tau - Y_fixed*theta_urdf_fixed = Y_mc * x_mc，x_mc 为 4*n_links 维。
输出 URDF: *_identified_mass_com_only.urdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_partial_common import (
    load_config,
    _find_config_path,
    setup_for_partial_identification,
    write_result_csvs_and_urdf,
    run_validation,
)

OUTPUT_URDF_BASENAME = "AR5-5_07R-W4C4A2_identified_mass_com_only.urdf"

# 每连杆 10 维中只辨识前 4 个: indices 0,1,2,3, 10,11,12,13, ...
def _mass_com_column_indices(n_links: int) -> list[int]:
    return [10 * j + k for j in range(n_links) for k in range(4)]


def main():
    parser = argparse.ArgumentParser(description="部分辨识：仅质量与质心 (m,mc)，惯量用 URDF")
    parser.add_argument("--config", default=None)
    parser.add_argument("data_file", nargs="?", default=None)
    parser.add_argument("urdf_file", nargs="?", default=None)
    args = parser.parse_args()
    config_path = args.config or _find_config_path()
    cfg = load_config(config_path)
    data_file = args.data_file or cfg["data_file"]
    urdf_file = args.urdf_file or cfg["urdf_file"]
    print("数据:", data_file)
    print("URDF:", urdf_file)
    print("模式: 只辨识质量与质心，惯量=URDF")
    print("========================================")

    (
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
    ) = setup_for_partial_identification(
        data_file, urdf_file, cfg, OUTPUT_URDF_BASENAME
    )

    # 摩擦本脚本不辨识，置 0；残差只用刚性部分
    # tau = Y*theta, theta 中 0:4,10:14,...,60:64 为未知 x_mc，其余=theta_urdf
    # tau = Y_mc @ x_mc + Y_fixed @ theta_urdf_fixed => tau - Y_fixed@theta_fixed = Y_mc @ x_mc
    col_inds = _mass_com_column_indices(n_links)
    Y_mc = Y_all[:, col_inds]
    fixed_inds = [i for i in range(n_params) if i not in col_inds]
    tau_resid = tau_all - Y_all[:, fixed_inds] @ theta_urdf[fixed_inds]
    x_prior = theta_urdf[col_inds]
    lam = float(cfg.get("lambda_rel", 1e-2)) * (np.trace(Y_mc.T @ Y_mc) / max(len(col_inds), 1))
    m_min = float(cfg.get("m_min", 1e-4))
    n_mc = len(col_inds)
    # 带先验: min ||Y_mc@x - tau_resid||^2 + lam*||x - x_prior||^2，避免 (m,mc) 偏离 URDF 过远
    P = 2.0 * (Y_mc.T @ Y_mc + lam * np.eye(n_mc))
    q = -2.0 * (Y_mc.T @ tau_resid + lam * x_prior)
    x_mc = np.linalg.solve(P / 2.0, -q / 2.0)
    if OSQP_AVAILABLE and n_links == 7:
        # 约束: x[4*j] >= m_min => -x[4*j] <= -m_min
        A_ineq = np.zeros((n_links, n_mc))
        for j in range(n_links):
            A_ineq[j, 4 * j] = -1.0
        l_ineq = np.full(n_links, -1e30)
        u_ineq = np.full(n_links, -m_min)
        prob = osqp.OSQP()
        prob.setup(
            P=sparse.csc_matrix(P),
            q=q,
            A=sparse.csc_matrix(A_ineq),
            l=l_ineq,
            u=u_ineq,
            verbose=False,
            polish=True,
        )
        r = prob.solve()
        if r.info.status in ("solved", "solved inaccurate"):
            x_mc = r.x
            print("  已用 QP（质量下界 m>=m_min）求解 (m,mc)")
        else:
            print("  QP 未收敛，使用无约束 LS 解（可能非物理）")
    else:
        for j in range(n_links):
            if x_mc[4 * j] < m_min:
                x_mc[4 * j] = m_min
                print("  已将连杆 {} 质量钳制为 m_min".format(j))

    theta_estimated = theta_urdf.copy()
    for j in range(n_links):
        base = 10 * j
        theta_estimated[base : base + 4] = x_mc[4 * j : 4 * j + 4]
    Fv = np.zeros(n_joints)
    Fc = np.zeros(n_joints)

    tau_pred = Y_all @ theta_estimated
    rmse = np.sqrt(np.mean((tau_all - tau_pred) ** 2))
    result_summary = f"部分辨识(仅质量与质心)\n惯量用 URDF，辨识 (m,mc) 每连杆 4 维\nRMSE(tau): {rmse:.6e}\n"
    write_result_csvs_and_urdf(
        out,
        theta_estimated,
        Fv,
        Fc,
        theta_urdf,
        n_params,
        n_joints,
        urdf_file,
        model,
        cfg,
        result_summary=result_summary,
    )
    run_validation(out, collected, theta_estimated, Fv, Fc, n_params, n_joints, urdf_file)
    print(f"\n  完成。输出 URDF: {out['output_urdf']}")


if __name__ == "__main__":
    main()
