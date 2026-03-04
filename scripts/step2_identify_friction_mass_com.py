# -*- coding: utf-8 -*-
"""
部分辨识：同时辨识摩擦力 (Fv, Fc) 与质量质心 (m, mc)，转动惯量使用 URDF 默认值。
模型: τ = Y*θ + D_visc*Fv + D_coul*Fc，θ 中仅 (m,mc) 每连杆 4 维为未知，惯量 I 固定。
即 tau - Y_fixed*theta_urdf_fixed = [Y_mc, W_frict] * [x_mc; Fv; Fc]。
约束: m >= m_min, Fv >= 0, Fc >= 0。

转动惯量约定（与主 step2 一致）：
- 输入：theta_urdf 来自 Pinocchio 的 toDynamicParameters()，本项目中 10 维里惯量 6 元为绕连杆原点 I^o，
  不做“关节处→质心处”转换，直接用于回归（Y 的列与 I^o 对应）。
- 输出：写 URDF 时在 generate_identified_urdf_com 内将 I^o 转为质心处 I_com 再写入 <inertia>。
输出 URDF: *_identified_friction_mass_com.urdf
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

OUTPUT_URDF_BASENAME = "AR5-5_07R-W4C4A2_identified_friction_mass_com.urdf"


def _mass_com_column_indices(n_links: int) -> list[int]:
    return [10 * j + k for j in range(n_links) for k in range(4)]


def main():
    parser = argparse.ArgumentParser(
        description="部分辨识：摩擦 + 质量质心，惯量用 URDF，约束 m≥m_min, Fv≥0, Fc≥0"
    )
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
    print("模式: 辨识摩擦 + 质量质心，惯量=URDF，Fv/Fc≥0, m≥m_min")
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

    n_frict = 2 * n_joints
    col_inds = _mass_com_column_indices(n_links)
    # fixed_inds：惯量等固定维（每连杆 4:9 为 I^o），来自 theta_urdf，不做关节→质心转换
    fixed_inds = [i for i in range(n_params) if i not in col_inds]
    tau_resid = tau_all - Y_all[:, fixed_inds] @ theta_urdf[fixed_inds]

    Y_mc = Y_all[:, col_inds]
    W_frict = W_all[:, n_params : n_params + n_frict]
    W_ext = np.hstack([Y_mc, W_frict])
    n_mc = len(col_inds)
    n_x = n_mc + n_frict

    lam_mc = float(cfg.get("lambda_rel", 1e-2)) * (np.trace(Y_mc.T @ Y_mc) / max(n_mc, 1))
    lam_frict = float(cfg.get("lam_friction", 1e-8))
    L_diag = np.ones(n_x)
    L_diag[:n_mc] = lam_mc
    L_diag[n_mc:] = lam_frict
    x_prior = np.zeros(n_x)
    x_prior[:n_mc] = theta_urdf[col_inds]

    P = 2.0 * (W_ext.T @ W_ext + np.diag(L_diag))
    q = -2.0 * (W_ext.T @ tau_resid + L_diag * x_prior)
    m_min = float(cfg.get("m_min", 1e-4))

    if OSQP_AVAILABLE and n_links == 7:
        # 约束: x[4*j] >= m_min (7), x[n_mc+j] >= 0 (Fv), x[n_mc+n_joints+j] >= 0 (Fc)
        n_ineq = 7 + 7 + 7
        A_ineq = np.zeros((n_ineq, n_x))
        l_ineq = np.full(n_ineq, -1e30)
        u_ineq = np.zeros(n_ineq)
        row = 0
        for j in range(n_links):
            A_ineq[row, 4 * j] = -1.0
            u_ineq[row] = -m_min
            row += 1
        for j in range(n_joints):
            A_ineq[row, n_mc + j] = -1.0
            u_ineq[row] = 0.0
            row += 1
        for j in range(n_joints):
            A_ineq[row, n_mc + n_joints + j] = -1.0
            u_ineq[row] = 0.0
            row += 1
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
            x = r.x
            print("  已用 QP（m≥m_min, Fv≥0, Fc≥0）求解 (m,mc)+摩擦")
        else:
            x = np.linalg.solve(P / 2.0, -q / 2.0)
            for j in range(n_links):
                if x[4 * j] < m_min:
                    x[4 * j] = m_min
            x[n_mc:] = np.maximum(x[n_mc:], 0.0)
            print("  QP 未收敛，使用无约束解并钳制")
    else:
        x = np.linalg.solve(P / 2.0, -q / 2.0)
        for j in range(n_links):
            if x[4 * j] < m_min:
                x[4 * j] = m_min
        x[n_mc:] = np.maximum(x[n_mc:], 0.0)
        print("  无 OSQP 或 n_links≠7，使用无约束解并钳制")

    theta_estimated = theta_urdf.copy()
    for j in range(n_links):
        base = 10 * j
        theta_estimated[base : base + 4] = x[4 * j : 4 * j + 4]
    Fv = np.asarray(x[n_mc : n_mc + n_joints], dtype=np.float64)
    Fc = np.asarray(x[n_mc + n_joints :], dtype=np.float64)

    tau_pred = Y_all @ theta_estimated + W_frict @ x[n_mc:]
    rmse = np.sqrt(np.mean((tau_all - tau_pred) ** 2))
    result_summary = (
        "部分辨识(摩擦 + 质量质心)\n"
        "惯量用 URDF，辨识 (m,mc)+Fv/Fc，约束 m≥m_min, Fv≥0, Fc≥0\n"
        f"RMSE(tau): {rmse:.6e}\n"
    )
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
