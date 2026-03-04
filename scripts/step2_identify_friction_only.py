# -*- coding: utf-8 -*-
"""
部分辨识（1）：只辨识摩擦，θ 完全用 URDF/CAD。
模型: τ = Y*θ_urdf + D_visc*Fv + D_coul*Fc => 残差 τ - Y*θ_urdf = W_frict * [Fv;Fc]。
约束: Fv >= 0, Fc >= 0（阻尼/库伦摩擦非负，保证物理合理）。
输出 URDF: *_identified_friction_only.urdf
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

# 公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_partial_common import (
    load_config,
    _find_config_path,
    setup_for_partial_identification,
    write_result_csvs_and_urdf,
    run_validation,
    check_inertia_positive_definite,
)

OUTPUT_URDF_BASENAME = "AR5-5_07R-W4C4A2_identified_friction_only.urdf"

# 参数（直接在本文件中设置，不通过 config 导入）
LAM_FRICTION = 1e-8
I_EPS = 1e-6


def main():
    parser = argparse.ArgumentParser(description="部分辨识：仅摩擦 Fv, Fc，θ 用 URDF，Fv/Fc≥0")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("data_file", nargs="?", default=None)
    parser.add_argument("urdf_file", nargs="?", default=None)
    args = parser.parse_args()
    config_path = args.config or _find_config_path()
    cfg = load_config(config_path)
    data_file = args.data_file or cfg["data_file"]
    urdf_file = args.urdf_file or cfg["urdf_file"]
    print("数据:", data_file)
    print("URDF:", urdf_file)
    print("模式: 只辨识摩擦，θ=URDF，约束 Fv≥0 Fc≥0")
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
    W_frict = W_all[:, n_params : n_params + n_frict]
    tau_resid = tau_all - Y_all @ theta_urdf
    lam = LAM_FRICTION
    P = 2.0 * (W_frict.T @ W_frict + lam * np.eye(n_frict))
    q = -2.0 * (W_frict.T @ tau_resid)

    if OSQP_AVAILABLE:
        # 约束: Fv[i] >= 0, Fc[i] >= 0 => x >= 0，即 -x <= 0 => 用 l <= A x <= u: A=-I, l=-inf, u=0
        A_ineq = -np.eye(n_frict)
        l_ineq = np.full(n_frict, -1e30)
        u_ineq = np.zeros(n_frict)
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
            print("  已用 QP（Fv≥0, Fc≥0）求解摩擦")
        else:
            x = np.linalg.solve(P / 2.0, -q / 2.0)
            x = np.maximum(x, 0.0)
            print("  QP 未收敛，使用无约束解并钳制为非负")
    else:
        x = np.linalg.solve(P / 2.0, -q / 2.0)
        x = np.maximum(x, 0.0)
        print("  无 OSQP，使用无约束解并钳制 Fv/Fc≥0")

    Fv = np.asarray(x[:n_joints], dtype=np.float64)
    Fc = np.asarray(x[n_joints:], dtype=np.float64)
    theta_estimated = np.asarray(theta_urdf, dtype=np.float64).copy()

    tau_pred = Y_all @ theta_estimated + W_frict @ x
    rmse = np.sqrt(np.mean((tau_all - tau_pred) ** 2))
    result_summary = f"部分辨识(仅摩擦)\nθ 使用 URDF，辨识 Fv/Fc\nRMSE(tau): {rmse:.6e}\n"
    cfg["I_eps"] = I_EPS
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
    check_inertia_positive_definite(theta_estimated, n_links)
    print(f"\n  完成。输出 URDF: {out['output_urdf']}")


if __name__ == "__main__":
    main()
