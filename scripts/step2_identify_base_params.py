# -*- coding: utf-8 -*-
"""
部分辨识（4）：基参数/最小参数集。τ = Y*θ，θ 可由基参数 β 线性表出：θ = V*β（由 Y 的 SVD 得到）。
辨识 min || Y_base*β - tau ||^2，Y_base = Y*V_r，再 θ = V_r^T * β。
输出 URDF: *_identified_base_params.urdf（θ 可能非物理，写 URDF 前可做投影）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_partial_common import (
    load_config,
    _find_config_path,
    setup_for_partial_identification,
    write_result_csvs_and_urdf,
    run_validation,
)
from step2_dynamics_parameter_estimation_friction_sdp import project_theta_to_physical

OUTPUT_URDF_BASENAME = "AR5-5_07R-W4C4A2_identified_base_params.urdf"


def main():
    parser = argparse.ArgumentParser(description="部分辨识：基参数/最小参数集")
    parser.add_argument("--config", default=None)
    parser.add_argument("--project", action="store_true", help="写 URDF 前将 θ 投影到物理可行域")
    parser.add_argument("data_file", nargs="?", default=None)
    parser.add_argument("urdf_file", nargs="?", default=None)
    args = parser.parse_args()
    config_path = args.config or _find_config_path()
    cfg = load_config(config_path)
    data_file = args.data_file or cfg["data_file"]
    urdf_file = args.urdf_file or cfg["urdf_file"]
    do_project = args.project
    print("数据:", data_file)
    print("URDF:", urdf_file)
    print("模式: 基参数/最小参数集")
    if do_project:
        print("写 URDF 前将对 θ 做物理投影")
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

    # SVD: Y_all = U @ diag(s) @ Vh，取 rank 个主成分，Y_base = Y_all @ Vh[:rank].T，θ = Vh[:rank].T @ β
    U, s, Vh = np.linalg.svd(Y_all, full_matrices=False)
    tol = 1e-6 * (s[0] if s.size else 0)
    rank = int(np.sum(s > tol))
    print(f"  Y 秩: {rank} / {n_params}")

    Vr = Vh[:rank].T  # (n_params, rank)
    Y_base = Y_all @ Vr  # (n_samples*7, rank)
    lam = float(cfg.get("lambda_rel", 1e-2)) * (np.trace(Y_base.T @ Y_base) / max(rank, 1))
    A = Y_base.T @ Y_base + lam * np.eye(rank)
    b = Y_base.T @ tau_all
    beta = np.linalg.solve(A, b)
    theta_estimated = (Vr @ beta).astype(np.float64)

    if do_project and theta_urdf.size == n_params:
        project_theta_to_physical(
            theta_estimated,
            n_params,
            cfg.get("m_min", 1e-4),
            cfg.get("I_eps", 1e-6),
            theta_urdf_fallback=theta_urdf,
            I_trace_min=cfg.get("I_trace_min", 1e-3),
        )

    Fv = np.zeros(n_joints)
    Fc = np.zeros(n_joints)

    tau_pred = Y_all @ theta_estimated
    rmse = np.sqrt(np.mean((tau_all - tau_pred) ** 2))
    result_summary = f"部分辨识(基参数)\n秩={rank}, 基参数维数={rank}\nRMSE(tau): {rmse:.6e}\n"
    if do_project:
        result_summary += "已对 θ 做物理投影后写 URDF\n"
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
