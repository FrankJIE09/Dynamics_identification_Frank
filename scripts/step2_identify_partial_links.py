# -*- coding: utf-8 -*-
"""
部分辨识（3）：只辨识末端若干连杆的 10D 参数，其余连杆用 URDF。
配置: identify_links 例如 [5, 6] 表示只辨识第 5、6 根连杆（0 起序）。
模型: tau - Y_fixed*theta_urdf_fixed = Y_part * theta_part。
输出 URDF: *_identified_partial_links.urdf
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

OUTPUT_URDF_BASENAME = "AR5-5_07R-W4C4A2_identified_partial_links.urdf"


def main():
    parser = argparse.ArgumentParser(description="部分辨识：仅末端若干连杆 10D，其余用 URDF")
    parser.add_argument("--config", default=None)
    parser.add_argument("--links", default="5,6", help="要辨识的连杆索引(0起), 逗号分隔, 如 5,6 表示末端 2 根")
    parser.add_argument("data_file", nargs="?", default=None)
    parser.add_argument("urdf_file", nargs="?", default=None)
    args = parser.parse_args()
    config_path = args.config or _find_config_path()
    cfg = load_config(config_path)
    data_file = args.data_file or cfg["data_file"]
    urdf_file = args.urdf_file or cfg["urdf_file"]
    identify_links = [int(x.strip()) for x in args.links.split(",") if x.strip()]
    print("数据:", data_file)
    print("URDF:", urdf_file)
    print("模式: 只辨识连杆", identify_links, "的 10D 参数")
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

    # 列索引: 对每个 j in identify_links，取 10*j .. 10*j+9
    col_inds = []
    for j in identify_links:
        if 0 <= j < n_links:
            col_inds.extend(range(10 * j, 10 * j + 10))
    col_inds = sorted(set(col_inds))
    fixed_inds = [i for i in range(n_params) if i not in col_inds]

    Y_part = Y_all[:, col_inds]
    tau_resid = tau_all - Y_all[:, fixed_inds] @ theta_urdf[fixed_inds]
    lam = float(cfg.get("lambda_rel", 1e-2)) * (np.trace(Y_part.T @ Y_part) / max(len(col_inds), 1))
    A = Y_part.T @ Y_part + lam * np.eye(Y_part.shape[1])
    b = Y_part.T @ tau_resid
    x_part = np.linalg.solve(A, b)

    theta_estimated = theta_urdf.copy()
    for ii, idx in enumerate(col_inds):
        theta_estimated[idx] = x_part[ii]
    # 写 URDF 前对 θ 做物理投影，避免负惯量/非正定
    if theta_urdf.size == n_params:
        project_theta_to_physical(
            theta_estimated,
            n_params,
            float(cfg.get("m_min", 1e-4)),
            float(cfg.get("I_eps", 1e-6)),
            theta_urdf_fallback=theta_urdf,
            I_trace_min=float(cfg.get("I_trace_min", 1e-3)),
        )
        print("  已对 θ 做物理投影后写 URDF")
    Fv = np.zeros(n_joints)
    Fc = np.zeros(n_joints)

    tau_pred = Y_all @ theta_estimated
    rmse = np.sqrt(np.mean((tau_all - tau_pred) ** 2))
    result_summary = f"部分辨识(仅连杆 {identify_links} 的 10D)\n其余连杆用 URDF\nRMSE(tau): {rmse:.6e}\n"
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
