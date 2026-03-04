#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1.2：用 q,dq,ddq 通过 MuJoCo 逆动力学算 tau，与 CSV 中的 tau 对比

读取：
- MuJoCo XML 模型（与 step1 相同重力设置）
- dynamics_identification_data.csv（time, q0..q6, dq0..dq6, ddq0..ddq6, tau0..tau6）

对每一行：设 qpos,qvel,qacc → mj_inverse → qfrc_inverse，与 CSV 的 tau 比较，输出误差统计。

依赖: pip install mujoco numpy scipy
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco
except ImportError:
    print("请安装 mujoco: pip install mujoco")
    sys.exit(1)


def load_csv(csv_path: str):
    """加载 step1 输出的 CSV，返回 list of dict: time, q[7], dq[7], ddq[7], tau[7]"""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 1 + 7 * 4:
                continue
            try:
                t = float(row[0])
                q = [float(row[1 + i]) for i in range(7)]
                dq = [float(row[8 + i]) for i in range(7)]
                ddq = [float(row[15 + i]) for i in range(7)]
                tau = [float(row[22 + i]) for i in range(7)]
                data.append({"time": t, "q": q, "dq": dq, "ddq": ddq, "tau": tau})
            except (ValueError, IndexError):
                continue
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Step1.2: 用 q,dq,ddq 逆动力学算 tau 与 CSV 中 tau 对比"
    )
    parser.add_argument(
        "--model", "-m",
        default="./AR5-5_07R-W4C4A2/AR5-5_07R-W4C4A2.xml",
        help="MuJoCo XML 模型路径",
    )
    parser.add_argument(
        "--csv", "-c",
        default="./dynamics_identification_data.csv",
        help="step1 输出的 CSV 路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="可选：对比结果输出 CSV（time, tau_csv, tau_id, err 等）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="最多处理前 N 行（默认全部）",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(script_dir, args.model)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(script_dir, args.csv)

    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}")
        sys.exit(1)
    if not os.path.exists(csv_path):
        print(f"错误: CSV 不存在 {csv_path}")
        sys.exit(1)

    # 加载模型，重力与 step1 / step2 copy 一致
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    R_base = Rotation.from_euler("yz", [np.pi / 2, -np.pi / 2]).as_matrix()
    gravity_world = np.array([0.0, 0.0, -9.81])
    model.opt.gravity[:] =  gravity_world

    if model.nq < 7 or model.nv < 7:
        print(f"错误: 期望至少 7 关节，当前 nq={model.nq}, nv={model.nv}")
        sys.exit(1)

    rows = load_csv(csv_path)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        print("错误: CSV 无有效数据")
        sys.exit(1)

    print(f"模型: {model_path}")
    print(f"CSV:  {csv_path}，共 {len(rows)} 行")
    print("----------------------------------------")

    tau_csv = np.array([r["tau"] for r in rows])
    tau_id = np.zeros_like(tau_csv)

    for i, row in enumerate(rows):
        data.qpos[:7] = row["q"]
        data.qvel[:7] = row["dq"]
        data.qacc[:7] = row["ddq"]
        mujoco.mj_inverse(model, data)
        tau_id[i, :] = data.qfrc_inverse[:7]

    err = tau_id - tau_csv
    abs_err = np.abs(err)

    print("按关节统计误差 (tau_inverse_dynamics - tau_csv):")
    print("  关节     mean(err)   std(err)   max|err|   mean|err|")
    for j in range(7):
        print(f"  joint{j}  {err[:, j].mean():+8.4f}   {err[:, j].std():8.4f}   {abs_err[:, j].max():8.4f}   {abs_err[:, j].mean():8.4f}")
    print("----------------------------------------")
    print(f"  全体    {err.mean():+8.4f}   {err.std():8.4f}   {abs_err.max():8.4f}   {abs_err.mean():8.4f}")

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("time,")
            f.write(",".join(f"tau_csv_{j}" for j in range(7)) + ",")
            f.write(",".join(f"tau_id_{j}" for j in range(7)) + ",")
            f.write(",".join(f"err_{j}" for j in range(7)) + "\n")
            for i, row in enumerate(rows):
                parts = [f"{row['time']:.6f}"]
                for j in range(7):
                    parts.append(f"{tau_csv[i, j]:.6f}")
                for j in range(7):
                    parts.append(f"{tau_id[i, j]:.6f}")
                for j in range(7):
                    parts.append(f"{err[i, j]:.6f}")
                f.write(",".join(parts) + "\n")
        print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
