# -*- coding: utf-8 -*-
"""
对比 CSV 实测关节力矩与辨识 URDF 计算的力矩。
用法: python compare_tau_csv_urdf.py <数据CSV> <辨识URDF> [摩擦系数CSV] [输出目录]
  τ_urdf = RNEA(q,dq,ddq) + Fv*dq + Fc*sign(dq)，与 CSV 中的 tau 逐样本对比，输出 RMSE、按关节统计及对比 CSV。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.pop("PYTHONPATH", None)
sys.path[:] = [p for p in sys.path if p and ("/opt/ros/" not in p)]

try:
    import pinocchio as pin
except ImportError:
    pin = None


def build_model_and_data(urdf_file: str):
    if pin is None:
        raise RuntimeError("需要 Pinocchio 库")
    model = pin.buildModelFromUrdf(urdf_file)
    data = model.createData()
    R_base = Rotation.from_euler("yz", [np.pi / 2, -np.pi / 2]).as_matrix()
    gravity_world = np.array([0.0, 0.0, -9.81])
    model.gravity.linear[:] = R_base.T @ gravity_world
    return model, data


def load_csv(filename: str) -> list[dict]:
    """CSV 格式: timestamp, q1..q7, dq1..dq7, ddq1..ddq7, tau1..tau7"""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 1 + 7 * 4:
                continue
            try:
                data.append({
                    "q": [float(row[1 + i]) for i in range(7)],
                    "dq": [float(row[8 + i]) for i in range(7)],
                    "ddq": [float(row[15 + i]) for i in range(7)],
                    "tau": [float(row[22 + i]) for i in range(7)],
                })
            except (ValueError, IndexError):
                continue
    return data


def load_friction_csv(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """读取 dynamics_parameters_friction.csv：header joint,Fv,Fc，每行 j, Fv_j, Fc_j。"""
    Fv = np.zeros(7)
    Fc = np.zeros(7)
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                j = int(row[0])
                if 0 <= j < 7:
                    Fv[j] = float(row[1])
                    Fc[j] = float(row[2])
            except (ValueError, IndexError):
                continue
    return Fv, Fc


def main():
    _script_dir = Path(__file__).resolve().parent
    _default_csv = _script_dir.parent / "src" / "config" / "dynamics_identification_data.csv"
    _default_urdf = _script_dir / "data_output" / "AR5-5_07R-W4C4A2_identified_inertia_com_friction.urdf"
    _default_friction = _script_dir / "data_output" / "dynamics_parameters_friction.csv"
    _default_out = _script_dir / "data_output"

    parser = argparse.ArgumentParser(description="对比 CSV 实测 τ 与辨识 URDF 计算的 τ")
    parser.add_argument("csv_file", nargs="?", default=str(_default_csv), help=f"采集数据 CSV，默认 {_default_csv.name}")
    parser.add_argument("urdf_file", nargs="?", default=str(_default_urdf), help=f"辨识 URDF，默认 data_output/AR5-5_07R-W4C4A2_identified_inertia_com_friction.urdf")
    parser.add_argument("friction_csv", nargs="?", default=None, help="摩擦系数 CSV（joint,Fv,Fc），默认 data_output/dynamics_parameters_friction.csv")
    parser.add_argument("out_dir", nargs="?", default=None, help="输出目录，默认 data_output")
    args = parser.parse_args()

    if args.friction_csv is None and (_default_friction.exists()):
        args.friction_csv = str(_default_friction)
    if args.out_dir is None:
        args.out_dir = str(_default_out)

    if pin is None:
        print("错误: 需要安装 pinocchio", file=sys.stderr)
        sys.exit(1)

    csv_path = os.path.abspath(args.csv_file)
    urdf_path = os.path.abspath(args.urdf_file)
    if not os.path.isfile(csv_path):
        print(f"错误: CSV 不存在 {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(urdf_path):
        print(f"错误: URDF 不存在 {urdf_path}", file=sys.stderr)
        sys.exit(1)

    Fv = np.zeros(7)
    Fc = np.zeros(7)
    if args.friction_csv and os.path.isfile(args.friction_csv):
        Fv, Fc = load_friction_csv(args.friction_csv)
        print(f"  已加载摩擦系数: {args.friction_csv}")
    else:
        print("  未提供摩擦系数 CSV，使用 Fv=Fc=0")

    data = load_csv(csv_path)
    if not data:
        print("错误: CSV 无有效数据", file=sys.stderr)
        sys.exit(1)
    print(f"  加载 {len(data)} 条样本: {csv_path}")

    model, model_data = build_model_and_data(urdf_path)
    print(f"  已加载辨识 URDF: {urdf_path}")

    n = len(data)
    tau_meas = np.zeros((n, 7))
    tau_urdf = np.zeros((n, 7))
    for i, point in enumerate(data):
        q = np.array(point["q"], dtype=np.float64)
        v = np.array(point["dq"], dtype=np.float64)
        a = np.array(point["ddq"], dtype=np.float64)
        tau_meas[i] = point["tau"]
        tau_rnea = pin.rnea(model, model_data, q, v, a)
        tau_urdf[i] = np.array(tau_rnea) + Fv * v + Fc * np.sign(v)

    err = tau_meas - tau_urdf
    rmse = np.sqrt(np.mean(err ** 2))
    rmse_per_j = np.sqrt(np.mean(err ** 2, axis=0))
    max_abs = np.max(np.abs(err))
    mean_abs = np.mean(np.abs(err))

    print("\n========================================")
    print("CSV 实测 τ vs 辨识 URDF 计算 τ")
    print("========================================")
    print(f"  样本数: {n}")
    print(f"  整体 RMSE: {rmse:.6f} Nm")
    print(f"  最大绝对误差: {max_abs:.6f} Nm")
    print(f"  平均绝对误差: {mean_abs:.6f} Nm")
    print("  按关节 RMSE [Nm]:", " ".join(f"J{j}={rmse_per_j[j]:.4f}" for j in range(7)))

    out_dir = args.out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, f"compare_tau_{base}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sample_idx"]
        for j in range(7):
            header += [f"tau{j+1}_meas", f"tau{j+1}_urdf", f"err{j+1}"]
        w.writerow(header)
        for i in range(n):
            row = [i]
            for j in range(7):
                row += [tau_meas[i, j], tau_urdf[i, j], err[i, j]]
            w.writerow(row)
    print(f"\n  对比结果已保存: {out_csv}")

    out_txt = os.path.join(out_dir, f"compare_tau_{base}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(f"URDF: {urdf_path}\n")
        f.write(f"样本数: {n}\n")
        f.write(f"整体 RMSE: {rmse:.6f} Nm\n")
        f.write(f"最大绝对误差: {max_abs:.6f} Nm\n")
        f.write(f"按关节 RMSE: " + " ".join(f"J{j}={rmse_per_j[j]:.6f}" for j in range(7)) + "\n")
    print(f"  摘要已保存: {out_txt}")


if __name__ == "__main__":
    main()
