#!/usr/bin/env python3
"""
读取 C++ 与 Python 保存的 Y_all 输入数据 (q,dq,ddq)，以及 Y_all 矩阵，
写入汇总 CSV 并计算逐样本/逐列差值。
用法（在项目根目录）:
  python scripts/compare_Y_input_and_diff.py
  python scripts/compare_Y_input_and_diff.py --build-dir build --output-dir build_outputs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def load_input_csv(path: str) -> tuple[list[int], np.ndarray]:
    """读取 Y_input_*.csv，返回 sample_idx 列表和 (N, 21) 数组 [q1..q7, dq1..dq7, ddq1..ddq7]。"""
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx_col = header.index("sample_idx")
        rows = []
        indices = []
        for row in r:
            if not row:
                continue
            indices.append(int(row[idx_col]))
            vals = [float(row[i]) for i in range(len(row)) if i != idx_col]
            rows.append(vals)
    return indices, np.array(rows, dtype=np.float64)


def load_Y_all_bin(path: str) -> np.ndarray | None:
    """读取 Y_all 二进制：int32 rows, int32 cols, row-major float64。"""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        rows = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        cols = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        data = np.frombuffer(f.read(rows * cols * 8), dtype=np.float64)
    return data.reshape(rows, cols, order="C")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 C++/Python 的 Y_all 输入数据并求差")
    parser.add_argument("--build-dir", default="build", help="C++ 输出目录（含 Y_input_cpp.csv, Y_all_cpp.bin）")
    parser.add_argument("--output-dir", default="build_outputs", help="Python 输出目录（含 Y_input_py.csv, Y_all_py.bin）")
    parser.add_argument("--out-csv", default=None, help="汇总与差值 CSV 输出路径，默认 output_dir/compare_Y_input_diff.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    build_dir = project_root / args.build_dir
    output_dir = project_root / args.output_dir
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / "compare_Y_input_diff.csv"

    cpp_input = build_dir / "Y_input_cpp.csv"
    py_input = output_dir / "Y_input_py.csv"
    cpp_bin = build_dir / "Y_all_cpp.bin"
    py_bin = output_dir / "Y_all_py.bin"

    if not cpp_input.is_file():
        print(f"未找到 C++ 输入: {cpp_input}，请先编译并运行 C++ step2。", file=sys.stderr)
        sys.exit(1)
    if not py_input.is_file():
        print(f"未找到 Python 输入: {py_input}，请先运行 Python step2。", file=sys.stderr)
        sys.exit(1)

    idx_cpp, data_cpp = load_input_csv(str(cpp_input))
    idx_py, data_py = load_input_csv(str(py_input))

    n_cpp, n_py = len(idx_cpp), len(idx_py)
    n_common = min(n_cpp, n_py)
    if n_cpp != n_py:
        print(f"  样本数不一致: C++={n_cpp}, Python={n_py}，仅对比前 {n_common} 个样本。")

    # 对齐：按 sample_idx 逐行对比（假定两边 sample_idx 均为 0,1,2,...）
    N = n_common
    data_cpp_cut = data_cpp[:N]
    data_py_cut = data_py[:N]
    diff = data_cpp_cut - data_py_cut

    col_names = (
        ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
        + ["dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7"]
        + ["ddq1", "ddq2", "ddq3", "ddq4", "ddq5", "ddq6", "ddq7"]
    )

    # 统计
    max_abs_diff = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))
    max_abs_per_col = np.max(np.abs(diff), axis=0)
    mean_abs_per_col = np.mean(np.abs(diff), axis=0)

    print("========================================")
    print("Y_all 输入数据 (q,dq,ddq) 对比")
    print("========================================")
    print(f"  C++ 样本数: {n_cpp}, Python 样本数: {n_py}, 对比样本数: {N}")
    print(f"  全表 最大绝对差: {max_abs_diff:.6e}")
    print(f"  全表 平均绝对差: {mean_abs_diff:.6e}")
    print("  各列最大绝对差:")
    for i, name in enumerate(col_names):
        print(f"    {name}: {max_abs_per_col[i]:.6e} (平均: {mean_abs_per_col[i]:.6e})")

    # 写入 CSV：sample_idx, q1_cpp, q1_py, q1_diff, ...
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sample_idx"]
        for name in col_names:
            header.append(f"{name}_cpp")
        for name in col_names:
            header.append(f"{name}_py")
        for name in col_names:
            header.append(f"{name}_diff")
        w.writerow(header)
        for i in range(N):
            row = [i]
            row.extend(data_cpp_cut[i].tolist())
            row.extend(data_py_cut[i].tolist())
            row.extend(diff[i].tolist())
            w.writerow(row)
    print(f"\n  汇总与差值已写入: {out_csv}")

    # 追加或单独写 Y_all 矩阵对比
    Y_cpp = load_Y_all_bin(str(cpp_bin))
    Y_py = load_Y_all_bin(str(py_bin))
    if Y_cpp is not None and Y_py is not None:
        if Y_cpp.shape != Y_py.shape:
            print(f"  Y_all 形状不一致: C++ {Y_cpp.shape}, Python {Y_py.shape}，跳过矩阵逐元求差。")
        else:
            Y_diff = Y_cpp - Y_py
            y_max = np.max(np.abs(Y_diff))
            y_mean = np.mean(np.abs(Y_diff))
            y_fro = np.linalg.norm(Y_diff, "fro")
            y_fro_rel = y_fro / (np.linalg.norm(Y_cpp, "fro") + 1e-20)
            print("\n========================================")
            print("Y_all 矩阵对比")
            print("========================================")
            print(f"  形状: {Y_cpp.shape}")
            print(f"  最大绝对差: {y_max:.6e}")
            print(f"  平均绝对差: {y_mean:.6e}")
            print(f"  Frobenius 差: {y_fro:.6e} (相对: {y_fro_rel:.6e})")

            # 将 Y_all (C++、Python、差值) 写入 CSV 做逐元对比
            n_rows, n_cols = Y_cpp.shape
            y_all_csv = output_dir / "compare_Y_all_compare.csv"
            with open(y_all_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                header = ["row_idx"]
                header += [f"cpp_{j}" for j in range(n_cols)]
                header += [f"py_{j}" for j in range(n_cols)]
                header += [f"diff_{j}" for j in range(n_cols)]
                w.writerow(header)
                for i in range(n_rows):
                    row = [i]
                    row.extend(Y_cpp[i].tolist())
                    row.extend(Y_py[i].tolist())
                    row.extend(Y_diff[i].tolist())
                    w.writerow(row)
            print(f"  Y_all 逐行对比已写入: {y_all_csv} (行数={n_rows}, 列数 C++/Python/diff 各 {n_cols})")

            # 把 Y_all 统计写入同一目录下的单独文件，便于记录
            summary_path = output_dir / "compare_Y_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as sf:
                sf.write("Y_all 输入 (q,dq,ddq) 对比\n")
                sf.write(f"  对比样本数: {N}\n")
                sf.write(f"  全表最大绝对差: {max_abs_diff:.6e}\n")
                sf.write(f"  全表平均绝对差: {mean_abs_diff:.6e}\n\n")
                sf.write("Y_all 矩阵对比\n")
                sf.write(f"  形状: {Y_cpp.shape}\n")
                sf.write(f"  最大绝对差: {y_max:.6e}\n")
                sf.write(f"  平均绝对差: {y_mean:.6e}\n")
                sf.write(f"  Frobenius差: {y_fro:.6e}, 相对: {y_fro_rel:.6e}\n")
            print(f"  统计摘要已写入: {summary_path}")
    else:
        if Y_cpp is None:
            print(f"  未找到 {cpp_bin}，跳过 Y_all 矩阵对比。")
        if Y_py is None:
            print(f"  未找到 {py_bin}，跳过 Y_all 矩阵对比。")


if __name__ == "__main__":
    main()
