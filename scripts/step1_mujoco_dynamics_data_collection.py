#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo 动力学辨识数据采集（关节位置控制）

参考 src/step1_dynamics_data_collection_joint.cpp 的逻辑：
- 正弦激励轨迹
- 关节位置控制
- 输出 CSV：time,q0..q6,dq0..dq6,ddq0..ddq6,tau0..tau6，供 step2 使用

依赖: pip install mujoco numpy
"""

import os
import sys
import argparse
import csv
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("请安装 mujoco: pip install mujoco")
    sys.exit(1)


# 与 step1_dynamics_data_collection_joint.cpp 一致的初始位置 (rad)
Q_INIT = np.array([
    -0.0,   # 一轴
    -0.0,   # 二轴
    -0.0,   # 三轴
    0.0,    # 四轴
    0.0,    # 五轴
    -0.0,   # 六轴
    -0.0,   # 七轴
], dtype=np.float64)

# 激励轨迹参数（与 C++ 一致）
AMPLITUDES = np.array([0.45, 0.45, 0.45, 0.5, 0.5, 0.5, 0.5])
FREQUENCIES = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 3.14])
# 全 0：保证 t=0 时期望位置为 0；各关节仍由不同 frequency 错峰激励
PHASES = np.zeros(7)

# 保存时丢弃前 N 个采样点（启动阶段不稳定）
DROP_FIRST_N_SAMPLES = 100

# XML 使用 position 执行器时：ctrl 为目标关节位置 (rad)，kp/kv 在模型内设置


def generate_sine_trajectory(t: float, q0: np.ndarray, amplitudes: np.ndarray,
                             frequencies: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """生成正弦激励目标关节位置 (rad)。"""
    return q0 + amplitudes * np.sin(frequencies * t + phases)


def run_simulation(
    model_path: str,
    output_csv: str,
    duration: float = 60.0,
    sim_dt: float = 0.001,
    sample_interval: int = 10,
    show_viewer: bool = True,
) -> None:
    """
    加载 MuJoCo 模型，运行位置控制仿真，并写入 CSV。

    - sim_dt: 仿真步长 (s)，默认 1ms
    - sample_interval: 每 N 个仿真步记录一次，默认 10 -> 100Hz
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 重力方向与 step2_dynamics_parameter_estimation_friction_sdp copy.py 一致：R_base 转到基座系
    R_base = Rotation.from_euler("yz", [np.pi / 2, -np.pi / 2]).as_matrix()
    gravity_world = np.array([0.0, 0.0, -9.81])
    model.opt.gravity[:] = R_base.T @ gravity_world

    nq = model.nq
    nv = model.nv
    nu = model.nu
    if nq < 7 or nv < 7 or nu < 7:
        raise RuntimeError(f"期望 7 关节，当前 nq={nq}, nv={nv}, nu={nu}")

    # 初始位置
    data.qpos[:7] = Q_INIT
    data.qvel[:7] = 0.0

    collected = []
    sim_time = 0.0
    step_count = 0
    sample_period = sim_dt * sample_interval
    last_print_time = -0.5  # 便于 t=0 时首次打印

    viewer = None
    if show_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"查看器启动失败: {e}，继续无界面运行")

    print("========================================")
    print("MuJoCo 动力学数据采集（位置控制）")
    print("========================================")
    print(f"模型: {model_path}")
    print(f"采集时长: {duration} s, 仿真步长: {sim_dt} s")
    print(f"采样间隔: 每 {sample_interval} 步 -> {1.0/sample_period:.0f} Hz")
    print(f"输出: {output_csv}")
    print("========================================")

    while sim_time < duration:
        # 目标位置（正弦激励）；position 执行器下 ctrl 即为目标关节位置 (rad)
        q_target = generate_sine_trajectory(sim_time, Q_INIT, AMPLITUDES, FREQUENCIES, PHASES)
        data.ctrl[:] = q_target

        if step_count % sample_interval == 0:
            # 采样时刻：在 mj_step 前记录，保证 (q, dq, ddq, tau) 同一时刻
            # 先 mj_forward 使 qacc、qfrc_actuator 与当前 (qpos, qvel) 一致
            mujoco.mj_forward(model, data)
            q = data.qpos[:7].copy()
            dq = data.qvel[:7].copy()
            ddq = data.qacc[:7].copy()
            tau_meas = data.qfrc_actuator[:7].copy()
            collected.append({
                "time": sim_time,
                "q": q.copy(),
                "dq": dq.copy(),
                "ddq": ddq.copy(),
                "tau": tau_meas.copy(),
            })

        mujoco.mj_step(model, data)

        sim_time += sim_dt
        step_count += 1

        # show_viewer 时每 0.5s 打印期望位置、实际位置与各关节力矩
        if show_viewer and (sim_time - last_print_time >= 0.5):
            mujoco.mj_forward(model, data)
            q_actual = data.qpos[:7]
            tau_actual = data.qfrc_actuator[:7]
            print(f"[t={sim_time:.1f}s] 期望(rad): {q_target.round(4)} | 实际(rad): {np.array(q_actual).round(4)}")
            print(f"        关节力矩(Nm): tau0={tau_actual[0]:.3f} tau1={tau_actual[1]:.3f} tau2={tau_actual[2]:.3f} tau3={tau_actual[3]:.3f} tau4={tau_actual[4]:.3f} tau5={tau_actual[5]:.3f} tau6={tau_actual[6]:.3f}")
            last_print_time = sim_time

        if show_viewer and viewer is not None and not viewer.is_running():
            break
        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    # 写入 CSV（与 step1 输出格式一致，供 step2 读取）；丢弃前 DROP_FIRST_N_SAMPLES 条
    to_save = collected[DROP_FIRST_N_SAMPLES:]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        f.write("time,q0,q1,q2,q3,q4,q5,q6,")
        f.write("dq0,dq1,dq2,dq3,dq4,dq5,dq6,")
        f.write("ddq0,ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,")
        f.write("tau0,tau1,tau2,tau3,tau4,tau5,tau6\n")
        for row in to_save:
            t, q, dq, ddq, tau = row["time"], row["q"], row["dq"], row["ddq"], row["tau"]
            parts = [f"{t:.6f}"]
            for x in (q, dq, ddq, tau):
                parts.extend(f"{v:.6f}" for v in x)
            f.write(",".join(parts) + "\n")

    print(f"采集完成，共 {len(collected)} 条，丢弃前 {DROP_FIRST_N_SAMPLES} 条，已保存 {len(to_save)} 条: {output_csv}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_xml = os.path.join(script_dir, "AR5-5_07R-W4C4A2", "AR5-5_07R-W4C4A2.xml")
    parser = argparse.ArgumentParser(description="MuJoCo 动力学辨识数据采集（位置控制）")
    parser.add_argument(
        "--model", "-m",
        default=default_xml,
        help="MuJoCo XML 模型路径（默认: 改好的 AR5-5_07R-W4C4A2.xml，含 option/contact/position）",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出 CSV 路径（默认: 脚本同目录 dynamics_identification_data.csv）",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=60.0,
        help="采集时长 (秒)，默认 60",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="仿真步长 (秒)，默认 0.001",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="每 N 步记录一次，默认 10（1ms 步长时约 100Hz）",
    )
    parser.add_argument(
        "--viewer", "-v",
        action="store_true",
        help="打开 MuJoCo 查看器",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="无界面运行（不打开查看器），便于批量评估跟踪",
    )
    args = parser.parse_args()

    output_csv = args.output
    if output_csv is None:
        output_csv = os.path.join(script_dir, "dynamics_identification_data.csv")

    show_viewer = args.viewer and not args.no_viewer
    run_simulation(
        model_path=args.model,
        output_csv=output_csv,
        duration=args.duration,
        sim_dt=args.dt,
        sample_interval=args.sample_interval,
        show_viewer=show_viewer,
    )


if __name__ == "__main__":
    main()
