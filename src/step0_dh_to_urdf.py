#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step0: 根据 DH 参数和标称 URDF 生成新 URDF 并保存.

DH 表（图中辨识/标定结果，仅 D 带小数）:
  轴1: Alpha=0,   A=0,    D=174.522,  Theta=0
  轴2: Alpha=-90, A=0,    D=0,        Theta=0
  轴3: Alpha=90,  A=0,    D=314.726,  Theta=0
  轴4: Alpha=-90, A=10,   D=0,        Theta=0
  轴5: Alpha=90,  A=-10,  D=272.656,  Theta=0
  轴6: Alpha=-90, A=0,    D=0,        Theta=-90
  轴7: Alpha=-90, A=0,    D=0,        Theta=90
  轴8: Alpha=90,  A=0,    D=96.8883,  Theta=90

标准 DH: T_i = Rz(theta)*Tz(d)*Tx(a)*Rx(alpha).
关节 i 的 URDF origin（theta=0 时父到子）: 平移 (a, -d*sin(alpha), d*cos(alpha))，旋转 Rx(alpha).
单位: A/D 为 mm，输出 URDF 为 m；角度转弧度.
"""

import re
import sys
import os

# DH 参数表: (Alpha_deg, A_mm, D_mm, Theta_deg)，轴1~8
DH_TABLE = [
    (0,    0,      174.522,  0),    # 轴1
    (-90,  0,      0,        0),    # 轴2
    (90,   0,      314.726,  0),    # 轴3
    (-90,  10,     0,        0),    # 轴4
    (90,   -10,    272.656,  0),    # 轴5
    (-90,  0,      0,        -90),  # 轴6
    (-90,  0,      0,        90),   # 轴7
    (90,   0,      96.8883,  90),   # 轴8 (TCP 固定关节)
]

# 关节名与 DH 行的对应: URDF 中 joint_1..joint_7 用 DH 行 1..7，tcp_joint 用行 8
JOINT_DH_INDEX = {
    "AR5-5_07R-W4C4A2_joint_1": 0,
    "AR5-5_07R-W4C4A2_joint_2": 1,
    "AR5-5_07R-W4C4A2_joint_3": 2,
    "AR5-5_07R-W4C4A2_joint_4": 3,
    "AR5-5_07R-W4C4A2_joint_5": 4,
    "AR5-5_07R-W4C4A2_joint_6": 5,
    "AR5-5_07R-W4C4A2_joint_7": 6,
    "AR5-5_07R-W4C4A2_tcp_joint": 7,
}


def dh_to_origin(alpha_deg: float, a_mm: float, d_mm: float, theta_deg: float, is_fixed: bool = False):
    """
    标准 DH 转 URDF origin (xyz, rpy).
    关节角 theta=0 时父系到子系的变换: T = Tz(d)*Tx(a)*Rx(alpha); 若为固定关节可含 Rz(theta).
    """
    import math
    alpha_rad = math.radians(alpha_deg)
    # 平移 (在父系): 先 Rx(alpha) 得中间系，再 Tx(a) 得 (a,0,0)，再 Tz(d) 沿中间系 z
    # 中间系 z 在父系中为 (0, -sin(alpha), cos(alpha))，故平移 = (a, 0, 0) + d*(0, -sin(alpha), cos(alpha))
    x = a_mm / 1000.0
    y = -d_mm / 1000.0 * math.sin(alpha_rad)
    z = d_mm / 1000.0 * math.cos(alpha_rad)
    # 旋转: Rx(alpha) -> rpy = (alpha, 0, 0)
    roll = alpha_rad
    pitch = 0.0
    yaw = 0.0
    if is_fixed:
        # 轴8 固定关节带 Theta=90°: 在 DH 中为 Rz(90°)，URDF 中放在 rpy 的 yaw
        yaw = math.radians(theta_deg)
    return (x, y, z), (roll, pitch, yaw)


def replace_joint_origin(content: str, joint_name: str, xyz: tuple, rpy: tuple) -> str:
    """将 URDF 中指定 joint 的 <origin ... /> 替换为给定 xyz 和 rpy."""
    # 先定位该 joint 块: 从 <joint name="..."> 到 </joint>
    start_m = re.search(
        r'<joint\s+name="' + re.escape(joint_name) + r'"\s[^>]*>',
        content
    )
    if not start_m:
        return content
    start = start_m.end()
    end_m = re.search(r'\s*</joint>', content[start:])
    if not end_m:
        return content
    end = start + end_m.start()
    block = content[start:end]
    # 替换 block 内任意顺序的 origin 行
    new_origin = '<origin rpy="{:.6f} {:.6f} {:.6f}" xyz="{:.6f} {:.6f} {:.6f}"/>'.format(
        rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]
    )
    if re.search(r'<origin\s+', block):
        block = re.sub(
            r'<origin\s+rpy="[^"]*"\s+xyz="[^"]*"\s*/>',
            new_origin,
            block,
            count=1
        )
        block = re.sub(
            r'<origin\s+xyz="[^"]*"\s+rpy="[^"]*"\s*/>',
            new_origin,
            block,
            count=1
        )
    return content[:start] + block + content[end:]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nominal_urdf = os.path.join(script_dir, "AR5-5_07R-W4C4A2.urdf")
    output_urdf = os.path.join(script_dir, "AR5-5_07R-W4C4A2_from_dh.urdf")

    if len(sys.argv) >= 2:
        nominal_urdf = sys.argv[1]
    if len(sys.argv) >= 3:
        output_urdf = sys.argv[2]

    if not os.path.isfile(nominal_urdf):
        print("错误: 未找到标称 URDF:", nominal_urdf)
        sys.exit(1)

    with open(nominal_urdf, "r", encoding="utf-8") as f:
        content = f.read()

    for joint_name, dh_idx in JOINT_DH_INDEX.items():
        if joint_name not in content:
            print("  跳过(未找到):", joint_name)
            continue
        row = DH_TABLE[dh_idx]
        alpha_deg, a_mm, d_mm, theta_deg = row
        is_fixed = joint_name == "AR5-5_07R-W4C4A2_tcp_joint"
        xyz, rpy = dh_to_origin(alpha_deg, a_mm, d_mm, theta_deg, is_fixed=is_fixed)
        content = replace_joint_origin(content, joint_name, xyz, rpy)
        print("  {} -> xyz=({:.6f},{:.6f},{:.6f}) rpy=({:.6f},{:.6f},{:.6f})".format(
            joint_name, xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]))

    with open(output_urdf, "w", encoding="utf-8") as f:
        f.write(content)
    print("已保存:", output_urdf)


if __name__ == "__main__":
    main()
