#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step0: 根据 DH 参数和标称 URDF 生成新 URDF 并保存（Modified DH 约定，DH 参数表不变）.

每一行 j 对应的下标约定（并非同一下标）:
  列:  a_{j-1}   α_{j-1}   θ_j    d_j
  行1:    0       0°       q1    0.174522
  行2:    0     -90°       q2    0
  行3:    0      90°       q3    0.314726
  行4:  0.01    -90°       q4    0
  行5: -0.01     90°       q5    0.272656
  行6:    0     -90°   q6-90°    0
  行7:    0     -90°   q7+90°    0
  行8:    0      90°   q8+90°    0.0968883

Modified DH: T_j = Rx(α_{j-1})*Tx(a_{j-1})*Rz(θ_j)*Tz(d_j).
URDF 约定: <origin> 为关节变量=0 时子系相对父系的位姿；<axis> 为关节轴在子系下的方向。
DH 中关节绕自身 Z 旋转，故 <axis> 固定为 xyz="0 0 1"。α 与 θ 偏置均写入 <origin> 的 rpy。
单位: a/d 为 mm，输出 URDF 为 m；角度转弧度.
"""

import re
import sys
import os

# DH 参数表: 每行 j 为 (α_{j-1}_deg, a_{j-1}_mm, d_j_mm, θ_j_offset_deg)
DH_TABLE = [
    (0,    0,      174.522,  0),    # 行1: α_0, a_0, d_1, θ_1
    (-90,  0,      0,        0),    # 行2: α_1, a_1, d_2, θ_2
    (90,   0,      314.726,  0),    # 行3: α_2, a_2, d_3, θ_3
    (-90,  10,     0,        0),    # 行4: α_3, a_3, d_4, θ_4
    (90,   -10,    272.656,  0),    # 行5: α_4, a_4, d_5, θ_5
    (-90,  0,      0,        -90),  # 行6: α_5, a_5, d_6, θ_6
    (-90,  0,      0,        90),   # 行7: α_6, a_6, d_7, θ_7
    (90,   0,      96.8883,  90),   # 行8: α_7, a_7, d_8, θ_8 (TCP)
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


def mdh_to_origin(alpha_deg: float, a_mm: float, d_mm: float, theta_deg: float, is_fixed: bool = False):
    """
    Modified DH 转 URDF origin: 平移 (a, -d*sin(α), d*cos(α))（米）；
    旋转全部写入 rpy：旋转关节 rpy=(α,0,0)，固定关节 rpy 由 R_x(α)*R_z(θ) 转为 RPY。
    """
    import math
    alpha_rad = math.radians(alpha_deg)
    theta_rad = math.radians(theta_deg)
    x = a_mm / 1000.0
    y = -d_mm / 1000.0 * math.sin(alpha_rad)
    z = d_mm / 1000.0 * math.cos(alpha_rad)
    if is_fixed:
        # R = R_x(α)*R_z(θ)，提取 RPY（URDF: R = R_z(yaw)*R_y(pitch)*R_x(roll)）
        ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)
        ct, st = math.cos(theta_rad), math.sin(theta_rad)
        roll = math.atan2(sa * ct, ca)
        pitch = math.asin(-sa * st)
        yaw = math.atan2(ca * st, ct)
        rpy = (roll, pitch, yaw)
    else:
        # 旋转关节零位时父到子旋转为 R_x(α)，θ 由关节变量表达
        rpy = (alpha_rad, 0.0, 0.0)
    return (x, y, z), rpy


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


def replace_joint_axis(content: str, joint_name: str, axis_xyz: tuple) -> str:
    """将 URDF 中指定 joint 的 <axis xyz="..." /> 替换为给定 axis_xyz."""
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
    new_axis = '<axis xyz="{:.6f} {:.6f} {:.6f}"/>'.format(
        axis_xyz[0], axis_xyz[1], axis_xyz[2]
    )
    if re.search(r'<axis\s+', block):
        block = re.sub(r'<axis\s+xyz="[^"]*"\s*/>', new_axis, block, count=1)
    return content[:start] + block + content[end:]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nominal_urdf = os.path.join(script_dir, "AR5-5_07R-W4C4A2.urdf")
    output_urdf = os.path.join(script_dir, "AR5-5_07R-W4C4A2_from_mdh.urdf")

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
        xyz, rpy = mdh_to_origin(alpha_deg, a_mm, d_mm, theta_deg, is_fixed=is_fixed)
        content = replace_joint_origin(content, joint_name, xyz, rpy)
        if not is_fixed:
            content = replace_joint_axis(content, joint_name, (0.0, 0.0, 1.0))
            print("  {} -> xyz=({:.6f},{:.6f},{:.6f}) rpy=({:.6f},{:.6f},{:.6f}) axis=(0,0,1)".format(
                joint_name, xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]))
        else:
            print("  {} -> xyz=({:.6f},{:.6f},{:.6f}) rpy=({:.6f},{:.6f},{:.6f}) (TCP)".format(
                joint_name, xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]))

    with open(output_urdf, "w", encoding="utf-8") as f:
        f.write(content)
    print("已保存:", output_urdf)


if __name__ == "__main__":
    main()
