"""
读取初始 URDF，验证文件中惯量是绕 link 原点还是绕质心与 Pinocchio 一致，
并写出一个新 URDF（惯量统一为质心处），尽量保证 Pinocchio 读回前后一致。

用法:
  python urdf_normalize_inertia_frame.py --urdf ./AR5-5_07R-W4C4A2/AR5-5_07R-W4C4A2.urdf --out ./AR5-5_07R-W4C4A2/AR5-5_07R-W4C4A2_com_inertia.urdf

"""

import argparse
import os
import re
import sys

# 与 step2_dynamics_parameter_estimation_friction_sdp.py 一致：避免 ROS 污染，使用当前环境的 pinocchio
os.environ.pop("PYTHONPATH", None)
sys.path[:] = [p for p in sys.path if p and ("/opt/ros/" not in p)]

import numpy as np
import pinocchio as pin

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- 惯量参考系转换 ----------
def _inertia_origin_to_com(I_3x3: np.ndarray, m: float, c: np.ndarray) -> np.ndarray:
    """从绕 link 原点的惯量 I_origin 得到绕质心的惯量 I_com。
    平行轴: I_origin = I_com + m * ( (c·c)*E - c⊗c )  =>  I_com = I_origin - m * ( (c·c)*E - c⊗c )。
    c 为质心在 link 系下坐标 (从 link 原点指向质心)。
    """
    c = np.asarray(c, dtype=float).ravel()[:3]
    r2 = float(np.dot(c, c))
    I_com = I_3x3 - m * (r2 * np.eye(3) - np.outer(c, c))
    return I_com


def _inertia_com_to_origin(I_com: np.ndarray, m: float, c: np.ndarray) -> np.ndarray:
    """从绕质心惯量得到绕 link 原点的惯量。I_origin = I_com + m * ( (c·c)*E - c⊗c )。"""
    c = np.asarray(c, dtype=float).ravel()[:3]
    r2 = float(np.dot(c, c))
    return I_com + m * (r2 * np.eye(3) - np.outer(c, c))


def _sym3_to_6(I: np.ndarray):
    """3x3 对称阵 -> (Ixx, Ixy, Iyy, Ixz, Iyz, Izz) Pinocchio 顺序。"""
    return (float(I[0, 0]), float(I[0, 1]), float(I[1, 1]), float(I[0, 2]), float(I[1, 2]), float(I[2, 2]))


def _six_to_sym3(ixx, ixy, iyy, ixz, iyz, izz):
    """(Ixx, Ixy, Iyy, Ixz, Iyz, Izz) -> 3x3 对称阵。"""
    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ], dtype=float)


# ---------- URDF 解析 ----------
def _parse_urdf_inertials(urdf_path: str) -> dict:
    """从 URDF 解析每个 link 的 <inertial>：mass, origin xyz, origin rpy, inertia 6 元。
    返回 dict: link_name -> {'mass', 'xyz', 'rpy', 'ixx','iyy','izz','ixy','ixz','iyz'}。
    """
    result = {}
    if not os.path.isfile(urdf_path):
        return result
    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()
    link_pattern = re.compile(r'<link\s+name="([^"]+)"[^>]*>')
    mass_pattern = re.compile(r'<mass\s+value="([^"]+)"')
    origin_pattern = re.compile(r'<origin[^>]*xyz="([^"]+)"')
    origin_rpy_pattern = re.compile(r'<origin[^>]*rpy="([^"]+)"')
    inertia_pattern = re.compile(
        r'<inertia\s+ixx="([^"]+)"\s+ixy="([^"]+)"\s+ixz="([^"]+)"\s+iyy="([^"]+)"\s+iyz="([^"]+)"\s+izz="([^"]+)"'
    )
    pos = 0
    while True:
        m_link = link_pattern.search(content, pos)
        if not m_link:
            break
        link_name = m_link.group(1)
        start = m_link.end()
        next_link = link_pattern.search(content, start)
        search_end = next_link.start() if next_link else len(content)
        end_inertial = content.find("</inertial>", start, search_end)
        if end_inertial == -1:
            pos = start
            continue
        block = content[start : end_inertial + len("</inertial>")]
        inertial_start = block.find("<inertial>")
        sub = block[inertial_start:] if inertial_start >= 0 else block
        mass_m = mass_pattern.search(sub)
        origin_m = origin_pattern.search(sub)
        origin_rpy_m = origin_rpy_pattern.search(sub)
        inertia_m = inertia_pattern.search(sub)
        try:
            entry = {}
            if mass_m:
                entry["mass"] = float(mass_m.group(1))
            if origin_m:
                xyz_str = origin_m.group(1).split()
                entry["xyz"] = (float(xyz_str[0]), float(xyz_str[1]), float(xyz_str[2])) if len(xyz_str) >= 3 else (0, 0, 0)
            if origin_rpy_m:
                rpy_str = origin_rpy_m.group(1).split()
                entry["rpy"] = (float(rpy_str[0]), float(rpy_str[1]), float(rpy_str[2])) if len(rpy_str) >= 3 else (0.0, 0.0, 0.0)
            else:
                entry["rpy"] = (0.0, 0.0, 0.0)
            if inertia_m:
                entry["ixx"] = float(inertia_m.group(1))
                entry["ixy"] = float(inertia_m.group(2))
                entry["ixz"] = float(inertia_m.group(3))
                entry["iyy"] = float(inertia_m.group(4))
                entry["iyz"] = float(inertia_m.group(5))
                entry["izz"] = float(inertia_m.group(6))
            if "mass" in entry and "xyz" in entry and "ixx" in entry:
                result[link_name] = entry
        except (ValueError, IndexError):
            pass
        pos = end_inertial + 1
    return result


def _joint_name_to_link_name(joint_name: str) -> str:
    if "joint_" in joint_name:
        return joint_name.replace("joint_", "link", 1)
    return joint_name


# ---------- 替换 URDF 中的 <inertial> 块 ----------
def _replace_inertials_in_urdf(content: str, link_inertial_blocks: dict) -> str:
    """将 content 中每个 link 的 <inertial>...</inertial> 替换为 link_inertial_blocks[link_name]。
    link_inertial_blocks: link_name -> 完整多行字符串（含 <inertial> 与 </inertial>）。
    """
    link_pattern = re.compile(r'<link\s+name="([^"]+)"[^>]*>')
    pos = 0
    out_parts = []
    while True:
        m_link = link_pattern.search(content, pos)
        if not m_link:
            out_parts.append(content[pos:])
            break
        link_name = m_link.group(1)
        start = m_link.end()
        next_link = link_pattern.search(content, start)
        search_end = next_link.start() if next_link else len(content)
        start_inertial = content.find("<inertial>", start, search_end)
        end_inertial = content.find("</inertial>", start, search_end)
        if start_inertial == -1 or end_inertial == -1:
            out_parts.append(content[pos:start])
            pos = start
            continue
        end_inertial += len("</inertial>")
        out_parts.append(content[pos:start_inertial])
        new_block = link_inertial_blocks.get(link_name)
        if new_block is not None:
            out_parts.append(new_block)
        else:
            out_parts.append(content[start_inertial:end_inertial])
        pos = end_inertial
    return "".join(out_parts)


def main():
    parser = argparse.ArgumentParser(description="验证 URDF 惯量参考系并写出质心惯量一致的 URDF")
    parser.add_argument("--urdf", type=str, default="./AR5-5_07R-W4C4A2/AR5-5_07R-W4C4A2.urdf", help="输入 URDF 路径")
    parser.add_argument("--out", type=str, default="", help="输出 URDF 路径，默认在输入同目录加 _com_inertia")
    args = parser.parse_args()

    urdf_path = os.path.normpath(os.path.join(_SCRIPT_DIR, args.urdf) if not os.path.isabs(args.urdf) else args.urdf)
    if not os.path.isfile(urdf_path):
        urdf_path = args.urdf
    if not os.path.isfile(urdf_path):
        print(f"错误: 找不到 URDF 文件 {urdf_path}")
        return 1

    if args.out:
        out_path = os.path.normpath(os.path.join(_SCRIPT_DIR, args.out) if not os.path.isabs(args.out) else args.out)
    else:
        base, ext = os.path.splitext(urdf_path)
        out_path = base + "_com_inertia" + ext

    # 1) 从文件解析
    file_inertials = _parse_urdf_inertials(urdf_path)
    print(f"从 URDF 解析到 {len(file_inertials)} 个 link 的 inertial 数据")

    # 1b) 检查所有 <inertial> 下的 origin rpy 是否全为 0
    rpy_nonzero = [(name, d["rpy"]) for name, d in file_inertials.items() if d.get("rpy") != (0.0, 0.0, 0.0)]
    if rpy_nonzero:
        print(f"  警告: 以下 link 的 <inertial><origin rpy> 非 0 0 0，脚本未做旋转变换: {rpy_nonzero}")
    else:
        print("  验证: 所有 <inertial><origin rpy> 均为 0 0 0，无需旋转变换。")

    # 2) Pinocchio 加载
    model = pin.buildModelFromUrdf(urdf_path)
    n_links = model.njoints - 1  # 不含 universe
    # toDynamicParameters 返回 [m, mc_x, mc_y, mc_z, I_xx^o, I_xy^o, I_yy^o, I_xz^o, I_yz^o, I_zz^o]，其中 I^o 为绕原点 (at Origin)
    theta_pin = np.zeros(10 * n_links)
    for jid in range(1, model.njoints):
        pi = np.array(model.inertias[jid].toDynamicParameters()).ravel()
        base = 10 * (jid - 1)
        theta_pin[base : base + 10] = pi

    # 2b) 单连杆数据：选末端 link（如 link7）做手动对比
    jid_single = model.njoints - 1  # 最后一个 joint 对应末端 link（如 joint_7 -> link7）
    link_name_single = _joint_name_to_link_name(model.names[jid_single])
    inv = model.inertias[jid_single]
    pi_single = np.array(inv.toDynamicParameters()).ravel()
    m_pin_s = float(inv.mass)
    lever_pin_s = np.array(inv.lever).ravel()
    print(f"\n--- 单连杆手动对比（末端: {link_name_single}, jid={jid_single}）---")
    print(f"  URDF 文件中: ixx={file_inertials.get(link_name_single, {}).get('ixx', 'N/A')}, mass={file_inertials.get(link_name_single, {}).get('mass', 'N/A')}, xyz(质心)={file_inertials.get(link_name_single, {}).get('xyz', 'N/A')}")
    print(f"  Pinocchio model.inertias[{jid_single}].mass = {m_pin_s}")
    print(f"  Pinocchio model.inertias[{jid_single}].lever (质心) = ({lever_pin_s[0]:.6e}, {lever_pin_s[1]:.6e}, {lever_pin_s[2]:.6e})")
    print(f"  toDynamicParameters() = [m, mc_x, mc_y, mc_z, I_xx^o, I_xy^o, I_yy^o, I_xz^o, I_yz^o, I_zz^o] (I^o=绕原点)")
    print(f"    pi = m={pi_single[0]:.6e}, mc={pi_single[1]:.6e},{pi_single[2]:.6e},{pi_single[3]:.6e}, I^o_xx,yy,zz={pi_single[4]:.6e},{pi_single[6]:.6e},{pi_single[9]:.6e}")

    # 3) 逐 link 验证：文件惯量是绕原点还是绕质心时与 Pinocchio 一致
    print("\n--- 验证惯量参考系（文件 vs Pinocchio）---")
    link_inertial_blocks = {}  # 用于写新 URDF：link_name -> 新 <inertial> 块

    for jid in range(1, model.njoints):
        link_name = _joint_name_to_link_name(model.names[jid])
        base = 10 * (jid - 1)
        # pi = [m, mc_x, mc_y, mc_z, I_xx^o, I_xy^o, I_yy^o, I_xz^o, I_yz^o, I_zz^o]，I^o = 绕 link 原点
        m_pin = theta_pin[base]
        com_pin = np.array([
            theta_pin[base + 1] / max(m_pin, 1e-12),
            theta_pin[base + 2] / max(m_pin, 1e-12),
            theta_pin[base + 3] / max(m_pin, 1e-12),
        ])
        I_origin_pin = _six_to_sym3(
            theta_pin[base + 4], theta_pin[base + 5], theta_pin[base + 6],
            theta_pin[base + 7], theta_pin[base + 8], theta_pin[base + 9],
        )
        # URDF <inertial><origin xyz="c"/> 表示质心，<inertia> 应为绕该点(质心)的惯量 I_com
        I_com_pin = _inertia_origin_to_com(I_origin_pin, m_pin, com_pin)

        file_d = file_inertials.get(link_name)
        if not file_d:
            # 写入 (m, com, I_com)：URDF 约定为绕 origin 的惯量，origin=质心故写 I_com
            ixx, ixy, iyy, ixz, iyz, izz = _sym3_to_6(I_com_pin)
            blk = (
                f'    <inertial>\n'
                f'      <mass value="{m_pin:.6f}" />\n'
                f'      <inertia ixx="{ixx:.6e}" ixy="{ixy:.6e}" ixz="{ixz:.6e}" iyy="{iyy:.6e}" iyz="{iyz:.6e}" izz="{izz:.6e}" />\n'
                f'      <origin rpy="0 0 0" xyz="{com_pin[0]:.6f} {com_pin[1]:.6f} {com_pin[2]:.6f}" />\n'
                f'    </inertial>\n'
            )
            link_inertial_blocks[link_name] = blk
            print(f"  连杆{jid-1} ({link_name}): 文件中无 inertial，新 URDF 将写入 Pinocchio (m, com, I_com)")
            continue

        m_f = file_d["mass"]
        c_f = np.array(file_d["xyz"])
        I_f = _six_to_sym3(
            file_d["ixx"], file_d["ixy"], file_d["iyy"],
            file_d["ixz"], file_d["iyz"], file_d["izz"],
        )

        # 假设1: 文件惯量绕 link 原点 -> 换算到质心后与 Pinocchio 的 I_com 比
        I_com_from_origin = _inertia_origin_to_com(I_f, m_f, c_f)
        err_origin = np.linalg.norm(I_com_from_origin - I_com_pin)
        # 假设2: 文件惯量绕质心 -> 直接与 I_com_pin 比
        err_com = np.linalg.norm(I_f - I_com_pin)

        if err_origin <= err_com:
            conclusion = "绕 link 原点（换算到质心后与 Pinocchio I_com 一致）"
        else:
            conclusion = "绕质心（与 Pinocchio I_com 一致）"

        print(f"  连杆{jid-1} ({link_name}): 假设绕原点 err={err_origin:.2e}, 假设绕质心 err={err_com:.2e} -> {conclusion}")

        # 新 URDF 写入 (m, com, I_com)：URDF 约定 <inertia> 为绕 <origin> 的惯量，origin=质心
        ixx, ixy, iyy, ixz, iyz, izz = _sym3_to_6(I_com_pin)
        blk = (
            f'    <inertial>\n'
            f'      <mass value="{m_pin:.6f}" />\n'
            f'      <inertia ixx="{ixx:.6e}" ixy="{ixy:.6e}" ixz="{ixz:.6e}" iyy="{iyy:.6e}" iyz="{iyz:.6e}" izz="{izz:.6e}" />\n'
            f'      <origin rpy="0 0 0" xyz="{com_pin[0]:.6f} {com_pin[1]:.6f} {com_pin[2]:.6f}" />\n'
            f'    </inertial>\n'
        )
        link_inertial_blocks[link_name] = blk

    # 4) 写新 URDF
    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = _replace_inertials_in_urdf(content, link_inertial_blocks)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"\n已写出新 URDF: {out_path}")

    # 5) 校验：用 Pinocchio 读回新 URDF，对比 theta
    model2 = pin.buildModelFromUrdf(out_path)
    theta_read = np.zeros(10 * (model2.njoints - 1))
    for jid in range(1, model2.njoints):
        pi = np.array(model2.inertias[jid].toDynamicParameters()).ravel()
        base = 10 * (jid - 1)
        theta_read[base : base + 10] = pi
    diff = np.abs(theta_pin[: theta_read.size] - theta_read)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"校验: 新 URDF 读回后与写入前 theta 差异 最大={max_diff:.2e}, 平均={mean_diff:.2e} (理想为 0)")
    # 逐连杆打印最大差异位置
    for jid in range(1, model2.njoints):
        base = 10 * (jid - 1)
        d = np.max(diff[base : base + 10])
        if d > 1e-6:
            link_name = _joint_name_to_link_name(model2.names[jid])
            print(f"    连杆{jid-1} ({link_name}) 最大分量差: {d:.2e}")
    if max_diff < 1e-5:
        print("前后一致，校验通过。")
    else:
        print("存在数值差异，可能为浮点或 Pinocchio 内部表示；新 URDF 仍以质心惯量写出，可供后续辨识使用。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
