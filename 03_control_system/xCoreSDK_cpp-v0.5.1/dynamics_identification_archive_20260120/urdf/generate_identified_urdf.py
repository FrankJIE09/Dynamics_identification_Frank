#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a new URDF by replacing link inertial parameters using identified results.

Input:
  - dynamics_physical_parameters_identified.txt (produced by dynamics_parameter_estimation)
  - original URDF

Output:
  - identified URDF with updated <inertial> blocks for link1..link7

Notes (IMPORTANT):
  - The identified file prints inertia matrix from Pinocchio Inertia::inertia(), which is the
    rotational inertia about the link origin (not about CoM).
  - URDF <inertial> typically sets <origin xyz="com"> and inertia about that origin (CoM),
    so we convert: I_C = I_origin - m * S(c)^T * S(c)
  - The identified inertias may be non-physical (not positive definite). We symmetrize and
    project I_C to be positive definite by clamping eigenvalues.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _skew(v):
    x, y, z = v
    return [
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ]


def _mat_t(A):
    return [list(row) for row in zip(*A)]


def _mat_mul(A, B):
    out = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            aik = A[i][k]
            for j in range(len(B[0])):
                out[i][j] += aik * B[k][j]
    return out


def _mat_add(A, B, a=1.0, b=1.0):
    return [[a * A[i][j] + b * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_sym(A):
    AT = _mat_t(A)
    return _mat_add(A, AT, a=0.5, b=0.5)


def _mat_eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_scale(A, s):
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_to_tuple(A):
    return tuple(tuple(float(x) for x in row) for row in A)


def _eig_sym_3x3(A):
    """
    Eigen-decomposition for symmetric 3x3 using numpy if available; else fallback to analytic-ish.
    Returns (eigenvalues, eigenvectors_as_columns).
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is None:
        # Minimal fallback: if no numpy, do diagonal shift until PD using Gershgorin.
        # We won't output eigenvectors; caller can just use diagonal shift method.
        return None

    M = np.array(A, dtype=float)
    w, V = np.linalg.eigh(M)
    return w.tolist(), V.tolist()  # V is columns


def _project_pd_3x3(A, eps=1e-9):
    """
    Ensure symmetric positive definite.
    - Symmetrize.
    - If numpy available: clamp eigenvalues to eps.
    - Else: diagonal shift.
    """
    A = _mat_sym(A)

    eig = _eig_sym_3x3(A)
    if eig is None:
        # Gershgorin diagonal shift: make diagonals large enough.
        # Compute lower bound for min eigenvalue.
        min_lb = float("inf")
        for i in range(3):
            r = sum(abs(A[i][j]) for j in range(3) if j != i)
            lb = A[i][i] - r
            min_lb = min(min_lb, lb)
        if min_lb >= eps:
            return A, 0.0
        shift = (eps - min_lb) + eps
        I = _mat_eye(3)
        return _mat_add(A, _mat_scale(I, shift)), shift

    w, V = eig
    import numpy as np  # type: ignore

    w2 = [max(float(x), eps) for x in w]
    M = np.array(A, dtype=float)
    w_np = np.array(w2, dtype=float)
    V_np = np.array(V, dtype=float)
    # V returned as list-of-lists; ensure it is 3x3, columns are eigenvectors
    # numpy.linalg.eigh returns V with columns = eigenvectors already.
    # But our V is that V; reconstruction: V * diag(w) * V^T
    A_pd = (V_np @ np.diag(w_np) @ V_np.T)
    shift = max(0.0, eps - min(w))
    return A_pd.tolist(), shift


def parse_identified_txt(path: Path):
    """
    Parse dynamics_physical_parameters_identified.txt
    Returns dict: idx -> {m, com(x,y,z), I_origin(3x3)}
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # Joint block header:
    # 关节 1 (AR5-5_07R-W4C4A2_joint_1):
    joint_re = re.compile(r"^关节\s+(\d+)\s+\(([^)]+)\):\s*$", re.M)
    # Identified mass:
    mass_re = re.compile(r"辨识质量\(kg\):\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
    # Identified CoM:
    com_re = re.compile(r"辨识质心\(m\):\s*\[([^\]]+)\]")
    # Identified inertia block starts after "辨识 惯性"
    inertia_block_re = re.compile(
        r"辨识 惯性\(kg·m²\):\s*\n"
        r"\s*\[([^\]]+)\]\s*\n"
        r"\s*\[([^\]]+)\]\s*\n"
        r"\s*\[([^\]]+)\]\s*\n",
        re.M
    )

    out = {}
    pos = 0
    matches = list(joint_re.finditer(txt))
    for i, m in enumerate(matches):
        jid = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        block = txt[start:end]

        mass_m = mass_re.search(block)
        com_m = com_re.search(block)
        I_m = inertia_block_re.search(block)
        if not (mass_m and com_m and I_m):
            raise RuntimeError(f"Failed parsing joint {jid} block in {path}")

        mass = float(mass_m.group(1))
        com_vals = [float(x.strip()) for x in com_m.group(1).split(",")]
        if len(com_vals) != 3:
            raise RuntimeError(f"Bad CoM for joint {jid}: {com_m.group(1)}")

        rows = []
        for r in (I_m.group(1), I_m.group(2), I_m.group(3)):
            vals = [float(x.strip()) for x in r.split(",")]
            if len(vals) != 3:
                raise RuntimeError(f"Bad inertia row for joint {jid}: {r}")
            rows.append(vals)

        out[jid] = {"mass": mass, "com": com_vals, "I_origin": rows}

    return out


def inertia_origin_to_com(mass, com, I_origin):
    S = _skew(com)
    STS = _mat_mul(_mat_t(S), S)
    I_c = _mat_add(I_origin, _mat_scale(STS, mass), a=1.0, b=-1.0)  # I_origin - m*STS
    return I_c


def format_float(x):
    # URDF usually uses decimal; keep enough precision
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.9g}"


def update_link_inertial(link_elem: ET.Element, mass, com_xyz, I_com_3x3):
    inertial = link_elem.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link_elem, "inertial")

    mass_elem = inertial.find("mass")
    if mass_elem is None:
        mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.set("value", format_float(mass))

    origin_elem = inertial.find("origin")
    if origin_elem is None:
        origin_elem = ET.SubElement(inertial, "origin")
    origin_elem.set("xyz", f"{format_float(com_xyz[0])} {format_float(com_xyz[1])} {format_float(com_xyz[2])}")
    # keep rpy = 0 0 0 (we don't estimate inertia principal axes here)
    if "rpy" not in origin_elem.attrib:
        origin_elem.set("rpy", "0 0 0")
    else:
        origin_elem.set("rpy", "0 0 0")

    inertia_elem = inertial.find("inertia")
    if inertia_elem is None:
        inertia_elem = ET.SubElement(inertial, "inertia")

    I = _mat_sym(I_com_3x3)
    ixx = I[0][0]
    ixy = I[0][1]
    ixz = I[0][2]
    iyy = I[1][1]
    iyz = I[1][2]
    izz = I[2][2]

    inertia_elem.set("ixx", format_float(ixx))
    inertia_elem.set("ixy", format_float(ixy))
    inertia_elem.set("ixz", format_float(ixz))
    inertia_elem.set("iyy", format_float(iyy))
    inertia_elem.set("iyz", format_float(iyz))
    inertia_elem.set("izz", format_float(izz))


def read_link_inertial(link_elem: ET.Element):
    """Read URDF inertial (assumes inertia is expressed about the inertial origin)."""
    inertial = link_elem.find("inertial")
    if inertial is None:
        return None

    mass_elem = inertial.find("mass")
    inertia_elem = inertial.find("inertia")
    origin_elem = inertial.find("origin")
    if mass_elem is None or inertia_elem is None or origin_elem is None:
        return None

    m = float(mass_elem.get("value"))
    xyz = [float(x) for x in origin_elem.get("xyz").split()]
    # URDF inertia is symmetric
    ixx = float(inertia_elem.get("ixx"))
    ixy = float(inertia_elem.get("ixy"))
    ixz = float(inertia_elem.get("ixz"))
    iyy = float(inertia_elem.get("iyy"))
    iyz = float(inertia_elem.get("iyz"))
    izz = float(inertia_elem.get("izz"))
    I = [
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ]
    return {"mass": m, "com": xyz, "I": I}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identified_txt", required=True, help="dynamics_physical_parameters_identified.txt")
    ap.add_argument("--input_urdf", required=True, help="original URDF path")
    ap.add_argument("--output_urdf", required=True, help="output URDF path")
    ap.add_argument("--eps", type=float, default=1e-9, help="min eigenvalue for PD projection")
    ap.add_argument("--blend", type=float, default=0.02, help="blend ratio with URDF inertia when identified inertia is ill-conditioned (0..1)")
    ap.add_argument("--shift_thresh", type=float, default=0.2, help="if PD-projection shift exceeds (shift_thresh * trace(I_urdf)), fallback to URDF inertia (optionally with small blend)")
    args = ap.parse_args()

    identified_txt = Path(args.identified_txt).expanduser().resolve()
    input_urdf = Path(args.input_urdf).expanduser().resolve()
    output_urdf = Path(args.output_urdf).expanduser().resolve()

    if not identified_txt.exists():
        print(f"ERROR: identified_txt not found: {identified_txt}", file=sys.stderr)
        return 2
    if not input_urdf.exists():
        print(f"ERROR: input_urdf not found: {input_urdf}", file=sys.stderr)
        return 2

    id_map = parse_identified_txt(identified_txt)

    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # Update link1..link7 corresponding to joint 1..7
    warnings = []
    for jid in range(1, 8):
        if jid not in id_map:
            raise RuntimeError(f"Missing joint {jid} in identified file")
        link_name = f"AR5-5_07R-W4C4A2_link{jid}"
        link_elem = root.find(f".//link[@name='{link_name}']")
        if link_elem is None:
            raise RuntimeError(f"Cannot find link element {link_name} in URDF")

        urdf_inert = read_link_inertial(link_elem)
        if urdf_inert is None:
            raise RuntimeError(f"Cannot read <inertial> for {link_name} from URDF")

        mass = float(id_map[jid]["mass"])
        com = [float(x) for x in id_map[jid]["com"]]
        I_origin = [[float(x) for x in row] for row in id_map[jid]["I_origin"]]

        # Identified I_C from identified (I_origin -> I_C)
        I_com_id = inertia_origin_to_com(mass, com, I_origin)
        I_com_id = _mat_sym(I_com_id)

        # URDF inertia: treat it as I about its inertial origin.
        # Most URDFs place inertial origin at CoM, so we use it as I_C prior.
        I_com_urdf = _mat_sym(urdf_inert["I"])

        # First try PD projection on identified I_C
        I_com_pd, shift = _project_pd_3x3(I_com_id, eps=args.eps)

        # If projection is too aggressive, fallback to URDF inertia (optionally with a tiny blend),
        # because a large PD shift means the identified inertia is strongly non-physical.
        trace_urdf = I_com_urdf[0][0] + I_com_urdf[1][1] + I_com_urdf[2][2]
        if trace_urdf <= 1e-12:
            trace_urdf = 1.0
        if shift > args.shift_thresh * trace_urdf:
            w = max(0.0, min(1.0, args.blend))
            if w > 0.0:
                I_blend = _mat_add(I_com_urdf, I_com_id, a=(1.0 - w), b=w)
                I_com_pd, shift2 = _project_pd_3x3(I_blend, eps=args.eps)
                # If it still collapses (trace too small), fully fallback to URDF inertia.
                tr = I_com_pd[0][0] + I_com_pd[1][1] + I_com_pd[2][2]
                if tr < 0.1 * trace_urdf:
                    I_com_pd, _ = _project_pd_3x3(I_com_urdf, eps=args.eps)
                warnings.append((jid, max(shift, shift2)))
            else:
                I_com_pd, _ = _project_pd_3x3(I_com_urdf, eps=args.eps)
                warnings.append((jid, shift))
        elif shift > 0.0:
            warnings.append((jid, shift))

        update_link_inertial(link_elem, mass, com, I_com_pd)

    # Add a comment at top (ET doesn't preserve comments well; we add as first child)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comment = ET.Comment(
        f" Generated by generate_identified_urdf.py at {stamp}. "
        f"Source identified: {identified_txt.name}. "
        f"Note: inertia written about CoM; PD-projection applied (eps={args.eps}). "
    )
    root.insert(0, comment)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)

    print(f"OK: wrote {output_urdf}")
    if warnings:
        print("WARNING: PD-projection shifts applied to I_C (added to diagonals via eigenvalue clamp):")
        for jid, shift in warnings:
            print(f"  joint {jid}: shift~{shift:g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


