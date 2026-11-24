"""Builds the MuJoCo model: untouched SO-101 arm + table + round blind hole + peg + cameras."""
from __future__ import annotations

import math
import mujoco
import numpy as np
import yaml

G = mujoco.mjtGeom


def load_params(path: str = "sim_params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def xyaxes_to_quat(xyaxes) -> np.ndarray:
    """MuJoCo xyaxes -> quaternion. Camera looks along its own -z; z = x cross y."""
    x = np.asarray(xyaxes[:3], float)
    y = np.asarray(xyaxes[3:], float)
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)  # columns = camera axes in parent frame
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _wedge_points(r_in: float, half: float, a0: float, a1: float) -> list[tuple[float, float]]:
    """2D corner points of the convex wedge between the bore and the square block."""
    rv = r_in / math.cos((a1 - a0) / 2)  # vertex radius so the chord is tangent at r_in
    pts = [(rv * math.cos(a), rv * math.sin(a)) for a in (a0, a1)]
    for a in (a0, a1):  # where the wedge edges hit the square boundary
        d = half / max(abs(math.cos(a)), abs(math.sin(a)))
        pts.append((d * math.cos(a), d * math.sin(a)))
    for sx in (1, -1):  # square corners inside this wedge
        for sy in (1, -1):
            ang = math.atan2(sy * half, sx * half) % (2 * math.pi)
            if a0 < ang < a1:
                pts.append((sx * half, sy * half))
    return pts


def add_hole_block(spec, P, clearance_mm, offset_mm=(0.0, 0.0), hole_mu=None):
    hb, c = P["scene"]["hole_block"], P["contact"]
    half = hb["size_m"][0] / 2
    height = hb["size_m"][2]
    r_in = 0.010 + clearance_mm * 1e-3
    z_floor = height - hb["hole_depth_m"]
    cx = hb["center_xy_m"][0] + offset_mm[0] * 1e-3
    cy = hb["center_xy_m"][1] + offset_mm[1] * 1e-3
    mu = hole_mu if hole_mu is not None else c["peg_hole_friction"]

    body = spec.worldbody.add_body(name="hole_block", pos=[cx, cy, 0.0])
    common = dict(
        friction=[mu, c["torsional"], c["rolling"]],
        solref=c["solref"],
        solimp=c["solimp"],
        condim=c["hole_condim"],
        priority=1,
    )

    body.add_geom(
        name="hole_floor",
        type=G.mjGEOM_BOX,
        size=[half, half, z_floor / 2],
        pos=[0, 0, z_floor / 2],
        rgba=[0.25, 0.45, 0.75, 1],
        **common,
    )

    n = hb["ring_segments"]
    da = 2 * math.pi / n
    for k in range(n):
        pts = _wedge_points(r_in, half, k * da, (k + 1) * da)
        verts = [v for (x, y) in pts for z in (z_floor, height) for v in (x, y, z)]
        mesh = spec.add_mesh(name=f"hole_wedge_{k}", uservert=verts)  # hull built at compile
        body.add_geom(
            name=f"hole_wall_{k}",
            type=G.mjGEOM_MESH,
            meshname=mesh.name,
            rgba=[0.25, 0.45, 0.75, 1],
            **common,
        )


def add_peg(spec, P):
    p, c = P["scene"]["peg"], P["contact"]
    st = P["scene"]["start_state"]
    # Placeholder pose above the hole; final start pose relative to gripperframe is set in reset().
    pos = [
        P["scene"]["hole_block"]["center_xy_m"][0] + st["peg_tip_x_offset_from_hole_m"],
        0.0,
        0.030 + st["peg_tip_above_opening_m"] + p["length_m"] / 2,
    ]
    body = spec.worldbody.add_body(name="peg", pos=pos)
    body.add_freejoint(name="peg_free")
    body.add_geom(
        name="peg_geom",
        type=G.mjGEOM_CYLINDER,
        size=[p["radius_m"], p["length_m"] / 2, 0],
        density=p["density_kg_m3"],
        friction=[c["peg_friction"], c["torsional"], c["rolling"]],
        solref=c["solref"],
        solimp=c["solimp"],
        condim=3,
        priority=0,
        rgba=[0.85, 0.55, 0.20, 1],
    )


def build_model(
    P: dict,
    clearance_mm: float | None = None,
    offset_mm=(0.0, 0.0),
    hole_mu: float | None = None,
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(P["robot"]["model_path"])  # arm joints/actuators untouched
    S, C = P["sim"], P["contact"]

    # Solver / timestep: dt derived so control steps hold an integer number of substeps
    spec.option.timestep = 1.0 / (S["control_hz"] * S["substeps"])
    spec.option.iterations = S["iterations"]
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.cone = (
        mujoco.mjtCone.mjCONE_ELLIPTIC
        if S["cone"] == "elliptic"
        else mujoco.mjtCone.mjCONE_PYRAMIDAL
    )
    spec.option.impratio = S["impratio"]
    spec.visual.global_.offwidth = S["render_width"]
    spec.visual.global_.offheight = S["render_height"]

    # Gripper pads: higher priority so peg-pad contact uses exact pad friction
    for gname in P["robot"]["gripper_pad_geoms"]:
        g = spec.geom(gname)
        g.priority = 1
        g.friction = [C["peg_gripper_friction"], C["torsional"], C["rolling"]]
        g.condim = C["pad_condim"]
        g.solref, g.solimp = C["solref"], C["solimp"]

    # Table (top surface at z = 0)
    T = P["scene"]["table"]["half_size_m"]
    spec.worldbody.add_geom(
        name="table_top",
        type=G.mjGEOM_BOX,
        size=T,
        pos=[0.2, 0, -T[2]],
        friction=[C["table_friction"], C["torsional"], C["rolling"]],
        solref=C["solref"],
        solimp=C["solimp"],
        rgba=[0.75, 0.75, 0.75, 1],
    )

    spec.worldbody.add_light(
        name="main_light",
        pos=[0.2, 0, 0.8],
        dir=[0, 0, -1],
        directional=1,
        diffuse=[0.9, 0.9, 0.9],
        specular=[0.2, 0.2, 0.2],
    )

    add_hole_block(
        spec,
        P,
        clearance_mm if clearance_mm is not None else P["scene"]["default_radial_clearance_mm"],
        offset_mm,
        hole_mu,
    )
    add_peg(spec, P)

    # Cameras (spec values, converted from xyaxes to quaternions)
    W, O = P["cameras"]["wrist"], P["cameras"]["overhead"]
    spec.body(P["robot"]["wrist_camera_body"]).add_camera(
        name="wrist",
        pos=[0, 0, 0],
        quat=xyaxes_to_quat(W["xyaxes"]),
        fovy=W["fovy"],
    )
    spec.worldbody.add_camera(
        name="overhead",
        pos=O["pos"],
        quat=xyaxes_to_quat(O["xyaxes"]),
        fovy=O["fovy"],
    )
    return spec.compile()
