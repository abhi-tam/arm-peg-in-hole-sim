"""Diagnose why the peg is ejected: where are the jaw pads relative to the gripperframe site?"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim.env import PegHoleEnv, rodrigues, PEG_TOP_BEHIND
from sim.scene import load_params

Path("outputs").mkdir(exist_ok=True)
P = load_params()
env = PegHoleEnv(P)
m, d = env.m, env.d
st = P["scene"]["start_state"]
gb = {m.body(n).id for n in P["robot"]["gripper_bodies"]}
print("gripper bodies:", P["robot"]["gripper_bodies"])
print("gripper joint range (rad):", env.g_lo, env.g_hi)

tip_t = np.array([env.hole_xy[0] + st["peg_tip_x_offset_from_hole_m"],
                  env.hole_xy[1], env.hole_top + st["peg_tip_above_opening_m"]])
q, err = env.solve_ik_multi(tip_t)


def setup(gripper_rad):
    mujoco.mj_resetData(m, d)
    d.qpos[env.qadr] = q
    d.qpos[env.g_qadr] = gripper_rad
    d.ctrl[env.act[:5]] = q
    d.ctrl[env.act[5]] = gripper_rad
    mujoco.mj_forward(m, d)


def park_peg():
    d.qpos[env.peg_qadr:env.peg_qadr + 7] = [1.0, 1.0, 1.0, 1, 0, 0, 0]
    d.qvel[env.peg_qadr:env.peg_qadr + 6] = 0


def dump(label):
    S = d.site_xmat[env.site].reshape(3, 3)
    sp = d.site_xpos[env.site]
    print(f"\n=== collision geoms of gripper bodies, {label} (site frame, mm; x = pointing) ===")
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] not in gb:
            continue
        if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
            continue
        gtype = mujoco.mjtGeom(m.geom_type[gid]).name
        gx, gR = d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3)
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[gid]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V = m.mesh_vert[a:a + n]
            loc = (S.T @ (gx + V @ gR.T - sp).T).T * 1e3
            lo, hi = loc.min(0), loc.max(0)
            print(f"  {m.geom(gid).name or gid} [{m.body(m.geom_bodyid[gid]).name}] {gtype} "
                  f"min={lo.round(1)} max={hi.round(1)}")
        else:
            c = S.T @ (gx - sp) * 1e3
            print(f"  {m.geom(gid).name or gid} [{m.body(m.geom_bodyid[gid]).name}] {gtype} "
                  f"size={m.geom_size[gid].round(4)} centre={c.round(1)}")


# 1) open and fully closed, no peg
setup(env.g_hi)
park_peg()
dump("OPEN")
setup(env.pct_to_rad(0))
park_peg()
for _ in range(int(0.5 / m.opt.timestep)):
    park_peg()
    mujoco.mj_step(m, d)
dump("CLOSED (no peg)")

# 2) pin the peg exactly as reset() does and close on it
setup(env.g_hi)
S = d.site_xmat[env.site].reshape(3, 3)
p = env.point_dir(d)
center = d.site_xpos[env.site] + (-PEG_TOP_BEHIND + env.peg_half) * p
quat = np.zeros(4)
mujoco.mju_quatZ2Vec(quat, -p)
pose = np.concatenate([center, quat])
print("\npeg centre in site frame (mm):", (S.T @ (center - d.site_xpos[env.site]) * 1e3).round(1))
print("peg quat:", quat.round(3), "(identity expected for a vertical peg)")

d.ctrl[env.act[5]] = env.pct_to_rad(0)
nsteps = int(0.7 / m.opt.timestep)
print("\n=== closing with peg pinned ===")
for i in range(nsteps):
    d.qpos[env.peg_qadr:env.peg_qadr + 7] = pose
    d.qvel[env.peg_qadr:env.peg_qadr + 6] = 0
    mujoco.mj_step(m, d)
    if i % (nsteps // 7) == 0 or i == nsteps - 1:
        f6 = np.zeros(6)
        rows = []
        for j in range(d.ncon):
            c = d.contact[j]
            if env.peg_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == env.peg_gid else c.geom1
                mujoco.mj_contactForce(m, d, j, f6)
                rows.append((m.geom(other).name or str(other), round(c.dist * 1e3, 3), round(abs(f6[0]), 1)))
        print(f"t={i * m.opt.timestep:.2f}s gripper={np.degrees(d.qpos[env.g_qadr]):6.1f} deg "
              f"peg contacts (geom, dist mm, N): {rows}")
img = env.render("wrist")
Image.fromarray(img).save("outputs/diag_pinned_wrist.png")

# 3) release
print("\n=== released: peg centre in site frame (mm) ===")
for i in range(int(0.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)
    if i % int(0.05 / m.opt.timestep) == 0:
        c = S.T @ (d.xpos[env.peg_bid] - d.site_xpos[env.site]) * 1e3
        print(f"t={i * m.opt.timestep:.2f}s  centre={c.round(1)}  gripper={np.degrees(d.qpos[env.g_qadr]):.1f} deg  ncon={d.ncon}")
Image.fromarray(env.render("wrist")).save("outputs/diag_released_wrist.png")
print("\nwrote outputs/diag_pinned_wrist.png and outputs/diag_released_wrist.png")