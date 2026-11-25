"""Hole-contact validation harness.

Holds the peg upright with a compliant spring (a stand-in for the gripper), lowers it at
10 mm/s over the hole at several lateral offsets, and reports depth, force, penetration, tilt.

Changes vs the first version (the old harness caused the blow-ups):
  - damping is applied through the MODEL (implicit, cannot diverge) instead of explicit forces
  - descent command is clamped at ~27 mm below the start, so the spring cannot wind up
  - holder force is clipped to a gripper-like maximum
  - the run stops cleanly if the sim goes non-finite and names the strongest contact pair
"""
import numpy as np
import mujoco
from sim.scene import build_model, load_params

P = load_params()

KP = 500.0              # N/m   holder translational stiffness
KD = 4.4                # N s/m translational damping (implicit, via dof_damping)
KR = 0.3                # N m/rad holder rotational stiffness
KDR = 1.6e-3            # N m s/rad rotational damping (implicit, via dof_damping)
F_MAX = 15.0            # N, cap on holder force
V = 0.010               # m/s descent speed (demonstrator spec)
R_PEG, H = 0.010, 0.025 # peg radius, half-length
T_END = 4.2             # s
MAX_DESCENT = 0.027     # m, stop commanding below this (tip ~1 mm below the hole floor)

f6 = np.zeros(6)


def normal_force(m, d, j):
    mujoco.mj_contactForce(m, d, j, f6)
    return abs(f6[0])


for clr in (2.0, 1.5, 1.0, 0.5):
    m = build_model(P, clearance_mm=clr)
    d = mujoco.MjData(m)
    bid = m.body("peg").id
    qadr = m.joint("peg_free").qposadr[0]
    dof = m.joint("peg_free").dofadr[0]

    # implicit damping on the peg's free joint
    m.dof_damping[dof:dof + 3] = KD
    m.dof_damping[dof + 3:dof + 6] = KDR

    for k in (0.5, 0.9, 1.5, 2.5):                       # lateral offset / clearance
        mujoco.mj_resetData(m, d)
        x0 = 0.22 + k * clr * 1e-3
        z0 = 0.030 + 0.001 + H                           # tip 1 mm above the opening
        d.qpos[qadr:qadr + 7] = [x0, 0, z0, 1, 0, 0, 0]
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)

        peak_F, steady_pen, max_tilt = 0.0, 0.0, 0.0
        logged, unstable = False, False
        n = int(T_END / m.opt.timestep)

        for i in range(n):
            t = i * m.opt.timestep
            tz = z0 - min(V * t, MAX_DESCENT)

            q = d.xquat[bid]
            qerr = 2.0 * q[1:4] * np.sign(q[0])          # small-angle orientation error
            f = KP * (np.array([x0, 0.0, tz]) - d.xpos[bid])
            d.xfrc_applied[bid, :3] = np.clip(f, -F_MAX, F_MAX)
            d.xfrc_applied[bid, 3:] = -KR * qerr

            mujoco.mj_step(m, d)

            if not np.isfinite(d.qacc).all() or not np.isfinite(d.qpos).all():
                print(f"  UNSTABLE at t={t:.3f}s")
                unstable = True
                break

            Fn = sum(normal_force(m, d, j) for j in range(d.ncon))
            peak_F = max(peak_F, Fn)

            if Fn > 100 and not logged and d.ncon:
                j = max(range(d.ncon), key=lambda c: normal_force(m, d, c))
                g1, g2 = d.contact[j].geom1, d.contact[j].geom2
                print(f"  strongest pair at t={t:.3f}s: {m.geom(g1).name or g1} <-> "
                      f"{m.geom(g2).name or g2}  (F={normal_force(m, d, j):.0f} N)")
                logged = True

            Rm = d.xmat[bid].reshape(3, 3)
            max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(Rm[2, 2], -1, 1))))
            if i > n - 300 and d.ncon:
                steady_pen = max(steady_pen, -min(c.dist for c in d.contact[:d.ncon]))

        if unstable:
            print(f"clr={clr} off={k}*clr  UNSTABLE")
            continue

        Rm = d.xmat[bid].reshape(3, 3)
        tip_z = (d.xpos[bid] - Rm[:, 2] * H)[2]
        print(f"clr={clr} off={k}*clr  tip depth={(0.030 - tip_z) * 1e3:5.1f} mm  "
              f"peakF={peak_F:6.1f} N  steady pen={steady_pen * 1e3:.3f} mm  "
              f"max tilt={max_tilt:4.1f} deg")