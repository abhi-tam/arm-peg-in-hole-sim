import numpy as np, mujoco
from sim.scene import build_model, load_params

P = load_params()
KP, KD = 500.0, 4.4          # N/m, N s/m   (peg ~0.0195 kg)
KR, KDR = 0.3, 1.6e-3        # N m/rad, N m s/rad  (peg transverse inertia ~4.5e-6)
V = 0.010                    # 10 mm/s descent
R_PEG, H = 0.010, 0.025      # peg radius, half-length

for clr in (2.0, 1.5, 1.0, 0.5):
    m = build_model(P, clearance_mm=clr); d = mujoco.MjData(m)
    bid = m.body("peg").id; a = m.joint("peg_free").qposadr[0]
    for k in (0.5, 0.9, 1.5, 2.5):                      # lateral offset / clearance
        mujoco.mj_resetData(m, d)
        x0, z0 = 0.22 + k * clr * 1e-3, 0.030 + 0.001 + H   # tip 1 mm above opening
        d.qpos[a:a+7] = [x0, 0, z0, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        res = np.zeros(6); f6 = np.zeros(6)
        peak_F = 0.0; steady_pen = 0.0; max_tilt = 0.0
        n = int(4.2 / m.opt.timestep)
        for i in range(n):
            t = i * m.opt.timestep
            tz = z0 - V * t
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
            w, v = res[:3], res[3:]
            q = d.xquat[bid]; qerr = 2 * q[1:4] * np.sign(q[0])
            d.xfrc_applied[bid, :3] = KP * (np.array([x0, 0, tz]) - d.xpos[bid]) - KD * v
            d.xfrc_applied[bid, 3:] = -KR * qerr - KDR * w
            mujoco.mj_step(m, d)
            Fn = 0.0
            for c in range(d.ncon):
                mujoco.mj_contactForce(m, d, c, f6); Fn += abs(f6[0])
            peak_F = max(peak_F, Fn)
            Rm = d.xmat[bid].reshape(3, 3)
            max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(Rm[2, 2], -1, 1))))
            if i > n - 300 and d.ncon:
                steady_pen = max(steady_pen, -min(c.dist for c in d.contact[:d.ncon]))
        Rm = d.xmat[bid].reshape(3, 3)
        tip_z = (d.xpos[bid] - Rm[:, 2] * H)[2]
        print(f"clr={clr} off={k}*clr  tip depth={(0.030 - tip_z)*1e3:5.1f} mm  "
              f"peakF={peak_F:5.1f} N  steady pen={steady_pen*1e3:.3f} mm  max tilt={max_tilt:4.1f} deg")
              