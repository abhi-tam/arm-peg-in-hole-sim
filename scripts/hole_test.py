import mujoco
from sim.scene import build_model, load_params
P = load_params()
for clr in (2.0, 1.5, 1.0, 0.5):
    m = build_model(P, clearance_mm=clr); d = mujoco.MjData(m)
    a = m.joint("peg_free").qposadr[0]
    for k in (0.0, 0.5, 1.5):                       # lateral offset as multiple of clearance
        mujoco.mj_resetData(m, d)
        d.qpos[a:a+7] = [0.22 + k*clr*1e-3, 0, 0.058, 1, 0, 0, 0]   # upright, tip 3 mm above opening
        mujoco.mj_forward(m, d)
        pen = 0.0
        for _ in range(1200):                       # 2 s
            mujoco.mj_step(m, d)
            if d.ncon: pen = max(pen, -min(c.dist for c in d.contact[:d.ncon]))
        depth = (0.030 - (d.qpos[a+2] - 0.025)) * 1e3
        print(f"clr={clr} offset={k}*clr  tip depth={depth:5.1f} mm  max pen={pen*1e3:.3f} mm  ncon={d.ncon}")