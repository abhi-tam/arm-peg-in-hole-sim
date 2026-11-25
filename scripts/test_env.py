from pathlib import Path

import numpy as np
from PIL import Image

from sim.env import PegHoleEnv
from sim.scene import load_params

Path("outputs").mkdir(exist_ok=True)
P = load_params()
env = PegHoleEnv(P)
obs = env.reset()

site = env.d.site_xpos[env.site]
print("state (deg, gripper %):", obs["observation.state"].round(2))
print("IK tilt (deg):", round(env.last_ori_err, 3))
print("gripper tip :", env.tip_pos().round(4))
print("peg tip     :", env.peg_tip().round(4))
print("pointing dir (want ~[0,0,-1]):", env.point_dir(env.d).round(3))
print("peg tip - site dist (want 0.035):", np.linalg.norm(env.peg_tip() - site).round(4))
print("peg axis tilt (deg, want ~0):",
      np.degrees(np.arccos(np.clip(env.d.xmat[env.peg_bid].reshape(3, 3)[2, 2], -1, 1))).round(2))
print("peg tip mm above hole opening (want 15):", ((env.peg_tip()[2] - env.hole_top) * 1e3).round(2))
print("peg tip x - hole x, mm (want -30):", ((env.peg_tip()[0] - env.hole_xy[0]) * 1e3).round(2))

# hold the nominal action for 2 s: peg must stay in the gripper
hold = obs["observation.state"].copy()
hold[5] = 0.0
tip0 = env.peg_tip() - site
for _ in range(60):
    env.step(hold)
drift = (env.peg_tip() - env.d.site_xpos[env.site]) - tip0
print("peg drift in gripper after 2 s (mm):", (drift * 1e3).round(2))

# IK sanity: descend the tip 10 mm
tgt = env.tip_pos() - np.array([0, 0, 0.010])
q, err = env.solve_ik_multi(tgt)
print("IK 10 mm descent err (mm):", round(err * 1e3, 4), "| tilt:", round(env.last_ori_err, 3))

for cam in ("wrist", "overhead"):
    Image.fromarray(env.render(cam)).save(f"outputs/env_{cam}.png")
print("wrote outputs/env_wrist.png / env_overhead.png")