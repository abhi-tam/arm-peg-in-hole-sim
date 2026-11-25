from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from sim.scene import build_model, load_params

Path("outputs").mkdir(exist_ok=True)
P = load_params()
model = build_model(P)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("dt:", model.opt.timestep, "| substeps per control step:", round((1 / 30) / model.opt.timestep))
print("joints:", [model.joint(i).name for i in range(model.njnt)])
print("cameras:", [model.cam(i).name for i in range(model.ncam)])
print("contacts at start (should be ~0):", data.ncon)
print("peg mass (want ~0.0195 kg):", model.body("peg").mass)
assert np.isclose(model.body("peg").mass, 0.0195, atol=1e-3)

# Overhead camera: right/up/back axes in world frame
cid = model.cam("overhead").id
R = data.cam_xmat[cid].reshape(3, 3)
print("overhead up axis (want +x):", R[:, 1].round(3), "| back axis (want +z):", R[:, 2].round(3))
assert np.allclose(R[:, 1], [1, 0, 0], atol=1e-6) and np.allclose(R[:, 2], [0, 0, 1], atol=1e-6)


def project(p, cam, f):
    c = model.cam(cam).id
    Rc, t = data.cam_xmat[c].reshape(3, 3), data.cam_xpos[c]
    x, y, z = Rc.T @ (np.asarray(p) - t)
    return 320 + f * x / -z, 240 - f * y / -z


u, v = project([0.22, 0, 0.03], "overhead", 457)
print("hole centre in overhead image (want ~320, 240):", round(u), round(v))
_, v2 = project([0.27, 0, 0.03], "overhead", 457)
assert v2 < v, "+x should be UP in the overhead image"

# Wrist camera: view direction vs gripperframe axes
view = -data.cam_xmat[model.cam("wrist").id].reshape(3, 3)[:, 2]
S = data.site_xmat[model.site(P["robot"]["gripperframe_site"]).id].reshape(3, 3)
for i, ax in enumerate("xyz"):
    angle_deg = np.degrees(np.arccos(np.clip(view @ S[:, i], -1, 1)))
    print(f"wrist view vs gripperframe {ax}-axis: {angle_deg:.1f} deg")
print("gripperframe world pos:", data.site_xpos[model.site(P["robot"]["gripperframe_site"]).id].round(4))

with mujoco.Renderer(model, height=480, width=640) as r:
    for cam in ("wrist", "overhead"):
        r.update_scene(data, camera=cam)
        img = r.render()
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        Image.fromarray(img).save(f"outputs/smoke_{cam}.png")

print("OK: wrote outputs/smoke_wrist.png and outputs/smoke_overhead.png")
