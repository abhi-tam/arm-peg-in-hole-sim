import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("assets/SO101/so101_new_calib_camera.xml")
n = lambda t, i: mujoco.mj_id2name(m, t, i)
O = mujoco.mjtObj

print("dt:", m.opt.timestep, "| nq:", m.nq, "| nu:", m.nu)

print("\nJOINTS (name, type, range raw, range deg)")
for i in range(m.njnt):
    r = m.jnt_range[i]
    print(f"  {n(O.mjOBJ_JOINT, i)}  type={m.jnt_type[i]}  {r.round(4)}  {np.degrees(r).round(1)}")

print("\nACTUATORS (name, ctrlrange, gain, bias)  <- DO NOT CHANGE")
for i in range(m.nu):
    print(
        f"  {n(O.mjOBJ_ACTUATOR, i)}  {m.actuator_ctrlrange[i]}  "
        f"{m.actuator_gainprm[i, :3]}  {m.actuator_biasprm[i, :3]}"
    )

print("\nBODIES:", [n(O.mjOBJ_BODY, i) for i in range(m.nbody)])
print("SITES :", [n(O.mjOBJ_SITE, i) for i in range(m.nsite)])
print("CAMERAS:", [n(O.mjOBJ_CAMERA, i) for i in range(m.ncam)])

print("\nGEOMS on gripper/jaw bodies (candidates for finger pads)")
for g in range(m.ngeom):
    b = n(O.mjOBJ_BODY, m.geom_bodyid[g])
    if any(k in (b or "").lower() for k in ("gripper", "jaw", "finger")):
        print(
            f"  geom {g} name={n(O.mjOBJ_GEOM, g)} body={b} "
            f"contype={m.geom_contype[g]} conaffinity={m.geom_conaffinity[g]} "
            f"friction={m.geom_friction[g]}"
        )
