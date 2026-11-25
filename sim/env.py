"""PegHoleEnv: 30 Hz LeRobot-style wrapper around the MuJoCo model."""
from __future__ import annotations

import mujoco
import numpy as np

from sim.scene import build_model

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM + ["gripper"]
TIP_FROM_SITE = 0.035
PEG_TOP_BEHIND = 0.015


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class PegHoleEnv:
    def __init__(self, P, clearance_mm=None, offset_mm=(0.0, 0.0), hole_mu=None, latency_ms=0.0):
        self.P = P
        self.m = build_model(P, clearance_mm, offset_mm, hole_mu)
        self.d = mujoco.MjData(self.m)
        self.scratch = mujoco.MjData(self.m)
        self.latency_ms = float(latency_ms)
        m = self.m

        self.qadr = np.array([m.joint(n).qposadr[0] for n in ARM])
        self.dadr = np.array([m.joint(n).dofadr[0] for n in ARM])
        self.jlo = np.array([m.joint(n).range[0] for n in ARM])
        self.jhi = np.array([m.joint(n).range[1] for n in ARM])
        g = m.joint("gripper")
        self.g_qadr, self.g_lo, self.g_hi = g.qposadr[0], g.range[0], g.range[1]
        self.act = [m.actuator(n).id for n in JOINTS]  # assumes actuator names == joint names

        self.site = m.site(P["robot"]["gripperframe_site"]).id

        # Auto-detect the gripper pointing axis: the wrist camera view direction is ~25 deg off it.
        mujoco.mj_forward(m, self.d)
        view = -self.d.cam_xmat[m.cam("wrist").id].reshape(3, 3)[:, 2]
        S0 = self.d.site_xmat[self.site].reshape(3, 3)
        c = S0.T @ view
        self.point_axis = int(np.argmax(np.abs(c)))
        self.point_sign = float(np.sign(c[self.point_axis]))

        self.peg_bid = m.body("peg").id
        self.peg_gid = m.geom("peg_geom").id
        self.peg_qadr = m.joint("peg_free").qposadr[0]
        self.peg_half = P["scene"]["peg"]["length_m"] / 2

        hb = P["scene"]["hole_block"]
        self.hole_top = hb["size_m"][2]
        self.hole_xy = m.body("hole_block").pos[:2].copy()

        self.dt_ctrl = 1.0 / P["sim"]["control_hz"]
        self.n_sub = int(round(self.dt_ctrl / m.opt.timestep))
        self.renderer = None
        self.t = 0.0
        self.peak_force = 0.0
        self.last_q_cmd = None
        self.last_ori_err = 0.0
        self.hist = []

    # ---------- kinematics ----------
    def point_dir(self, d):
        return self.point_sign * d.site_xmat[self.site].reshape(3, 3)[:, self.point_axis]

    def tip_pos(self, d=None):
        d = self.d if d is None else d
        return d.site_xpos[self.site] + TIP_FROM_SITE * self.point_dir(d)

    def peg_tip(self):
        R = self.d.xmat[self.peg_bid].reshape(3, 3)
        return self.d.xpos[self.peg_bid] - R[:, 2] * self.peg_half

    def peg_axis_xy(self):
        return self.d.xpos[self.peg_bid][:2].copy()

    def ik(self, tip_target, q_init, iters=300, tol=1e-5):
        """Damped least squares: tip position (3) + gripper pointing straight down (3, rank 2).
        Returns (q, pos_err_m, tilt_deg)."""
        m, s = self.m, self.scratch
        down = np.array([0, 0, -1.0])
        L = TIP_FROM_SITE
        q = np.array(q_init, float)
        s.qpos[:] = self.d.qpos
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_fwdPosition(m, s)
            p = self.point_dir(s)
            tip = s.site_xpos[self.site] + L * p
            mujoco.mj_jacSite(m, s, jacp, jacr, self.site)
            Jr = jacr[:, self.dadr]
            Jp = jacp[:, self.dadr] - L * skew(p) @ Jr
            Jo = -skew(p) @ Jr
            e_pos = tip_target - tip
            e_ori = down - p
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_ori) < 1e-4:
                break
            J = np.vstack([Jp, Jo])
            r = np.concatenate([e_pos, e_ori])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), r)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.jlo, self.jhi)
        s.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(m, s)
        err = float(np.linalg.norm(tip_target - self.tip_pos(s)))
        tilt = float(np.degrees(np.arccos(np.clip(self.point_dir(s) @ down, -1, 1))))
        return q, err, tilt

    def solve_ik_multi(self, tip_target, q_hint=None):
        rng = np.random.default_rng(0)
        seeds = []
        if q_hint is not None:
            seeds.append(np.asarray(q_hint, float))
        seeds.append(self.d.qpos[self.qadr].copy())
        seeds += [np.zeros(5),
                  np.array([0, -0.8, 0.8, 0.8, 0]),
                  np.array([0, 0.3, 0.8, 1.0, 0]),
                  np.array([0, -1.0, 1.2, 0.5, 0]),
                  np.array([0, 0.5, 0.5, 1.2, 0])]
        seeds += [rng.uniform(self.jlo, self.jhi) for _ in range(40)]
        best = None
        for sd in seeds:
            q, err, tilt = self.ik(tip_target, sd)
            score = err + 1e-3 * np.radians(tilt)
            if best is None or score < best[0]:
                best = (score, q, err, tilt)
            if err < 1e-4 and tilt < 0.1:
                break
        _, q, err, tilt = best
        self.last_ori_err = tilt
        return q, err

    # ---------- units ----------
    def pct_to_rad(self, pct):
        return self.g_lo + np.clip(pct, 0, 100) / 100.0 * (self.g_hi - self.g_lo)

    def get_state(self):
        q = np.degrees(self.d.qpos[self.qadr])
        gp = 100.0 * (self.d.qpos[self.g_qadr] - self.g_lo) / (self.g_hi - self.g_lo)
        return np.concatenate([q, [gp]]).astype(np.float32)

    # ---------- reset ----------
    def reset(self, peg_in_gripper=None, with_images=True):
        m, d, P = self.m, self.d, self.P
        pig = {"dx_mm": 0, "dy_mm": 0, "dz_mm": 0, "tilt_deg": 0, **(peg_in_gripper or {})}
        st = P["scene"]["start_state"]
        mujoco.mj_resetData(m, d)
        tip_t = np.array([
            self.hole_xy[0] + st["peg_tip_x_offset_from_hole_m"],
            self.hole_xy[1],
            self.hole_top + st["peg_tip_above_opening_m"],
        ])
        q, err = self.solve_ik_multi(tip_t)
        assert err < 1e-3 and self.last_ori_err < 0.5, (
            f"IK failed for start pose (pos err={err*1e3:.2f} mm, tilt={self.last_ori_err:.2f} deg)")
        d.qpos[self.qadr] = q
        d.qpos[self.g_qadr] = self.g_hi
        d.ctrl[self.act[:5]] = q
        d.ctrl[self.act[5]] = self.g_hi
        mujoco.mj_forward(m, d)

        S = d.site_xmat[self.site].reshape(3, 3)
        p = self.point_dir(d)
        p = rodrigues(S[:, 2], np.radians(pig["tilt_deg"])) @ p
        center = (d.site_xpos[self.site]
                  + (-PEG_TOP_BEHIND + self.peg_half) * p
                  + S @ (np.array([pig["dx_mm"], pig["dy_mm"], pig["dz_mm"]]) * 1e-3))
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, -p)
        pose = np.concatenate([center, quat])

        d.ctrl[self.act[5]] = self.pct_to_rad(0)
        for _ in range(int(0.7 / m.opt.timestep)):
            d.qpos[self.peg_qadr:self.peg_qadr + 7] = pose
            d.qvel[self.peg_qadr:self.peg_qadr + 6] = 0
            mujoco.mj_step(m, d)
        for _ in range(int(0.3 / m.opt.timestep)):
            mujoco.mj_step(m, d)

        self.t, self.peak_force = 0.0, 0.0
        self.last_q_cmd = np.concatenate([np.degrees(q), [0.0]])
        self.hist = [self.last_q_cmd.copy()]
        return self.get_obs(with_images)

    # ---------- stepping ----------
    def _apply(self, a):
        lo, hi = self.m.actuator_ctrlrange[self.act[:5]].T
        self.d.ctrl[self.act[:5]] = np.clip(np.radians(a[:5]), lo, hi)
        self.d.ctrl[self.act[5]] = self.pct_to_rad(a[5])

    def step(self, action_deg, with_images=True):
        a = np.asarray(action_deg, float)
        self.last_q_cmd = a.copy()
        self.hist.append(a.copy())
        self.hist = self.hist[-100:]
        dt_ms = self.dt_ctrl * 1e3
        k = int(self.latency_ms // dt_ms)
        frac_s = (self.latency_ms - k * dt_ms) * 1e-3
        new = self.hist[-1 - k] if len(self.hist) > k else self.hist[0]
        old = self.hist[-2 - k] if len(self.hist) > k + 1 else self.hist[0]
        n_old = int(round(frac_s / self.m.opt.timestep))
        f6 = np.zeros(6)
        for i in range(self.n_sub):
            self._apply(old if i < n_old else new)
            mujoco.mj_step(self.m, self.d)
            tot = 0.0
            for j in range(self.d.ncon):
                c = self.d.contact[j]
                if self.peg_gid in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == self.peg_gid else c.geom1
                    if self.m.geom(other).name.startswith("hole_"):
                        mujoco.mj_contactForce(self.m, self.d, j, f6)
                        tot += abs(f6[0])
            self.peak_force = max(self.peak_force, tot)
        self.t += self.dt_ctrl
        return self.get_obs(with_images)

    # ---------- obs / metrics ----------
    def render(self, cam):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, 480, 640)
        self.renderer.update_scene(self.d, camera=cam)
        return self.renderer.render().copy()

    def get_obs(self, with_images=True):
        obs = {"observation.state": self.get_state(), "task": "Insert the peg into the hole."}
        if with_images:
            obs["observation.images.wrist"] = self.render("wrist")
            obs["observation.images.overhead"] = self.render("overhead")
        return obs

    def depth_mm(self):
        return (self.hole_top - self.peg_tip()[2]) * 1e3

    def success(self):
        return self.depth_mm() >= 20.0