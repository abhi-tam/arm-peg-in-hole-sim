"""Scripted demonstrator: Align -> Descend -> (Search spiral) -> Insert.

Controls the PEG TIP (measured from sim state) by IK on the gripper tip. The commanded tip
height never leads the measured height by more than 2 mm, so the arm cannot wind up force
while resting on the rim (this is also the "command 2 mm below current height" of the Search).

Modification vs the spec: the hole position the demonstrator aims at has a per-episode
calibration bias (default 0.3 mm sigma per axis). With perfect knowledge the peg slides in
without touching the walls at any clearance, which makes the task trivial and the Search
phase dead code. The bias models camera/hand-eye calibration error.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

V_ALIGN = 0.050      # m/s
V_DESCEND = 0.010    # m/s
V_SPIRAL = 0.005     # m/s along path
PITCH = 0.5e-3       # m per turn
R_MAX = 3e-3         # m
ABOVE = 0.015        # m, tip above opening during align
LEAD = 0.002         # m, max command lead below measured tip
INSERT_DEPTH = 0.022
SETTLE_S = 0.3
STALL_WIN_S = 0.3
STALL_PROGRESS = 0.5e-3
GIVE_UP_S = 20.0
DEFAULT_BIAS_MM = 0.3


class Demonstrator:
    def __init__(self, env, bias_mm=DEFAULT_BIAS_MM, rng=None):
        self.env = env
        rng = rng or np.random.default_rng(0)
        self.bias = rng.normal(0.0, bias_mm * 1e-3, 2)
        self.reset()

    def reset(self):
        e = self.env
        self.dt = e.dt_ctrl
        self.t = 0.0
        self.phase = "align"
        self.cmd = e.peg_tip().copy()
        self.corr = np.zeros(3)
        self.q = e.d.qpos[e.qadr].copy()
        self.settle_t = 0.0
        self.desc_t = 0.0
        self.zhist = deque(maxlen=int(STALL_WIN_S / self.dt) + 1)
        self.theta = 0.0
        self.n_search = 0

    def _depth(self):
        return (self.env.hole_top - self.env.peg_tip()[2])

    def _action(self):
        e = self.env
        offset = e.peg_tip() - e.tip_pos()
        target = self.cmd - offset + self.corr
        q, err, tilt = e.ik(target, self.q, iters=60)
        self.q = q
        return np.concatenate([np.degrees(q), [0.0]])

    def act(self):
        e = self.env
        hx, hy = e.hole_xy + self.bias       # estimated hole position (biased)
        top = e.hole_top
        actual = e.peg_tip()
        self.t += self.dt

        if self.t > GIVE_UP_S:
            self.phase = "gave_up"

        if self.phase == "align":
            goal = np.array([hx, hy, top + ABOVE])
            vec = goal - self.cmd
            dist = np.linalg.norm(vec)
            step = V_ALIGN * self.dt
            if dist > step:
                self.cmd = self.cmd + vec / dist * step
            else:
                self.cmd = goal
                self.settle_t += self.dt
            self.corr = np.clip(self.corr + 0.2 * (self.cmd - actual), -0.008, 0.008)
            if self.settle_t >= SETTLE_S:
                self.phase = "descend"
                self.cmd[2] = actual[2]

        elif self.phase == "descend":
            self.desc_t += self.dt
            self.cmd[:2] = [hx, hy]
            self.cmd[2] = max(self.cmd[2] - V_DESCEND * self.dt, actual[2] - LEAD)
            self.zhist.append(actual[2])
            if self._depth() >= 0.002:
                self.phase = "insert"
            elif (self.desc_t > 0.5 and len(self.zhist) == self.zhist.maxlen
                  and self.zhist[0] - actual[2] < STALL_PROGRESS and actual[2] > top):
                self.phase = "search"
                self.theta = 0.0
                self.n_search += 1

        elif self.phase == "search":
            b = PITCH / (2 * math.pi)
            r = b * self.theta
            self.theta += V_SPIRAL * self.dt / math.sqrt(r * r + b * b)
            r = b * self.theta
            if r > R_MAX:
                self.theta = 0.0
                r = 0.0
            self.cmd[0] = hx + r * math.cos(self.theta)
            self.cmd[1] = hy + r * math.sin(self.theta)
            self.cmd[2] = actual[2] - LEAD
            if self._depth() >= 0.002:
                self.phase = "insert"

        elif self.phase == "insert":
            self.cmd[:2] = [hx, hy]
            floor = top - INSERT_DEPTH
            self.cmd[2] = max(self.cmd[2] - V_DESCEND * self.dt, actual[2] - LEAD, floor)
            if self._depth() >= INSERT_DEPTH:
                self.phase = "done"

        return self._action()