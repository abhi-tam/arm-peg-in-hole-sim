"""evaluate.py --variations <yaml> --episodes N --policy demonstrator|vla|final"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sim.demonstrator import Demonstrator, DEFAULT_BIAS_MM
from sim.env import PegHoleEnv
from sim.scene import load_params

MAX_STEPS = 20 * 30
VISUAL_KEYS = {"lighting", "textures", "camera_offset", "distractors"}  # TODO: not implemented yet


def run_episode(env, var, seed, policy_name, save_dir, name, save_frames):
    rng = np.random.default_rng(seed)
    pig = dict(var.get("peg_in_gripper") or {})
    pig["dx_mm"] = pig.get("dx_mm", 0.0) + rng.uniform(-0.5, 0.5)
    pig["dy_mm"] = pig.get("dy_mm", 0.0) + rng.uniform(-0.5, 0.5)
    pig["tilt_deg"] = pig.get("tilt_deg", 0.0) + rng.uniform(-0.5, 0.5)

    if policy_name == "demonstrator":
        env.reset(pig, with_images=False)
        demo = Demonstrator(env, bias_mm=var.get("hole_est_bias_mm", DEFAULT_BIAS_MM), rng=rng)
        t_succ = None
        for _ in range(MAX_STEPS):
            env.step(demo.act(), with_images=False)
            if env.success():
                t_succ = env.t
                break
    else:
        raise NotImplementedError(f"policy '{policy_name}' not wired up yet")

    if save_frames:
        for cam in ("wrist", "overhead"):
            Image.fromarray(env.render(cam)).save(save_dir / f"{name}_{cam}.png")

    return {
        "variation": name,
        "seed": int(seed),
        "success": bool(t_succ is not None),
        "insertion_depth_mm": round(float(env.depth_mm()), 3),
        "insertion_time_s": None if t_succ is None else round(float(t_succ), 3),
        "peak_contact_force_N": round(float(env.peak_force), 2),
        "final_phase": demo.phase,
        "n_search": demo.n_search,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variations", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--policy", choices=["demonstrator", "vla", "final"], default="demonstrator")
    ap.add_argument("--out", default="outputs/results.json")
    ap.add_argument("--params", default="sim_params.yaml")
    args = ap.parse_args()

    P = load_params(args.params)
    variations = yaml.safe_load(open(args.variations))
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for var in variations:
        name = var.get("name", "unnamed")
        ignored = VISUAL_KEYS & set(var)
        if ignored:
            print(f"[{name}] WARNING: visual fields not implemented yet, ignored: {sorted(ignored)}")
        ho = var.get("hole_offset_mm") or {}
        env = PegHoleEnv(
            P,
            clearance_mm=var.get("radial_clearance_mm", P["scene"]["default_radial_clearance_mm"]),
            offset_mm=(ho.get("x", 0.0), ho.get("y", 0.0)),
            hole_mu=var.get("peg_hole_friction"),
            latency_ms=var.get("action_latency_ms", 0.0),
        )
        base_seed = int(var.get("seed", 0))
        for ep in range(args.episodes):
            rec = run_episode(env, var, base_seed + ep, args.policy, out_dir, name, ep == 0)
            records.append(rec)
            print(f"  [{name}] seed={rec['seed']} success={rec['success']} depth={rec['insertion_depth_mm']}mm "
                  f"t={rec['insertion_time_s']} F={rec['peak_contact_force_N']}N "
                  f"phase={rec['final_phase']} searches={rec['n_search']}")

    json.dump(records, open(args.out, "w"), indent=2)

    print("\n=== success rate per variation ===")
    by = defaultdict(list)
    for r in records:
        by[r["variation"]].append(r)
    for k, rs in by.items():
        s = np.mean([r["success"] for r in rs])
        ts = [r["insertion_time_s"] for r in rs if r["insertion_time_s"] is not None]
        print(f"{k:24s} {100*s:5.1f}%  ({sum(r['success'] for r in rs)}/{len(rs)})  "
              f"mean time {np.mean(ts) if ts else float('nan'):.1f}s")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()