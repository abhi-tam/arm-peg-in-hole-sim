# Decision log (becomes the 2-page report)
Started: $(date)   (4-day clock)

## Sim choices
- Engine: MuJoCo. CPU contact solver, EGL rendering, leaves GPU memory available for VLA workloads.
- Robot & Timestep: Upstream SO101 arm kept untouched; simulation uses `dt = 1/600 s` (~0.00167 s) instead of upstream's `0.002 s`, giving 20 substeps per 30 Hz control step (integer, no drift). The spec leaves the timestep open, and because `1/600 s < 0.002 s`, the upstream actuator damping remains at least as numerically stable.
- Contact: elliptic cone, impratio 10, solref [0.004, 1], condim 3 (hole) / 4 (pads).
- Collision: hole = 48 convex wedge meshes (exact round bore) + floor box.

## Demonstrator / data

## VLA fine-tuning

## Tight-clearance improvement

## Sim-to-real gap / next week
