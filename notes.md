# Engineering Decisions

## 1. Sim Choices & Justifications
- Engine: MuJoCo (CPU physics + EGL rendering leaves ~7.5 GB of 8 GB VRAM for SmolVLA).
- Timestep: 0.002 s (500 Hz) with Newton solver (50 iters).
- Collision geometry: Primitive box/cylinder decomposition for hole so there are no mesh edge artifacts.

## 2. Demonstrator & Data Generation Decisions
- (Record success rates across 2.0 mm, 1.5 mm, 1.0 mm, 0.5 mm clearances here)

## 3. VLA Fine-Tuning & Tight-Clearance Architecture
- (Record frozen vs. unfrozen layers, batch size, VRAM usage, and improvement method here)

## 4. Sim-to-Real Gap & Next Steps (If Given +1 Week)
- Real pretraining vs. synthetic renders, gear backlash/compliance on low-cost SO-101 servos, camera latency.