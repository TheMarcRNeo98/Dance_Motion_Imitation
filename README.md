# Humanoid Dance Motion Imitation via Reinforcement Learning

<p align="center">
  <img src="docs/assets/all_metrics.png" width="100%" alt="Policy evaluation dashboard showing 50-second dance tracking">
</p>

A reinforcement learning system that trains a **Unitree H1 humanoid robot** to physically imitate a 50-second dance motion clip in simulation. The policy controls 19 joints through residual position commands while maintaining balance entirely through learned physics—no root forcing, no motion replay.

Built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [skrl](https://github.com/Toni-SM/skrl).

---

## Key results

| Metric | Value |
|--------|-------|
| Dance duration reproduced | **50.0 s** (1500 frames, full clip) |
| Root position tracking | **12.1 cm** mean 3D error |
| Root orientation tracking | **6.5°** mean error |
| Joint tracking (legs) | **9.8°** MAE |
| Joint tracking (arms) | **12.9°** MAE |
| Falls during full clip | **0** |
| Training time (RTX 4090) | ~12 hours total |

---

## Overview

This project addresses the problem of training a bipedal humanoid to reproduce a reference dance motion through physically-grounded reinforcement learning. Unlike kinematic replay (which directly sets joint/root states each timestep), the learned policy must produce joint torques that result in the correct motion *through physics simulation*—meaning the robot must simultaneously track the reference trajectory and maintain dynamic balance.

The core challenge is that dance motions involve rapid weight shifts, deep crouches, large arm movements, and sustained single-leg phases—all of which stress the limits of bipedal balance. A policy that perfectly tracks joint angles will still fall if it doesn't implicitly learn center-of-mass management, ground reaction force distribution, and recovery strategies.

### What this project implements

- **Custom IsaacLab Direct environment** for single-clip motion imitation on the H1 platform
- **Multiplicative reward formulation** that prevents the policy from ignoring any tracking objective
- **Biomechanically-motivated actuator configuration** with differentiated gains for load-bearing vs. non-load-bearing joints
- **Projected gravity observation** as an explicit balance signal for the policy
- **Staged curriculum** that progressively expands the training window from a short prefix to the full clip
- **Evaluation pipeline** with per-joint metrics, root tracking analysis, and automated plot generation

### What comes from prior work

The learning algorithm is **Adversarial Motion Priors (AMP)** from [Peng et al., 2021](https://arxiv.org/abs/2104.02180), used through the [skrl](https://skrl.readthedocs.io) library. The simulation platform is [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/). The robot model is the [Unitree H1](https://www.unitree.com/h1). The overall paradigm of physics-based motion imitation follows from [DeepMimic (Peng et al., 2018)](https://arxiv.org/abs/1804.02717).

This project does **not** claim novelty in the RL algorithm itself. The contributions are in the environment design, reward formulation, actuator modeling, and the engineering required to make a real humanoid platform track a full dance clip without falling.

---

## Mathematical formulation

### Problem setup

The environment is a discrete-time Markov Decision Process (MDP) where the policy $\pi_\theta(a_t | o_t)$ maps observations to actions at 30 Hz. The physics simulation runs at 60 Hz (decimation factor of 2).

**State**: Full articulated state of the H1 robot (joint positions, joint velocities, root pose, root velocities) plus reference motion features.

**Action**: Residual joint position commands $a_t \in \mathbb{R}^{19}$, applied as:

$$q_t^{\text{cmd}} = q_t^{\text{ref}} + \alpha \cdot a_t$$

where $q_t^{\text{ref}}$ is the reference joint position at the current frame and $\alpha = 0.25$ is the action scale. The robot's PD controllers then track $q_t^{\text{cmd}}$. This residual formulation provides a strong prior (the reference motion) while allowing the policy to make corrections for balance.

### Observation space (111 dimensions)

The policy observation concatenates:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Current AMP features | 53 | Joint pos (19) + joint vel (19) + root pos (3) + root orientation as tangent/normal (6) + root lin vel (3) + root ang vel (3) |
| Reference AMP features | 53 | Same features from the reference motion at the current frame |
| Phase encoding | 2 | $(\sin\phi_t,\; \cos\phi_t)$ where $\phi_t = 2\pi \cdot \frac{t - t_{\text{start}}}{T}$ |
| Projected gravity | 3 | Gravity vector rotated into the root body frame: $g_{\text{body}} = R(q_t^{\text{root}})^{-1} \cdot [0, 0, -1]^T$ |

The projected gravity vector is critical for balance—it tells the policy which direction is "down" relative to its torso, enabling it to learn tilt-correction strategies analogous to the vestibular system in biological locomotion.

### Reward: multiplicative tracking

A standard approach sums individual reward terms, but this allows the policy to achieve high reward by excelling at easy terms while ignoring hard ones (e.g., getting perfect joint tracking while the root drifts). We use a **weighted geometric mean** that forces all terms to be satisfied simultaneously:

$$r_t^{\text{core}} = \prod_{i} r_i^{w_i}$$

where each sub-reward $r_i \in [0, 1]$ is an exponential tracking kernel:

$$r_i = \exp(-k_i \cdot e_i)$$

| Sub-reward $r_i$ | Error $e_i$ | Kernel $k_i$ | Weight $w_i$ |
|---|---|---|---|
| Pose tracking | $\frac{1}{n}\sum_j (q_j - q_j^{\text{ref}})^2$ | 3.0 | 0.10 |
| Joint velocity | $\frac{1}{n}\sum_j (\dot{q}_j - \dot{q}_j^{\text{ref}})^2$ | 0.3 | 0.05 |
| Root position | $\frac{1}{3}\|p - p^{\text{ref}}\|^2$ | 25.0 | **0.35** |
| Root orientation | $\angle(q^{\text{root}},\; q^{\text{root,ref}})^2$ | 10.0 | 0.20 |
| Root linear velocity | $\frac{1}{3}\|\dot{p} - \dot{p}^{\text{ref}}\|^2$ | 2.0 | 0.15 |
| Root angular velocity | $\frac{1}{3}\|\omega - \omega^{\text{ref}}\|^2$ | 1.0 | 0.10 |
| Alive | 1 (constant) | — | 0.05 |

Root position receives the highest weight (0.35) because root drift is the primary failure mode—if the center of mass deviates from the reference trajectory, balance is quickly lost regardless of joint tracking quality.

The total reward adds shaping terms:

$$r_t = c \cdot r_t^{\text{core}} + \lambda_{\text{act}} \|a_t\|^2 + \lambda_{\text{rate}} \|a_t - a_{t-1}\|^2 + \lambda_{\text{slip}} \cdot s_t + \lambda_{\text{contact}} \cdot b_t + \lambda_{\text{term}} \cdot d_t$$

where $s_t$ is a foot-slip penalty (penalizing horizontal foot velocity when the foot is on the ground), $b_t$ is a ground-contact bonus, and $d_t$ is a termination penalty.

### Differentiated actuator gains

The H1 weighs approximately 47 kg. A knee joint supporting half the body weight at a typical flexion angle through a 0.4 m lever arm requires ~90 Nm of torque. With uniform PD gains (the standard IsaacLab default), the gains are either too weak for legs (robot sinks) or too stiff for arms (unnatural motion).

We split the actuator configuration by biomechanical role:

| Joint group | Stiffness (Nm/rad) | Damping (Nms/rad) | Rationale |
|-------------|--------------------|--------------------|-----------|
| Legs (hip, knee, ankle) | 500 | 50 | Must support body weight against gravity |
| Torso | 350 | 35 | Moderate load, postural stability |
| Arms (shoulder, elbow) | 150 | 15 | Low load, free-swinging motion |

### Staged curriculum

Rather than training on the full 50-second clip from the start (which dilutes per-frame training density), we use a curriculum that progressively expands the training window:

| Stage | Frames | Duration | Purpose | Training time |
|-------|--------|----------|---------|---------------|
| 1 | 0–180 | 6 s | Learn balance fundamentals | ~2 h |
| 2 | 0–1499 | 50 s | Extend to full clip (resume from Stage 1) | ~4 h |
| 3 | 0–1499 | 50 s | Increase per-frame density (resume from Stage 2) | ~6 h |

Each stage resumes from the previous checkpoint, transferring learned balance skills to new motion segments.

---

## Project structure

```
Dance_Motion_Imitation/                         # OUTER project root
├── assets/
│   └── h1/
│       └── usd/
│           └── h1.usd                          # Unitree H1 robot model
├── data/
│   └── motions/
│       └── h1_dance.npy                        # Reference dance clip (1500 frames, 30 Hz)
├── Dance_Motion_Imitation/                     # INNER project (IsaacLab extension)
│   ├── pyproject.toml                          # Linting / formatting config
│   ├── scripts/
│   │   ├── skrl/
│   │   │   ├── train.py                        # Training entry point
│   │   │   ├── play.py                         # Visual inference
│   │   │   └── log_policy_rollout.py           # Rollout logging with CSV export
│   │   ├── replay_h1_motion.py                 # Replay reference motion (no policy)
│   │   ├── list_envs.py                        # List registered environments
│   │   ├── zero_agent.py                       # Zero-action debug agent
│   │   └── random_agent.py                     # Random-action debug agent
│   └── source/
│       └── Dance_Motion_Imitation/
│           └── Dance_Motion_Imitation/
│               └── tasks/
│                   └── direct/
│                       └── dance_motion_imitation/
│                           ├── __init__.py
│                           ├── dance_motion_imitation_env.py       # Environment implementation
│                           ├── dance_motion_imitation_env_cfg.py   # Environment configuration
│                           └── agents/
│                               └── skrl_amp_cfg.yaml               # AMP hyperparameters
├── tools/
│   └── evaluate_policy.py                      # Evaluation metrics and plots
├── docs/
│   └── assets/
│       └── all_metrics.png                     # Evaluation dashboard
├── .gitignore
├── .gitattributes
├── LICENSE                                     # MIT License
└── README.md
```

### Motion data format

`h1_dance.npy` is a NumPy dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `dt` | scalar | Timestep: 0.0333 s (30 Hz) |
| `joint_order` | (19,) | Joint name ordering |
| `joint_pos` | (1500, 19) | Joint angles in radians |
| `root_pos` | (1500, 3) | World-frame position of `imu_link` |
| `root_quat` | (1500, 4) | Orientation as quaternion (xyzw) |
| `root_name` | string | Reference body: `"imu_link"` |

---

## Installation

### Prerequisites

- NVIDIA GPU (RTX 3090 or better; tested on RTX 4090)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (tested with Isaac Sim 4.x)
- Python 3.10+
- [skrl](https://skrl.readthedocs.io) ≥ 1.4.3

### Setup

```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/Dance_Motion_Imitation.git
cd Dance_Motion_Imitation

# Ensure IsaacLab is installed and activated
# (follow https://isaac-sim.github.io/IsaacLab/source/setup/installation.html)

# Install this project as an IsaacLab extension
cd Dance_Motion_Imitation  # inner project directory
python -m pip install -e .
```

---

## Usage

All commands are run from the IsaacLab root directory (`~/IsaacLab`).

### Training

**Stage 1 — Short prefix (balance learning):**
```bash
./isaaclab.sh -p <path>/scripts/skrl/train.py \
    --task Isaac-H1-Dance-AMP-Direct-v0 \
    --algorithm AMP \
    --num_envs 512 \
    --max_iterations 5000 \
    --headless
```

**Stage 2+ — Full clip (resume from previous checkpoint):**
```bash
./isaaclab.sh -p <path>/scripts/skrl/train.py \
    --task Isaac-H1-Dance-AMP-Direct-v0 \
    --algorithm AMP \
    --num_envs 512 \
    --max_iterations 30000 \
    --headless \
    --checkpoint <path_to_best_agent.pt>
```

### Inference

**Visual playback:**
```bash
./isaaclab.sh -p <path>/scripts/skrl/log_policy_rollout.py \
    --task Isaac-H1-Dance-AMP-Direct-v0 \
    --algorithm AMP \
    --max_steps 1500
```

**Headless with CSV export:**
```bash
./isaaclab.sh -p <path>/scripts/skrl/log_policy_rollout.py \
    --task Isaac-H1-Dance-AMP-Direct-v0 \
    --algorithm AMP \
    --max_steps 1500 \
    --headless
```

### Evaluation

```bash
pip install matplotlib
python tools/evaluate_policy.py \
    --csv logs/policy_debug/policy_vs_reference.csv \
    --output eval_results/
```

Produces per-joint tracking plots, root trajectory analysis, reward curves, and a summary dashboard.

---

## Results

The trained policy successfully imitates the full 50-second dance clip without falling. Tracking quality across the entire clip:

<p align="center">
  <img src="docs/assets/all_metrics.png" width="100%">
</p>

**Summary statistics (full 50-second clip):**

| Metric | Value |
|--------|-------|
| Mean episode reward | 12.0 |
| Root 3D position error | 12.1 cm (mean), 39.0 cm (max) |
| Root height error | 8.1 cm (mean) |
| Root orientation error | 6.5° (mean), 26.3° (max) |
| Leg joint MAE | 9.8° |
| Arm joint MAE | 12.9° |
| Torso joint MAE | 8.3° |

---

## References

- **DeepMimic**: Peng, X. B., Abbeel, P., Levine, S., & van de Panne, M. (2018). *DeepMimic: Example-guided deep reinforcement learning of physics-based character skills.* ACM Transactions on Graphics (TOG), 37(4). [[Paper]](https://arxiv.org/abs/1804.02717)

- **AMP**: Peng, X. B., Ma, Z., Abbeel, P., Levine, S., & Kanazawa, A. (2021). *AMP: Adversarial motion priors for stylized physics-based character animation.* ACM Transactions on Graphics (TOG), 40(4). [[Paper]](https://arxiv.org/abs/2104.02180)

- **skrl**: Serrano-Muñoz, A. (2022). *skrl: Modular and flexible library for reinforcement learning.* Journal of Machine Learning Research. [[Docs]](https://skrl.readthedocs.io)

- **Isaac Lab**: Mittal, M., et al. (2023). *Orbit: A unified simulation framework for interactive robot learning environments.* IEEE Robotics and Automation Letters. [[GitHub]](https://github.com/isaac-sim/IsaacLab)

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The Unitree H1 robot model is subject to Unitree's own licensing terms.
