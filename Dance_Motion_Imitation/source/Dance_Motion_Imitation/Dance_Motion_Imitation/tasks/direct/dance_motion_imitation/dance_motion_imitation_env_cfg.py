# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

PROJECT_ROOT = Path(__file__).resolve().parents[7]
H1_USD_PATH = PROJECT_ROOT / "assets" / "h1" / "usd" / "h1.usd"
H1_MOTION_PATH = PROJECT_ROOT / "data" / "motions" / "h1_dance.npy"


@configclass
class H1DanceMotionImitationEnvCfg(DirectRLEnvCfg):
    """Config for H1 dance motion imitation — STAGE 2: FULL CLIP

    Changes from Stage 1:
    - motion_frame_end = 1499  (full 50s dance)
    - episode_length_s = 55    (covers full clip)
    - Leg stiffness 400→500 to fix 14cm height deficit
    - Tighter root_pos_kernel 20→25
    """

    # env
    episode_length_s = 52.0             # slightly above 50s clip — episode ends before any wrap
    decimation = 2
    action_space = 19
    observation_space = 111
    state_space = 0

    # AMP
    num_amp_observations = 2
    amp_observation_space = 53

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(H1_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05)),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*hip_yaw_joint",
                    ".*hip_roll_joint",
                    ".*hip_pitch_joint",
                    ".*knee_joint",
                    ".*ankle_joint",
                ],
                velocity_limit=100.0,
                stiffness=500.0,        # was 400 in stage 1
                damping=50.0,           # was 40
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                velocity_limit=100.0,
                stiffness=350.0,
                damping=35.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*shoulder_pitch_joint",
                    ".*shoulder_roll_joint",
                    ".*shoulder_yaw_joint",
                    ".*elbow_joint",
                ],
                velocity_limit=100.0,
                stiffness=150.0,
                damping=15.0,
            ),
        },
    )

    # dataset / task
    motion_file: str = str(H1_MOTION_PATH)
    reference_body: str = "imu_link"
    joint_names: list[str] = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "torso_joint",
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_joint",
    ]

    # ── FULL CLIP ──
    motion_frame_start: int = 0
    motion_frame_end: int = 1499

    action_scale: float = 0.25
    early_termination: bool = True

    termination_height: float = 0.55
    termination_root_pos_err: float = 0.45
    termination_root_ori_err: float = 0.90

    reset_strategy: str = "random"

    # ── Reward kernels ──
    pose_kernel: float = 3.0
    velocity_kernel: float = 0.3
    root_pos_kernel: float = 25.0       # tighter than stage 1
    root_ori_kernel: float = 10.0
    root_lin_vel_kernel: float = 2.0
    root_ang_vel_kernel: float = 1.0

    w_pose: float = 0.10
    w_velocity: float = 0.05
    w_root_pos: float = 0.35
    w_root_ori: float = 0.20
    w_root_lin_vel: float = 0.15
    w_root_ang_vel: float = 0.10
    w_alive: float = 0.05

    rew_scale_core: float = 15.0
    rew_scale_action_l2: float = -0.001
    rew_scale_action_rate: float = -0.002
    rew_scale_foot_slip: float = -0.20
    rew_scale_termination: float = -10.0
    rew_scale_foot_contact_balance: float = 0.5

    foot_height_contact_thresh: float = 0.08
    foot_vertical_speed_contact_thresh: float = 0.25

    pose_weight_arms: float = 0.3
    pose_weight_torso_legs: float = 2.0
