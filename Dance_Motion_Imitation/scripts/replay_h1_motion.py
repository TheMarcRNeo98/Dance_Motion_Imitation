from __future__ import annotations

"""
Replay the provided H1 motion file directly in IsaacLab / Isaac Sim.

This is a visualization script only.
It does NOT train a policy.
It does NOT use a controller.
It writes the reference root pose and joint states frame-by-frame into simulation.

Usage example:
./isaaclab.sh -p /home/bi-manual/Dance_Motion_Imitation/scripts/replay_h1_motion.py \
    --robot /home/bi-manual/Dance_Motion_Imitation/assets/h1/usd/h1.usd \
    --motion /home/bi-manual/Dance_Motion_Imitation/data/motions/h1_dance.npy
"""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT = PROJECT_ROOT / "assets" / "h1" / "usd" / "h1.usd"
DEFAULT_MOTION = PROJECT_ROOT / "data" / "motions" / "h1_dance.npy"

parser = argparse.ArgumentParser(description="Replay H1 motion file in IsaacLab.")
parser.add_argument(
    "--robot",
    type=str,
    default=str(DEFAULT_ROBOT),
    help="Absolute path to H1 USD file.",
)
parser.add_argument(
    "--motion",
    type=str,
    default=str(DEFAULT_MOTION),
    help="Absolute path to h1_dance.npy.",
)
parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0, help="Physics timestep.")
parser.add_argument("--quat_order", type=str, default="xyzw", choices=["xyzw", "wxyz"], help="Quaternion order in the motion file.")
parser.add_argument("--root_z_offset", type=float, default=0.0, help="Extra z offset added to motion root position.")
parser.add_argument("--loop", action="store_true", help="Loop the motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

MOTION_JOINT_NAMES = [
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


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[:-1] = (x[1:] - x[:-1]) / dt
    out[-1] = out[-2]
    return out


def quaternion_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    return np.concatenate(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quaternion_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quaternion_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    n = quat_wxyz.shape[0]
    ang_vel = np.zeros((n, 3), dtype=np.float32)

    for i in range(n - 1):
        dq = quaternion_multiply_wxyz(
            quat_wxyz[i + 1 : i + 2],
            quaternion_conjugate_wxyz(quat_wxyz[i : i + 1]),
        )[0]

        if dq[0] < 0.0:
            dq = -dq

        vec = dq[1:]
        norm = np.linalg.norm(vec)

        if norm < 1e-8:
            ang_vel[i] = 0.0
            continue

        angle = 2.0 * math.atan2(norm, max(float(dq[0]), 1e-8))
        axis = vec / norm
        ang_vel[i] = axis * angle / dt

    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def load_motion_file(motion_path: str, quat_order: str):
    motion = np.load(motion_path, allow_pickle=True).item()

    required = {"dt", "joint_order", "joint_pos", "root_pos", "root_quat", "root_name"}
    missing = required - set(motion.keys())
    if missing:
        raise KeyError(f"Motion file is missing keys: {sorted(missing)}")

    joint_order = list(motion["joint_order"])
    if joint_order != MOTION_JOINT_NAMES:
        raise ValueError(
            "Joint order in motion file does not match the expected 19 H1 joints.\n"
            f"Expected: {MOTION_JOINT_NAMES}\n"
            f"Got:      {joint_order}"
        )

    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_quat = np.asarray(motion["root_quat"], dtype=np.float32)
    dt = float(motion["dt"])
    root_name = str(motion["root_name"])

    if quat_order == "xyzw":
        root_quat = np.concatenate((root_quat[:, 3:4], root_quat[:, :3]), axis=-1)

    joint_vel = finite_difference(joint_pos, dt)
    root_lin_vel = finite_difference(root_pos, dt)
    root_ang_vel = quaternion_angular_velocity(root_quat, dt)

    return {
        "dt": dt,
        "root_name": root_name,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
    }


def main():
    robot_path = Path(args_cli.robot).expanduser().resolve()
    motion_path = Path(args_cli.motion).expanduser().resolve()

    if not robot_path.exists():
        raise FileNotFoundError(f"USD file not found: {robot_path}")
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    motion = load_motion_file(str(motion_path), args_cli.quat_order)

    print("[INFO] Motion loaded")
    print(f"  root_name: {motion['root_name']}")
    print(f"  dt:        {motion['dt']}")
    print(f"  frames:    {motion['joint_pos'].shape[0]}")
    print(f"  joints:    {motion['joint_pos'].shape[1]}")

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 2.5, 2.0], target=[0.0, 0.0, 1.0])

    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(robot_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05)),
    actuators={},
    )

    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    joint_ids = [robot.find_joints(name)[0][0] for name in MOTION_JOINT_NAMES]
    print("[INFO] Found all motion joints in the USD articulation")

    if motion["root_name"] != "imu_link":
        print(f"[WARN] Motion file root_name is '{motion['root_name']}', expected 'imu_link' for this H1 asset")

    num_frames = motion["joint_pos"].shape[0]
    motion_dt = motion["dt"]
    steps_per_frame = max(1, int(round(motion_dt / args_cli.physics_dt)))
    print(f"[INFO] Replay settings: physics_dt={args_cli.physics_dt}, motion_dt={motion_dt}, steps_per_frame={steps_per_frame}")

    device = sim.device

    joint_pos = torch.tensor(motion["joint_pos"], dtype=torch.float32, device=device)
    joint_vel = torch.tensor(motion["joint_vel"], dtype=torch.float32, device=device)
    root_pos = torch.tensor(motion["root_pos"], dtype=torch.float32, device=device)
    root_pos[:, 2] += args_cli.root_z_offset
    root_quat = torch.tensor(motion["root_quat"], dtype=torch.float32, device=device)
    root_lin_vel = torch.tensor(motion["root_lin_vel"], dtype=torch.float32, device=device)
    root_ang_vel = torch.tensor(motion["root_ang_vel"], dtype=torch.float32, device=device)

    frame_id = 0
    step_count = 0

    print("[INFO] Starting replay... Close the Isaac Sim window or Ctrl+C to stop.")

    while simulation_app.is_running():
        if step_count % steps_per_frame == 0:
            root_pose = torch.cat((root_pos[frame_id : frame_id + 1], root_quat[frame_id : frame_id + 1]), dim=-1)
            root_vel = torch.cat((root_lin_vel[frame_id : frame_id + 1], root_ang_vel[frame_id : frame_id + 1]), dim=-1)

            robot.write_root_link_pose_to_sim(root_pose)
            robot.write_root_com_velocity_to_sim(root_vel)
            robot.write_joint_state_to_sim(
                joint_pos[frame_id : frame_id + 1],
                joint_vel[frame_id : frame_id + 1],
                joint_ids=joint_ids,
            )

            frame_id += 1
            if frame_id >= num_frames:
                if args_cli.loop:
                    frame_id = 0
                else:
                    print("[INFO] Replay finished.")
                    break

        sim.step()
        robot.update(args_cli.physics_dt)
        step_count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
