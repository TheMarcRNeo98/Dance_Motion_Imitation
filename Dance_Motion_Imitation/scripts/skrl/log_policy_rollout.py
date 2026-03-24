# log_policy_rollout.py
# Run one-env inference and log:
# - sim joint positions / velocities
# - ref joint positions
# - sim root pos / quat
# - ref root pos / quat
# - ref frame index
#
# This script is made for the custom H1 dance env used in this chat.

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint and log sim vs reference trajectories.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default="Isaac-H1-Dance-AMP-Direct-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If omitted, auto-pick latest.")
parser.add_argument("--seed", type=int, default=None, help="Seed for env.")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 env for logging.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="AMP", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=1800, help="Max env steps to log.")
parser.add_argument("--output_dir", type=str, default="logs/policy_debug", help="Output dir for csv / npz.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"Unsupported skrl version {skrl.__version__}. Need >={SKRL_VERSION}")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    raise RuntimeError("Unsupported ML framework")

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Dance_Motion_Imitation.tasks  # noqa: F401

if args_cli.algorithm is None:
    raise RuntimeError("Please pass --algorithm")
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def quat_wxyz_to_xyzw_torch(q):
    return torch.cat((q[..., 1:], q[..., 0:1]), dim=-1)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    env_cfg.reset_strategy = "start-only"
    env_cfg.early_termination = False

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    if algorithm == "amp" and hasattr(env.unwrapped, "amp_observation_space"):
        env.amp_observation_space = env.unwrapped.amp_observation_space

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    base_env = env.unwrapped

    required_attrs = [
        "robot",
        "joint_ids",
        "ref_body_index",
        "cfg",
        "ref_frame_idx",
        "motion_joint_pos",
        "motion_root_pos",
        "motion_root_quat",
    ]
    missing = [name for name in required_attrs if not hasattr(base_env, name)]
    if missing:
        raise RuntimeError(
            "Your current local env does not expose the expected attributes for logging: "
            + ", ".join(missing)
        )

    robot = base_env.robot
    joint_ids = base_env.joint_ids
    ref_body_index = base_env.ref_body_index
    joint_names = list(base_env.cfg.joint_names)

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()

    timestep = 0
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        start_time = time.time()

        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            obs, rew, terminated, truncated, info = env.step(actions)

            ref_idx = int(base_env.ref_frame_idx[0].item())

            sim_joint_pos = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
            sim_joint_vel = robot.data.joint_vel[0, joint_ids].detach().cpu().numpy()

            ref_joint_pos = base_env.motion_joint_pos[ref_idx].detach().cpu().numpy()

            sim_root_pos = (robot.data.body_pos_w[0, ref_body_index] - base_env.scene.env_origins[0]).detach().cpu().numpy()
            sim_root_quat_wxyz = robot.data.body_quat_w[0, ref_body_index].detach().cpu()
            sim_root_quat_xyzw = quat_wxyz_to_xyzw_torch(sim_root_quat_wxyz.unsqueeze(0))[0].numpy()

            #ref_root_pos = base_env.motion_root_pos[ref_idx].detach().cpu().numpy()
            ref_root_pos = base_env.motion_root_pos[ref_idx] - base_env.scene.env_origins[0]
            ref_root_quat_wxyz = base_env.motion_root_quat[ref_idx].detach().cpu()
            ref_root_quat_xyzw = quat_wxyz_to_xyzw_torch(ref_root_quat_wxyz.unsqueeze(0))[0].numpy()

            row = {
                "step": timestep,
                "ref_frame_idx": ref_idx,
                "reward": float(rew[0].item()) if torch.is_tensor(rew) else float(rew),
                "sim_root_x": float(sim_root_pos[0]),
                "sim_root_y": float(sim_root_pos[1]),
                "sim_root_z": float(sim_root_pos[2]),
                "sim_root_qx": float(sim_root_quat_xyzw[0]),
                "sim_root_qy": float(sim_root_quat_xyzw[1]),
                "sim_root_qz": float(sim_root_quat_xyzw[2]),
                "sim_root_qw": float(sim_root_quat_xyzw[3]),
                "ref_root_x": float(ref_root_pos[0]),
                "ref_root_y": float(ref_root_pos[1]),
                "ref_root_z": float(ref_root_pos[2]),
                "ref_root_qx": float(ref_root_quat_xyzw[0]),
                "ref_root_qy": float(ref_root_quat_xyzw[1]),
                "ref_root_qz": float(ref_root_quat_xyzw[2]),
                "ref_root_qw": float(ref_root_quat_xyzw[3]),
            }

            for i, name in enumerate(joint_names):
                row[f"sim_{name}"] = float(sim_joint_pos[i])
                row[f"sim_vel_{name}"] = float(sim_joint_vel[i])
                row[f"ref_{name}"] = float(ref_joint_pos[i])
                row[f"err_{name}"] = float(sim_joint_pos[i] - ref_joint_pos[i])

            rows.append(row)

        timestep += 1

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    csv_path = output_dir / "policy_vs_reference.csv"
    npz_path = output_dir / "policy_vs_reference.npz"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # save a compact npz too
        npz_dict = {k: np.array([row[k] for row in rows]) for k in rows[0].keys()}
        np.savez(npz_path, **npz_dict)

    print(f"[INFO] Logged {len(rows)} steps in {time.time() - t0:.2f} s")
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] NPZ saved to: {npz_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
