from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .dance_motion_imitation_env_cfg import H1DanceMotionImitationEnvCfg


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
        angle = 2.0 * math.atan2(norm, max(dq[0], 1e-8))
        axis = vec / norm
        ang_vel[i] = axis * angle / dt
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


class DanceMotionImitationEnv(DirectRLEnv):
    cfg: H1DanceMotionImitationEnvCfg

    def __init__(self, cfg: H1DanceMotionImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_ids = [self.robot.find_joints(name)[0][0] for name in self.cfg.joint_names]
        self.num_dofs = len(self.joint_ids)
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)

        self.left_foot_body_idx = self._find_body_index(["left", "ankle"])
        self.right_foot_body_idx = self._find_body_index(["right", "ankle"])

        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids, 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids, 1]

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ref_frame_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self._load_motion(self.cfg.motion_file)

        self.motion_start = int(getattr(self.cfg, "motion_frame_start", 0))
        self.motion_end = min(
            int(getattr(self.cfg, "motion_frame_end", self.num_motion_frames - 1)),
            self.num_motion_frames - 1,
        )
        if self.motion_end < self.motion_start:
            raise ValueError("motion_frame_end must be >= motion_frame_start")

        self.single_amp_observation_size = self.cfg.amp_observation_space
        self.num_amp_observations = self.cfg.num_amp_observations
        self.amp_observation_size = self.num_amp_observations * self.single_amp_observation_size

        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.amp_observation_size,), dtype=np.float32,
        )

        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.num_amp_observations, self.single_amp_observation_size),
            dtype=torch.float32, device=self.device,
        )

    def _find_body_index(self, required_keywords: list[str]) -> int:
        names = list(self.robot.data.body_names)
        for idx, name in enumerate(names):
            lname = name.lower()
            if all(keyword in lname for keyword in required_keywords):
                return idx
        raise RuntimeError(
            f"Could not infer body index for keywords={required_keywords}. "
            f"Available body names: {names}"
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                )
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.ref_frame_idx = self.ref_frame_idx + 1
        # Clamp at motion_end instead of wrapping — avoids the discontinuity
        # crash when the robot is at end-of-clip position but reference jumps
        # back to the start (which is ~1.2m away, instantly triggering termination).
        # During training with random resets, envs that reach the end simply
        # hold the last frame until episode timeout triggers a natural reset.
        self.ref_frame_idx = torch.clamp(self.ref_frame_idx, self.motion_start, self.motion_end)
        self.actions = torch.clamp(actions, -1.0, 1.0).clone()

    def _apply_action(self) -> None:
        ref_joint_pos = self.motion_joint_pos[self.ref_frame_idx]
        joint_target = ref_joint_pos + self.cfg.action_scale * self.actions
        joint_target = torch.clamp(joint_target, self.joint_lower_limits, self.joint_upper_limits)
        self.robot.set_joint_position_target(joint_target, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        current_amp_obs = self._build_current_amp_obs()
        ref_amp_obs = self._build_reference_amp_obs(self.ref_frame_idx)

        segment_len = float(self.motion_end - self.motion_start + 1)
        phase = 2.0 * math.pi * (self.ref_frame_idx.float() - float(self.motion_start)) / segment_len
        phase_obs = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)

        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        gravity_world = torch.zeros((self.num_envs, 3), device=self.device)
        gravity_world[:, 2] = -1.0
        proj_gravity = quat_apply_inverse(root_quat, gravity_world)

        policy_obs = torch.cat((current_amp_obs, ref_amp_obs, phase_obs, proj_gravity), dim=-1)

        for i in reversed(range(self.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = current_amp_obs

        self.extras = {"amp_obs": self.get_amp_observations()}
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]

        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_vel = self.robot.data.body_ang_vel_w[:, self.ref_body_index]

        ref_joint_pos = self.motion_joint_pos[self.ref_frame_idx]
        ref_joint_vel = self.motion_joint_vel[self.ref_frame_idx]
        ref_root_pos = self.motion_root_pos[self.ref_frame_idx]
        ref_root_quat = self.motion_root_quat[self.ref_frame_idx]
        ref_root_lin_vel = self.motion_root_lin_vel[self.ref_frame_idx]
        ref_root_ang_vel = self.motion_root_ang_vel[self.ref_frame_idx]

        num_arm_joints = min(8, joint_pos.shape[1])
        if joint_pos.shape[1] > num_arm_joints:
            arm_err = torch.mean(torch.square(joint_pos[:, :num_arm_joints] - ref_joint_pos[:, :num_arm_joints]), dim=-1)
            torso_leg_err = torch.mean(torch.square(joint_pos[:, num_arm_joints:] - ref_joint_pos[:, num_arm_joints:]), dim=-1)
            w_arms = float(self.cfg.pose_weight_arms)
            w_torso_legs = float(self.cfg.pose_weight_torso_legs)
            pose_err = (w_arms * arm_err + w_torso_legs * torso_leg_err) / max(w_arms + w_torso_legs, 1e-6)
        else:
            pose_err = torch.mean(torch.square(joint_pos - ref_joint_pos), dim=-1)

        joint_vel_err = torch.mean(torch.square(joint_vel - ref_joint_vel), dim=-1)
        root_pos_err_sq_mean = torch.mean(torch.square(root_pos - ref_root_pos), dim=-1)
        root_lin_vel_err = torch.mean(torch.square(root_lin_vel - ref_root_lin_vel), dim=-1)
        root_ang_vel_err = torch.mean(torch.square(root_ang_vel - ref_root_ang_vel), dim=-1)
        root_ori_err = quaternion_distance_angle(root_quat, ref_root_quat)

        k = self.cfg
        r_pose     = torch.exp(-k.pose_kernel * pose_err)
        r_velocity = torch.exp(-k.velocity_kernel * joint_vel_err)
        r_root_pos = torch.exp(-k.root_pos_kernel * root_pos_err_sq_mean)
        r_root_ori = torch.exp(-k.root_ori_kernel * torch.square(root_ori_err))
        r_root_lv  = torch.exp(-k.root_lin_vel_kernel * root_lin_vel_err)
        r_root_av  = torch.exp(-k.root_ang_vel_kernel * root_ang_vel_err)
        r_alive    = torch.ones_like(r_pose)

        core_reward = (
            r_pose.pow(k.w_pose)
            * r_velocity.pow(k.w_velocity)
            * r_root_pos.pow(k.w_root_pos)
            * r_root_ori.pow(k.w_root_ori)
            * r_root_lv.pow(k.w_root_lin_vel)
            * r_root_av.pow(k.w_root_ang_vel)
            * r_alive.pow(k.w_alive)
        )

        action_l2 = torch.mean(torch.square(self.actions), dim=-1)
        action_rate = torch.mean(torch.square(self.actions - self.prev_actions), dim=-1)

        left_foot_pos = self.robot.data.body_pos_w[:, self.left_foot_body_idx] - self.scene.env_origins
        right_foot_pos = self.robot.data.body_pos_w[:, self.right_foot_body_idx] - self.scene.env_origins
        left_foot_vel = self.robot.data.body_lin_vel_w[:, self.left_foot_body_idx]
        right_foot_vel = self.robot.data.body_lin_vel_w[:, self.right_foot_body_idx]

        fh = float(k.foot_height_contact_thresh)
        fv = float(k.foot_vertical_speed_contact_thresh)

        left_contact = ((left_foot_pos[:, 2] < fh) & (torch.abs(left_foot_vel[:, 2]) < fv)).float()
        right_contact = ((right_foot_pos[:, 2] < fh) & (torch.abs(right_foot_vel[:, 2]) < fv)).float()

        left_slip = (torch.square(left_foot_vel[:, 0]) + torch.square(left_foot_vel[:, 1])) * left_contact
        right_slip = (torch.square(right_foot_vel[:, 0]) + torch.square(right_foot_vel[:, 1])) * right_contact
        foot_slip = 0.5 * (left_slip + right_slip)

        has_ground_contact = torch.clamp(left_contact + right_contact, 0.0, 1.0)
        foot_contact_bonus = has_ground_contact

        root_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        root_pos_l2 = torch.sqrt(torch.sum(torch.square(root_pos - ref_root_pos), dim=-1))

        termination_flag = (
            (root_height < self.cfg.termination_height)
            | (root_pos_l2 > self.cfg.termination_root_pos_err)
            | (root_ori_err > self.cfg.termination_root_ori_err)
        ).float()

        reward = (
            float(k.rew_scale_core) * core_reward
            + float(k.rew_scale_action_l2) * action_l2
            + float(k.rew_scale_action_rate) * action_rate
            + float(k.rew_scale_foot_slip) * foot_slip
            + float(k.rew_scale_foot_contact_balance) * foot_contact_bonus
            + float(k.rew_scale_termination) * termination_flag
        )

        self.prev_actions[:] = self.actions
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            root_pos = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
            root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
            ref_root_pos = self.motion_root_pos[self.ref_frame_idx]
            ref_root_quat = self.motion_root_quat[self.ref_frame_idx]
            root_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
            root_pos_l2 = torch.sqrt(torch.sum(torch.square(root_pos - ref_root_pos), dim=-1))
            root_ori_err = quaternion_distance_angle(root_quat, ref_root_quat)
            died = (
                (root_height < self.cfg.termination_height)
                | (root_pos_l2 > self.cfg.termination_root_pos_err)
                | (root_ori_err > self.cfg.termination_root_ori_err)
            )
        else:
            died = torch.zeros_like(time_out)

        invalid = (
            torch.isnan(self.robot.data.joint_pos).any(dim=-1)
            | torch.isnan(self.robot.data.body_pos_w).any(dim=(1, 2))
        )
        died = died | invalid
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "start-only":
            sampled_frames = torch.full(
                (len(env_ids),), self.motion_start, dtype=torch.long, device=self.device
            )
        elif self.cfg.reset_strategy == "random":
            sampled_frames = torch.randint(
                low=self.motion_start, high=self.motion_end + 1,
                size=(len(env_ids),), device=self.device,
            )
        else:
            raise ValueError(f"Unknown reset_strategy: {self.cfg.reset_strategy}")

        self.ref_frame_idx[env_ids] = sampled_frames
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        root_pos = self.motion_root_pos[sampled_frames] + self.scene.env_origins[env_ids]
        root_quat = self.motion_root_quat[sampled_frames]
        root_lin_vel = self.motion_root_lin_vel[sampled_frames]
        root_ang_vel = self.motion_root_ang_vel[sampled_frames]

        root_pose = torch.cat((root_pos, root_quat), dim=-1)
        root_vel = torch.cat((root_lin_vel, root_ang_vel), dim=-1)

        joint_pos = self.motion_joint_pos[sampled_frames]
        joint_vel = self.motion_joint_vel[sampled_frames]

        self.robot.write_root_link_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self.joint_ids, env_ids)

        self.amp_observation_buffer[env_ids] = self.collect_reference_motions(
            len(env_ids), sampled_frames
        ).view(len(env_ids), self.num_amp_observations, self.single_amp_observation_size)

    def get_amp_observations(self) -> torch.Tensor:
        return self.amp_observation_buffer.view(self.num_envs, -1)

    def collect_reference_motions(self, num_samples: int, frame_ids: torch.Tensor | None = None) -> torch.Tensor:
        if frame_ids is None:
            frame_ids = torch.randint(
                low=self.motion_start, high=self.motion_end + 1,
                size=(num_samples,), device=self.device,
            )
        offsets = torch.arange(self.num_amp_observations, device=self.device).view(1, -1)
        segment_len = self.motion_end - self.motion_start + 1
        hist_frames = self.motion_start + ((frame_ids.view(-1, 1) - self.motion_start - offsets) % segment_len)
        flat_frames = hist_frames.reshape(-1)
        amp_obs = self._build_reference_amp_obs(flat_frames)
        return amp_obs.view(num_samples, -1)

    def _build_current_amp_obs(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]
        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_vel = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        return torch.cat((joint_pos, joint_vel, root_pos, quaternion_to_tangent_and_normal(root_quat), root_lin_vel, root_ang_vel), dim=-1)

    def _build_reference_amp_obs(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            self.motion_joint_pos[frame_ids],
            self.motion_joint_vel[frame_ids],
            self.motion_root_pos[frame_ids],
            quaternion_to_tangent_and_normal(self.motion_root_quat[frame_ids]),
            self.motion_root_lin_vel[frame_ids],
            self.motion_root_ang_vel[frame_ids],
        ), dim=-1)

    def _load_motion(self, motion_file: str) -> None:
        motion_dict = np.load(motion_file, allow_pickle=True).item()
        required_keys = {"dt", "joint_order", "joint_pos", "root_name", "root_pos", "root_quat"}
        missing = required_keys - set(motion_dict.keys())
        if missing:
            raise KeyError(f"Motion file is missing required keys: {sorted(missing)}")
        if motion_dict["root_name"] != self.cfg.reference_body:
            raise ValueError(
                f"Expected root_name='{self.cfg.reference_body}' in motion file, "
                f"but got '{motion_dict['root_name']}'"
            )
        motion_joint_order = list(motion_dict["joint_order"])
        reorder = [motion_joint_order.index(name) for name in self.cfg.joint_names]
        joint_pos = np.asarray(motion_dict["joint_pos"], dtype=np.float32)[:, reorder]
        root_pos = np.asarray(motion_dict["root_pos"], dtype=np.float32)
        root_pos[:, 2] += 0.05
        root_quat_xyzw = np.asarray(motion_dict["root_quat"], dtype=np.float32)
        dt = float(motion_dict["dt"])
        root_quat = np.concatenate((root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]), axis=-1)
        joint_vel = finite_difference(joint_pos, dt)
        root_lin_vel = finite_difference(root_pos, dt)
        root_ang_vel = quaternion_angular_velocity(root_quat, dt)
        self.motion_dt = dt
        self.num_motion_frames = joint_pos.shape[0]
        self.motion_joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self.device)
        self.motion_joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=self.device)
        self.motion_root_pos = torch.tensor(root_pos, dtype=torch.float32, device=self.device)
        self.motion_root_quat = torch.tensor(root_quat, dtype=torch.float32, device=self.device)
        self.motion_root_lin_vel = torch.tensor(root_lin_vel, dtype=torch.float32, device=self.device)
        self.motion_root_ang_vel = torch.tensor(root_ang_vel, dtype=torch.float32, device=self.device)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1.0
    ref_normal[..., 2] = 1.0
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat((tangent, normal), dim=-1)


@torch.jit.script
def quaternion_distance_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], -q2[:, 1], -q2[:, 2], -q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    imag_norm = torch.sqrt(torch.clamp(x * x + y * y + z * z, min=1.0e-8))
    return 2.0 * torch.atan2(imag_norm, torch.abs(w) + 1.0e-8)
