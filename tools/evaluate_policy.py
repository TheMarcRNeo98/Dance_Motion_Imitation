#!/usr/bin/env python3
"""
Evaluate a trained H1 dance imitation policy from a rollout CSV.

Usage:
    python evaluate_policy.py --csv /path/to/policy_vs_reference.csv --output /path/to/output_dir

Produces:
    - root_height_tracking.png
    - root_position_error.png
    - joint_tracking_mae.png
    - reward_over_time.png
    - root_orientation_error.png
    - summary_metrics.txt
    - all_metrics.png  (combined dashboard)
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib --break-system-packages")
    sys.exit(1)


JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "torso_joint",
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
]

SHORT_NAMES = [
    "L sh pitch", "L sh roll", "L sh yaw", "L elbow",
    "R sh pitch", "R sh roll", "R sh yaw", "R elbow",
    "Torso",
    "L hip yaw", "L hip roll", "L hip pitch", "L knee", "L ankle",
    "R hip yaw", "R hip roll", "R hip pitch", "R knee", "R ankle",
]

GROUPS = {
    "Arms": list(range(0, 8)),
    "Torso": [8],
    "Legs": list(range(9, 19)),
}


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def quat_angle(r):
    q1 = np.array([r["sim_root_qx"], r["sim_root_qy"], r["sim_root_qz"], r["sim_root_qw"]])
    q2 = np.array([r["ref_root_qx"], r["ref_root_qy"], r["ref_root_qz"], r["ref_root_qw"]])
    dot = min(abs(np.dot(q1, q2)), 1.0)
    return 2.0 * np.arccos(dot)


def compute_metrics(rows):
    n = len(rows)
    steps = np.arange(n)
    time_s = steps / 30.0  # 30 Hz control

    sim_z = np.array([r["sim_root_z"] for r in rows])
    ref_z = np.array([r["ref_root_z"] for r in rows])
    rewards = np.array([r["reward"] for r in rows])
    ref_frames = np.array([int(r["ref_frame_idx"]) for r in rows])

    pos_err_3d = np.array([
        np.sqrt((r["sim_root_x"] - r["ref_root_x"])**2 +
                (r["sim_root_y"] - r["ref_root_y"])**2 +
                (r["sim_root_z"] - r["ref_root_z"])**2)
        for r in rows
    ])

    ori_err = np.degrees(np.array([quat_angle(r) for r in rows]))

    joint_mae_rad = {}
    joint_mae_deg = {}
    for jn in JOINT_NAMES:
        errs = np.array([abs(r[f"err_{jn}"]) for r in rows])
        joint_mae_rad[jn] = np.mean(errs)
        joint_mae_deg[jn] = np.degrees(np.mean(errs))

    return {
        "steps": steps,
        "time_s": time_s,
        "sim_z": sim_z,
        "ref_z": ref_z,
        "rewards": rewards,
        "ref_frames": ref_frames,
        "pos_err_3d": pos_err_3d,
        "ori_err": ori_err,
        "joint_mae_deg": joint_mae_deg,
        "n": n,
    }


def plot_dashboard(m, output_dir):
    fig = plt.figure(figsize=(18, 22), facecolor="white")
    fig.suptitle("H1 Dance Motion Imitation — Policy Evaluation", fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.95, top=0.94, bottom=0.04)

    colors = {
        "sim": "#2563EB",
        "ref": "#16A34A",
        "err": "#DC2626",
        "reward": "#7C3AED",
        "bar_good": "#16A34A",
        "bar_mid": "#F59E0B",
        "bar_bad": "#DC2626",
    }

    # 1. Root height tracking
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(m["time_s"], m["sim_z"], color=colors["sim"], linewidth=1.2, label="Sim height")
    ax1.plot(m["time_s"], m["ref_z"], color=colors["ref"], linewidth=1.2, linestyle="--", label="Ref height")
    ax1.fill_between(m["time_s"], m["sim_z"], m["ref_z"], alpha=0.1, color=colors["err"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Root height (m)")
    ax1.set_title("Root height tracking")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_ylim(0.8, 1.5)
    ax1.grid(True, alpha=0.2)

    # 2. Root 3D position error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(m["time_s"], m["pos_err_3d"], color=colors["err"], linewidth=1.0)
    ax2.axhline(np.mean(m["pos_err_3d"]), color=colors["err"], linestyle="--", alpha=0.5,
                label=f'Mean: {np.mean(m["pos_err_3d"]):.3f} m')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("3D position error (m)")
    ax2.set_title("Root position error")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # 3. Root orientation error
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(m["time_s"], m["ori_err"], color="#9333EA", linewidth=1.0)
    ax3.axhline(np.mean(m["ori_err"]), color="#9333EA", linestyle="--", alpha=0.5,
                label=f'Mean: {np.mean(m["ori_err"]):.1f}°')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Orientation error (deg)")
    ax3.set_title("Root orientation error")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    # 4. Reward over time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(m["time_s"], m["rewards"], color=colors["reward"], linewidth=1.0)
    ax4.axhline(np.mean(m["rewards"]), color=colors["reward"], linestyle="--", alpha=0.5,
                label=f'Mean: {np.mean(m["rewards"]):.1f}')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Reward")
    ax4.set_title("Episode reward")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)

    # 5. Joint MAE bar chart
    ax5 = fig.add_subplot(gs[2, :])
    mae_vals = [m["joint_mae_deg"][jn] for jn in JOINT_NAMES]
    bar_colors = [colors["bar_good"] if v < 10 else colors["bar_mid"] if v < 14 else colors["bar_bad"] for v in mae_vals]
    bars = ax5.barh(SHORT_NAMES, mae_vals, color=bar_colors, height=0.7)
    ax5.set_xlabel("MAE (degrees)")
    ax5.set_title("Per-joint tracking error (green <10°, amber 10–14°, red >14°)")
    ax5.invert_yaxis()
    ax5.axvline(10, color="gray", linestyle=":", alpha=0.4)
    ax5.axvline(14, color="gray", linestyle=":", alpha=0.4)
    for bar, val in zip(bars, mae_vals):
        ax5.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}°", va="center", fontsize=8)
    ax5.grid(True, axis="x", alpha=0.2)

    # 6. Summary metrics box
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis("off")

    summary_lines = [
        f"Duration: {m['time_s'][-1]:.1f} s  |  Steps: {m['n']}  |  Control freq: 30 Hz",
        f"Mean reward: {np.mean(m['rewards']):.2f}  |  Min reward: {np.min(m['rewards']):.2f}  |  Max reward: {np.max(m['rewards']):.2f}",
        f"Root 3D error: {np.mean(m['pos_err_3d'])*100:.1f} cm (mean)  |  {np.max(m['pos_err_3d'])*100:.1f} cm (max)",
        f"Root height error: {np.mean(np.abs(m['sim_z'] - m['ref_z']))*100:.1f} cm (mean)",
        f"Root orientation error: {np.mean(m['ori_err']):.1f}° (mean)  |  {np.max(m['ori_err']):.1f}° (max)",
        f"Joint MAE — Arms: {np.mean([m['joint_mae_deg'][JOINT_NAMES[i]] for i in GROUPS['Arms']]):.1f}°  |  "
        f"Torso: {m['joint_mae_deg']['torso_joint']:.1f}°  |  "
        f"Legs: {np.mean([m['joint_mae_deg'][JOINT_NAMES[i]] for i in GROUPS['Legs']]):.1f}°",
    ]

    for i, line in enumerate(summary_lines):
        ax6.text(0.5, 0.85 - i * 0.16, line, transform=ax6.transAxes,
                 fontsize=12, ha="center", va="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#DEE2E6") if i == 0 else None)

    # Save
    dashboard_path = os.path.join(output_dir, "all_metrics.png")
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {dashboard_path}")

    # Also save individual plots
    for name, title, ydata, ylabel, color in [
        ("root_height_tracking", "Root height", None, None, None),
        ("root_position_error", "Root 3D error", m["pos_err_3d"], "Error (m)", colors["err"]),
        ("root_orientation_error", "Root orientation error", m["ori_err"], "Error (deg)", "#9333EA"),
        ("reward_over_time", "Reward", m["rewards"], "Reward", colors["reward"]),
    ]:
        fig2, ax = plt.subplots(figsize=(10, 4))
        if name == "root_height_tracking":
            ax.plot(m["time_s"], m["sim_z"], color=colors["sim"], linewidth=1.5, label="Sim")
            ax.plot(m["time_s"], m["ref_z"], color=colors["ref"], linewidth=1.5, linestyle="--", label="Ref")
            ax.legend()
            ax.set_ylabel("Height (m)")
        else:
            ax.plot(m["time_s"], ydata, color=color, linewidth=1.0)
            ax.axhline(np.mean(ydata), color=color, linestyle="--", alpha=0.5)
            ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        p = os.path.join(output_dir, f"{name}.png")
        fig2.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {p}")

    # Summary text
    txt_path = os.path.join(output_dir, "summary_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("H1 Dance Motion Imitation — Policy Evaluation\n")
        f.write("=" * 50 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nPer-joint MAE (degrees):\n")
        for jn, sn in zip(JOINT_NAMES, SHORT_NAMES):
            f.write(f"  {sn:15s}  {m['joint_mae_deg'][jn]:.2f}°\n")
    print(f"Saved: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate H1 dance policy from rollout CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to policy_vs_reference.csv")
    parser.add_argument("--output", type=str, default="eval_plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.csv}")
    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} steps ({len(rows)/30:.1f} seconds)")

    m = compute_metrics(rows)
    plot_dashboard(m, args.output)
    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
