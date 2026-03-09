"""
Multimodal evaluation script for PushT.

Runs N rollouts from the SAME initial state and visualizes all agent
trajectories overlaid on the environment's rendered initial frame.

Usage:
    python flare/eval_multimodal.py policy=a2a task=pusht \
        checkpoint_dir=flare_outputs/lerobot_pusht/a2a/test/checkpoints

    python flare/eval_multimodal.py policy=a2a_noise task=pusht \
        checkpoint_dir=flare_outputs/lerobot_pusht/a2a_noise/test/checkpoints \
        +n_rollouts=20
"""

import torch
import hydra
import logging
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig

from flare.factory import get_policy_class
from flare.utils.checkpoints import get_best_checkpoint, get_latest_checkpoint, load_model_weights
from flare.utils.dataset_utils import create_dataset_stats
from flare.utils.eval import create_eval_env, rollout
from flare.utils.utils import write_video

logger = logging.getLogger(__name__)

# PushT environment is 512x512. Coordinate system: (0,0) top-left, y down.
#
# Classic Diffusion Policy multimodal test layout:
# All three objects on the SAME diagonal line (y = -x + 512):
#   - Agent (blue):     lower-left   (156, 356)
#   - Goal T (green):   center       (256, 256), angle pi/4
#   - Initial T (gray): upper-right  (306, 156), angle pi/4
#
# The agent is at lower-left, must go around the T to push it into the goal.
INITIAL_STATE = np.array([156.0, 356.0, 306.0, 157.0, np.pi / 4])
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])


def load_policy(cfg):
    """Load policy and checkpoint weights."""
    dataset_meta, stats = create_dataset_stats(cfg)
    policy_cls = get_policy_class(cfg.policy.name)
    policy = policy_cls(cfg, stats)

    if cfg.checkpoint_path:
        checkpoint_path = Path(cfg.checkpoint_path)
    else:
        checkpoint_path = get_best_checkpoint(cfg.checkpoint_dir)
        if checkpoint_path is None:
            checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found.")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    load_model_weights(policy, checkpoint_path, cfg.device)
    policy.to(cfg.device)
    policy.eval()
    return policy, checkpoint_path


def run_multimodal_eval(cfg, policy, n_rollouts, output_dir):
    """Run N rollouts from the same initial state and collect trajectories + videos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_eval_env(
        eval_n_envs=1,
        env_package=cfg.task.env_package,
        env_name=cfg.task.env_name,
        env_kwargs=cfg.task.env_kwargs,
    )
    fps = env.unwrapped.metadata.get("render_fps", 10)

    all_trajectories = []
    all_successes = []
    initial_frame = None

    for i in range(n_rollouts):
        logger.info(f"Rollout {i+1}/{n_rollouts}")

        # Reset torch seeds so policy internals are identical each run
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        # Reset environment to our fixed initial state using the official API
        policy.reset()
        observation, info = env.envs[0].reset(
            options={"reset_to_state": INITIAL_STATE.tolist()}
        )

        # Capture the environment's rendered initial frame (first rollout only)
        if initial_frame is None:
            initial_frame = env.envs[0].render()

        ep_frames = []
        all_states = []
        all_rewards_ep = []
        all_successes_ep = []
        done = False

        max_steps = env.envs[0].spec.max_episode_steps or 300
        frame = env.envs[0].render()
        ep_frames.append(frame)

        def add_batch_dim(obs):
            """Add batch dimension to single-env observation for preprocess_observation."""
            if isinstance(obs, dict):
                return {k: np.expand_dims(v, 0) if isinstance(v, np.ndarray) else v
                        for k, v in obs.items()}
            return np.expand_dims(obs, 0)

        for step in range(max_steps):
            if done:
                break

            # Preprocess observation (needs batch dim)
            from flare.utils.policy_utils import preprocess_observation
            batched_obs = add_batch_dim(observation)
            proc_obs = preprocess_observation(batched_obs)

            # Record agent position (shape: (1, state_dim) -> (state_dim,))
            state_val = proc_obs[cfg.task.state_key][0]
            all_states.append(state_val.cpu().numpy() if isinstance(state_val, torch.Tensor) else state_val)

            # Move to device (already has batch dim from preprocess)
            device_obs = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                          for k, v in proc_obs.items()}

            with torch.inference_mode():
                action = policy.select_action(device_obs)

            action_np = action.squeeze(0).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()

            observation, reward, terminated, truncated, info = env.envs[0].step(action_np)
            frame = env.envs[0].render()
            ep_frames.append(frame)

            all_rewards_ep.append(reward)
            success = info.get("is_success", False)
            all_successes_ep.append(success)
            done = terminated or truncated

        # Collect final state
        batched_obs = add_batch_dim(observation)
        proc_obs = preprocess_observation(batched_obs)
        state_val = proc_obs[cfg.task.state_key][0]
        all_states.append(state_val.cpu().numpy() if isinstance(state_val, torch.Tensor) else state_val)

        traj = np.stack(all_states)  # (T, state_dim)
        traj_xy = traj[:, :2]
        all_trajectories.append(traj_xy)

        success = any(all_successes_ep)
        all_successes.append(success)
        logger.info(f"  Success: {success}, Steps: {len(all_states)}")

        # Save video
        if len(ep_frames) > 0:
            video_path = output_dir / f"rollout_{i}.mp4"
            frames = np.stack(ep_frames)
            write_video(str(video_path), frames, fps)
            logger.info(f"  Video saved: {video_path}")

    env.close()

    # Plot all trajectories with environment render as background
    plot_trajectories(all_trajectories, all_successes, n_rollouts, output_dir,
                      initial_frame)

    # Diversity metrics
    log_diversity_metrics(all_trajectories)

    return all_trajectories, all_successes


def plot_trajectories(trajectories, successes, n_rollouts, output_dir,
                      initial_frame):
    """Plot trajectories overlaid on the environment's rendered initial frame.
    Uses the actual PushT render as background for pixel-perfect accuracy."""
    n = len(trajectories)
    h, w = initial_frame.shape[:2]
    # PushT world is 512x512; rendered frame may be different size (e.g. 680x680)
    world_size = 512.0
    scale_x = w / world_size
    scale_y = h / world_size

    fig, ax = plt.subplots(figsize=(7, 7))

    # Use environment render as background (shows T-blocks, goal, agent exactly as pymunk draws them)
    ax.imshow(initial_frame, extent=[0, w, h, 0], zorder=0)

    # Draw trajectories with varying colors, scaling from world coords to pixel coords
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, n))
    for i, (traj, success) in enumerate(zip(trajectories, successes)):
        ax.plot(traj[:, 0] * scale_x, traj[:, 1] * scale_y, "-", color=colors[i],
                alpha=0.8, linewidth=2.5, zorder=3, solid_capstyle="round")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # y-axis flipped: (0,0) top-left, matching PushT coords
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.5)

    save_path = output_dir / "trajectory_overlay.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Trajectory plot saved: {save_path}")


def log_diversity_metrics(trajectories):
    """Log diversity metrics across trajectories."""
    if len(trajectories) < 2:
        return

    max_len = max(len(t) for t in trajectories)
    padded = []
    for t in trajectories:
        if len(t) < max_len:
            pad = np.tile(t[-1:], (max_len - len(t), 1))
            padded.append(np.concatenate([t, pad], axis=0))
        else:
            padded.append(t)
    padded = np.stack(padded)  # (N, T, 2)

    n = len(padded)
    pairwise_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(padded[i] - padded[j], axis=1).mean()
            pairwise_dists.append(dist)

    mean_pairwise = np.mean(pairwise_dists)
    std_endpoints = np.std(padded[:, -1, :], axis=0).mean()

    logger.info(f"Diversity metrics:")
    logger.info(f"  Mean pairwise trajectory distance: {mean_pairwise:.4f}")
    logger.info(f"  Endpoint std (x,y avg): {std_endpoints:.4f}")
    if mean_pairwise < 1e-3:
        logger.info(f"  => Trajectories are nearly identical (deterministic policy)")
    else:
        logger.info(f"  => Trajectories show diversity (multimodal behavior)")


@hydra.main(config_path="configs", config_name="default_policy", version_base="1.3")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Starting multimodal evaluation...")

    n_rollouts = cfg.get("n_rollouts", 10)

    # Set global seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    output_dir = Path(cfg.output_dir) / "multimodal_eval"

    policy, ckpt_path = load_policy(cfg)
    logger.info(f"Using checkpoint: {ckpt_path}")
    logger.info(f"N rollouts: {n_rollouts}")
    logger.info(f"Initial state: agent=({INITIAL_STATE[0]}, {INITIAL_STATE[1]}), "
                f"block=({INITIAL_STATE[2]}, {INITIAL_STATE[3]}, angle={INITIAL_STATE[4]:.2f})")
    logger.info(f"Goal: ({GOAL_POSE[0]}, {GOAL_POSE[1]}, angle={GOAL_POSE[2]:.2f})")

    trajectories, successes = run_multimodal_eval(cfg, policy, n_rollouts, output_dir)

    n_success = sum(successes)
    logger.info(f"Success rate: {n_success}/{len(successes)} ({100*n_success/len(successes):.1f}%)")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Multimodal evaluation completed!")


if __name__ == "__main__":
    main()