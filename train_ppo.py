"""
train_ppo.py — Train a PPO Agent to Navigate a 2D Drone Environment
======================================================================

WHAT IS PPO?
------------
Proximal Policy Optimization (PPO) is a policy-gradient reinforcement
learning algorithm developed by OpenAI (2017).  It is the industry
default for continuous-control and robotics tasks because it:

  • Is stable to train (small trust-region updates)
  • Works with discrete AND continuous action spaces
  • Has few hyperparameters to tune compared to SAC or TD3
  • Parallelises easily across multiple CPU environments

PPO CORE IDEA
-------------
The agent has two neural networks:

  Actor  (policy π)  — maps observation → action probabilities
  Critic (value V)   — maps observation → expected total reward

At each iteration PPO:
  1. Collects n_steps of experience by running the current policy
  2. Estimates advantages: A(s,a) = actual_return - V(s)
  3. Updates the policy using a clipped surrogate objective so no single
     update changes the policy too drastically (the "proximal" part)
  4. Discards the experience and repeats

WHY PPO FOR THIS TASK?
----------------------
  • Simple discrete action space (4 actions) — PPO handles this well
  • Dense reward signal from our shaped reward function
  • Environment is deterministic (no sensor noise), which helps PPO converge
  • Training on CPU is fast enough for a 40×40 grid

HOW TO RUN
----------
    cd rl_drone_path_planning
    python train_ppo.py

Outputs saved to:
    models/drone_ppo_final       — final trained model
    models/drone_ppo_best/       — best checkpoint during training
    outputs/training_rewards.png — learning curve plot
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Stable-Baselines3 imports
# ---------------------------------------------------------------------------
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.results_plotter import load_results, ts2xy

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.drone_env import DroneNavEnv

# ===========================================================================
#  Configuration — change these to experiment
# ===========================================================================

MAP_PATH  = os.path.join(PROJECT_ROOT, "maps", "training_map.pgm")
START_POS = (3, 3)
GOAL_POS  = (35, 35)
MAX_STEPS = 1000

# PPO hyperparameters
TOTAL_TIMESTEPS    = 600_000   # more steps — harder task with random obstacles
N_ENVS             = 4         # parallel environments (speeds up data collection)
N_STEPS            = 2048      # longer rollouts → better value estimates
BATCH_SIZE         = 512       # N_STEPS × N_ENVS / 16
N_EPOCHS           = 10        # gradient update passes per iteration
LEARNING_RATE      = 2.5e-4    # RATE - too low -train slow , too high - oscillations and policy fails (default is adam's coeff) - Range {1e-5 to 3e-4}
GAMMA              = 0.995     # Discount Factor - (higher discount → long-range goal stays relevant) - Range {0 to 1}
ENT_COEF           = 0.02      # Exploration - (more entropy → more exploration around obstacles) - Range {0 to 0.1}
CLIP_RANGE         = 0.2       # PPO clipping parameter (the "proximal" constraint) - Range{0.05 – 0.4}

# Domain randomisation — random obstacles scattered each episode during training.
# Teaches the agent to navigate around unseen obstacles at test time.
# 0 = disabled (trains on clean map only).
N_RANDOM_OBSTACLES = 5

# Save / log paths
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "outputs", "logs")
OUTPUTS_DIR     = os.path.join(PROJECT_ROOT, "outputs")


# ===========================================================================
#  Reward-logging callback
# ===========================================================================

class RewardLoggerCallback(BaseCallback):
    """
    Custom callback that prints training progress every N timesteps.

    Stable-Baselines3 provides BaseCallback so you can hook into the
    training loop without modifying library code.  This is how you would
    add custom logging, early stopping, curriculum learning, etc.
    """

    def __init__(self, print_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self._last_print = 0

    def _on_step(self) -> bool:
        """Called after every environment step.  Return False to stop training."""
        if self.num_timesteps - self._last_print >= self.print_every:
            # Access the Monitor wrapper's episode info
            ep_rewards = [
                ep["r"]
                for ep in self.model.ep_info_buffer
                if "r" in ep
            ]
            if ep_rewards:
                mean_r  = np.mean(ep_rewards)
                best_r  = np.max(ep_rewards)
                pct     = 100 * self.num_timesteps / TOTAL_TIMESTEPS
                print(
                    f"  [{pct:5.1f}%]  step {self.num_timesteps:>8,}  |  "
                    f"mean_ep_reward = {mean_r:7.2f}  |  "
                    f"best_ep_reward = {best_r:7.2f}"
                )
            self._last_print = self.num_timesteps
        return True   # True = continue training


# ===========================================================================
#  Environment factory
# ===========================================================================

def make_env(rank: int = 0, seed: int = 0):
    """
    Factory function that creates one wrapped training environment.

    We use Monitor to record episode rewards and lengths so we can plot
    the learning curve later.  Each parallel env gets a unique seed so
    the random number generators don't all produce identical sequences.
    """
    def _init():
        env = DroneNavEnv(
            map_path            = MAP_PATH,
            start_pos           = START_POS,
            goal_pos            = GOAL_POS,
            max_steps           = MAX_STEPS,
            n_random_obstacles  = N_RANDOM_OBSTACLES,
        )
        log_dir = os.path.join(LOGS_DIR, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        env.reset(seed=seed + rank)
        return env

    return _init


# ===========================================================================
#  Plot learning curve
# ===========================================================================

def plot_learning_curve(log_dirs: list, save_path: str):
    """
    Aggregate Monitor logs from all parallel environments and plot the
    episodic reward over training steps.

    The rolling mean window (100 episodes) smooths out the noise so
    you can see the learning trend clearly.
    """
    all_steps, all_rewards = [], []

    for log_dir in log_dirs:
        monitor_file = os.path.join(log_dir, "monitor.csv")
        if not os.path.exists(monitor_file):
            continue
        try:
            x, y = ts2xy(load_results(log_dir), "timesteps")
            all_steps.extend(x.tolist())
            all_rewards.extend(y.tolist())
        except Exception:
            pass

    if not all_steps:
        print("  No monitor data found — skipping plot.")
        return

    # Sort by timestep
    pairs = sorted(zip(all_steps, all_rewards))
    steps   = np.array([p[0] for p in pairs])
    rewards = np.array([p[1] for p in pairs])

    # Rolling mean
    window  = 100
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        steps_s = steps[window - 1:]
    else:
        smooth  = rewards
        steps_s = steps

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, alpha=0.2, color="#90CAF9", label="episode reward")
    ax.plot(steps_s, smooth, color="#1565C0", linewidth=2,
            label=f"rolling mean ({window} eps)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title("PPO Learning Curve — Drone Navigation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Learning curve saved: {save_path}")
    plt.close()


# ===========================================================================
#  Main training routine
# ===========================================================================

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  RL Drone Path Planning — PPO Training")
    print("=" * 60)
    print(f"  Map:               {MAP_PATH}")
    print(f"  Start:             {START_POS}  →  Goal: {GOAL_POS}")
    print(f"  Timesteps:         {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs:     {N_ENVS}")
    print(f"  Random obstacles:  {N_RANDOM_OBSTACLES} per episode (domain randomisation)")
    print(f"  Network:           MLP [128, 128]")
    print()

    # ------------------------------------------------------------------
    # Build vectorised training environment
    # ------------------------------------------------------------------
    # make_vec_env creates N_ENVS copies of the env and runs them in
    # parallel on separate threads, multiplying data throughput.
    print("Creating training environments …")
    train_env = make_vec_env(
        make_env(rank=0, seed=42),
        n_envs=N_ENVS,
        seed=42,
    )

    # Separate evaluation environment (no Monitor wrapper noise)
    eval_env = Monitor(
        DroneNavEnv(
            map_path  = MAP_PATH,
            start_pos = START_POS,
            goal_pos  = GOAL_POS,
            max_steps = MAX_STEPS,
        )
    )

    # ------------------------------------------------------------------
    # Define callbacks
    # ------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODELS_DIR, "drone_ppo_best"),
        log_path             = os.path.join(LOGS_DIR,   "eval"),
        eval_freq            = max(10_000 // N_ENVS, 1),
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 0,
    )

    reward_logger = RewardLoggerCallback(print_every=20_000)

    # ------------------------------------------------------------------
    # Instantiate PPO
    # ------------------------------------------------------------------
    # "MlpPolicy" = Multi-Layer Perceptron (plain feed-forward neural net)
    # Suitable for our 8-dimensional observation vector.
    # For image observations you would use "CnnPolicy" instead.
    print("Initialising PPO …")
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        learning_rate = LEARNING_RATE,
        gamma         = GAMMA,
        ent_coef      = ENT_COEF,
        clip_range    = CLIP_RANGE,
        policy_kwargs = dict(net_arch=[128, 128]),  # wider than default [64,64]
        verbose       = 0,           # set to 1 for SB3 built-in logging
        tensorboard_log = os.path.join(LOGS_DIR, "tensorboard"),
    )

    print(f"\nPolicy network architecture:")
    print(f"  {model.policy}\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps …")
    print(f"  (This may take a few minutes on CPU)\n")

    t0 = time.time()
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, reward_logger],
        progress_bar    = True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  "
          f"({TOTAL_TIMESTEPS / elapsed:.0f} steps/sec)")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_path = os.path.join(MODELS_DIR, "drone_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}.zip")

    # ------------------------------------------------------------------
    # Plot learning curve
    # ------------------------------------------------------------------
    log_dirs = [os.path.join(LOGS_DIR, f"env_{i}") for i in range(N_ENVS)]
    plot_learning_curve(
        log_dirs,
        os.path.join(OUTPUTS_DIR, "training_rewards.png"),
    )

    # ------------------------------------------------------------------
    # Quick evaluation of the saved model
    # ------------------------------------------------------------------
    print("\nQuick evaluation (50 episodes) …")
    successes, lengths, rewards = 0, [], []

    obs, _ = eval_env.reset()
    ep_reward, ep_length = 0.0, 0

    for _ in range(50 * MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        ep_reward += reward
        ep_length += 1

        if terminated or truncated:
            if terminated:
                successes += 1
            lengths.append(ep_length)
            rewards.append(ep_reward)
            ep_reward, ep_length = 0.0, 0
            obs, _ = eval_env.reset()

            if len(rewards) >= 50:
                break

    if rewards:
        print(f"  Success rate:  {successes}/{len(rewards)}  "
              f"({100*successes/len(rewards):.0f}%)")
        print(f"  Mean reward:   {np.mean(rewards):.2f}")
        print(f"  Mean length:   {np.mean(lengths):.1f} steps")

    train_env.close()
    eval_env.close()
    print("\nDone. Run  python test_rl.py  to see the agent in action.")


# ===========================================================================
#  Entry point
# ===========================================================================
if __name__ == "__main__":
    train()
