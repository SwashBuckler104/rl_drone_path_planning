"""
student_exercises/train_ppo_template.py
=========================================
STUDENT EXERCISE 3 — Train a PPO Agent

Your task is to wire up a PPO training loop using Stable-Baselines3.

LEARNING GOALS
--------------
  1. Understand how to wrap a custom environment for SB3.
  2. Understand key PPO hyperparameters and what they control.
  3. Use callbacks for evaluation and checkpointing.
  4. Load and run inference on a trained model.

BEFORE YOU CODE — KEY CONCEPTS
--------------------------------

  Policy Network (Actor):
    Maps observation (8 floats) → action probabilities (4 actions).
    During training: samples actions proportionally to probabilities.
    During inference: selects the highest-probability action (deterministic=True).

  Value Network (Critic):
    Maps observation → expected total future reward V(s).
    Used to compute advantage: A(s,a) = actual_return - V(s)
    Positive advantage → action was better than expected → increase its probability.

  Key Hyperparameters to understand:
    n_steps       — how many steps each env collects before an update
    batch_size    — mini-batch size for gradient updates (must divide n_steps * n_envs)
    n_epochs      — how many times to reuse the collected data per update
    gamma         — discount factor (0=greedy, 1=long-horizon)
    ent_coef      — entropy bonus (higher → more exploration)
    clip_range    — the "proximal" constraint (0.1-0.3 typical)

REFERENCE
---------
Compare your solution with  train_ppo.py  after trying yourself.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor   import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from envs.drone_env import DroneNavEnv

# ===========================================================================
#  Configuration — fill these in
# ===========================================================================

MAP_PATH        = os.path.join(PROJECT_ROOT, "maps", "training_map.pgm")
START_POS       = (2, 2)
GOAL_POS        = (37, 37)
MAX_STEPS       = 1000
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR     = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "outputs", "logs")


# ===========================================================================
#  PART A — Create the environment
# ===========================================================================

def make_training_env():
    """
    Create and wrap a single DroneNavEnv for training.

    The Monitor wrapper records episode rewards and lengths so we can
    plot the learning curve later.
    """
    # ------------------------------------------------------------------
    # TODO 1 — Create a DroneNavEnv with the config above, wrap with Monitor
    # ------------------------------------------------------------------
    # env = DroneNavEnv(map_path=..., start_pos=..., goal_pos=..., max_steps=...)
    # os.makedirs(LOGS_DIR, exist_ok=True)
    # env = Monitor(env, LOGS_DIR)
    # return env
    # ------------------------------------------------------------------
    pass  # TODO


def make_eval_env():
    """Create a separate environment for evaluation during training."""
    # ------------------------------------------------------------------
    # TODO 2 — Create a DroneNavEnv wrapped with Monitor (separate log dir)
    # ------------------------------------------------------------------
    pass  # TODO


# ===========================================================================
#  PART B — Define PPO hyperparameters
# ===========================================================================

PPO_HYPERPARAMS = {
    # ------------------------------------------------------------------
    # TODO 3 — Fill in reasonable PPO hyperparameters
    # ------------------------------------------------------------------
    # Hint: start with the defaults below and experiment.
    #
    # "n_steps":      1024,    # steps collected per env per iteration
    # "batch_size":   256,     # must divide (n_steps * n_envs)
    # "n_epochs":     10,      # gradient passes per collected batch
    # "learning_rate": 3e-4,   # Adam learning rate
    # "gamma":        0.99,    # discount factor
    # "ent_coef":     0.01,    # entropy bonus (exploration)
    # "clip_range":   0.2,     # PPO clipping threshold
    # ------------------------------------------------------------------
}


# ===========================================================================
#  PART C — Training loop
# ===========================================================================

def train(total_timesteps: int = 200_000, n_envs: int = 2):
    """
    Train a PPO agent on the drone navigation task.

    Args:
        total_timesteps: how many environment steps to train for
        n_envs:          number of parallel environments
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"Training PPO for {total_timesteps:,} timesteps …")

    # ------------------------------------------------------------------
    # TODO 4 — Create vectorised training environment
    # ------------------------------------------------------------------
    # make_vec_env takes a callable that returns one env.
    # It creates n_envs copies of it.
    #
    # train_env = make_vec_env(make_training_env, n_envs=n_envs)
    # ------------------------------------------------------------------
    train_env = None  # TODO

    # ------------------------------------------------------------------
    # TODO 5 — Create the PPO model
    # ------------------------------------------------------------------
    # Use "MlpPolicy" (Multi-Layer Perceptron).
    # Pass train_env and PPO_HYPERPARAMS.
    #
    # model = PPO("MlpPolicy", train_env, verbose=1, **PPO_HYPERPARAMS)
    # ------------------------------------------------------------------
    model = None  # TODO

    # ------------------------------------------------------------------
    # TODO 6 — Create an EvalCallback
    # ------------------------------------------------------------------
    # eval_env     = make_eval_env()
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path = os.path.join(MODELS_DIR, "best"),
    #     eval_freq  = 10_000,
    #     n_eval_episodes = 10,
    #     deterministic = True,
    #     verbose = 0,
    # )
    # ------------------------------------------------------------------
    eval_callback = None  # TODO

    # ------------------------------------------------------------------
    # TODO 7 — Train the model
    # ------------------------------------------------------------------
    # model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 8 — Save the final model
    # ------------------------------------------------------------------
    # model.save(os.path.join(MODELS_DIR, "drone_ppo_student"))
    # print("Model saved!")
    # ------------------------------------------------------------------

    if train_env is not None:
        train_env.close()
    return model


# ===========================================================================
#  PART D — Evaluate the trained model
# ===========================================================================

def evaluate(model, n_episodes: int = 10):
    """
    Run n_episodes with the trained model and print the success rate.

    Args:
        model:      Trained PPO model
        n_episodes: Number of test episodes
    """
    if model is None:
        print("  No model provided — skipping evaluation.")
        return

    env = DroneNavEnv(
        map_path  = MAP_PATH,
        start_pos = START_POS,
        goal_pos  = GOAL_POS,
        max_steps = MAX_STEPS,
    )

    successes, steps_list = 0, []
    print(f"\nEvaluating model over {n_episodes} episodes …")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(MAX_STEPS):
            # ------------------------------------------------------------------
            # TODO 9 — Use model.predict() to get the action
            # ------------------------------------------------------------------
            # action, _ = model.predict(obs, deterministic=True)
            # obs, reward, terminated, truncated, _ = env.step(action)
            # ep_reward += reward
            # if terminated or truncated:
            #     if terminated: successes += 1
            #     steps_list.append(env.step_count)
            #     break
            # ------------------------------------------------------------------
            pass  # TODO

    if steps_list:
        print(f"  Success rate: {successes}/{n_episodes} "
              f"({100*successes/n_episodes:.0f}%)")
        print(f"  Mean steps:   {np.mean(steps_list):.1f}")
    else:
        print("  No episodes completed — check TODO 9.")

    env.close()


# ===========================================================================
#  Entry point
# ===========================================================================
if __name__ == "__main__":
    # Quick training run to verify your implementation
    model = train(total_timesteps=50_000, n_envs=2)
    evaluate(model, n_episodes=10)

    print("\nIf everything worked, run the full training script:")
    print("  python train_ppo.py")
