"""
test_rl.py — Test the Trained RL Agent with a Dynamic Obstacle
================================================================

THE CORE EXPERIMENT
-------------------
This script demonstrates the key educational difference between A* and RL:

  Step 1.  The trained RL agent begins navigating from start to goal.
  Step 2.  After a configurable number of steps, a NEW obstacle is
           injected onto the planned path — something the agent was
           NOT trained to handle.
  Step 3.  The agent's sensor readings (distances) change immediately
           when the obstacle appears.
  Step 4.  Because the RL policy maps observations → actions, it can
           REACT to the new sensor readings without any explicit
           replanning step.
  Step 5.  A* is run on the same modified map to show that, without
           replanning, its stored path is now INVALID.

KEY INSIGHT
-----------
  A* generates a plan once and follows it blindly.  If the environment
  changes, the plan must be recomputed from scratch.

  The RL agent has no explicit plan — it reacts to whatever it currently
  observes.  This makes it naturally adaptive, but it may take a suboptimal
  detour and success is not guaranteed.

  Neither approach is universally better.  They have complementary strengths.

HOW TO RUN
----------
    cd rl_drone_path_planning
    python test_rl.py

    # To test without the dynamic obstacle:
    python test_rl.py --no-obstacle

    # To run more episodes:
    python test_rl.py --n-episodes 5

PREREQUISITES
-------------
    python train_ppo.py   # must be run first to produce the model
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend for saved figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from envs.drone_env    import DroneNavEnv
from astar.astar_planner import AStarPlanner

# ===========================================================================
#  Configuration
# ===========================================================================

MAP_PATH   = os.path.join(PROJECT_ROOT, "maps", "training_map.pgm")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "drone_ppo_final")
START_POS = (3, 3)
GOAL_POS  = (35, 35)
MAX_STEPS = 1000

# Dynamic obstacle config — format: (row, col, inject_step)
# Each obstacle has its own step at which it appears.  Add as many as you like.
DYNAMIC_OBSTACLES = [
    (3, 20, 5), 
    (4, 20, 2),      
    (15, 3, 10), 
    (20, 3, 10),
    (15, 32, 20), 
    (20, 31, 20),
    (35, 33, 50),
    (20, 33, 20),
    (26, 35, 20),  
    (30, 34, 50),
    (32, 35, 50),
    (28, 34, 50),     # Fails seems exploration is lacking
]

# A* replanning: if True, A* is re-run on the map with all dynamic obstacles added,
# showing the detour it would take.  If False, A* keeps its original path — which
# will now pass through blocked cells, illustrating what happens without replanning.
ASTAR_REPLAN = True

# GIF animation: skip if the trajectory is longer than this many steps (OOM protection).
# Set to 0 to always skip GIF generation.
MAX_GIF_STEPS = 300

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Set to True to generate a V(s) heatmap after the test episodes.
# The PPO critic scores every free cell — warm = agent expects high reward,
# cool = agent expects to struggle.  Saved as outputs/reward_map.png.
SHOW_REWARD_MAP = True


# ===========================================================================
#  Run one RL episode (with optional dynamic obstacle)
# ===========================================================================

def run_episode(
    model:           PPO,
    env:             DroneNavEnv,
    inject_obstacle: bool = True,
    verbose:         bool = True,
) -> dict:
    """
    Run one episode of the RL agent and record the trajectory.

    Args:
        model:            Trained PPO model.
        env:              DroneNavEnv instance.
        inject_obstacle:  Whether to add a dynamic obstacle mid-episode.
        verbose:          Print step-by-step commentary.

    Returns:
        dict with keys: trajectory, success, steps, total_reward,
                        obstacle_injected_at, obstacle_pos
    """
    obs, _ = env.reset()
    total_reward       = 0.0
    first_inject_step  = None                  # step at which the first obstacle appeared
    placed_positions   = []                    # (row, col) of all obstacles placed so far
    pending            = list(DYNAMIC_OBSTACLES)  # obstacles not yet injected

    if verbose:
        print(f"\n  Episode start: drone at {START_POS}, goal at {GOAL_POS}")

    for step in range(MAX_STEPS):

        # ---- Dynamic obstacle injection ----
        if inject_obstacle and pending:
            newly_placed = []
            still_pending = []
            for (obr, obc, obstep) in pending:
                if step == obstep:
                    if env.add_obstacle(obr, obc):
                        newly_placed.append((obr, obc))
                        placed_positions.append((obr, obc))
                else:
                    still_pending.append((obr, obc, obstep))
            pending = still_pending

            if newly_placed:
                if first_inject_step is None:
                    first_inject_step = step
                if verbose:
                    print(f"\n  *** {len(newly_placed)} DYNAMIC OBSTACLE(S) injected"
                          f" at step {step}: {newly_placed} ***")
                    print(f"      Drone is at {tuple(env.drone_pos)}.  "
                          f"Sensor readings will update on next step.")

        # ---- Agent acts ----
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if verbose and step < 5:
            print(f"  step {step:3d}: pos={tuple(env.drone_pos)}  "
                  f"action={['UP','DOWN','LEFT','RIGHT'][int(action)]}  "
                  f"reward={reward:+.3f}")

        if verbose and step == 5:
            print("  … (suppressing step-by-step output)")

        if terminated:
            if verbose:
                print(f"\n  SUCCESS in {step+1} steps!  "
                      f"Total reward = {total_reward:.2f}")
            return {
                "trajectory":           list(env.trajectory),
                "success":              True,
                "steps":                step + 1,
                "total_reward":         total_reward,
                "obstacle_injected_at": first_inject_step,
                "obstacle_positions":   placed_positions,
            }

        if truncated:
            if verbose:
                print(f"\n  TIMEOUT after {step+1} steps.  "
                      f"Total reward = {total_reward:.2f}")
            break

    return {
        "trajectory":           list(env.trajectory),
        "success":              False,
        "steps":                env.step_count,
        "total_reward":         total_reward,
        "obstacle_injected_at": first_inject_step,
        "obstacle_positions":   placed_positions,
    }


# ===========================================================================
#  Visualisation
# ===========================================================================

def visualise_episode(
    grid:        np.ndarray,
    rl_result:   dict,
    astar_path:  list,
    astar_path_blocked: list,
    save_path:   str,
):
    """
    Create a two-panel figure comparing:
      Left  — RL agent trajectory + dynamic obstacle
      Right — A* path: original (blocked) vs replanned

    Colour key:
      Dark grey  — obstacle (wall)
      White      — free space
      Sky-blue   — RL trajectory
      Red dot    — drone (RL) final position
      Blue line  — A* original path
      Orange     — A* replanned path (after obstacle added)
      Bright red — dynamic obstacle
      Green star — goal
      Green dot  — start
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Dynamic Obstacle Experiment: RL (reactive) vs A* (plan-based)",
        fontsize=13, fontweight="bold"
    )

    for ax in axes:
        # Draw base map
        bg = np.where(grid == 0, 240, 40).astype(np.uint8)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, origin="upper",
                  interpolation="nearest", aspect="equal")

    # ------- Left panel: RL agent -------
    ax_rl = axes[0]
    ax_rl.set_title("RL Agent  (reacts to obstacle via sensor readings)",
                    fontsize=10)

    # Dynamic obstacles (one label in legend, one patch per obstacle)
    obstacle_positions = rl_result["obstacle_positions"]
    for i, (drow, dcol) in enumerate(obstacle_positions):
        ax_rl.add_patch(plt.Rectangle(
            (dcol - 0.5, drow - 0.5), 1, 1,
            facecolor="#F44336", edgecolor="darkred", linewidth=1.5, zorder=3,
            label="Dynamic obstacle" if i == 0 else "_nolegend_",
        ))
        ax_rl.text(dcol, drow, "✕", ha="center", va="center",
                   fontsize=9, color="white", fontweight="bold", zorder=4)

    # RL trajectory
    traj = rl_result["trajectory"]
    if len(traj) > 1:
        tr = [p[0] for p in traj]
        tc = [p[1] for p in traj]
        ax_rl.plot(tc, tr, "-o", color="#2196F3",
                   markersize=2, linewidth=1.5, label="RL trajectory", zorder=2)

    # Marker for injection point
    inj = rl_result["obstacle_injected_at"]
    if inj is not None and inj < len(traj):
        ax_rl.plot(traj[inj][1], traj[inj][0], "D",
                   color="#FF9800", markersize=8, zorder=5,
                   label=f"Obstacle appeared\n(step {inj})")

    # Start / goal / final
    ax_rl.plot(START_POS[1], START_POS[0], "o",
               color="#4CAF50", markersize=12, markeredgecolor="white",
               markeredgewidth=1.5, label="Start", zorder=6)
    ax_rl.plot(GOAL_POS[1],  GOAL_POS[0],  "*",
               color="#FFC107", markersize=16, markeredgecolor="white",
               markeredgewidth=1.5, label="Goal", zorder=6)
    ax_rl.plot(traj[-1][1], traj[-1][0], "^",
               color="#9C27B0", markersize=10, markeredgecolor="white",
               markeredgewidth=1.5, label="Drone (end)", zorder=7)

    outcome = "SUCCESS ✓" if rl_result["success"] else "TIMEOUT ✗"
    ax_rl.set_xlabel(
        f"{outcome}  |  Steps: {rl_result['steps']}  |  "
        f"Reward: {rl_result['total_reward']:.1f}",
        fontsize=9
    )
    ax_rl.legend(loc="upper right", fontsize=7)
    ax_rl.axis("off")

    # ------- Right panel: A* comparison -------
    ax_as = axes[1]
    ax_as.set_title("A* Planner  (original plan becomes invalid after obstacle)",
                    fontsize=10)

    # Draw dynamic obstacles on A* panel too
    obstacle_set = set(map(tuple, obstacle_positions))
    for drow, dcol in obstacle_positions:
        ax_as.add_patch(plt.Rectangle(
            (dcol - 0.5, drow - 0.5), 1, 1,
            facecolor="#F44336", edgecolor="darkred", linewidth=1.5, zorder=3,
        ))
        ax_as.text(dcol, drow, "✕", ha="center", va="center",
                   fontsize=9, color="white", fontweight="bold", zorder=4)

    # Original A* path (now blocked)
    if astar_path:
        tr = [p[0] for p in astar_path]
        tc = [p[1] for p in astar_path]
        ax_as.plot(tc, tr, "--", color="#B0BEC5",
                   linewidth=2, label="Original A* path (blocked)", zorder=2)
        # Mark where it hits any dynamic obstacle
        for p in astar_path:
            if tuple(p) in obstacle_set:
                ax_as.plot(p[1], p[0], "x",
                           color="#F44336", markersize=12, markeredgewidth=3,
                           label="Path blocked here", zorder=5)
                break

    # Replanned A* path
    if astar_path_blocked:
        tr = [p[0] for p in astar_path_blocked]
        tc = [p[1] for p in astar_path_blocked]
        ax_as.plot(tc, tr, "-", color="#FF9800",
                   linewidth=2.5, label="A* replanned path", zorder=3)

    # Start / goal
    ax_as.plot(START_POS[1], START_POS[0], "o",
               color="#4CAF50", markersize=12, markeredgecolor="white",
               markeredgewidth=1.5, label="Start", zorder=6)
    ax_as.plot(GOAL_POS[1],  GOAL_POS[0],  "*",
               color="#FFC107", markersize=16, markeredgecolor="white",
               markeredgewidth=1.5, label="Goal", zorder=6)

    original_len  = len(astar_path)        if astar_path         else 0
    replanned_len = len(astar_path_blocked) if astar_path_blocked else 0
    ax_as.set_xlabel(
        f"Original path: {original_len} cells  |  "
        f"Replanned path: {replanned_len} cells\n"
        f"(A* must be re-run to find the new path)",
        fontsize=9
    )
    ax_as.legend(loc="upper right", fontsize=7)
    ax_as.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure: {save_path}")
    plt.close()


def animate_episode(
    grid:      np.ndarray,
    result:    dict,
    save_path: str,
    fps:       int = 6,
):
    """
    Save an animated GIF showing the RL agent moving step-by-step,
    with the dynamic obstacle appearing mid-way.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not found — skipping GIF animation.")
        return

    traj        = result["trajectory"]
    obst_list   = result["obstacle_positions"]   # list of (row, col)
    H, W        = grid.shape
    scale       = max(2, min(10, 800 // max(H, W)))  # auto-scale: smaller for big maps

    # Safety: skip GIF if trajectory is too long — prevents OOM on large maps
    if MAX_GIF_STEPS > 0 and len(traj) > MAX_GIF_STEPS:
        print(f"  Skipping GIF: trajectory has {len(traj)} steps "
              f"(limit = MAX_GIF_STEPS = {MAX_GIF_STEPS}).")
        return

    # Build a per-step lookup: which obstacles are visible at each step?
    # obstacle (row,col) becomes visible at the inject_step listed in DYNAMIC_OBSTACLES.
    obst_step_map = {(r, c): s for r, c, s in DYNAMIC_OBSTACLES}

    frames = []

    for step_idx, pos in enumerate(traj):
        img = np.full((H * scale, W * scale, 3), 230, dtype=np.uint8)

        # Draw map
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 1:
                    img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = [40, 40, 40]

        # Draw each dynamic obstacle from the step it was injected onward
        for dr, dc in obst_list:
            appear_at = obst_step_map.get((dr, dc), 0)
            if step_idx >= appear_at:
                img[dr*scale:(dr+1)*scale, dc*scale:(dc+1)*scale] = [244, 67, 54]

        # Draw trajectory trail
        for tr in traj[:step_idx]:
            img[tr[0]*scale:(tr[0]+1)*scale, tr[1]*scale:(tr[1]+1)*scale] = \
                [135, 206, 235]

        # Draw goal
        gr, gc = GOAL_POS
        img[gr*scale:(gr+1)*scale, gc*scale:(gc+1)*scale] = [76, 175, 80]

        # Draw drone
        pr, pc = pos
        img[pr*scale:(pr+1)*scale, pc*scale:(pc+1)*scale] = [244, 67, 54]

        frames.append(Image.fromarray(img))

    if frames:
        delay = int(1000 / fps)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=delay,
        )
        print(f"  Saved animation: {save_path}")


# ===========================================================================
#  Reward / Value Map
# ===========================================================================

def plot_reward_map(
    model:     "PPO",
    env:       "DroneNavEnv",
    base_grid: np.ndarray,
    save_path: str,
) -> None:
    """
    Render a PPO value-function heatmap over the occupancy grid.

    For every free cell (r, c) the function builds the observation the drone
    would have if it were standing there on the CLEAN (no dynamic obstacles)
    map, then queries the PPO critic network for V(s) — the expected
    cumulative discounted reward from that position.

    Colour scale (RdYlGn):
      Green  (high V)  →  agent expects to reach the goal from here
      Red    (low V)   →  agent expects poor performance from here
      Dark   (NaN)     →  obstacle / wall cell

    The gradient should point roughly toward the goal with high values near
    the goal and lower values in areas blocked by walls.
    Flat or noisy regions indicate the agent hasn't learned well there.
    """
    import torch

    H, W = base_grid.shape
    value_map = np.full((H, W), np.nan, dtype=np.float32)

    # Use the clean base map for raycasts so dynamic obstacles don't distort
    # the picture.
    saved_grid      = env.grid
    env.grid        = base_grid.copy()

    goal_x = env.goal_pos[1] / env._norm_w
    goal_y = env.goal_pos[0] / env._norm_h

    print("\n  Computing value map (querying PPO critic for every free cell) …")
    for r in range(H):
        for c in range(W):
            if base_grid[r, c] == env.OBSTACLE:
                continue
            obs = np.array([
                c / env._norm_w,
                r / env._norm_h,
                goal_x,
                goal_y,
                env._raycast(r, c, -1,  0) / env._norm_ray,
                env._raycast(r, c, +1,  0) / env._norm_ray,
                env._raycast(r, c,  0, -1) / env._norm_ray,
                env._raycast(r, c,  0, +1) / env._norm_ray,
            ], dtype=np.float32)
            obs_t = torch.tensor(obs[None]).to(model.device)
            with torch.no_grad():
                value_map[r, c] = model.policy.predict_values(obs_t).item()

    env.grid = saved_grid  # restore working grid

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 8))

    # Heatmap — NaN cells (obstacles) show as white background
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#222222")   # obstacles render dark grey
    im = ax.imshow(value_map, cmap=cmap, origin="upper",
                   interpolation="nearest", aspect="equal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("V(s) — expected cumulative reward", fontsize=9)

    # Start / goal markers
    ax.plot(START_POS[1], START_POS[0], "o", color="#00E5FF",
            markersize=12, markeredgecolor="white", markeredgewidth=1.5,
            label="Start", zorder=5)
    ax.plot(GOAL_POS[1],  GOAL_POS[0],  "*", color="#FFC107",
            markersize=16, markeredgecolor="white", markeredgewidth=1.5,
            label="Goal", zorder=5)

    ax.set_title("PPO Value Map  —  V(s) per grid cell\n"
                 "(green = agent expects high reward, red = expects to struggle)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved reward map: {save_path}")
    plt.close()


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test trained RL drone agent with dynamic obstacle."
    )
    parser.add_argument("--no-obstacle", action="store_true",
                        help="Run without injecting a dynamic obstacle.")
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="Number of test episodes to run.")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to the trained model (without .zip).")
    args = parser.parse_args()

    # Load model
    model_path = args.model
    if not os.path.exists(model_path + ".zip"):
        print(f"\nERROR: No trained model found at '{model_path}.zip'")
        print("  Run  python train_ppo.py  first to train the agent.")
        sys.exit(1)

    print(f"Loading model: {model_path}.zip")
    model = PPO.load(model_path)

    # Create environment
    env = DroneNavEnv(
        map_path  = MAP_PATH,
        start_pos = START_POS,
        goal_pos  = GOAL_POS,
        max_steps = MAX_STEPS,
    )

    # Set up A* planners
    from maps.generate_map import load_pgm, create_training_map
    if os.path.exists(MAP_PATH):
        base_grid = load_pgm(MAP_PATH)
    else:
        base_grid = create_training_map(40, 40)

    planner_original = AStarPlanner(base_grid)
    astar_path       = planner_original.plan(START_POS, GOAL_POS)

    # A* on the modified map — controlled by ASTAR_REPLAN
    if ASTAR_REPLAN:
        planner_blocked = AStarPlanner(base_grid)
        for obr, obc, _ in DYNAMIC_OBSTACLES:
            planner_blocked.add_obstacle(obr, obc)
        astar_path_blocked = planner_blocked.plan(START_POS, GOAL_POS)
        replan_label = "replanned"
    else:
        # Keep the original path — it may now pass through obstacle cells
        astar_path_blocked = astar_path
        replan_label = "original (no replan)"

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"\nA* original path length:  {len(astar_path) if astar_path else 'NO PATH'}")
    print(f"A* {replan_label} path length: "
          f"{len(astar_path_blocked) if astar_path_blocked else 'NO PATH'}")

    # ---- Run episodes ----
    inject = not args.no_obstacle
    results = []

    for ep in range(args.n_episodes):
        print(f"\n{'='*50}")
        print(f"  Episode {ep+1}/{args.n_episodes}  "
              f"({'with' if inject else 'without'} dynamic obstacle)")
        print(f"{'='*50}")

        result = run_episode(model, env, inject_obstacle=inject, verbose=True)
        results.append(result)

        # Save visualisation for each episode
        fig_path = os.path.join(OUTPUTS_DIR, f"episode_{ep+1}_comparison.png")
        visualise_episode(
            grid               = env.grid,
            rl_result          = result,
            astar_path         = astar_path,
            astar_path_blocked = astar_path_blocked,
            save_path          = fig_path,
        )

        # Save animation for the first episode only
        if ep == 0:
            gif_path = os.path.join(OUTPUTS_DIR, "episode_1_animation.gif")
            animate_episode(base_grid, result, gif_path, fps=8)

    # ---- Summary ----
    successes = sum(r["success"] for r in results)
    mean_steps = np.mean([r["steps"] for r in results])
    print(f"\n{'='*50}")
    print(f"  SUMMARY  ({args.n_episodes} episodes, "
          f"{'with' if inject else 'without'} obstacle)")
    print(f"{'='*50}")
    print(f"  RL success rate:  {successes}/{args.n_episodes}")
    print(f"  RL mean steps:    {mean_steps:.1f}")
    if astar_path:
        print(f"  A* path length:   {len(astar_path)} cells")
    print(f"\n  Interpretation:")
    if inject:
        print("  • A* computed a path on the original map — "
              "that path now runs through the dynamic obstacle.")
        print("  • Without replanning, A* would try to follow a blocked route.")
        print("  • The RL agent's sensor readings changed when the obstacle")
        print("    appeared, allowing it to attempt to navigate around it.")
        print("  • RL success here is NOT guaranteed — it depends on whether")
        print("    the agent encountered similar situations during training.")
    else:
        print("  • On a static map, A* is optimal and always succeeds.")
        print("  • RL is typically near-optimal but may take a longer path.")

    if SHOW_REWARD_MAP:
        plot_reward_map(
            model     = model,
            env       = env,
            base_grid = base_grid,
            save_path = os.path.join(OUTPUTS_DIR, "reward_map.png"),
        )

    env.close()


if __name__ == "__main__":
    main()
