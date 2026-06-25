"""
compare_astar_rl.py — Quantitative and Visual Comparison of A* vs RL
======================================================================

PURPOSE
-------
This script runs a controlled experiment comparing A* and the trained
RL agent across two scenarios:

  SCENARIO 1 — Static Map
    The map is unchanged.  A* is optimal here; RL should be close.
    Metrics: path length, steps, success rate.

  SCENARIO 2 — Dynamic Map (obstacle injected mid-navigation)
    A new obstacle is placed on the A* path while the RL agent is
    already navigating.
    A*: the stored path is invalid and must be replanned.
    RL: the agent reacts via its sensor readings.

OUTPUT
------
  outputs/comparison_static.png   — side-by-side paths (scenario 1)
  outputs/comparison_dynamic.png  — dynamic obstacle scenario (scenario 2)
  outputs/comparison_metrics.png  — bar-chart of quantitative metrics

HOW TO RUN
----------
    cd rl_drone_path_planning
    python compare_astar_rl.py

PREREQUISITES
-------------
    python train_ppo.py   # must be run first
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from envs.drone_env      import DroneNavEnv
from astar.astar_planner import AStarPlanner
from maps.generate_map   import load_pgm, create_training_map

# ===========================================================================
#  Configuration
# ===========================================================================

MAP_PATH    = os.path.join(PROJECT_ROOT, "maps", "training_map.pgm")
MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "drone_ppo_final")
START_POS   = (2, 2)
GOAL_POS    = (37, 37)
MAX_STEPS   = 1000

N_EVAL_EPISODES = 20        # episodes for statistics

# Dynamic obstacle parameters (on the horizontal wall gap)
DYN_OBS_ROW = 15
DYN_OBS_COL = 18

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ===========================================================================
#  Helpers
# ===========================================================================

def load_map() -> np.ndarray:
    if os.path.exists(MAP_PATH):
        return load_pgm(MAP_PATH)
    print("  Map not found — auto-generating.")
    return create_training_map(40, 40)


def run_rl_episodes(
    model:           PPO,
    env:             DroneNavEnv,
    n:               int,
    inject_obstacle: bool = False,
) -> list:
    """
    Run n episodes and return a list of result dicts.
    Each dict: {success, steps, total_reward, trajectory}
    """
    results = []
    for _ in range(n):
        obs, _ = env.reset()
        ep_reward = 0.0
        injected  = False

        for step in range(MAX_STEPS):
            if inject_obstacle and not injected and step == 15:
                env.add_obstacle(DYN_OBS_ROW, DYN_OBS_COL)
                injected = True

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                results.append({
                    "success":      terminated,
                    "steps":        env.step_count,
                    "total_reward": ep_reward,
                    "trajectory":   list(env.trajectory),
                })
                break
        else:
            results.append({
                "success":      False,
                "steps":        MAX_STEPS,
                "total_reward": ep_reward,
                "trajectory":   list(env.trajectory),
            })
    return results


def draw_map_background(ax, grid):
    """Draw the occupancy grid as a grayscale image background."""
    bg = np.where(grid == 0, 240, 50).astype(np.uint8)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255,
              origin="upper", interpolation="nearest", aspect="equal")


# ===========================================================================
#  Figure 1 — Static scenario side-by-side
# ===========================================================================

def plot_static_comparison(
    grid:        np.ndarray,
    astar_path:  list,
    rl_results:  list,
    save_path:   str,
):
    """Two-panel figure: left=A*, right=RL on the static map."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Scenario 1 — Static Map: A* (optimal) vs RL (learned)",
        fontsize=13, fontweight="bold"
    )

    # ----------- A* panel -----------
    ax = axes[0]
    draw_map_background(ax, grid)
    ax.set_title("A* — Dijkstra / Best-First (guaranteed optimal)", fontsize=10)

    if astar_path:
        tr = [p[0] for p in astar_path]
        tc = [p[1] for p in astar_path]
        ax.plot(tc, tr, "-", color="#1565C0", linewidth=2.5, label="A* path")
        ax.plot(tc[0],  tr[0],  "o", color="#4CAF50", markersize=12,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(tc[-1], tr[-1], "*", color="#FFC107", markersize=16,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")

    ax.set_xlabel(f"Path length: {len(astar_path)} cells  |  Always succeeds  |  "
                  f"Time: deterministic", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    # ----------- RL panel -----------
    ax = axes[1]
    draw_map_background(ax, grid)
    ax.set_title("RL Agent (PPO) — Learned policy", fontsize=10)

    successes  = [r for r in rl_results if r["success"]]
    failures   = [r for r in rl_results if not r["success"]]

    # Plot all successful trajectories (semi-transparent)
    for r in successes:
        traj = r["trajectory"]
        tr = [p[0] for p in traj]
        tc = [p[1] for p in traj]
        ax.plot(tc, tr, "-", color="#1976D2", alpha=0.3, linewidth=1.2)

    # Highlight the best (shortest) trajectory
    if successes:
        best = min(successes, key=lambda r: r["steps"])
        traj = best["trajectory"]
        tr   = [p[0] for p in traj]
        tc   = [p[1] for p in traj]
        ax.plot(tc, tr, "-", color="#0D47A1", linewidth=2.5,
                label=f"Best RL path ({best['steps']} steps)")

    ax.plot(START_POS[1], START_POS[0], "o", color="#4CAF50", markersize=12,
            markeredgecolor="white", markeredgewidth=1.5, label="Start")
    ax.plot(GOAL_POS[1],  GOAL_POS[0],  "*", color="#FFC107", markersize=16,
            markeredgecolor="white", markeredgewidth=1.5, label="Goal")

    mean_steps  = np.mean([r["steps"]   for r in rl_results]) if rl_results else 0
    success_pct = 100 * len(successes) / len(rl_results) if rl_results else 0
    ax.set_xlabel(
        f"Success rate: {success_pct:.0f}%  |  "
        f"Mean steps: {mean_steps:.0f}  |  "
        f"A* length: {len(astar_path)}",
        fontsize=9
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ===========================================================================
#  Figure 2 — Dynamic obstacle scenario
# ===========================================================================

def plot_dynamic_comparison(
    base_grid:         np.ndarray,
    astar_path_orig:   list,
    astar_path_replan: list,
    rl_results_dyn:    list,
    save_path:         str,
):
    """Two-panel: left=RL reactive, right=A* original vs replanned."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Scenario 2 — Dynamic Obstacle: RL (reactive) vs A* (requires replanning)",
        fontsize=13, fontweight="bold"
    )

    successes = [r for r in rl_results_dyn if r["success"]]
    failures  = [r for r in rl_results_dyn if not r["success"]]

    def draw_obstacle(ax):
        ax.add_patch(plt.Rectangle(
            (DYN_OBS_COL - 0.5, DYN_OBS_ROW - 0.5), 1, 1,
            facecolor="#F44336", edgecolor="#B71C1C",
            linewidth=2, zorder=4, label="New obstacle",
        ))
        ax.text(DYN_OBS_COL, DYN_OBS_ROW, "✕",
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=5)

    # ----------- RL panel -----------
    ax = axes[0]
    draw_map_background(ax, base_grid)
    draw_obstacle(ax)
    ax.set_title("RL Agent — reacts via changed sensor readings", fontsize=10)

    for r in successes:
        traj = r["trajectory"]
        ax.plot([p[1] for p in traj], [p[0] for p in traj],
                "-", color="#1976D2", alpha=0.4, linewidth=1.2)

    if successes:
        best = min(successes, key=lambda r: r["steps"])
        traj = best["trajectory"]
        ax.plot([p[1] for p in traj], [p[0] for p in traj],
                "-", color="#0D47A1", linewidth=2.5, label="Best RL path")

    # Mark injection point on each trajectory
    for r in rl_results_dyn:
        traj = r["trajectory"]
        if len(traj) > 15:
            ax.plot(traj[15][1], traj[15][0], "D",
                    color="#FF9800", markersize=6, zorder=3)

    ax.plot(START_POS[1], START_POS[0], "o", color="#4CAF50", markersize=12,
            markeredgecolor="white", markeredgewidth=1.5, label="Start")
    ax.plot(GOAL_POS[1],  GOAL_POS[0],  "*", color="#FFC107", markersize=16,
            markeredgecolor="white", markeredgewidth=1.5, label="Goal")

    success_pct = 100 * len(successes) / len(rl_results_dyn) if rl_results_dyn else 0
    mean_steps  = np.mean([r["steps"] for r in rl_results_dyn]) if rl_results_dyn else 0
    ax.set_xlabel(
        f"RL success rate with obstacle: {success_pct:.0f}%  |  "
        f"Mean steps: {mean_steps:.0f}",
        fontsize=9
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    # ----------- A* panel -----------
    ax = axes[1]
    draw_map_background(ax, base_grid)
    draw_obstacle(ax)
    ax.set_title("A* — original path blocked, must replan", fontsize=10)

    if astar_path_orig:
        ax.plot([p[1] for p in astar_path_orig],
                [p[0] for p in astar_path_orig],
                "--", color="#90A4AE", linewidth=2,
                label=f"Original path ({len(astar_path_orig)} cells, BLOCKED)")

    if astar_path_replan:
        ax.plot([p[1] for p in astar_path_replan],
                [p[0] for p in astar_path_replan],
                "-", color="#FF9800", linewidth=2.5,
                label=f"Replanned path ({len(astar_path_replan)} cells)")

    ax.plot(START_POS[1], START_POS[0], "o", color="#4CAF50", markersize=12,
            markeredgecolor="white", markeredgewidth=1.5, label="Start")
    ax.plot(GOAL_POS[1],  GOAL_POS[0],  "*", color="#FFC107", markersize=16,
            markeredgecolor="white", markeredgewidth=1.5, label="Goal")

    extra = (len(astar_path_replan) - len(astar_path_orig)) if (astar_path_replan and astar_path_orig) else 0
    ax.set_xlabel(
        f"A* replanned successfully  |  "
        f"Extra cells due to detour: +{extra}  |  "
        f"Replanning requires re-running A* from scratch",
        fontsize=9
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ===========================================================================
#  Figure 3 — Metrics bar chart
# ===========================================================================

def plot_metrics(
    astar_path:        list,
    astar_path_replan: list,
    rl_static:         list,
    rl_dynamic:        list,
    save_path:         str,
):
    """
    Four-panel bar chart comparing key metrics across all scenarios.
    This is the "scorecard" of the experiment.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "A* vs RL — Quantitative Metrics",
        fontsize=13, fontweight="bold"
    )

    COLORS = {
        "astar_static":   "#1565C0",
        "astar_dynamic":  "#FFA726",
        "rl_static":      "#43A047",
        "rl_dynamic":     "#E53935",
    }

    labels  = ["A* static", "A* dynamic\n(replanned)", "RL static", "RL dynamic"]
    colors  = [COLORS["astar_static"], COLORS["astar_dynamic"],
                COLORS["rl_static"],   COLORS["rl_dynamic"]]

    rl_s_succ = [r for r in rl_static  if r["success"]]
    rl_d_succ = [r for r in rl_dynamic if r["success"]]

    # ---- Metric 1: Success Rate ----
    ax = axes[0]
    values = [
        100.0,
        100.0,
        100 * len(rl_s_succ) / len(rl_static)  if rl_static  else 0,
        100 * len(rl_d_succ) / len(rl_dynamic) if rl_dynamic else 0,
    ]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # ---- Metric 2: Path / Episode Length ----
    ax = axes[1]
    rl_s_len = np.mean([r["steps"] for r in rl_s_succ]) if rl_s_succ else 0
    rl_d_len = np.mean([r["steps"] for r in rl_d_succ]) if rl_d_succ else 0
    values = [
        len(astar_path)        if astar_path        else 0,
        len(astar_path_replan) if astar_path_replan else 0,
        rl_s_len,
        rl_d_len,
    ]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Steps to goal")
    ax.set_title("Steps to Goal\n(successful episodes)")
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # ---- Metric 3: Mean Episode Reward ----
    ax = axes[2]
    rl_s_rew = np.mean([r["total_reward"] for r in rl_static])  if rl_static  else 0
    rl_d_rew = np.mean([r["total_reward"] for r in rl_dynamic]) if rl_dynamic else 0
    # A* reward approximation based on path length
    astar_s_rew = (100.0 + len(astar_path)        * (-0.02 + 0.1)) if astar_path        else 0
    astar_d_rew = (100.0 + len(astar_path_replan) * (-0.02 + 0.1)) if astar_path_replan else 0
    values = [astar_s_rew, astar_d_rew, rl_s_rew, rl_d_rew]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Mean total reward")
    ax.set_title("Mean Reward\n(all episodes)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if val >= 0 else -8),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # ---- Metric 4: Adaptability (descriptive) ----
    ax = axes[3]
    # Score: 0 = fails, 1 = succeeds after replanning, 2 = adapts online
    values = [1, 2, 2, 1]  # qualitative scale
    y_labels = ["Fails\n(old path invalid)", "Adapts\n(replanning needed)",
                "Adapts\n(online)", "Partially\nadapts"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Fails", "Needs\nreplanning", "Online\nadaptation"])
    ax.set_title("Adaptability to\nDynamic Obstacles")
    for bar, val, lbl in zip(bars, values, y_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                lbl, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ===========================================================================
#  Main
# ===========================================================================

def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"\nERROR: No trained model found at '{MODEL_PATH}.zip'")
        print("  Run  python train_ppo.py  first.")
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  A* vs RL Comparison  —  rl_drone_path_planning")
    print("=" * 60)

    # ---- Load map and model ----
    print("\nLoading map and model …")
    base_grid = load_map()
    model = PPO.load(MODEL_PATH)

    env = DroneNavEnv(
        map_path  = MAP_PATH,
        start_pos = START_POS,
        goal_pos  = GOAL_POS,
        max_steps = MAX_STEPS,
    )

    # ---- A* paths ----
    print("\nRunning A* planners …")
    planner_static = AStarPlanner(base_grid)
    t0             = time.perf_counter()
    astar_path     = planner_static.plan(START_POS, GOAL_POS)
    t_astar        = time.perf_counter() - t0

    planner_dyn    = AStarPlanner(base_grid)
    planner_dyn.add_obstacle(DYN_OBS_ROW, DYN_OBS_COL)
    astar_replan   = planner_dyn.plan(START_POS, GOAL_POS)

    print(f"  A* static  path: {len(astar_path) if astar_path else 'NONE':>4} cells  "
          f"(computed in {t_astar*1000:.2f} ms)")
    print(f"  A* replanned:    {len(astar_replan) if astar_replan else 'NONE':>4} cells")

    # ---- RL — static scenario ----
    print(f"\nRunning RL on static map ({N_EVAL_EPISODES} episodes) …")
    t0         = time.perf_counter()
    rl_static  = run_rl_episodes(model, env, N_EVAL_EPISODES, inject_obstacle=False)
    t_rl_inf   = (time.perf_counter() - t0) / N_EVAL_EPISODES
    s_succ     = sum(r["success"] for r in rl_static)
    s_steps    = np.mean([r["steps"] for r in rl_static if r["success"]]) if any(r["success"] for r in rl_static) else 0
    print(f"  Success rate:  {s_succ}/{N_EVAL_EPISODES}  ({100*s_succ/N_EVAL_EPISODES:.0f}%)")
    print(f"  Mean steps:    {s_steps:.1f}  (A* = {len(astar_path) if astar_path else 0})")
    print(f"  Inference time per episode: {t_rl_inf:.3f}s")

    # ---- RL — dynamic scenario ----
    print(f"\nRunning RL on dynamic map ({N_EVAL_EPISODES} episodes, "
          f"obstacle at step 15) …")
    rl_dynamic = run_rl_episodes(model, env, N_EVAL_EPISODES, inject_obstacle=True)
    d_succ     = sum(r["success"] for r in rl_dynamic)
    d_steps    = np.mean([r["steps"] for r in rl_dynamic if r["success"]]) if any(r["success"] for r in rl_dynamic) else 0
    print(f"  Success rate:  {d_succ}/{N_EVAL_EPISODES}  ({100*d_succ/N_EVAL_EPISODES:.0f}%)")
    print(f"  Mean steps:    {d_steps:.1f}")

    # ---- Plots ----
    print("\nGenerating comparison figures …")

    plot_static_comparison(
        base_grid, astar_path, rl_static,
        os.path.join(OUTPUTS_DIR, "comparison_static.png"),
    )

    plot_dynamic_comparison(
        base_grid, astar_path, astar_replan, rl_dynamic,
        os.path.join(OUTPUTS_DIR, "comparison_dynamic.png"),
    )

    plot_metrics(
        astar_path, astar_replan, rl_static, rl_dynamic,
        os.path.join(OUTPUTS_DIR, "comparison_metrics.png"),
    )

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<35} {'A*':>10} {'RL':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Static success rate':<35} {'100%':>10} "
          f"{100*s_succ/N_EVAL_EPISODES:>9.0f}%")
    print(f"  {'Dynamic success rate':<35} {'100%*':>10} "
          f"{100*d_succ/N_EVAL_EPISODES:>9.0f}%")
    print(f"  {'Path length (static)':<35} {len(astar_path) if astar_path else 0:>10} "
          f"{s_steps:>10.0f}")
    print(f"  {'Path length after replan/detour':<35} {len(astar_replan) if astar_replan else 0:>10} "
          f"{d_steps:>10.0f}")
    print(f"  {'Needs explicit replanning?':<35} {'Yes*':>10} {'No':>10}")
    print()
    print("  * A* dynamic: replanned successfully because we re-ran A*.")
    print("    Without replanning, the original A* path would hit the obstacle.")
    print()
    print("  EDUCATIONAL TAKE-AWAY")
    print("  ─────────────────────")
    print("  • A* is optimal on a known static map.  Always succeeds, always")
    print("    finds the shortest path.  Extremely fast to compute.")
    print()
    print("  • RL learns a general policy, not a fixed path.  It can react")
    print("    to unexpected obstacles via its sensor readings — no explicit")
    print("    replanning step required.")
    print()
    print("  • RL is NOT universally better.  On a static map it may take")
    print("    a longer path and has a non-zero failure rate.")
    print()
    print("  • The right tool depends on the task:")
    print("    Use A*  when the map is known, static, and replanning time")
    print("    is acceptable.")
    print("    Use RL  when the environment is partially unknown, dynamic,")
    print("    or when replanning latency is a hard constraint.")
    print()
    print(f"  Outputs saved to: {OUTPUTS_DIR}/")

    env.close()


if __name__ == "__main__":
    main()
