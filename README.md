# RL Drone Path Planning
### Module 6 — Reinforcement Learning for 2D Drone Navigation

A practical tutorial project comparing **A\*** (classical) and **PPO** (reinforcement learning) for navigating a drone through a 2D occupancy grid — including a live dynamic-obstacle experiment.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv && source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate                             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate map files
python maps/generate_map.py

# 4. Train the RL agent  (~5–15 min on CPU)
python train_ppo.py

# 5. Dynamic obstacle test
python test_rl.py

```

---

## Learning Objectives

- Implement a custom **Gymnasium** environment for a robotics task
- Understand the **Gymnasium API**: `observation_space`, `action_space`, `reset()`, `step()`
- Design an **observation vector** using local sensor (range) readings
- Understand **reward shaping**: dense feedback vs sparse rewards
- Train a **PPO** agent with Stable-Baselines3
- Implement **A\*** path planning from scratch on a 2D grid
- Run the **dynamic obstacle experiment** and observe the difference in agent behaviour
- Understand when RL adds value over classical planning — and when it does not

---

## Project Structure

```
rl_drone_path_planning/
├── maps/
│   ├── generate_map.py        ← Build and save PGM occupancy grids
│   ├── training_map.pgm       ← 40×40 map for PPO training  (generated)
│   └── simple_map.pgm         ← 20×20 map for quick tests   (generated)
│
├── envs/
│   └── drone_env.py           ← Custom Gymnasium environment  ★
│
├── astar/
│   └── astar_planner.py       ← A* planner + AStarPlanner class  ★
│
├── models/                    ← Saved PPO models (created by train_ppo.py)
├── outputs/                   ← Saved figures and animations
│
├── train_ppo.py               ← PPO training script  ★
├── test_rl.py                 ← Dynamic obstacle test
│
├── student_exercises/
│   ├── drone_env_template.py      ← Exercise 1: build the Gym env
│   ├── astar_planner_template.py  ← Exercise 2: implement A*
│   └── train_ppo_template.py      ← Exercise 3: wire up PPO training
│
├── requirements.txt
├── requirements.md            ← Full setup guide (Linux + Windows)
└── README.md                  ← This file
```

---

## RL Concepts for Drone Engineers

### The Core Loop

```
Environment                Agent
    │                        │
    │ ── observation ──────► │
    │                        │  (neural network)
    │ ◄─── action ───────── │
    │                        │
    │ ── reward + next obs ► │  ← agent updates its weights
    │                        │
```

The agent never sees the map.  It only sees 8 numbers:

```
[drone_x, drone_y, goal_x, goal_y,
 dist_up, dist_down, dist_left, dist_right]
```

All values are normalised to [0, 1] so the neural network trains stably.

### Why Local Observations?

A real drone has **sensors**, not a downloaded map.  Local observations
(4 range readings) are more realistic and also generalise across different
maps — the same policy can navigate maps it was never trained on.

### What Does the Agent Learn?

The PPO agent learns a **policy**: a function from observations to actions.
After training, it has implicitly learned:

- "If the goal is to my right and there's nothing blocking me → move right"
- "If there's an obstacle close in front → turn before moving forward"
- "If I'm drifting away from the goal → correct course"

No explicit map, no replanning — just pattern matching from observations.

---

## Environment Description

| Property | Value |
|----------|-------|
| Map | 40×40 occupancy grid loaded from PGM |
| Start | `(2, 2)` — top-left area |
| Goal | `(37, 37)` — bottom-right area |
| Obstacles | Border walls + 2 internal walls with gaps + 1 cluster |
| Action space | `Discrete(4)` — UP / DOWN / LEFT / RIGHT |
| Observation space | `Box(8,)` — all float32, normalised to [0, 1] |
| Max steps | 1000 per episode |

### Map Layout

```
S . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  row 15: ████████████████████  GAP  █████████████████████  │wall  . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . │      . . . .
. . . . . . . . . █████ . . . . . . . . . . . . . . . . . . │      . . . .
. . . . . . . . . █████ . . . . . . . . . . . . . . . . . . GAP    . . . .
. . . . . . . . . █████ . . . . . . . . . . . . . . . . . . │      . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G

S = Start (2,2)    G = Goal (37,37)    ██ = obstacle    GAP = traversable opening
```

---

## Reward Function

| Event | Reward | Why |
|-------|--------|-----|
| Reach goal | **+100.0** | Strong positive signal — what we want to maximise |
| Collision | **−1.0** | Penalises hitting walls; drone stays in place |
| Each step | **−0.02** | Time penalty — encourages finding the shortest path |
| Moved closer to goal | **+0.1** | Dense feedback — shapes learning direction |
| Moved away from goal | **−0.1** | Dense feedback — discourages backtracking |

### Why Reward Shaping?

Without the ±0.1 shaping, the agent would only learn from the +100 when it
accidentally reaches the goal.  On a 40×40 map with 1000 steps, a random
agent reaches the goal roughly 1 in 10,000 episodes.  Shaping gives feedback
on every step, giving the agent a gradient to follow even before it ever
reaches the goal.

**Trade-off:** dense shaping can cause reward hacking (e.g. oscillating in
place at the closest point without making progress).  The step penalty
−0.02 counteracts this.

---

## Training Instructions

```bash
# Full training (recommended)
python train_ppo.py

# Training is configured at the top of train_ppo.py:
TOTAL_TIMESTEPS = 300_000    # increase to 500k–1M for better performance
N_ENVS          = 4          # parallel environments (reduce if RAM-limited)
```

Training produces:
- `models/drone_ppo_final.zip` — the final model
- `models/drone_ppo_best/`     — the best checkpoint (highest eval reward)
- `outputs/training_rewards.png` — learning curve

### Expected Training Progress

| Timestep | Typical Mean Reward | What's Happening |
|----------|--------------------|--------------------|
| 0–30k    | −10 to −5          | Random exploration, many timeouts |
| 30k–80k  | −5 to +20          | Agent learns to avoid walls |
| 80k–150k | +20 to +80         | Agent finds the goal occasionally |
| 150k+    | +80 to +95         | Policy refines, shorter paths |

---

## Dynamic Obstacle Experiment

This is the central experiment of the tutorial.

```
Timeline:

  Step 0    Drone starts at (2,2). A* path computed on original map.
  Step 1–14 Both A* and RL navigate normally.
  Step 15   ★ NEW OBSTACLE placed at (15, 18) — blocking the gap in
              the horizontal wall.
  Step 16+  • A*: its stored path now passes through the obstacle cell.
              A* fails silently unless we re-run it.
            • RL: the dist_up sensor reading changes (smaller).
              The policy maps this new observation to a different action
              — typically sidestepping the blockage.
```

### How to Run

```bash
python test_rl.py                # default: injects obstacle at step 15
python test_rl.py --no-obstacle  # static map only
python test_rl.py --n-episodes 5 # run 5 episodes
```

### Key Observation

The RL agent does NOT have special obstacle-avoidance code.  It simply
observes that one direction is suddenly blocked (shorter raycast) and
responds according to its trained policy.  The adaptation is implicit.

---

## A* vs RL Discussion

| Criteria | A* | RL (PPO) |
|----------|----|----------|
| **Optimal path** | ✓ Guaranteed | ✗ Near-optimal at best |
| **Success rate (static map)** | 100% | ~80–95% (depends on training) |
| **Computation (at runtime)** | Fast planning once | Fast inference always |
| **Needs training** | ✗ None | ✓ Requires ~300k steps |
| **Known map required** | ✓ Yes | ✗ Not needed |
| **Dynamic obstacles** | ✗ Requires replanning | ✓ Reacts via sensors |
| **Generalises to new maps** | ✗ Must replan | ✓ Often generalises |
| **Explainability** | ✓ Inspectable path | ✗ Black-box policy |

### Use A\* when:
- The map is fully known and static
- You need a guaranteed shortest path
- Replanning time is acceptable when the environment changes
- Explainability is important (safety-critical systems)

### Use RL when:
- The environment changes faster than you can replan
- The map is partially unknown at runtime
- The sensor model is complex and hard to embed in a planner
- You want a policy that generalises across map variations

### The Honest Truth

**RL is not a magic solution.**  In this tutorial, A* outperforms RL on
a static map — shorter path, 100% success rate, no training needed.  The
RL agent's advantage only appears when the environment changes unexpectedly
and replanning is unavailable or too slow.

A real drone system would likely use **both**:  A* for initial planning on
a known map, and an RL reactive layer for local obstacle avoidance when
unexpected objects appear.

---

## Student Exercises

Work through the exercises in `student_exercises/` in order:

| File | Exercise | Concepts |
|------|----------|---------|
| `drone_env_template.py` | Build the Gym environment | Spaces, step(), reward shaping |
| `astar_planner_template.py` | Implement A* | Priority queue, heuristic, path reconstruction |
| `train_ppo_template.py` | Wire up PPO training | SB3 API, hyperparameters, callbacks |

Each file has:
- Detailed docstrings explaining the concept
- `# TODO` markers at each implementation step
- A smoke-test at the bottom to verify your work

---

## Expected Output

After running all four scripts, `outputs/` will contain:

```
outputs/
├── training_rewards.png       — PPO learning curve
├── episode_1_comparison.png   — RL vs A* side-by-side (episode 1)
├── episode_1_animation.gif    — animated RL trajectory
├── comparison_static.png      — static map comparison
├── comparison_dynamic.png     — dynamic obstacle scenario
└── comparison_metrics.png     — quantitative bar charts
```
