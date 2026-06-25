"""
drone_env.py — Custom Gymnasium Environment for 2D Drone Navigation
====================================================================

LEARNING GOALS
--------------
  1. Understand the Gymnasium API: reset(), step(), observation_space,
     action_space — the universal interface for RL environments.
  2. Understand observation design: why LOCAL sensor readings generalise
     better than giving the agent the full map.
  3. Understand reward shaping: how to give the agent useful feedback on
     every step, not just at the goal.
  4. Understand dynamic environments: how to inject new obstacles at
     runtime, which is the key experiment in this tutorial.

HOW GYMNASIUM WORKS
-------------------
  Every RL environment must expose exactly three things:

    env.observation_space  — shape/type of what the agent sees
    env.action_space       — shape/type of what the agent can do
    obs, info  = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

  Your RL algorithm (PPO, SAC, …) calls reset() at the start of each
  episode and step() repeatedly until terminated or truncated is True.
  It never needs to know about maps, grids, or physics — only the
  numbers that come back from step().

EPISODE LIFECYCLE
-----------------
    reset()  →  step() → step() → … → step()  →  (terminated or truncated)

  terminated = True  when the drone reaches the goal (success)
  truncated  = True  when max_steps is exceeded (timeout)

  A collision does NOT end the episode — the drone stays in place and
  receives a negative reward, then the episode continues.  This gives
  the agent more experience per episode during training.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Path setup — allows this file to import from the maps package regardless
# of which directory the user launches the script from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maps.generate_map import load_pgm, create_training_map, save_pgm


class DroneNavEnv(gym.Env):
    """
    A discrete 2D grid-world environment for drone path-planning.

    The drone occupies a single cell on a 2D occupancy grid loaded from a
    PGM map file.  At each step it moves one cell in a cardinal direction.
    It must reach the goal cell while avoiding obstacles.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  OBSERVATION SPACE  — 8 floats, all normalised to [0, 1]           │
    │                                                                     │
    │  Index  Value                 Description                          │
    │  ─────  ──────────────────── ──────────────────────────────────── │
    │    0    drone_x              col / (width  - 1)                    │
    │    1    drone_y              row / (height - 1)                    │
    │    2    goal_x               goal_col / (width  - 1)              │
    │    3    goal_y               goal_row / (height - 1)              │
    │    4    dist_up              cells to obstacle looking UP          │
    │    5    dist_down            cells to obstacle looking DOWN        │
    │    6    dist_left            cells to obstacle looking LEFT        │
    │    7    dist_right           cells to obstacle looking RIGHT       │
    │                                                                     │
    │  WHY LOCAL OBSERVATIONS?                                           │
    │  A real drone has sensors (ultrasonic, lidar) — it doesn't have   │
    │  a downloaded copy of the map.  Local observations also generalise │
    │  across different maps, making the policy more robust.             │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  ACTION SPACE  — Discrete(4)                                       │
    │                                                                     │
    │   0 = UP    (row − 1)   2 = LEFT   (col − 1)                      │
    │   1 = DOWN  (row + 1)   3 = RIGHT  (col + 1)                      │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  REWARD FUNCTION                                                    │
    │                                                                     │
    │   +100.0   Reached the goal  (episode ends: terminated=True)       │
    │    −1.0    Collision with obstacle (drone stays in place)          │
    │    −0.02   Time penalty — every step (encourages efficiency)       │
    │    +0.1    Moved closer to goal  (Manhattan distance decreased)    │
    │    −0.1    Moved away from goal  (Manhattan distance increased)    │
    │                                                                     │
    │  WHY REWARD SHAPING?                                               │
    │  Without dense rewards the agent only learns from the rare moment  │
    │  it stumbles onto the goal.  Shaping gives feedback on every step  │
    │  so the agent can learn "am I on the right track?" from the start. │
    └─────────────────────────────────────────────────────────────────────┘
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # Grid cell values
    FREE     = 0
    OBSTACLE = 1

    # Action indices
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3

    # (Δrow, Δcol) for each action — indexed by action int, avoids dict per call
    _DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Reward constants — adjust these to experiment with agent behaviour
    R_GOAL      =  100.0
    R_COLLISION =   -1.0
    R_STEP      =   -0.02
    R_CLOSER    =    0.1
    R_FARTHER   =   -0.1

    def __init__(
        self,
        map_path: Optional[str] = None,
        start_pos: Tuple[int, int] = (2, 2),
        goal_pos: Optional[Tuple[int, int]] = None,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        n_random_obstacles: int = 0,
    ):
        """
        Initialise the drone navigation environment.

        Args:
            map_path:            Path to a .pgm file.  If None or not found, a
                                 40×40 training map is auto-generated and saved.
            start_pos:           Drone starting cell as (row, col).
            goal_pos:            Goal cell as (row, col).  Defaults to (H-3, W-3).
            max_steps:           Steps before the episode is truncated (timeout).
            render_mode:         'human' to display live, 'rgb_array' for arrays.
            n_random_obstacles:  Number of random obstacles to scatter on the map
                                 at the start of every episode.  0 = no extras.
                                 Use during training for domain randomization so
                                 the agent learns to navigate around obstacles.
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.start_pos   = tuple(start_pos)

        # ------------------------------------------------------------------
        # Load or generate the map
        # ------------------------------------------------------------------
        self.base_grid = self._load_or_generate_map(map_path)
        self.height, self.width = self.base_grid.shape

        # Working copy — dynamic obstacles are written here, not to base_grid
        self.grid = self.base_grid.copy()

        # ------------------------------------------------------------------
        # Goal position
        # ------------------------------------------------------------------
        if goal_pos is not None:
            self.goal_pos = tuple(goal_pos)
        else:
            self.goal_pos = (self.height - 3, self.width - 3)

        self._validate_positions()

        # ------------------------------------------------------------------
        # Gymnasium spaces
        # ------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # ------------------------------------------------------------------
        # Internal episode state (initialised properly in reset())
        # ------------------------------------------------------------------
        self.drone_pos:  list  = list(self.start_pos)
        self.step_count: int   = 0
        self.trajectory: list  = []
        self._prev_dist: float = 0.0

        self.n_random_obstacles = n_random_obstacles

        # Pre-compute normalisation constants (avoids dividing in hot path)
        self._norm_w   = float(self.width  - 1)
        self._norm_h   = float(self.height - 1)
        self._norm_ray = float(max(self.width, self.height))

    # ======================================================================
    #  Gymnasium API — required methods
    # ======================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to its initial state for a new episode.

        Called once at the start of every episode — by the training loop
        when an episode ends, or by you when you want a fresh run.

        Returns:
            observation: the initial 8-float observation vector
            info:        empty dict (Gymnasium convention)
        """
        super().reset(seed=seed)

        self.drone_pos  = list(self.start_pos)
        self.grid       = self.base_grid.copy()   # remove dynamic obstacles
        self.step_count = 0
        self.trajectory = [tuple(self.drone_pos)]

        # Domain randomisation: scatter random obstacles every episode.
        # Keeps start/goal and a safety ring of 3 cells around each clear.
        if self.n_random_obstacles > 0:
            free = np.argwhere(self.grid == 0)
            sr, sc = self.start_pos
            gr, gc = self.goal_pos
            exclude = {
                (sr + dr, sc + dc)
                for dr in range(-3, 4) for dc in range(-3, 4)
            } | {
                (gr + dr, gc + dc)
                for dr in range(-3, 4) for dc in range(-3, 4)
            }
            candidates = [(int(r), int(c)) for r, c in free
                          if (int(r), int(c)) not in exclude]
            if candidates:
                n = min(self.n_random_obstacles, len(candidates))
                chosen = self.np_random.choice(len(candidates), size=n, replace=False)
                for idx in chosen:
                    r, c = candidates[idx]
                    self.grid[r, c] = 1

        # Initialise distance for reward shaping on the first step
        self._prev_dist = self._manhattan(self.drone_pos, self.goal_pos)

        return self._observe(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one action and return the resulting transition.

        This is the heart of the Gymnasium API.  PPO calls this millions
        of times during training.

        Args:
            action: integer 0-3 (UP / DOWN / LEFT / RIGHT)

        Returns:
            observation:  new 8-float state vector
            reward:       scalar feedback signal
            terminated:   True if the drone reached the goal
            truncated:    True if max_steps was exceeded
            info:         dict with debug information
        """
        self.step_count += 1
        dr, dc = self._delta(action)

        new_row = self.drone_pos[0] + dr
        new_col = self.drone_pos[1] + dc

        # --- Boundary check ---
        out_of_bounds = (
            new_row < 0 or new_row >= self.height or
            new_col < 0 or new_col >= self.width
        )

        # --- Collision handling ---
        if out_of_bounds or self.grid[new_row, new_col] == self.OBSTACLE:
            # Drone stays in place; receives collision penalty.
            # The episode continues so the agent gets more learning signal.
            reward     = self.R_COLLISION
            terminated = False
            truncated  = self.step_count >= self.max_steps
            return self._observe(), reward, terminated, truncated, {"collision": True}

        # --- Move drone ---
        self.drone_pos = [new_row, new_col]
        self.trajectory.append(tuple(self.drone_pos))

        # --- Goal check ---
        if tuple(self.drone_pos) == self.goal_pos:
            return self._observe(), self.R_GOAL, True, False, {"success": True}

        # --- Reward shaping (dense feedback) ---
        curr_dist = self._manhattan(self.drone_pos, self.goal_pos)
        if curr_dist < self._prev_dist:
            shape = self.R_CLOSER
        elif curr_dist > self._prev_dist:
            shape = self.R_FARTHER
        else:
            shape = 0.0
        self._prev_dist = curr_dist

        reward     = self.R_STEP + shape
        truncated  = self.step_count >= self.max_steps
        terminated = False

        info = {
            "step":            self.step_count,
            "dist_to_goal":    curr_dist,
            "drone_pos":       tuple(self.drone_pos),
        }
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state as an RGB image.

        Colour coding:
          Black   — obstacle / wall
          White   — free space
          Green   — goal cell
          Red     — drone (current position)
          Sky-blue — visited trajectory cells
        """
        import matplotlib
        matplotlib.use("TkAgg" if self.render_mode == "human" else "Agg")
        import matplotlib.pyplot as plt

        scale = 12  # pixels per grid cell
        img = np.full((self.height * scale, self.width * scale, 3),
                      255, dtype=np.uint8)

        def fill(r, c, colour):
            rr, cc = r * scale, c * scale
            img[rr:rr + scale, cc:cc + scale] = colour

        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == self.OBSTACLE:
                    fill(r, c, [30, 30, 30])

        for (tr, tc) in self.trajectory[:-1]:
            fill(tr, tc, [135, 206, 235])   # sky-blue trail

        fill(*self.goal_pos,  [0,  200,  0])   # green goal
        fill(*self.drone_pos, [220,  50, 50])  # red drone

        if self.render_mode == "human":
            plt.figure("DroneNav", figsize=(6, 6))
            plt.clf()
            plt.imshow(img, origin="upper")
            plt.title(f"Step {self.step_count}  |  dist={self._prev_dist:.0f}")
            plt.axis("off")
            plt.tight_layout()
            plt.pause(0.05)

        return img

    def close(self):
        """Release any rendering resources."""
        try:
            import matplotlib.pyplot as plt
            plt.close("DroneNav")
        except Exception:
            pass

    # ======================================================================
    #  Dynamic obstacle API  (the key experiment in this tutorial!)
    # ======================================================================

    def add_obstacle(self, row: int, col: int) -> bool:
        """
        Inject a new obstacle into the running episode.

        This is the central API for the dynamic-obstacle experiment.
        The agent was NOT trained with this obstacle present — it must
        detect it via its sensor readings and react in real time.

        The obstacle is written to self.grid (the working copy), NOT to
        self.base_grid, so reset() removes it automatically.

        Args:
            row, col: grid cell to block

        Returns:
            True if the obstacle was placed, False if the cell was invalid
            (out of bounds, already the drone, or already the goal).
        """
        if (
            0 <= row < self.height
            and 0 <= col < self.width
            and [row, col] != self.drone_pos
            and (row, col) != self.goal_pos
        ):
            self.grid[row, col] = self.OBSTACLE
            return True
        return False

    def add_obstacle_block(
        self,
        row: int,
        col: int,
        size: int = 2,
    ) -> None:
        """
        Inject a rectangular block of obstacles (size×size) centred at (row, col).

        Larger blocks are more visible in visualisations and make the
        dynamic-obstacle experiment more dramatic.
        """
        for dr in range(-size // 2, size // 2 + 1):
            for dc in range(-size // 2, size // 2 + 1):
                self.add_obstacle(row + dr, col + dc)

    # ======================================================================
    #  Private helpers
    # ======================================================================

    def _observe(self) -> np.ndarray:
        """
        Build the 8-element observation vector from current state.

        Normalisation keeps all values in [0, 1], which is important for
        neural network training — unnormalised inputs slow convergence and
        can cause numerical instability.
        """
        r, c = self.drone_pos

        drone_x  = c / self._norm_w
        drone_y  = r / self._norm_h
        goal_x   = self.goal_pos[1] / self._norm_w
        goal_y   = self.goal_pos[0] / self._norm_h

        d_up    = self._raycast(r, c, -1,  0) / self._norm_ray
        d_down  = self._raycast(r, c, +1,  0) / self._norm_ray
        d_left  = self._raycast(r, c,  0, -1) / self._norm_ray
        d_right = self._raycast(r, c,  0, +1) / self._norm_ray

        return np.array(
            [drone_x, drone_y, goal_x, goal_y,
             d_up, d_down, d_left, d_right],
            dtype=np.float32,
        )

    def _raycast(self, row: int, col: int, dr: int, dc: int) -> int:
        """
        Count free cells in direction (dr, dc) until hitting an obstacle.

        Simulates a 1-D range sensor (ultrasonic / single-beam lidar).
        Returns the number of clear cells before the first obstacle.
        A reading of 0 means the next cell in that direction is blocked.
        """
        dist = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.height and 0 <= c < self.width:
            if self.grid[r, c] == self.OBSTACLE:
                return dist
            dist += 1
            r += dr
            c += dc
        return dist   # reached map edge without hitting an obstacle

    def _delta(self, action: int) -> Tuple[int, int]:
        """Map action integer to (Δrow, Δcol)."""
        return self._DELTAS[int(action)]

    def _manhattan(self, pos: list, goal: tuple) -> float:
        """Manhattan (L1) distance between two grid cells."""
        return float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    def _validate_positions(self):
        """Assert that start and goal are within bounds and on free cells."""
        sr, sc = self.start_pos
        gr, gc = self.goal_pos
        assert 0 <= sr < self.height and 0 <= sc < self.width, \
            f"Start {self.start_pos} is outside the map ({self.height}×{self.width})."
        assert 0 <= gr < self.height and 0 <= gc < self.width, \
            f"Goal {self.goal_pos} is outside the map ({self.height}×{self.width})."
        assert self.base_grid[sr, sc] == self.FREE, \
            f"Start position {self.start_pos} is on an obstacle!"
        assert self.base_grid[gr, gc] == self.FREE, \
            f"Goal position {self.goal_pos} is on an obstacle!"

    def _load_or_generate_map(self, map_path: Optional[str]) -> np.ndarray:
        """Load a PGM map or auto-generate one if the file does not exist."""
        if map_path is None:
            return create_training_map(40, 40)

        if os.path.exists(map_path):
            print(f"  Loading map: {map_path}")
            return load_pgm(map_path)

        # Auto-generate and save so the file exists for future runs
        print(f"  Map not found at '{map_path}' — generating a fresh map.")
        grid = create_training_map(40, 40)
        save_pgm(grid, map_path)
        return grid


# ===========================================================================
#  Quick smoke-test — run this file directly to verify the env works
# ===========================================================================
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    print("Creating environment …")
    env = DroneNavEnv()

    print("Running Gymnasium env_checker …")
    check_env(env, warn=True)
    print("  check_env passed!")

    print("Running a random-action episode …")
    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended after {step + 1} steps.  "
                  f"Reward = {total_reward:.2f}  "
                  f"{'SUCCESS' if terminated else 'TIMEOUT'}")
            break
    else:
        print(f"  Still running after 200 steps.  Reward so far = {total_reward:.2f}")

    env.close()
    print("Smoke-test complete.")
