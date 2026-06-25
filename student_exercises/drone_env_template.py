"""
student_exercises/drone_env_template.py
========================================
STUDENT EXERCISE 1 — Build the Gymnasium Environment

Your task is to implement a custom Gymnasium environment for drone
navigation on a 2D occupancy grid.

LEARNING GOALS
--------------
  1. Learn the Gymnasium API (observation_space, action_space, reset, step).
  2. Design an observation vector using local sensor readings.
  3. Implement a reward function that shapes behaviour toward the goal.
  4. Implement a dynamic obstacle injection method.

INSTRUCTIONS
------------
  Search for every  # TODO  comment and fill in the implementation.
  The docstrings tell you WHAT to do.  Think about WHY each piece
  matters before you write the code.

  When you are done, run this file directly:
      python student_exercises/drone_env_template.py

  It will run a quick smoke-test.  If it passes, your environment is
  compatible with the PPO training script.

REFERENCE
---------
  Compare your implementation against  envs/drone_env.py
  Only look at the reference after you have tried each section yourself!
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maps.generate_map import load_pgm, create_training_map, save_pgm


class DroneNavEnv(gym.Env):
    """
    2D grid-world environment for drone path planning.

    The drone starts at start_pos and must reach goal_pos without hitting
    any obstacles (grid cells with value 1).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # Grid constants
    FREE     = 0
    OBSTACLE = 1

    # Action constants
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3

    def __init__(
        self,
        map_path:    Optional[str] = None,
        start_pos:   Tuple[int, int] = (2, 2),
        goal_pos:    Optional[Tuple[int, int]] = None,
        max_steps:   int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.start_pos   = tuple(start_pos)

        # Load or auto-generate the map
        if map_path is not None and os.path.exists(map_path):
            self.base_grid = load_pgm(map_path)
        else:
            self.base_grid = create_training_map(40, 40)

        self.height, self.width = self.base_grid.shape
        self.grid = self.base_grid.copy()

        self.goal_pos = tuple(goal_pos) if goal_pos else (self.height - 3, self.width - 3)

        # ------------------------------------------------------------------
        # TODO 1 — Define the Observation Space
        # ------------------------------------------------------------------
        # The agent observes 8 values, all normalised to [0, 1]:
        #   drone_x, drone_y   — normalised column and row of the drone
        #   goal_x,  goal_y    — normalised column and row of the goal
        #   dist_up, dist_down, dist_left, dist_right
        #                      — normalised distance to nearest obstacle
        #                        in each cardinal direction
        #
        # Use  gymnasium.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        # ------------------------------------------------------------------
        self.observation_space = None  # TODO: replace with correct space

        # ------------------------------------------------------------------
        # TODO 2 — Define the Action Space
        # ------------------------------------------------------------------
        # Four discrete actions: UP=0, DOWN=1, LEFT=2, RIGHT=3
        # Use  gymnasium.spaces.Discrete(4)
        # ------------------------------------------------------------------
        self.action_space = None  # TODO: replace with correct space

        self.drone_pos:  list  = list(self.start_pos)
        self.step_count: int   = 0
        self.trajectory: list  = []
        self._prev_dist: float = 0.0

        self._norm_w   = float(self.width  - 1)
        self._norm_h   = float(self.height - 1)
        self._norm_ray = float(max(self.width, self.height))

    # ======================================================================
    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Returns:
            observation: initial 8-float vector from _observe()
            info:        empty dict {}
        """
        super().reset(seed=seed)

        # ------------------------------------------------------------------
        # TODO 3 — Reset episode state
        # ------------------------------------------------------------------
        # Set:
        #   self.drone_pos  ← list(self.start_pos)
        #   self.grid       ← self.base_grid.copy()   (removes dynamic obstacles)
        #   self.step_count ← 0
        #   self.trajectory ← [tuple(self.drone_pos)]
        #   self._prev_dist ← self._manhattan(self.drone_pos, self.goal_pos)
        # Then return self._observe(), {}
        # ------------------------------------------------------------------
        pass  # TODO: implement reset and return (obs, {})

    # ======================================================================
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one action.

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.step_count += 1

        # ------------------------------------------------------------------
        # TODO 4 — Compute new position using _delta(action)
        # ------------------------------------------------------------------
        dr, dc = self._delta(action)
        new_row = None  # TODO
        new_col = None  # TODO

        # ------------------------------------------------------------------
        # TODO 5 — Check for out-of-bounds or obstacle collision
        # ------------------------------------------------------------------
        # If the new cell is outside the grid boundaries OR it contains an
        # obstacle (self.grid[new_row, new_col] == self.OBSTACLE):
        #   • Do NOT move the drone
        #   • Return reward = -1.0  (collision penalty)
        #   • terminated = False, truncated = (step_count >= max_steps)
        # ------------------------------------------------------------------
        out_of_bounds = False  # TODO: check bounds
        # TODO: handle collision case (return early)

        # ------------------------------------------------------------------
        # TODO 6 — Move the drone and update trajectory
        # ------------------------------------------------------------------
        # self.drone_pos = [new_row, new_col]
        # self.trajectory.append(tuple(self.drone_pos))
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 7 — Check if goal is reached
        # ------------------------------------------------------------------
        # If tuple(self.drone_pos) == self.goal_pos:
        #   return self._observe(), +100.0, True, False, {"success": True}
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 8 — Reward shaping based on distance change
        # ------------------------------------------------------------------
        # Compute curr_dist = self._manhattan(self.drone_pos, self.goal_pos)
        # If curr_dist < self._prev_dist: shape = +0.1   (moved closer)
        # If curr_dist > self._prev_dist: shape = -0.1   (moved away)
        # Else:                           shape =  0.0
        # Update self._prev_dist = curr_dist
        # Total reward = -0.02 + shape   (step penalty + shaping)
        # ------------------------------------------------------------------
        reward   = 0.0  # TODO
        truncated  = self.step_count >= self.max_steps
        terminated = False

        return self._observe(), reward, terminated, truncated, {}

    # ======================================================================
    def add_obstacle(self, row: int, col: int) -> bool:
        """
        Inject a new obstacle into the RUNNING episode.

        This is the key method for the dynamic-obstacle experiment.
        Only write to self.grid, NOT to self.base_grid.

        Returns True if placed, False if invalid.
        """
        # ------------------------------------------------------------------
        # TODO 9 — Implement dynamic obstacle injection
        # ------------------------------------------------------------------
        # Check:
        #   • 0 <= row < self.height  AND  0 <= col < self.width
        #   • [row, col] is NOT the drone's current position
        #   • (row, col) is NOT the goal
        # If all checks pass: self.grid[row, col] = self.OBSTACLE; return True
        # Otherwise: return False
        # ------------------------------------------------------------------
        pass  # TODO

    # ======================================================================
    def _observe(self) -> np.ndarray:
        """Build the 8-element normalised observation vector."""
        r, c = self.drone_pos

        drone_x = c / self._norm_w
        drone_y = r / self._norm_h
        goal_x  = self.goal_pos[1] / self._norm_w
        goal_y  = self.goal_pos[0] / self._norm_h

        # ------------------------------------------------------------------
        # TODO 10 — Compute the four normalised raycast distances
        # ------------------------------------------------------------------
        # Use self._raycast(r, c, dr, dc) where (dr, dc) is the direction:
        #   UP    = (-1, 0)
        #   DOWN  = (+1, 0)
        #   LEFT  = ( 0,-1)
        #   RIGHT = ( 0,+1)
        # Normalise each by self._norm_ray
        # ------------------------------------------------------------------
        d_up    = 0.0  # TODO
        d_down  = 0.0  # TODO
        d_left  = 0.0  # TODO
        d_right = 0.0  # TODO

        return np.array(
            [drone_x, drone_y, goal_x, goal_y,
             d_up, d_down, d_left, d_right],
            dtype=np.float32,
        )

    # ======================================================================
    def _raycast(self, row: int, col: int, dr: int, dc: int) -> int:
        """
        Count free cells from (row, col) in direction (dr, dc) until
        hitting an obstacle or the map edge.
        """
        # ------------------------------------------------------------------
        # TODO 11 — Implement the raycast
        # ------------------------------------------------------------------
        # Start one cell ahead: r, c = row + dr, col + dc
        # Walk while in bounds AND cell is FREE, incrementing dist
        # Return dist when you hit an obstacle or go out of bounds
        # ------------------------------------------------------------------
        pass  # TODO

    # ======================================================================
    def _delta(self, action: int) -> Tuple[int, int]:
        """Map action integer to (Δrow, Δcol)."""
        return {
            self.UP:    (-1,  0),
            self.DOWN:  (+1,  0),
            self.LEFT:  ( 0, -1),
            self.RIGHT: ( 0, +1),
        }[action]

    def _manhattan(self, pos: list, goal: tuple) -> float:
        return float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    def close(self):
        pass


# ===========================================================================
#  Smoke-test — run directly to check your implementation
# ===========================================================================
if __name__ == "__main__":
    print("Testing your DroneNavEnv implementation …\n")

    # Test 1: Can we create the environment?
    try:
        env = DroneNavEnv()
        print("  [PASS] Environment created")
    except Exception as e:
        print(f"  [FAIL] Could not create environment: {e}")
        sys.exit(1)

    # Test 2: Do observation_space and action_space exist?
    if env.observation_space is None:
        print("  [FAIL] TODO 1 not done — observation_space is None")
        sys.exit(1)
    if env.action_space is None:
        print("  [FAIL] TODO 2 not done — action_space is None")
        sys.exit(1)
    print("  [PASS] Spaces defined")

    # Test 3: Does reset() return correct shapes?
    try:
        obs, info = env.reset()
        assert obs.shape == (8,), f"Expected shape (8,), got {obs.shape}"
        assert obs.dtype == np.float32
        print(f"  [PASS] reset() returns obs shape {obs.shape}")
    except Exception as e:
        print(f"  [FAIL] reset() failed: {e}")
        sys.exit(1)

    # Test 4: Does step() return correct types?
    try:
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, (int, float)), "reward must be a number"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        print(f"  [PASS] step() works  (reward={reward:.3f})")
    except Exception as e:
        print(f"  [FAIL] step() failed: {e}")
        sys.exit(1)

    # Test 5: Run Gymnasium's built-in checker
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("  [PASS] SB3 check_env passed!")
    except Exception as e:
        print(f"  [WARN] check_env raised: {e}")

    # Test 6: Random episode
    obs, _ = env.reset()
    total_r = 0.0
    for i in range(500):
        action = env.action_space.sample()
        obs, r, done, trunc, _ = env.step(action)
        total_r += r
        if done or trunc:
            print(f"  [PASS] Random episode ended at step {i+1}  "
                  f"(reward={total_r:.2f})")
            break
    else:
        print(f"  [NOTE] Random episode ran 500 steps without ending "
              f"(reward={total_r:.2f}).  This is OK.")

    # Test 7: Dynamic obstacle
    obs, _ = env.reset()
    placed = env.add_obstacle(5, 5)
    if placed:
        print("  [PASS] add_obstacle returned True for valid cell")
    else:
        print("  [FAIL] add_obstacle returned False for valid cell "
              "— check TODO 9")

    env.close()
    print("\nAll tests passed!  Your environment is ready for training.")
    print("Try:  python train_ppo.py")
