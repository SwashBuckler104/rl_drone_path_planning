"""
astar_planner.py — A* Path Planning on a 2D Occupancy Grid
============================================================

WHAT IS A*?
-----------
A* is a graph search algorithm that finds the shortest path between two
nodes.  On a grid, each cell is a node and edges connect adjacent cells.
A* is guaranteed to find the OPTIMAL (shortest) path if one exists.

HOW A* WORKS
------------
A* maintains two data structures:

  open_set  — cells not yet fully explored, sorted by f-score (priority queue)
  came_from — a map recording which cell we came from to reach each cell

For each cell it computes:
  g(n) = actual cost from start to n (number of steps taken so far)
  h(n) = heuristic estimate from n to goal (Manhattan distance)
  f(n) = g(n) + h(n)  ← A* always expands the cell with lowest f

WHY MANHATTAN DISTANCE AS HEURISTIC?
--------------------------------------
On a 4-connected grid (up/down/left/right only) every step costs 1.
The minimum steps from any cell to the goal is the Manhattan distance:
  |row_current - row_goal| + |col_current - col_goal|

This heuristic is ADMISSIBLE — it never overestimates the true cost —
which guarantees A* returns the optimal path.

WHY A* IS EXCELLENT FOR KNOWN STATIC MAPS
------------------------------------------
  ✓  Guaranteed optimal path
  ✓  Fast (polynomial in map size)
  ✓  No training needed — just run it
  ✓  Deterministic and explainable

WHY A* STRUGGLES WITH DYNAMIC ENVIRONMENTS
--------------------------------------------
  ✗  Replanning is required whenever the map changes
  ✗  If a new obstacle blocks the planned path, A* must be re-run
     from scratch (or with an incremental planner like D* Lite)
  ✗  The plan is computed once, offline — it has no sensors
"""

import heapq
import os
import sys
from itertools import count
from typing import List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maps.generate_map import load_pgm, create_training_map


# ===========================================================================
#  Core A* algorithm
# ===========================================================================

def astar(
    grid:  np.ndarray,
    start: Tuple[int, int],
    goal:  Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Find the shortest path on a 4-connected 2D grid using A*.

    Args:
        grid:  2-D uint8 array, 0=free, 1=obstacle  (shape: H×W)
        start: (row, col) start cell
        goal:  (row, col) goal cell

    Returns:
        path: list of (row, col) tuples from start to goal (inclusive),
              or None if no path exists.

    Complexity:
        Time:  O(W × H × log(W × H))  — priority queue operations
        Space: O(W × H)               — storing g_score and came_from
    """
    rows, cols = grid.shape

    # Validate start and goal
    if not _in_bounds(start, rows, cols) or not _in_bounds(goal, rows, cols):
        return None
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]

    # -----------------------------------------------------------------------
    # Priority queue entries: (f_score, tie_breaker, (row, col))
    #
    # We add a tie_breaker counter so Python never tries to compare tuples
    # as (row, col) when f_scores are equal.  Without this, tuples of ints
    # are fine, but in general this pattern is safer.
    # -----------------------------------------------------------------------
    tiebreak = count()
    open_set: list = []
    heapq.heappush(open_set, (0 + _h(start, goal), next(tiebreak), start))

    # g_score[cell] = cheapest known cost from start to cell
    g_score: dict = {start: 0}

    # came_from[cell] = which cell we arrived from (for path reconstruction)
    came_from: dict = {}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        # ---- Goal reached ----
        if current == goal:
            return _reconstruct(came_from, current)

        # ---- Expand neighbours (4-connected) ----
        for neighbour in _neighbours(current, rows, cols):
            nr, nc = neighbour

            # Skip obstacles
            if grid[nr, nc] == 1:
                continue

            # Each step costs 1
            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour]   = tentative_g
                f = tentative_g + _h(neighbour, goal)
                heapq.heappush(open_set, (f, next(tiebreak), neighbour))

    # Open set exhausted — no path exists
    return None


# ===========================================================================
#  Planner class (wraps the function with convenience methods)
# ===========================================================================

class AStarPlanner:
    """
    Stateful A* planner that operates on a fixed map grid.

    Typical usage:
        planner = AStarPlanner(grid)
        path = planner.plan((2, 2), (37, 37))
        print(f"Path length: {planner.path_length}")

    The planner supports updating the grid (e.g., to add a new obstacle)
    and re-running plan() without creating a new object.
    """

    def __init__(self, grid: np.ndarray):
        """
        Args:
            grid: 2-D uint8 occupancy array, 0=free, 1=obstacle.
                  The planner stores a COPY so external changes to the
                  original array do not affect it.
        """
        self.grid = grid.copy()
        self.last_path: Optional[List[Tuple[int, int]]] = None

    # ------------------------------------------------------------------
    def plan(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* and return the path (or None if unreachable).

        The result is also stored in self.last_path for later inspection.
        """
        self.last_path = astar(self.grid, start, goal)
        return self.last_path

    # ------------------------------------------------------------------
    def update_grid(self, grid: np.ndarray) -> None:
        """Replace the stored grid (e.g., after adding a new obstacle)."""
        self.grid = grid.copy()

    # ------------------------------------------------------------------
    def add_obstacle(self, row: int, col: int) -> None:
        """Add a single obstacle cell to the planner's map."""
        self.grid[row, col] = 1

    # ------------------------------------------------------------------
    @property
    def path_length(self) -> Optional[int]:
        """Number of cells in the last computed path (None if no path)."""
        return len(self.last_path) if self.last_path else None

    # ------------------------------------------------------------------
    def visualise(
        self,
        ax=None,
        title: str = "A* Path",
        show_costs: bool = False,
    ):
        """
        Draw the map and the most recently computed path on a matplotlib Axes.

        Args:
            ax:         matplotlib Axes to draw on.  Creates a new figure if None.
            title:      Axes title.
            show_costs: If True, annotate each path cell with its step index.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        # Map background — 0=free (white), 1=obstacle (dark grey)
        display = np.where(self.grid == 0, 255, 30).astype(np.uint8)
        ax.imshow(display, cmap="gray", vmin=0, vmax=255, origin="upper",
                  interpolation="nearest")

        if self.last_path:
            rows_p = [p[0] for p in self.last_path]
            cols_p = [p[1] for p in self.last_path]

            # Path line
            ax.plot(cols_p, rows_p, "-", color="#2196F3",
                    linewidth=2, label="A* path")

            # Start marker
            ax.plot(cols_p[0], rows_p[0], "o",
                    color="#4CAF50", markersize=10, label="Start",
                    markeredgecolor="white", markeredgewidth=1.5)

            # Goal marker
            ax.plot(cols_p[-1], rows_p[-1], "*",
                    color="#FF9800", markersize=14, label="Goal",
                    markeredgecolor="white", markeredgewidth=1.5)

            if show_costs:
                for i, (r, c) in enumerate(self.last_path):
                    ax.text(c, r, str(i), fontsize=5,
                            ha="center", va="center", color="white")

        ax.set_title(f"{title}  (length: {self.path_length})")
        ax.legend(loc="upper right", fontsize=8)
        ax.axis("off")


# ===========================================================================
#  Private helpers
# ===========================================================================

def _h(cell: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """Manhattan distance heuristic — admissible on 4-connected grids."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def _neighbours(cell: Tuple[int, int], rows: int, cols: int):
    """Yield the valid 4-connected neighbours of a cell."""
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def _in_bounds(cell: Tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _reconstruct(
    came_from: dict,
    current:   Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Walk back through came_from to reconstruct the path."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ===========================================================================
#  Smoke-test
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading / generating training map …")
    maps_dir = os.path.join(_PROJECT_ROOT, "maps")
    map_path = os.path.join(maps_dir, "training_map.pgm")

    if os.path.exists(map_path):
        grid = load_pgm(map_path)
    else:
        grid = create_training_map(40, 40)

    start = (2, 2)
    goal  = (37, 37)

    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    if path:
        print(f"Path found!  Length = {planner.path_length} cells")
        print(f"First 5 cells: {path[:5]}")
        print(f"Last  5 cells: {path[-5:]}")

        fig, ax = plt.subplots(figsize=(7, 7))
        planner.visualise(ax=ax, title="A* on Training Map")
        plt.tight_layout()
        out = os.path.join(_PROJECT_ROOT, "outputs", "astar_smoke_test.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"Saved: {out}")
        plt.show()
    else:
        print("No path found — check that start and goal are not blocked.")
