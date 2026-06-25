"""
student_exercises/astar_planner_template.py
============================================
STUDENT EXERCISE 2 — Implement A* Path Planning

Your task is to implement the A* algorithm on a 2D occupancy grid.

LEARNING GOALS
--------------
  1. Understand the A* algorithm: f(n) = g(n) + h(n).
  2. Implement a priority queue (Python's heapq).
  3. Implement path reconstruction via backtracking.
  4. Understand why Manhattan distance is an admissible heuristic.

BEFORE YOU CODE — READ THIS
----------------------------
A* works on a graph.  On our 2D grid:
  • Each cell (row, col) is a NODE
  • Cells share an EDGE if they are 4-connected (up/down/left/right)
  • Every edge costs 1 (one step)

A* maintains:
  g(n) = total steps taken to reach node n
  h(n) = estimated steps from n to goal (heuristic)
  f(n) = g(n) + h(n)    ← lower is better; expand lowest f first

It expands nodes in order of f-score using a MIN-HEAP (priority queue).
When you first reach the goal, you have found the optimal path.

KEY INVARIANT
-------------
The heuristic must be ADMISSIBLE: never overestimate the true cost.
Manhattan distance |Δrow| + |Δcol| is admissible on a 4-connected grid
because you need AT LEAST that many steps regardless of obstacles.

REFERENCE
---------
Compare your solution with  astar/astar_planner.py  after trying yourself.
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
#  PART A — Helper functions
# ===========================================================================

def manhattan_heuristic(cell: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Compute the Manhattan distance between two grid cells.

    Manhattan distance = |row1 - row2| + |col1 - col2|

    This is the minimum number of steps on a 4-connected grid with no
    obstacles.  It is therefore ADMISSIBLE (never overestimates).
    """
    # ------------------------------------------------------------------
    # TODO 1 — Implement the Manhattan distance
    # ------------------------------------------------------------------
    pass  # TODO: return |cell[0] - goal[0]| + |cell[1] - goal[1]|


def get_neighbours(
    cell: Tuple[int, int],
    rows: int,
    cols: int,
) -> List[Tuple[int, int]]:
    """
    Return the 4-connected valid neighbours of cell inside an (rows×cols) grid.

    A cell (r, c) has up to 4 neighbours: (r-1,c), (r+1,c), (r,c-1), (r,c+1).
    Only return neighbours that are WITHIN the grid boundaries.
    """
    # ------------------------------------------------------------------
    # TODO 2 — Implement neighbour generation
    # ------------------------------------------------------------------
    # For each delta in [(-1,0),(+1,0),(0,-1),(0,+1)]:
    #   nr, nc = cell[0] + dr, cell[1] + dc
    #   if 0 <= nr < rows and 0 <= nc < cols:  yield or append (nr, nc)
    # ------------------------------------------------------------------
    neighbours = []
    # TODO
    return neighbours


def reconstruct_path(
    came_from: dict,
    current:   Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from start to current by following came_from.

    came_from[cell] = the cell we arrived at 'cell' from.
    Start has no entry in came_from.
    """
    # ------------------------------------------------------------------
    # TODO 3 — Walk back through came_from
    # ------------------------------------------------------------------
    # path = [current]
    # while current in came_from:
    #     current = came_from[current]
    #     path.append(current)
    # path.reverse()
    # return path
    # ------------------------------------------------------------------
    pass  # TODO


# ===========================================================================
#  PART B — Core A* function
# ===========================================================================

def astar(
    grid:  np.ndarray,
    start: Tuple[int, int],
    goal:  Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Find the shortest path from start to goal on a 2D occupancy grid.

    Args:
        grid:  uint8 array shape (H, W), 0=free, 1=obstacle
        start: (row, col) start cell
        goal:  (row, col) goal cell

    Returns:
        List of (row, col) tuples from start to goal (inclusive),
        or None if no path exists.

    ALGORITHM OUTLINE
    -----------------
      1. Initialise open_set with (f_score=0+h, tie_break, start)
      2. g_score[start] = 0
      3. While open_set is not empty:
           a. Pop the cell with the lowest f_score
           b. If it's the goal → reconstruct_path() and return
           c. For each neighbour:
                i.  Skip if it's an obstacle
               ii.  tentative_g = g_score[current] + 1
              iii.  If tentative_g < g_score.get(neighbour, ∞):
                       • came_from[neighbour] = current
                       • g_score[neighbour]   = tentative_g
                       • f = tentative_g + h(neighbour, goal)
                       • Push (f, tie_break, neighbour) onto open_set
      4. Return None (no path)
    """
    rows, cols = grid.shape

    # Edge cases
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]

    # tie_break ensures Python never compares (row,col) tuples as numbers
    tie_break  = count()

    # ------------------------------------------------------------------
    # TODO 4 — Initialise the priority queue and g_score dict
    # ------------------------------------------------------------------
    # open_set: min-heap, entries = (f_score, next(tie_break), cell)
    # g_score:  dict mapping cell → cheapest cost from start
    # came_from: dict mapping cell → predecessor cell
    # Seed with the start node: f = 0 + h(start, goal)
    # ------------------------------------------------------------------
    open_set:  list = []
    g_score:   dict = {}
    came_from: dict = {}
    # TODO: heapq.heappush(open_set, (f_start, next(tie_break), start))
    # TODO: g_score[start] = 0

    while open_set:
        # ------------------------------------------------------------------
        # TODO 5 — Pop the lowest-f cell
        # ------------------------------------------------------------------
        # _, _, current = heapq.heappop(open_set)
        # ------------------------------------------------------------------
        current = None  # TODO

        # ------------------------------------------------------------------
        # TODO 6 — Check if we reached the goal
        # ------------------------------------------------------------------
        # if current == goal:
        #     return reconstruct_path(came_from, current)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 7 — Expand neighbours
        # ------------------------------------------------------------------
        # for neighbour in get_neighbours(current, rows, cols):
        #     if grid[neighbour] == 1:            # skip obstacles
        #         continue
        #     tentative_g = g_score[current] + 1
        #     if tentative_g < g_score.get(neighbour, float("inf")):
        #         came_from[neighbour] = current
        #         g_score[neighbour]   = tentative_g
        #         f = tentative_g + manhattan_heuristic(neighbour, goal)
        #         heapq.heappush(open_set, (f, next(tie_break), neighbour))
        # ------------------------------------------------------------------
        pass  # TODO: replace with the expansion logic above

    return None  # no path found


# ===========================================================================
#  PART C — AStarPlanner convenience class
# ===========================================================================

class AStarPlanner:
    """Wraps the astar() function with a state-keeping interface."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid.copy()
        self.last_path: Optional[List[Tuple[int, int]]] = None

    def plan(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Run A* and store the result in self.last_path."""
        # ------------------------------------------------------------------
        # TODO 8 — Call astar() and store result in self.last_path
        # ------------------------------------------------------------------
        pass  # TODO: self.last_path = astar(...); return self.last_path

    def add_obstacle(self, row: int, col: int) -> None:
        """Block a cell in the planner's map."""
        self.grid[row, col] = 1

    @property
    def path_length(self) -> Optional[int]:
        return len(self.last_path) if self.last_path else None


# ===========================================================================
#  Smoke-test
# ===========================================================================
if __name__ == "__main__":
    print("Testing your A* implementation …\n")

    # Simple 5×5 test
    test_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    start, goal = (0, 0), (4, 4)
    path = astar(test_grid, start, goal)

    if path is None:
        print("  [FAIL] astar() returned None for a solvable grid")
        print("         Check TODOs 4-7")
    else:
        print(f"  [PASS] Path found: length={len(path)}")
        print(f"         Path: {path}")

        # Verify start and goal
        if path[0] != start:
            print(f"  [FAIL] Path should start at {start}, got {path[0]}")
        elif path[-1] != goal:
            print(f"  [FAIL] Path should end at {goal}, got {path[-1]}")
        else:
            print(f"  [PASS] Path starts at {start}, ends at {goal}")

        # Verify no obstacles in path
        obs_in_path = [(r, c) for (r, c) in path if test_grid[r, c] == 1]
        if obs_in_path:
            print(f"  [FAIL] Path goes through obstacles: {obs_in_path}")
        else:
            print("  [PASS] No obstacles in path")

        # Verify continuity (each step is one cell)
        discontinuous = [
            (path[i], path[i+1])
            for i in range(len(path) - 1)
            if abs(path[i][0]-path[i+1][0]) + abs(path[i][1]-path[i+1][1]) != 1
        ]
        if discontinuous:
            print(f"  [FAIL] Path has jumps: {discontinuous}")
        else:
            print("  [PASS] Path is continuous (each step = 1 cell)")

    # Test on the no-path case
    blocked = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
    result = astar(blocked, (0, 0), (2, 2))
    if result is None:
        print("  [PASS] Returns None when no path exists")
    else:
        print("  [FAIL] Should return None when path is blocked")

    # Test on the training map
    print("\nTesting on training map …")
    map_path = os.path.join(_PROJECT_ROOT, "maps", "training_map.pgm")
    if os.path.exists(map_path):
        grid = load_pgm(map_path)
    else:
        grid = create_training_map(40, 40)

    planner = AStarPlanner(grid)
    path = planner.plan((2, 2), (37, 37))

    if path:
        print(f"  [PASS] Training map path found: {planner.path_length} cells")
    else:
        print("  [FAIL] No path found on training map — check your implementation")
