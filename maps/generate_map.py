"""
generate_map.py — Build and save 2D occupancy grid maps as PGM files.
======================================================================

WHAT IS A PGM FILE?
-------------------
PGM (Portable GrayMap) is a simple image format used widely in robotics
to represent 2D occupancy grids. ROS (Robot Operating System) uses it as
the standard map file format for its map_server.

Two PGM variants are common:
  P2  — ASCII text  (human-readable, our generated maps use this)
  P5  — binary data (compact, what ROS map_server produces)

OCCUPANCY CONVENTION IN THIS PROJECT
--------------------------------------
  In PGM file   →  In NumPy grid
  255 (white)   →  0  (free space)
  0   (black)   →  1  (obstacle / wall)

ROS maps may also contain a third value:
  205 (grey)    →  1  (unknown — treated as obstacle for safety)

USAGE
-----
    # Generate the synthetic tutorial maps:
    python maps/generate_map.py

    # Import an existing PGM map (e.g. from module 5):
    python maps/generate_map.py --import ../module5/path-planning-viz/map.pgm

    # From Python code:
    from maps.generate_map import load_pgm, import_map, create_training_map
"""

import os
import numpy as np
from typing import Optional, Tuple


# ===========================================================================
#  MAP CREATION  (synthetic tutorial maps)
# ===========================================================================

def create_training_map(width: int = 40, height: int = 40) -> np.ndarray:
    """
    Create a 40×40 occupancy grid with two wall structures and a small
    obstacle cluster, giving the drone a meaningful navigation challenge.

    Map layout (S = start (2,2), G = goal (37,37)):

        S . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        WALL (row 15): ############  GAP  ######################  WALL (col 30)
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  WALL . . .
        . . . . . . . . . . CLUSTER . . . . . . . . . . . . . . . .  GAP  . . .
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G

    Returns:
        grid: np.ndarray shape (height, width), dtype uint8, 0=free 1=obstacle
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[0, :]  = 1;  grid[-1, :] = 1   # border walls
    grid[:, 0]  = 1;  grid[:, -1] = 1
    grid[15, 4:29] = 1;  grid[15, 18] = 0;  grid[15, 19] = 0  # H-wall + gap
    grid[15:38, 30] = 1; grid[26, 30] = 0;  grid[27, 30] = 0  # V-wall + gap
    grid[28:35, 9:14] = 1                                       # obstacle cluster
    return grid


def create_simple_map(width: int = 20, height: int = 20) -> np.ndarray:
    """
    Minimal 20×20 map for quick debugging.
    Start (2,2) → Goal (17,17).
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[0, :] = 1;  grid[-1, :] = 1
    grid[:, 0] = 1;  grid[:, -1] = 1
    grid[8, 3:14] = 1;  grid[8, 8]    = 0
    grid[8:16, 14] = 1; grid[13, 14]  = 0
    return grid


# ===========================================================================
#  PGM I/O
# ===========================================================================

def save_pgm(grid: np.ndarray, filepath: str) -> None:
    """
    Write an occupancy grid to a P2 (ASCII) PGM file.

    Internal convention (0=free, 1=obstacle) → PGM convention (255=free, 0=obstacle).
    Saved as human-readable P2 so students can open it in a text editor.

    Args:
        grid:     2-D uint8 array, 0=free, 1=obstacle
        filepath: destination path (.pgm)
    """
    pgm = np.where(grid == 0, 255, 0).astype(np.uint8)
    height, width = pgm.shape
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("P2\n")
        f.write("# 2D Occupancy Grid — RL Drone Path Planning Tutorial\n")
        f.write("# Convention: 255=free space (white), 0=obstacle (black)\n")
        f.write(f"{width} {height}\n255\n")
        np.savetxt(f, pgm, fmt="%d", delimiter=" ")
    print(f"  Saved: {filepath}  ({width}×{height})")


def load_pgm(filepath: str) -> np.ndarray:
    """
    Load any PGM file (P2 ASCII **or** P5 binary) and return an occupancy grid.

    Handles two common pixel conventions automatically:
      • Standard binary map  (0=obstacle, 255=free)
          threshold: >= 250 → free,  else → obstacle
      • ROS three-value map  (0=obstacle, 205=unknown, 254=free)
          same threshold: >= 250 → free,  else → obstacle
          (unknown cells are treated as obstacles for safety)

    Args:
        filepath: path to a .pgm file (P2 or P5)

    Returns:
        grid: 2-D uint8 array, 0=free, 1=obstacle
    """
    with open(filepath, "rb") as f:
        # ---- Read header ----
        magic = _read_token(f).decode()
        if magic not in ("P2", "P5"):
            raise ValueError(f"Unsupported PGM format: '{magic}'. Expected P2 or P5.")

        width  = int(_read_token(f))
        height = int(_read_token(f))
        maxval = int(_read_token(f))

        # ---- Read pixel data ----
        if magic == "P2":
            # ASCII: read remaining text
            rest = f.read().decode()
            pixels = np.array(rest.split(), dtype=np.uint16).reshape(height, width)
        else:
            # P5 binary: one or two bytes per pixel depending on maxval
            if maxval < 256:
                raw    = f.read(height * width)
                pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
            else:
                # 16-bit big-endian
                raw    = f.read(height * width * 2)
                pixels = np.frombuffer(raw, dtype=">u2").reshape(height, width)
                pixels = (pixels * 255 // maxval).astype(np.uint8)

    # ---- Normalise to [0, 255] if maxval != 255 ----
    if maxval not in (255, 65535):
        pixels = (pixels.astype(np.uint32) * 255 // maxval).astype(np.uint8)

    # ---- Convert to occupancy grid ----
    # >= 250 → free (0),  < 250 → obstacle (1)
    # This correctly handles both simple binary maps AND ROS three-value maps
    # (where 205 = unknown is conservatively treated as obstacle).
    grid = np.where(pixels >= 250, 0, 1).astype(np.uint8)
    return grid


# ===========================================================================
#  IMPORT AN EXTERNAL MAP
# ===========================================================================

def import_map(
    source_path:      str,
    dest_path:        str,
    start_search_box: Tuple[int, int, int, int] = None,
    goal_search_box:  Tuple[int, int, int, int] = None,
    clear_radius:     int = 4,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Load an external PGM map, save a copy into this project's maps/ folder,
    and automatically suggest start and goal positions.

    This is the function to call when you want to reuse a map from another
    project (e.g. the path-planning map from module 5).

    Args:
        source_path:       Path to the source .pgm file (P2 or P5).
        dest_path:         Where to save the converted copy inside maps/.
        start_search_box:  (row_min, row_max, col_min, col_max) region to
                           search for the start position.  Defaults to the
                           top-left quadrant of the map.
        goal_search_box:   Same for the goal.  Defaults to bottom-right.
        clear_radius:      Minimum clear-cell radius around start/goal
                           (avoids placing them directly next to a wall).

    Returns:
        grid:      Occupancy grid (0=free, 1=obstacle).
        start_pos: Suggested (row, col) start position.
        goal_pos:  Suggested (row, col) goal position.

    Example:
        grid, start, goal = import_map(
            source_path = "../../module5/path-planning-viz/map.pgm",
            dest_path   = "maps/imported_map.pgm",
        )
        print(f"Start: {start}, Goal: {goal}")
    """
    print(f"  Loading source map: {source_path}")
    grid = load_pgm(source_path)
    h, w = grid.shape
    print(f"  Map size: {w}×{h}  |  "
          f"Free: {np.sum(grid==0)} cells  |  "
          f"Obstacle: {np.sum(grid==1)} cells")

    # ---- Find start position ----
    if start_search_box is None:
        start_search_box = (clear_radius, h // 4, clear_radius, w // 4)
    start_pos = _find_clear_cell(grid, *start_search_box, clear_radius)
    if start_pos is None:
        raise RuntimeError(
            "Could not find a clear start cell in the top-left quadrant. "
            "Try reducing clear_radius or specifying start_search_box manually."
        )

    # ---- Find goal position ----
    if goal_search_box is None:
        goal_search_box = (3 * h // 4, h - clear_radius,
                           3 * w // 4, w - clear_radius)
    goal_pos = _find_clear_cell(grid, *goal_search_box, clear_radius)
    if goal_pos is None:
        raise RuntimeError(
            "Could not find a clear goal cell in the bottom-right quadrant. "
            "Try reducing clear_radius or specifying goal_search_box manually."
        )

    # ---- Save the copy ----
    save_pgm(grid, dest_path)

    manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
    print(f"  Start:    {start_pos}")
    print(f"  Goal:     {goal_pos}")
    print(f"  Manhattan distance: {manhattan} steps")
    print(f"  Recommended max_steps: {manhattan * 3}")

    return grid, start_pos, goal_pos


# ===========================================================================
#  Helpers
# ===========================================================================

def _read_token(f) -> bytes:
    """
    Read the next whitespace-delimited token from a binary file handle,
    skipping comment lines (lines starting with '#').

    Used to parse PGM headers which may have embedded comments.
    """
    token = b""
    # Skip leading whitespace and comments
    while True:
        byte = f.read(1)
        if not byte:
            break
        if byte == b"#":
            f.readline()   # discard rest of comment line
        elif byte not in (b" ", b"\t", b"\r", b"\n"):
            token = byte
            break
    # Read remaining token characters
    while True:
        byte = f.read(1)
        if not byte or byte in (b" ", b"\t", b"\r", b"\n"):
            break
        token += byte
    return token


def _find_clear_cell(
    grid:     np.ndarray,
    row_min:  int,
    row_max:  int,
    col_min:  int,
    col_max:  int,
    radius:   int,
) -> Optional[Tuple[int, int]]:
    """
    Scan the bounding box for the first free cell whose neighbours are all
    free within the given radius.  Returns (row, col) or None.
    """
    h, w = grid.shape
    for r in range(max(row_min, radius), min(row_max, h - radius)):
        for c in range(max(col_min, radius), min(col_max, w - radius)):
            if grid[r - radius : r + radius + 1, c - radius : c + radius + 1].any():
                continue
            return (r, c)
    return None


def find_start_goal(
    grid:        np.ndarray,
    clear_radius: int = 4,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Convenience wrapper: auto-detect start (top-left area) and goal
    (bottom-right area) on any occupancy grid.

    Returns:
        (start_pos, goal_pos)  — each is (row, col) or None if not found.
    """
    h, w = grid.shape
    start = _find_clear_cell(grid, clear_radius, h // 4,
                              clear_radius, w // 4, clear_radius)
    goal  = _find_clear_cell(grid, 3 * h // 4, h - clear_radius,
                              3 * w // 4, w - clear_radius, clear_radius)
    return start, goal


# ===========================================================================
#  ADD CUSTOM OBSTACLES
# ===========================================================================

def add_obstacles(grid: np.ndarray, coords: list) -> np.ndarray:
    """
    Place obstacles at the specified (row, col) coordinates.

    Use this to customise any map without modifying the creation functions.
    Out-of-bounds coordinates are silently skipped.

    Args:
        grid:   2-D uint8 occupancy array (0=free, 1=obstacle).
        coords: List of (row, col) tuples, e.g. [(5, 10), (6, 10), (7, 10)].

    Returns:
        The same grid array with obstacles written in-place (also returned
        for convenience so you can chain calls).

    Example:
        grid = create_training_map()
        grid = add_obstacles(grid, CUSTOM_OBSTACLES)
        save_pgm(grid, "maps/training_map.pgm")
    """
    h, w = grid.shape
    placed, skipped = [], []
    for r, c in coords:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = 1
            placed.append((r, c))
        else:
            skipped.append((r, c))
    if placed:
        print(f"  Added {len(placed)} custom obstacle(s): {placed}")
    if skipped:
        print(f"  Skipped {len(skipped)} out-of-bounds coordinate(s): {skipped}")
    return grid


# ===========================================================================
#  MAIN — generate tutorial maps OR import an external map
# ===========================================================================

# -----------------------------------------------------------------------
# EDIT THIS LIST to place extra obstacles on the training map.
# Each entry is (row, col).  Rows and cols start at 0.
# Leave the list empty [] for the default map with no extra obstacles.
#
# Example — block the horizontal wall gap and add two more obstacles:
#   CUSTOM_OBSTACLES = [
#       (15, 18),   # blocks the H-wall gap
#       (15, 19),   # blocks second gap cell
#       (22, 25),   # freestanding obstacle mid-map
#   ]
# -----------------------------------------------------------------------
CUSTOM_OBSTACLES = [
    # (row, col),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate tutorial PGM maps, or import an existing map."
    )
    parser.add_argument(
        "--import", dest="import_path", default=None, metavar="PATH",
        help="Import an external .pgm file and save it as maps/imported_map.pgm"
    )
    parser.add_argument(
        "--obstacles", dest="obstacles", nargs="+", default=None,
        metavar="ROW,COL",
        help=(
            "Extra obstacles to add, as row,col pairs.  "
            "Example: --obstacles 15,18 20,25 30,10"
        ),
    )
    args = parser.parse_args()

    maps_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Parse --obstacles flag into (row, col) list ----
    cli_obstacles = []
    if args.obstacles:
        for token in args.obstacles:
            try:
                r, c = token.split(",")
                cli_obstacles.append((int(r), int(c)))
            except ValueError:
                print(f"  Warning: could not parse obstacle '{token}' — "
                      f"expected format ROW,COL (e.g. 15,18).  Skipping.")

    # Merge: variables defined above + anything passed on the command line
    all_extra_obstacles = CUSTOM_OBSTACLES + cli_obstacles

    if args.import_path:
        # ---- Import mode ----
        dest = os.path.join(maps_dir, "imported_map.pgm")
        print(f"\nImporting external map …")
        grid, start, goal = import_map(args.import_path, dest)

        if all_extra_obstacles:
            print("\nApplying custom obstacles to imported_map.pgm …")
            add_obstacles(grid, all_extra_obstacles)
            save_pgm(grid, dest)

        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

        print(f"""
Imported successfully!  To use this map in training and testing:

  In train_ppo.py  (top of file):
    MAP_PATH  = os.path.join(PROJECT_ROOT, "maps", "imported_map.pgm")
    START_POS = {start}
    GOAL_POS  = {goal}
    MAX_STEPS = {manhattan * 3}
    TOTAL_TIMESTEPS = 1_000_000   # larger map needs more training

  In test_rl.py  (top of file):
    MAP_PATH  = os.path.join(PROJECT_ROOT, "maps", "imported_map.pgm")
    START_POS = {start}
    GOAL_POS  = {goal}
    MAX_STEPS = {manhattan * 3}
    # Dynamic obstacle: pick a (row, col) that lies on the A* path.
    # Run python astar/astar_planner.py first to see the path, then choose.
""")

    else:
        # ---- Generate mode (default) ----
        print("Generating maps...")

        training_grid = create_training_map(40, 40)

        if all_extra_obstacles:
            print("\nApplying custom obstacles to training_map.pgm …")
            add_obstacles(training_grid, all_extra_obstacles)

        save_pgm(training_grid, os.path.join(maps_dir, "training_map.pgm"))

        simple_grid = create_simple_map(20, 20)
        save_pgm(simple_grid, os.path.join(maps_dir, "simple_map.pgm"))

        print("\nAll maps generated.")
        print("  maps/training_map.pgm  — 40×40, used for PPO training")
        print("  maps/simple_map.pgm    — 20×20, used for quick tests")
        print()
        print("To add custom obstacles (edit the list in this file):")
        print("  CUSTOM_OBSTACLES = [(15, 18), (20, 25), (30, 10)]")
        print()
        print("Or pass them on the command line:")
        print("  python maps/generate_map.py --obstacles 15,18 20,25 30,10")
        print()
        print("To import your own map (e.g. from module 5):")
        print("  python maps/generate_map.py "
              "--import ../../module5/path-planning-viz/map.pgm")
