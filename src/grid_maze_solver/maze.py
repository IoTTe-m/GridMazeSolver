from typing import List, Tuple, Optional, Sequence
import random
import numpy as np

# matplotlib optional for rendering
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # render disabled if matplotlib missing

# Directions (0=N,1=E,2=S,3=W) and their vectors (row, col)
DIRECTION_VECTORS: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

class Maze:
    """A simple grid maze with walls (1) and free cells (0).
    A perfect maze is generated with a recursive backtracker. Optionally,
    a small number of random walls can be removed after generation.
    """

    def __init__(self, width: int = 21, height: int = 21, seed: Optional[int] = None, remove_walls: int = 0) -> None:
        assert width % 2 == 1 and height % 2 == 1, "width and height must be odd"
        self.width = width
        self.height = height
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.grid: np.ndarray = np.ones((height, width), dtype=np.int8)
        self.start: Tuple[int, int] = (1, 1)
        self.goal: Tuple[int, int] = (height - 2, width - 2)
        self._generate()
        if remove_walls > 0:
            self._remove_random_walls(remove_walls)

    def _generate(self) -> None:
        H, W = self.height, self.width
        stack: List[Tuple[int, int]] = [(1, 1)]
        self.grid[1, 1] = 0
        while stack:
            x, y = stack[-1]
            neighbors: List[Tuple[int, int]] = []
            # check four directions, jump by 2 to find neighbouring cell
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 1 <= nx < H - 1 and 1 <= ny < W - 1 and self.grid[nx, ny] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                wx, wy = (x + nx) // 2, (y + ny) // 2
                self.grid[nx, ny] = 0
                self.grid[wx, wy] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _remove_random_walls(self, k: int) -> None:
        """Remove up to k random interior walls (not borders, not start/goal).
        Selects from cells currently equal to 1 (walls) and flips them to 0."""
        # candidate walls exclude border cells and start/goal
        candidates: List[Tuple[int, int]] = [
            (i, j)
            for i in range(2, self.height - 2, 2)
            for j in range(2, self.width - 2, 2)
            if self.grid[i, j] == 1 and (i, j) != self.start and (i, j) != self.goal
        ]
        if not candidates:
            return
        k = min(k, len(candidates))
        chosen = random.sample(candidates, k)
        for (i, j) in chosen:
            self.grid[i, j] = 0

    def set_start_goal(self, start: Tuple[int, int] = (1, 1), goal: Optional[Tuple[int, int]] = None) -> None:
        if goal is None:
            goal = (self.height - 2, self.width - 2)
        self.start, self.goal = start, goal

    def is_free(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x, y] == 0

    def render(self, path_positions: Optional[Sequence[Tuple[int, int]]] = None, savepath: Optional[str] = None, figsize: Tuple[int, int] = (6, 6)) -> None:
        if plt is None:
            print("matplotlib not available; render skipped.")
            return
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.grid, cmap="gray_r", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if path_positions:
            xs = [p[1] for p in path_positions]
            ys = [p[0] for p in path_positions]
            ax.plot(xs, ys, linewidth=2)
            ax.scatter([self.start[1], self.goal[1]], [self.start[0], self.goal[0]], c="red")
        if savepath:
            plt.savefig(savepath, bbox_inches="tight")
            print(f"Saved visual to {savepath}")
        plt.close(fig)
