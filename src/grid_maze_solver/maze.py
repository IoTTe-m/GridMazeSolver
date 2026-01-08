from typing import List, Tuple, Optional, Sequence
import random
import numpy as np

import matplotlib.pyplot as plt

# Directions (0=N,1=E,2=S,3=W) and their vectors (row, col)
DIRECTION_VECTORS: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

class Maze:
    """A grid maze with walls (1) and free cells (0), generated using recursive backtracking."""

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
        self.remove_walls_count = remove_walls
        self._generate()
        if remove_walls > 0:
            self._remove_random_walls(remove_walls)

    def regenerate(self, seed: Optional[int] = None) -> None:
        """Regenerate the maze with a new seed."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.grid[:] = 1
        self._generate()
        if hasattr(self, 'remove_walls_count') and self.remove_walls_count > 0:
            self._remove_random_walls(self.remove_walls_count)

    def _generate(self) -> None:
        grid_height, grid_width = self.height, self.width
        cell_stack: List[Tuple[int, int]] = [(1, 1)]
        self.grid[1, 1] = 0
        while cell_stack:
            current_row, current_col = cell_stack[-1]
            unvisited_neighbors: List[Tuple[int, int]] = []
            for delta_row, delta_col in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                neighbor_row, neighbor_col = current_row + delta_row, current_col + delta_col
                if 1 <= neighbor_row < grid_height - 1 and 1 <= neighbor_col < grid_width - 1 and self.grid[neighbor_row, neighbor_col] == 1:
                    unvisited_neighbors.append((neighbor_row, neighbor_col))
            if unvisited_neighbors:
                neighbor_row, neighbor_col = random.choice(unvisited_neighbors)
                wall_row, wall_col = (current_row + neighbor_row) // 2, (current_col + neighbor_col) // 2
                self.grid[neighbor_row, neighbor_col] = 0
                self.grid[wall_row, wall_col] = 0
                cell_stack.append((neighbor_row, neighbor_col))
            else:
                cell_stack.pop()

    def _remove_random_walls(self, count: int) -> None:
        """Remove up to k random interior walls."""
        wall_candidates: List[Tuple[int, int]] = []
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if (row + col) % 2 == 1 and self.grid[row, col] == 1:
                    if (row, col) != self.start and (row, col) != self.goal:
                        wall_candidates.append((row, col))
        if not wall_candidates:
            return
        count = min(count, len(wall_candidates))
        chosen_walls = random.sample(wall_candidates, count)
        for (row, col) in chosen_walls:
            self.grid[row, col] = 0

    def set_start_goal(self, start: Tuple[int, int] = (1, 1), goal: Optional[Tuple[int, int]] = None) -> None:
        if goal is None:
            goal = (self.height - 2, self.width - 2)
        self.start, self.goal = start, goal

    def is_free(self, pos: Tuple[int, int]) -> bool:
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width and self.grid[row, col] == 0

    def render(self, path_positions: Optional[Sequence[Tuple[int, int]]] = None, savepath: Optional[str] = None, figsize: Tuple[int, int] = (6, 6)) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.grid, cmap="gray_r", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if path_positions:
            x_coords = [pos[1] for pos in path_positions]
            y_coords = [pos[0] for pos in path_positions]
            ax.plot(x_coords, y_coords, linewidth=2)
            ax.scatter([self.start[1], self.goal[1]], [self.start[0], self.goal[0]], c="red")
        if savepath:
            plt.savefig(savepath, bbox_inches="tight")
            print(f"Saved visual to {savepath}")
        plt.close(fig)
