from typing import Tuple, Dict, Any, Sequence
import numpy as np
from .maze import Maze
from .simulator import Simulator

class Evaluator:
    """Wraps the simulator and provides a scoring function for genomes."""

    def __init__(self, maze: Maze, max_steps: int = 200, n_states: int = 8, n_stack_syms: int = 4) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.sim = Simulator(maze, max_steps=max_steps, n_states=n_states, n_stack_syms=n_stack_syms)

    def score(self, genome: Sequence[int]) -> Tuple[float, Dict[str, Any]]:
        out = self.sim.run_genome(genome)
        score = 0.0

        if out["reached"]:
            score += 100_000.0
            score += (self.max_steps - out["steps"]) * 10.0
            score -= out["collisions"] * 20.0
            score -= out["stack_underflow"] * 50.0
        else:
            path = np.array(out["path"], dtype=np.int32)  # shape (T,2)
            goal = np.array(self.maze.goal, dtype=np.int32)
            if path.size == 0:
                min_dist = abs(self.maze.start[0] - goal[0]) + abs(self.maze.start[1] - goal[1])
            else:
                manh = np.abs(path - goal).sum(axis=1)  # Manhattan distances per step
                min_dist = int(manh.min())

            score -= min_dist * 1000.0
            score -= out["collisions"] * 10.0
            score -= out["stack_underflow"] * 5.0

        return float(score), out
