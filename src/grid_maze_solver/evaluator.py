from typing import Tuple, Dict, Any, Sequence, List, Optional
import numpy as np

from .maze import Maze
from .simulator import Simulator


class Evaluator:
    """Wraps the simulator and provides a scoring function for genomes."""

    def __init__(self, maze: Maze, max_steps: int = 200, n_states: int = 8, n_stack_syms: int = 4) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.n_states = n_states
        self.n_stack_syms = n_stack_syms
        self.simulator = Simulator(maze, max_steps=max_steps, num_states=n_states, num_stack_symbols=n_stack_syms)
        self._extra_mazes: List[Maze] = []
    
    def set_mazes(self, mazes: List[Maze]) -> None:
        """Set multiple mazes for multi-maze evaluation."""
        if mazes:
            self.maze = mazes[0]
            self.simulator = Simulator(self.maze, max_steps=self.max_steps, 
                                       num_states=self.n_states, num_stack_symbols=self.n_stack_syms)
            self._extra_mazes = mazes[1:]
        else:
            self._extra_mazes = []
    
    def _score_single(self, genome: Sequence[int], maze: Maze, simulator: Simulator) -> Tuple[float, Dict[str, Any]]:
        """Score a genome on a single maze."""
        simulation_result = simulator.run_genome(genome)
        fitness_score = 0.0

        if simulation_result["reached"]:
            fitness_score += 100_000.0
            fitness_score += (self.max_steps - simulation_result["steps"]) * 60.0
            fitness_score -= simulation_result["collisions"] * 20.0
            fitness_score -= simulation_result["stack_underflow"] * 50.0
        else:
            path_array = np.array(simulation_result["path"], dtype=np.int32)
            goal_position = np.array(maze.goal, dtype=np.int32)
            if path_array.size == 0:
                min_distance = abs(maze.start[0] - goal_position[0]) + abs(maze.start[1] - goal_position[1])
            else:
                manhattan_distances = np.abs(path_array - goal_position).sum(axis=1)
                min_distance = int(manhattan_distances.min())

            fitness_score -= min_distance * 1000.0
            fitness_score -= simulation_result["collisions"] * 10.0
            fitness_score -= simulation_result["stack_underflow"] * 5.0
        
        fitness_score += simulation_result["unique_cells"] * 50.0

        return float(fitness_score), simulation_result

    def score(self, genome: Sequence[int]) -> Tuple[float, Dict[str, Any]]:
        """Score genome, averaging over multiple mazes if configured."""
        primary_score, primary_result = self._score_single(genome, self.maze, self.simulator)
        
        if not self._extra_mazes:
            return primary_score, primary_result
        
        total_score = primary_score
        for extra_maze in self._extra_mazes:
            extra_simulator = Simulator(extra_maze, max_steps=self.max_steps,
                                        num_states=self.n_states, num_stack_symbols=self.n_stack_syms)
            extra_score, _ = self._score_single(genome, extra_maze, extra_simulator)
            total_score += extra_score
        
        avg_score = total_score / (1 + len(self._extra_mazes))
        return avg_score, primary_result
