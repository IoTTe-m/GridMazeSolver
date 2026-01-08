from typing import List, Tuple, Sequence, Dict, Any

import numpy as np
from numba import njit

from .maze import Maze

ACTION_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_TO_CHAR = {ACTION_FORWARD: "^", ACTION_TURN_LEFT: "<", ACTION_TURN_RIGHT: ">"}

STACK_NOP = 0
STACK_PUSH = 1
STACK_POP = 2
STACK_OP_TO_STR = {STACK_NOP: "NOP", STACK_PUSH: "PUSH", STACK_POP: "POP"}

DIR_VECS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int32)

@njit(fastmath=True)
def _run_genome_numba(
    genome: np.ndarray,
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    max_steps: int,
    num_states: int,
    num_stack_symbols: int,
    per_state_block: int
):
    start_row, start_col = start_pos
    goal_row, goal_col = goal_pos
    grid_height, grid_width = grid.shape

    current_row, current_col = start_row, start_col
    direction_index = 0
    current_state = 0
    steps_taken = 0
    collision_count = 0
    stack_underflow_count = 0

    stack = [0]
    stack.pop()

    path_rows = [start_row]
    path_cols = [start_col]
    reached_goal = False
    
    # Track unique states visited
    states_visited = np.zeros(num_states, dtype=np.int32)
    states_visited[0] = 1
    
    # Track unique cells visited
    cells_visited = np.zeros((grid_height, grid_width), dtype=np.int32)
    cells_visited[start_row, start_col] = 1
    
    for _ in range(max_steps):
        if current_row == goal_row and current_col == goal_col:
            reached_goal = True
            break

        forward_delta_row = DIR_VECS[direction_index, 0]
        forward_delta_col = DIR_VECS[direction_index, 1]

        left_direction_index = (direction_index - 1) % 4
        left_delta_row = DIR_VECS[left_direction_index, 0]
        left_delta_col = DIR_VECS[left_direction_index, 1]

        right_direction_index = (direction_index + 1) % 4
        right_delta_row = DIR_VECS[right_direction_index, 0]
        right_delta_col = DIR_VECS[right_direction_index, 1]

        # Check directions
        next_row, next_col = current_row + forward_delta_row, current_col + forward_delta_col
        is_front_free = 0
        if 0 <= next_row < grid_height and 0 <= next_col < grid_width:
            if grid[next_row, next_col] == 0:
                is_front_free = 1

        next_row, next_col = current_row + left_delta_row, current_col + left_delta_col
        is_left_free = 0
        if 0 <= next_row < grid_height and 0 <= next_col < grid_width:
            if grid[next_row, next_col] == 0:
                is_left_free = 1

        next_row, next_col = current_row + right_delta_row, current_col + right_delta_col
        is_right_free = 0
        if 0 <= next_row < grid_height and 0 <= next_col < grid_width:
            if grid[next_row, next_col] == 0:
                is_right_free = 1

        top_symbol_index = num_stack_symbols
        if len(stack) > 0:
            top_symbol_index = stack[-1] % num_stack_symbols

        sensor_input_index = (is_left_free << 2) | (is_front_free << 1) | is_right_free
        genome_base_index = current_state * per_state_block + top_symbol_index * 32 + sensor_input_index * 4

        action = genome[genome_base_index + 0]
        next_state = genome[genome_base_index + 1]
        stack_operation = genome[genome_base_index + 2]
        stack_symbol = genome[genome_base_index + 3]

        if action == 0:
            if is_front_free:
                current_row = current_row + forward_delta_row
                current_col = current_col + forward_delta_col
                path_rows.append(current_row)
                path_cols.append(current_col)
                cells_visited[current_row, current_col] = 1
            else:
                collision_count += 1
        elif action == 1:
            direction_index = left_direction_index
        elif action == 2:
            direction_index = right_direction_index

        if stack_operation == 1:
            symbol_to_push = stack_symbol % num_stack_symbols
            stack.append(symbol_to_push)
        elif stack_operation == 2:
            if len(stack) > 0:
                stack.pop()
            else:
                stack_underflow_count += 1

        if num_states > 0:
            current_state = next_state % num_states
            states_visited[current_state] = 1
        steps_taken += 1

    unique_states_count = int(states_visited.sum())
    unique_cells_count = int(cells_visited.sum())
    
    return reached_goal, steps_taken, collision_count, stack_underflow_count, stack, path_rows, path_cols, unique_states_count, unique_cells_count

class Simulator:
    """Simulator for an agent controlled by a conditional PDA."""

    def __init__(self, maze: Maze, max_steps: int = 200, num_states: int = 8, num_stack_symbols: int = 4) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.num_states = max(1, num_states)
        self.num_stack_symbols = max(1, num_stack_symbols)
        self._per_state_block = 32 * (self.num_stack_symbols + 1)

    def run_genome(self, genome: Sequence[int]) -> Dict[str, Any]:
        """Run a genome for up to max_steps and return summary info."""
        expected_length = self.num_states * self._per_state_block
        if len(genome) != expected_length:
            raise ValueError(f"Genome length {len(genome)} != expected {expected_length}")

        genome_array = np.array(genome, dtype=np.int32) if not isinstance(genome, np.ndarray) else genome

        reached_goal, steps_taken, collisions, underflow, stack, path_rows, path_cols, unique_states, unique_cells = _run_genome_numba(
            genome_array, self.maze.grid,
            self.maze.start, self.maze.goal,
            self.max_steps, self.num_states, self.num_stack_symbols, self._per_state_block
        )

        path = [(path_rows[i], path_cols[i]) for i in range(len(path_rows))]

        return {
            "reached": bool(reached_goal),
            "steps": steps_taken,
            "collisions": collisions,
            "stack_underflow": underflow,
            "final_stack": list(stack),
            "path": path,
            "unique_states": unique_states,
            "unique_cells": unique_cells,
        }

