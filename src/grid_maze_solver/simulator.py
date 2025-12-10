from typing import List, Tuple, Sequence, Dict, Any, Optional
from .maze import Maze, DIRECTION_VECTORS

# --- Constants -----------------------------------------------------
# Actions
ACTION_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_TO_CHAR = {ACTION_FORWARD: "^", ACTION_TURN_LEFT: "<", ACTION_TURN_RIGHT: ">"}

# Stack ops
STACK_NOP = 0
STACK_PUSH = 1
STACK_POP = 2
STACK_OP_TO_STR = {STACK_NOP: "NOP", STACK_PUSH: "PUSH", STACK_POP: "POP"}

# --- Numba Acceleration -------------------------------------------
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    # Dummy decorator if numba is missing
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    HAVE_NUMBA = False

import numpy as np

# Flatten direction vectors for numba (tuple)
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0)) # East, West, South, North? 
# Wait, let's check original DIRECTION_VECTORS
# From maze.py: DIRECTION_VECTORS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # E, S, W, N
# We must match exactly.
# Let's hardcode the correct ones from maze.py to be safe or import if possible.
# Numba works best with locally defined constants or passed args.
# Let's redefine here to be self-contained.
# E: (0, 1), S: (1, 0), W: (0, -1), N: (-1, 0)
DIR_VECS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int32)

@njit(fastmath=True)
def _run_genome_numba(
    genome: np.ndarray,
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    max_steps: int,
    n_states: int,
    n_stack_syms: int,
    per_state_block: int
):
    # Unpack
    start_r, start_c = start_pos
    goal_r, goal_c = goal_pos
    H, W = grid.shape
    
    # State
    r, c = start_r, start_c
    dr_idx = 0 # 0=E, 1=S, 2=W, 3=N
    state = 0
    steps = 0
    collisions = 0
    stack_underflow = 0
    
    # Stack
    stack = [0] 
    stack.pop() 
    
    path_r = [start_r]
    path_c = [start_c]
    
    reached = False
    
    for _ in range(max_steps):
        if r == goal_r and c == goal_c:
            reached = True
            break
            
        # 1. Sense Front, Left, Right
        
        # Front
        dr_f = DIR_VECS[dr_idx, 0]
        dc_f = DIR_VECS[dr_idx, 1]
        
        # Left
        left_idx = (dr_idx - 1) % 4
        dr_l = DIR_VECS[left_idx, 0]
        dc_l = DIR_VECS[left_idx, 1]
        
        # Right
        right_idx = (dr_idx + 1) % 4
        dr_r = DIR_VECS[right_idx, 0]
        dc_r = DIR_VECS[right_idx, 1]
        
        # Check directions (0=Path, 1=Wall)
        # Note: grid is 0 for free, 1 for wall.
        # We want "Free" variable = 1 if free.
        
        # Front
        nr, nc = r + dr_f, c + dc_f
        front_free = 0
        if 0 <= nr < H and 0 <= nc < W:
            if grid[nr, nc] == 0: front_free = 1
            
        # Left
        nr, nc = r + dr_l, c + dc_l
        left_free = 0
        if 0 <= nr < H and 0 <= nc < W:
            if grid[nr, nc] == 0: left_free = 1

        # Right
        nr, nc = r + dr_r, c + dc_r
        right_free = 0
        if 0 <= nr < H and 0 <= nc < W:
            if grid[nr, nc] == 0: right_free = 1
        
        # 2. Top symbol
        top_idx = n_stack_syms # Default EMPTY
        if len(stack) > 0:
            top_idx = stack[-1] % n_stack_syms 
            
        # 3. Decode
        # Block structure: [StackSym][Input]
        # Inputs: 8 combinations.
        # Index = (left<<2) | (front<<1) | right
        input_idx = (left_free << 2) | (front_free << 1) | right_free
        
        # Per stack sym block: 8 inputs * 4 genes = 32 ints
        # base = state * per_state_block + top_idx * 32 + input_idx * 4
        base = state * per_state_block + top_idx * 32 + input_idx * 4
        
        action = genome[base + 0]
        next_state = genome[base + 1]
        stack_op = genome[base + 2]
        stack_sym = genome[base + 3]
        
        # 4. Action
        if action == 0: # FORWARD
            if front_free:
                r, c = r + dr_f, c + dc_f
                path_r.append(r)
                path_c.append(c)
            else:
                collisions += 1
        elif action == 1: # LEFT
            dr_idx = left_idx
        elif action == 2: # RIGHT
            dr_idx = right_idx
            
        # 5. Stack Op
        # 0=NOP, 1=PUSH, 2=POP
        if stack_op == 1: # PUSH
            push_val = stack_sym % n_stack_syms
            stack.append(push_val)
        elif stack_op == 2: # POP
            if len(stack) > 0:
                stack.pop()
            else:
                stack_underflow += 1
        
        # 6. Update State
        if n_states > 0:
            state = next_state % n_states
        steps += 1
        
    return reached, steps, collisions, stack_underflow, stack, path_r, path_c

class Simulator:
    """Simulator for an agent controlled by a conditional PDA.
    
    Uses Numba for acceleration if available.
    Senses: Left, Front, Right.
    """

    def __init__(self, maze: Maze, max_steps: int = 200, n_states: int = 8, n_stack_syms: int = 4) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.n_states = max(1, n_states) 
        self.n_stack_syms = max(1, n_stack_syms)
        # per-state block size: 
        # For each top-symbol (S+1):
        #   8 inputs (L,F,R) * 4 ints (Action, NextState, Op, Sym) = 32 ints
        self._per_state_block = 32 * (self.n_stack_syms + 1)


    def run_genome(self, genome: Sequence[int]) -> Dict[str, Any]:
        """Run a genome for up to max_steps and return summary info."""
        expected_len = self.n_states * self._per_state_block
        if len(genome) != expected_len:
            raise ValueError(f"Genome length {len(genome)} != expected {expected_len}")

        # Convert to numpy for Numba
        genome_arr = np.array(genome, dtype=np.int32) if not isinstance(genome, np.ndarray) else genome
        grid = self.maze.grid
        
        reached, steps, collisions, underflow, stack, pr, pc = _run_genome_numba(
            genome_arr, grid, 
            self.maze.start, self.maze.goal, 
            self.max_steps, self.n_states, self.n_stack_syms, self._per_state_block
        )
        
        # Reconstruct path list of tuples
        path = []
        for i in range(len(pr)):
            path.append((pr[i], pc[i]))
            
        return {
            "reached": bool(reached),
            "steps": steps,
            "collisions": collisions,
            "stack_underflow": underflow,
            "final_stack": list(stack),
            "path": path,
        }

