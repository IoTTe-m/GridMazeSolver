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

class Simulator:
    """Simulator for an agent controlled by a conditional PDA.

    Transitions are selected based on (front_free? blocked/free) and the current
    top-of-stack symbol (including special EMPTY index).
    """

    def __init__(self, maze: Maze, max_steps: int = 200, n_states: int = 8, n_stack_syms: int = 4) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.n_states = max(1, n_states) # Ensure at least 1 state
        # number of actual stack symbols; index `n_stack_syms` is reserved for EMPTY
        self.n_stack_syms = max(1, n_stack_syms)
        # per-state block size: for each top-symbol (S+1) we have 2 inputs * 4 ints = 8 ints
        self._per_state_block = 8 * (self.n_stack_syms + 1)
        self.reset()

    def reset(self) -> None:
        self.pos: Tuple[int, int] = tuple(self.maze.start)
        self.dir: int = 1  # face East
        self.state: int = 0
        self.steps: int = 0
        self.collisions: int = 0
        self.stack_underflow: int = 0
        self.stack: List[int] = []
        self.path: List[Tuple[int, int]] = [self.pos]
        self.reached: bool = False

    def _top_symbol_index(self) -> int:
        """Return the top-of-stack index: 0..S-1 for real symbols, S for EMPTY."""
        if not self.stack:
            return self.n_stack_syms  # special EMPTY index
        return int(self.stack[-1]) % self.n_stack_syms

    def _decode_transition(self, genome: Sequence[int], state_idx: int, top_sym_idx: int, front_free: int) -> Tuple[int, int, int, int]:
        """Return (action, next_state, stack_op, stack_sym) for the given triple."""
        if not (0 <= state_idx < self.n_states):
            raise IndexError("state index out of range in _decode_transition")
        # base for that state
        state_base = state_idx * self._per_state_block
        # each top-symbol block is 8 ints: [blocked:4][free:4]
        top_block_offset = top_sym_idx * 8
        input_offset = 4 if front_free else 0
        base = state_base + top_block_offset + input_offset
        return (int(genome[base + 0]), int(genome[base + 1]), int(genome[base + 2]), int(genome[base + 3]))

    def _apply_stack_op(self, op: int, sym: int) -> None:
        """Mutate the internal stack according to op and sym."""
        if op == STACK_NOP:
            return
        if op == STACK_PUSH:
            push_sym = int(sym) % self.n_stack_syms
            self.stack.append(push_sym)
            return
        if op == STACK_POP:
            if self.stack:
                self.stack.pop()
            else:
                self.stack_underflow += 1
            return
        # unknown op treated as NOP
        return

    def step(self, genome: Sequence[int]) -> None:
        """Sense, select transition using (front sensor, stack top), execute action,
        perform stack op, update state and counters."""
        dx, dy = DIRECTION_VECTORS[self.dir]
        look_pos = (self.pos[0] + dx, self.pos[1] + dy)
        front_free = 1 if self.maze.is_free(look_pos) else 0

        top_idx = self._top_symbol_index()
        action, next_state, stack_op, stack_sym = self._decode_transition(genome, self.state, top_idx, front_free)

        # action
        if action == ACTION_FORWARD:
            newpos = (self.pos[0] + dx, self.pos[1] + dy)
            if self.maze.is_free(newpos):
                self.pos = newpos
                self.path.append(self.pos)
            else:
                self.collisions += 1
        elif action == ACTION_TURN_LEFT:
            self.dir = (self.dir - 1) % 4
        elif action == ACTION_TURN_RIGHT:
            self.dir = (self.dir + 1) % 4
        else:
            # invalid action => treat as NOP
            pass

        # stack op (applied after movement)
        self._apply_stack_op(stack_op, stack_sym)

        # update
        self.state = int(next_state) % self.n_states
        self.steps += 1
        if self.pos == self.maze.goal:
            self.reached = True

    def run_genome(self, genome: Sequence[int]) -> Dict[str, Any]:
        """Run a genome for up to max_steps and return summary info."""
        expected_len = self.n_states * self._per_state_block
        if len(genome) != expected_len:
            raise ValueError(f"Genome length {len(genome)} != expected {expected_len} (n_states={self.n_states}, stack_syms={self.n_stack_syms})")

        self.reset()
        for _ in range(self.max_steps):
            if self.reached:
                break
            self.step(genome)

        return {
            "reached": self.reached,
            "steps": self.steps,
            "collisions": self.collisions,
            "stack_underflow": self.stack_underflow,
            "final_stack": list(self.stack),
            "path": list(self.path),
        }
