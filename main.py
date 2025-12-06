"""
Grid Maze Pathfinding — Conditional Pushdown / Stack Automaton (PDA) agent

This version upgrades the PDA so transitions depend on:
  (a) whether the cell in front is free (blocked/free)
  (b) the current top-of-stack symbol (including an explicit EMPTY symbol)

Genome encoding (flat list of ints):
 For N states and S stack symbols the genome length is:
    genome_len = N * ((S + 1) * 2 * 4) = N * 8 * (S + 1)

Explanation:
 - For each control state i (0..N-1)
   - for each possible top-symbol ts in 0..S (where S == EMPTY marker)
     - for each sensor value v in {blocked, free} (0,1)
       - store 4 integers for the transition:
         [action, next_state, stack_op, stack_sym]

 Field ranges:
   action        : 0..2   (0=Forward,1=TurnLeft,2=TurnRight)
   next_state    : 0..N-1
   stack_op      : 0..2   (0=NOP,1=PUSH,2=POP)
   stack_sym     : 0..S-1 (only used if op==PUSH; ignored for POP/NOP)

Special notes:
 - top-symbol index S is reserved for EMPTY stack.
 - maze generator has a new parameter `remove_walls` to remove a small number
   of random walls after maze carving to create loops / shortcuts.
"""
from typing import List, Tuple, Sequence, Dict, Any, Optional
import argparse
import random
from collections import deque
import sys

import numpy as np

# matplotlib optional for rendering
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # render disabled if matplotlib missing

# DEAP imports (must be installed to run evolution)
try:
    from deap import base, creator, tools, algorithms
except Exception:
    raise RuntimeError("This script requires DEAP. Install with: pip install deap")

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

# Directions (0=N,1=E,2=S,3=W) and their vectors (row, col)
DIRECTION_VECTORS: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Maze ----------------------------------------------------------
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

# --- Simulator (conditional PDA) ---------------------------------
class Simulator:
    """Simulator for an agent controlled by a conditional PDA.

    Transitions are selected based on (front_free? blocked/free) and the current
    top-of-stack symbol (including special EMPTY index).
    """

    def __init__(self, maze: Maze, max_steps: int = 200, n_states: int = 8, n_stack_syms: int = 4) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.n_states = n_states
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
        self.state = int(next_state) % max(1, self.n_states)
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

# --- Evaluator ----------------------------------------------------
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
            score += 10_000_000.0
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

# --- DEAP toolbox setup (conditional PDA genome) ------------------
def setup_deap_conditional_pda(n_states: int, n_stack_syms: int, eval_fn, seed: Optional[int] = None):
    """Create DEAP toolbox adapted to conditional PDA genomes.

    Genome length = n_states * (8 * (n_stack_syms + 1))
    per-state layout for each top-symbol ts in 0..n_stack_syms:
       [ blocked: (action, next, op, sym) , free: (action, next, op, sym) ]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    per_state_block = 8 * (max(1, n_stack_syms) + 1)
    genome_length = n_states * per_state_block

    def init_individual() -> List[int]:
        chrom: List[int] = []
        for _ in range(n_states):
            for _ts in range(n_stack_syms + 1):  # include special EMPTY top index
                # blocked transition
                chrom.append(random.randint(0, 2))               # action
                chrom.append(random.randint(0, n_states - 1))    # next state
                chrom.append(random.randint(0, 2))               # stack op
                chrom.append(random.randint(0, max(1, n_stack_syms) - 1))  # stack symbol
                # free transition
                chrom.append(random.randint(0, 2))
                chrom.append(random.randint(0, n_states - 1))
                chrom.append(random.randint(0, 2))
                chrom.append(random.randint(0, max(1, n_stack_syms) - 1))
        # sanity: length should match genome_length
        assert len(chrom) == genome_length
        return creator.Individual(chrom)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        sc, _ = eval_fn(individual)
        return (sc,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)

    # Mutation: each gene has probability indpb of being replaced with a valid random
    # Additionally: with a small probability rename_prob we perform a state rename:
    #   pick a source state index and insert its block at a new destination index,
    #   then remap all next_state references accordingly.
    def mutate(individual, indpb=0.05, rename_prob: float = 0.02):
        """
        Vectorized mutation:
          - per-gene mutation: vectorized random mask + batched random draws
          - optional state-rename: move one state-block and remap all next_state fields
        """
        per_state = per_state_block  # captured from outer scope
        L = len(individual)
        assert L == n_states * per_state

        # convert to numpy array for vectorized ops
        arr = np.asarray(individual, dtype=int)

        # ---------- Vectorized per-gene mutation ----------
        # draw mask of which positions mutate
        mut_mask = (np.random.rand(L) < indpb)
        if mut_mask.any():
            # compute position-in-transition for each index
            local = np.arange(L) % per_state
            pos_in_transition = (local % 4)  # 0..3

            # action positions (pos==0) -> random 0..2
            idx = mut_mask & (pos_in_transition == 0)
            if idx.any():
                arr[idx] = np.random.randint(0, 3, size=idx.sum())

            # next_state positions (pos==1) -> random 0..n_states-1
            idx = mut_mask & (pos_in_transition == 1)
            if idx.any():
                arr[idx] = np.random.randint(0, n_states, size=idx.sum())

            # stack_op positions (pos==2) -> random 0..2
            idx = mut_mask & (pos_in_transition == 2)
            if idx.any():
                arr[idx] = np.random.randint(0, 3, size=idx.sum())

            # stack_sym positions (pos==3) -> random 0..max(1,n_stack_syms)-1
            idx = mut_mask & (pos_in_transition == 3)
            if idx.any():
                max_sym = max(1, n_stack_syms)
                arr[idx] = np.random.randint(0, max_sym, size=idx.sum())

        # ---------- Possibly perform a rename / reindex ----------
        if (np.random.rand() < rename_prob) and (n_states > 1):
            # reshape to blocks (n_states x per_state)
            blocks = arr.reshape((n_states, per_state)).copy()  # copy to be safe

            src = np.random.randint(0, n_states)
            dst = np.random.randint(0, n_states)
            if dst != src:
                # move block src -> dst
                moved = blocks[src:src + 1]  # shape (1, per_state)
                remaining = np.delete(blocks, src, axis=0)  # shape (n_states-1, per_state)
                new_blocks = np.insert(remaining, dst, moved, axis=0)  # shape (n_states, per_state)

                # build mapping old_index -> new_index
                permuted = list(range(n_states))
                val = permuted.pop(src)
                permuted.insert(dst, val)
                permuted = np.asarray(permuted, dtype=int)
                mapping = np.empty(n_states, dtype=int)
                # mapping[old_state] = new_state
                mapping[permuted] = np.arange(n_states, dtype=int)

                # remap all next_state fields: columns where (col % 4 == 1)
                cols = np.arange(per_state)
                next_cols_mask = (cols % 4 == 1)
                # slice of next_state values (shape n_states x n_nexts_per_state)
                next_vals = new_blocks[:, next_cols_mask]
                # clamp to valid indices, then map
                clipped = np.clip(next_vals, 0, n_states - 1)
                # apply mapping via fancy indexing
                new_blocks[:, next_cols_mask] = mapping[clipped]

                # flatten back
                arr = new_blocks.reshape(-1)

        # write back into DEAP individual (in-place)
        for i in range(L):
            individual[i] = int(arr[i])

        return (individual,)

    toolbox.register("mutate", mutate, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# --- Evolution runners --------------------------------------------
def run_ga_conditional_pda(maze: Maze, n_states: int = 8, n_stack_syms: int = 4, pop_size: int = 200, gens: int = 200, cxpb: float = 0.6, mutpb: float = 0.3, seed: Optional[int] = None, ind_max_steps: int = 200):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states, n_stack_syms=n_stack_syms)
    toolbox = setup_deap_conditional_pda(n_states, n_stack_syms, evaluator.score, seed=seed)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(np.mean(fits)))
    stats.register("min", lambda fits: float(np.min(fits)))
    stats.register("max", lambda fits: float(np.max(fits)))

    print(f"Starting GA (conditional PDA): n_states={n_states} pop={pop_size} gens={gens} stack_syms={n_stack_syms}")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gens, stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, log


def run_es_conditional_pda(maze: Maze, n_states: int = 8, n_stack_syms: int = 4, mu: int = 100, lam: int = 200, gens: int = 200, cxpb: float = 0.6, mutpb: float = 0.3, seed: Optional[int] = None, ind_max_steps: int = 200):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states, n_stack_syms=n_stack_syms)
    toolbox = setup_deap_conditional_pda(n_states, n_stack_syms, evaluator.score, seed=seed)

    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(np.mean(fits)))
    stats.register("min", lambda fits: float(np.min(fits)))
    stats.register("max", lambda fits: float(np.max(fits)))

    print(f"Starting ES (mu+lambda) (conditional PDA): mu={mu} lambda={lam} gens={gens} stack_syms={n_stack_syms}")
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lam, cxpb, mutpb, ngen=gens, stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, log

# --- Utilities ----------------------------------------------------
def genome_to_conditional_pda_string(genome: Sequence[int], n_states: int, n_stack_syms: int) -> str:
    """Pretty print the conditional PDA genome: for each state and top-symbol list transitions."""
    lines: List[str] = []
    per_state_block = 8 * (n_stack_syms + 1)
    EMPTY_IDX = n_stack_syms
    for s in range(n_states):
        lines.append(f"State {s}:")
        state_base = s * per_state_block
        for ts in range(n_stack_syms + 1):
            ts_name = f"{ts}" if ts != EMPTY_IDX else "EMPTY"
            block_base = state_base + ts * 8
            blocked = genome[block_base:block_base + 4]
            free = genome[block_base + 4:block_base + 8]
            a_b, n_b, op_b, sym_b = blocked
            a_f, n_f, op_f, sym_f = free
            lines.append(
                f"  top={ts_name} | BLOCKED -> (act={ACTION_TO_CHAR.get(a_b,'?')}, next={n_b}, op={STACK_OP_TO_STR.get(op_b,'?')}, sym={sym_b})"
            )
            lines.append(
                f"  top={ts_name} | FREE    -> (act={ACTION_TO_CHAR.get(a_f,'?')}, next={n_f}, op={STACK_OP_TO_STR.get(op_f,'?')}, sym={sym_f})"
            )
    return "\n".join(lines)

# --- CLI ----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evolve conditional PDA agents to solve a grid maze (DEAP required)")
    p.add_argument("--algo", choices=["ga", "es"], default="ga", help="Algorithm to run (ga or es)")
    p.add_argument("--width", type=int, default=21, help="Maze width (odd)")
    p.add_argument("--height", type=int, default=21, help="Maze height (odd)")
    p.add_argument("--n_states", type=int, default=8, help="Number of control states in the PDA")
    p.add_argument("--n_stack_syms", type=int, default=4, help="Number of possible stack symbols (positive int)")
    p.add_argument("--max_steps", type=int, default=400, help="Max steps per evaluation")
    p.add_argument("--pop", type=int, default=200, help="Population size (GA) or mu (ES)")
    p.add_argument("--gens", type=int, default=200, help="Generations / iterations to run")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--remove_walls", type=int, default=0, help="Remove this many random interior walls after maze generation")
    p.add_argument("--out", type=str, default="best_path.png", help="Output filename for best path image")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # ensure odd sizes
    if args.width % 2 == 0:
        args.width += 1
    if args.height % 2 == 0:
        args.height += 1

    maze = Maze(width=args.width, height=args.height, seed=args.seed, remove_walls=args.remove_walls)
    maze.set_start_goal(start=(1, 1), goal=(args.height - 2, args.width - 2))

    print(f"Maze size: {args.width}x{args.height}. Start={maze.start} Goal={maze.goal}")
    print(f"Conditional PDA config: n_states={args.n_states} stack_symbols={args.n_stack_syms} max_steps={args.max_steps} remove_walls={args.remove_walls}")

    if args.algo == "ga":
        best, best_score, best_out, log = run_ga_conditional_pda(
            maze, n_states=args.n_states, n_stack_syms=args.n_stack_syms, pop_size=args.pop, gens=args.gens, seed=args.seed, ind_max_steps=args.max_steps
        )
    else:
        best, best_score, best_out, log = run_es_conditional_pda(
            maze, n_states=args.n_states, n_stack_syms=args.n_stack_syms, mu=args.pop, lam=args.pop * 2, gens=args.gens, seed=args.seed, ind_max_steps=args.max_steps
        )

    print("\n=== BEST SOLUTION ===")
    print("Score:", best_score)
    print("Reached goal:", best_out["reached"])
    print("Steps taken:", best_out["steps"])
    print("Collisions:", best_out["collisions"])
    print("Stack underflow:", best_out.get("stack_underflow", 0))
    print("\nConditional PDA description:")
    print(genome_to_conditional_pda_string(best, args.n_states, args.n_stack_syms))

    # Save visualization of the best path and maze
    if plt is not None:
        maze.render(path_positions=best_out["path"], savepath=args.out)
        maze.render(savepath="maze.png")
    else:
        print("matplotlib not available; skipping image save.")

    print("Done.")

if __name__ == "__main__":
    main()
