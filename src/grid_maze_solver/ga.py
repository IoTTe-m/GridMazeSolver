import random
import numpy as np
from typing import Optional, List, Sequence, Tuple, Dict, Any

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    raise RuntimeError("This script requires DEAP. Install with: pip install deap")

from .maze import Maze
from .evaluator import Evaluator

def setup_deap_conditional_pda(n_states: int, n_stack_syms: int, eval_fn, seed: Optional[int] = None):
    """Create DEAP toolbox adapted to conditional PDA genomes."""
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
            for _ts in range(n_stack_syms + 1):
                # blocked
                chrom.append(random.randint(0, 2))
                chrom.append(random.randint(0, n_states - 1) if n_states > 0 else 0)
                chrom.append(random.randint(0, 2))
                chrom.append(random.randint(0, max(1, n_stack_syms) - 1))
                # free
                chrom.append(random.randint(0, 2))
                chrom.append(random.randint(0, n_states - 1) if n_states > 0 else 0)
                chrom.append(random.randint(0, 2))
                chrom.append(random.randint(0, max(1, n_stack_syms) - 1))
        assert len(chrom) == genome_length
        return creator.Individual(chrom)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        sc, _ = eval_fn(individual)
        return (sc,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)

    def mutate(individual, indpb=0.05, rename_prob: float = 0.02):
        per_state = per_state_block
        L = len(individual)
        assert L == n_states * per_state

        arr = np.asarray(individual, dtype=int)
        mut_mask = (np.random.rand(L) < indpb)
        if mut_mask.any():
            local = np.arange(L) % per_state
            pos_in_transition = (local % 4)
            
            idx = mut_mask & (pos_in_transition == 0)
            if idx.any(): arr[idx] = np.random.randint(0, 3, size=idx.sum())
            
            idx = mut_mask & (pos_in_transition == 1)
            if idx.any() and n_states > 0: arr[idx] = np.random.randint(0, n_states, size=idx.sum())
            
            idx = mut_mask & (pos_in_transition == 2)
            if idx.any(): arr[idx] = np.random.randint(0, 3, size=idx.sum())
            
            idx = mut_mask & (pos_in_transition == 3)
            if idx.any(): 
                max_sym = max(1, n_stack_syms)
                arr[idx] = np.random.randint(0, max_sym, size=idx.sum())

        if (np.random.rand() < rename_prob) and (n_states > 1):
            blocks = arr.reshape((n_states, per_state)).copy()
            src = np.random.randint(0, n_states)
            dst = np.random.randint(0, n_states)
            if dst != src:
                moved = blocks[src:src + 1]
                remaining = np.delete(blocks, src, axis=0)
                new_blocks = np.insert(remaining, dst, moved, axis=0)

                permuted = list(range(n_states))
                val = permuted.pop(src)
                permuted.insert(dst, val)
                permuted = np.asarray(permuted, dtype=int)
                mapping = np.empty(n_states, dtype=int)
                mapping[permuted] = np.arange(n_states, dtype=int)

                cols = np.arange(per_state)
                next_cols_mask = (cols % 4 == 1)
                next_vals = new_blocks[:, next_cols_mask]
                clipped = np.clip(next_vals, 0, n_states - 1)
                new_blocks[:, next_cols_mask] = mapping[clipped]
                arr = new_blocks.reshape(-1)

        for i in range(L):
            individual[i] = int(arr[i])
        return (individual,)

    toolbox.register("mutate", mutate, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def run_ga_conditional_pda(maze: Maze, n_states: int, n_stack_syms: int, pop_size: int, gens: int, seed: Optional[int], ind_max_steps: int, cxpb=0.6, mutpb=0.3):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states, n_stack_syms=n_stack_syms)
    toolbox = setup_deap_conditional_pda(n_states, n_stack_syms, evaluator.score, seed=seed)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(np.mean(fits)))
    stats.register("min", lambda fits: float(np.min(fits)))
    stats.register("max", lambda fits: float(np.max(fits)))
    
    print(f"Starting GA: n_states={n_states} pop={pop_size} gens={gens} stack_syms={n_stack_syms}")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gens, stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, log

def run_es_conditional_pda(maze: Maze, n_states: int, n_stack_syms: int, mu: int, lam: int, gens: int, seed: Optional[int], ind_max_steps: int, cxpb=0.6, mutpb=0.3):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states, n_stack_syms=n_stack_syms)
    toolbox = setup_deap_conditional_pda(n_states, n_stack_syms, evaluator.score, seed=seed)
    
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(np.mean(fits)))
    stats.register("min", lambda fits: float(np.min(fits)))
    stats.register("max", lambda fits: float(np.max(fits)))
    
    print(f"Starting ES: mu={mu} lambda={lam} gens={gens} stack_syms={n_stack_syms}")
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lam, cxpb, mutpb, ngen=gens, stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, log
