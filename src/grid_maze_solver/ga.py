import random
import numpy as np
from typing import Optional, List, Sequence, Tuple, Dict, Any

from deap import base, creator, tools, algorithms

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

def run_ga_conditional_pda(maze: Maze, n_states: int, n_stack_syms: int, pop_size: int, gens: int, seed: Optional[int], ind_max_steps: int, cxpb=0.6, mutpb=0.3, dynamic_maze: bool = False, step_callback=None):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states, n_stack_syms=n_stack_syms)
    toolbox = setup_deap_conditional_pda(n_states, n_stack_syms, evaluator.score, seed=seed)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(np.mean(fits)))
    stats.register("min", lambda fits: float(np.min(fits)))
    stats.register("max", lambda fits: float(np.max(fits)))
    
    print(f"Starting GA: n_states={n_states} pop={pop_size} gens={gens} stack_syms={n_stack_syms} dynamic_maze={dynamic_maze}")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)
    
    record = stats.compile(pop)
    print(record)

    for g in range(1, gens + 1):
        # If dynamic, regenerate maze and re-evaluate EVERYONE?
        # In strict GA (eaSimple), we create offspring and replace parents.
        # So we only need to regenerate maze before evaluating offspring.
        if dynamic_maze:
             # Random int seed
             maze.regenerate(seed=random.randint(0, 1_000_000))
             # Note: standard eaSimple doesn't re-evaluate parents because they are discarded.
             # We just proceed to generate offspring.

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        # In dynamic mode, everyone is invalid effectively because maze changed,
        # but `del fitness` only happened for modified ones.
        # If dynamic_maze=True, we MUST force evaluation of everyone in offspring
        # even if they weren't mutated (clones of parents), because maze changed.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if dynamic_maze:
            # Re-evaluate ALL offspring
            invalid_ind = offspring
        
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        if hof is not None:
            hof.update(pop)
            
        record = stats.compile(pop)
        print(f"Gen {g}: {record}")
        
        if step_callback:
            if dynamic_maze:
                current_best = tools.selBest(pop, 1)[0]
            else:
                current_best = hof[0] if hof else pop[0]
            step_callback(g, record, current_best)

    best = hof[0]
    # If dynamic, the best individual might have a score from a random maze.
    # We might want to re-eval it on the current maze or just return it.
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, None

def run_es_conditional_pda(maze: Maze, n_states: int, n_stack_syms: int, mu: int, lam: int, gens: int, seed: Optional[int], ind_max_steps: int, cxpb=0.6, mutpb=0.3, dynamic_maze: bool = False, step_callback=None):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states, n_stack_syms=n_stack_syms)
    toolbox = setup_deap_conditional_pda(n_states, n_stack_syms, evaluator.score, seed=seed)
    
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(np.mean(fits)))
    stats.register("min", lambda fits: float(np.min(fits)))
    stats.register("max", lambda fits: float(np.max(fits)))
    
    print(f"Starting ES: mu={mu} lambda={lam} gens={gens} stack_syms={n_stack_syms} dynamic_maze={dynamic_maze}")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)
        
    record = stats.compile(pop)
    print(record)

    for g in range(1, gens + 1):
        if dynamic_maze:
            maze.regenerate(seed=random.randint(0, 1_000_000))
            # In ES (Mu + Lambda), parents survive. 
            # We MUST invalidate parents' fitness so they are re-compared fairly with offspring.
            for ind in pop:
                del ind.fitness.values

        # If dynamic, parents (pop) are now invalid.
        # Check invalid
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Generate offspring
        offspring = algorithms.varOr(pop, toolbox, lambda_=lam, cxpb=cxpb, mutpb=mutpb)
        
        # Evaluate validity
        # If dynamic, all offspring should be treated as invalid anyway?
        # varOr deletes fitness if modified. Clones keep fitness.
        # But clones have fitness from PARENTS (which was just updated on NEW maze? No).
        # Wait:
        # 1. Parents P re-evaluated on Maze M2.
        # 2. Offspring O created from P.
        #    - Mutated O: fitness deleted.
        #    - Cloned O: fitness copied from P (which is ON MAZE M2).
        # So clones are valid on Maze M2!
        # So specific invalidation of offspring is NOT needed if parents were updated first.
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Select the next generation population
        pop[:] = toolbox.select(pop + offspring, mu)

        if hof is not None:
            hof.update(pop)
            
        record = stats.compile(pop)
        print(f"Gen {g}: {record}")
        
        if step_callback:
            # For dynamic maze, HOF contains 'stale' bests from previous mazes.
            # We want to visualize the best of the CURRENT generation/maze.
            if dynamic_maze:
                current_best = tools.selBest(pop, 1)[0]
            else:
                current_best = hof[0] if hof else pop[0]
            
            step_callback(g, record, current_best)

    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, None
