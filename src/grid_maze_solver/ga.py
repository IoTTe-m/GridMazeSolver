import random
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from deap import base, creator, tools, algorithms

from .maze import Maze
from .evaluator import Evaluator
import multiprocessing

class FitnessWrapper:
    """Wrapper to make evaluation function picklable for multiprocessing."""
    def __init__(self, evaluation_function) -> None:
        self.evaluation_function = evaluation_function

    def __call__(self, individual) -> Tuple[float]:
        score, _ = self.evaluation_function(individual)
        return (score,)

def setup_deap_conditional_pda(num_states: int, num_stack_symbols: int, evaluation_function, seed: Optional[int] = None) -> base.Toolbox:
    """Create DEAP toolbox adapted to conditional PDA genomes."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    per_state_block = 32 * (max(1, num_stack_symbols) + 1)
    genome_length = num_states * per_state_block

    def init_individual() -> List[int]:
        chromosome: List[int] = []
        for _ in range(num_states):
            for _stack_symbol in range(num_stack_symbols + 1):
                for _input_combo in range(8):
                    chromosome.append(random.randint(0, 2))
                    chromosome.append(random.randint(0, num_states - 1) if num_states > 0 else 0)
                    chromosome.append(random.randint(0, 2))
                    chromosome.append(random.randint(0, max(1, num_stack_symbols) - 1))
        assert len(chromosome) == genome_length
        return creator.Individual(chromosome)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", FitnessWrapper(evaluation_function))
    toolbox.register("mate", tools.cxTwoPoint)

    def mutate(individual, mutation_probability: Optional[float] = None, state_rename_probability: float = 0.02):
        genes_per_state = per_state_block
        current_genome_length = len(individual)
        assert current_genome_length == num_states * genes_per_state
        
        # Adaptive mutation: default to ~2 mutations per individual on average
        if mutation_probability is None:
            mutation_probability = 2.0 / current_genome_length

        genes = np.asarray(individual, dtype=int)
        mutation_mask = (np.random.rand(current_genome_length) < mutation_probability)
        if mutation_mask.any():
            position_in_state = np.arange(current_genome_length) % genes_per_state
            gene_type = (position_in_state % 4)

            action_mask = mutation_mask & (gene_type == 0)
            if action_mask.any():
                genes[action_mask] = np.random.randint(0, 3, size=action_mask.sum())

            next_state_mask = mutation_mask & (gene_type == 1)
            if next_state_mask.any() and num_states > 0:
                genes[next_state_mask] = np.random.randint(0, num_states, size=next_state_mask.sum())

            stack_op_mask = mutation_mask & (gene_type == 2)
            if stack_op_mask.any():
                genes[stack_op_mask] = np.random.randint(0, 3, size=stack_op_mask.sum())

            stack_symbol_mask = mutation_mask & (gene_type == 3)
            if stack_symbol_mask.any():
                max_symbol = max(1, num_stack_symbols)
                genes[stack_symbol_mask] = np.random.randint(0, max_symbol, size=stack_symbol_mask.sum())

        if (np.random.rand() < state_rename_probability) and (num_states > 1):
            state_blocks = genes.reshape((num_states, genes_per_state)).copy()
            source_state = np.random.randint(0, num_states)
            destination_state = np.random.randint(0, num_states)
            if destination_state != source_state:
                moved_block = state_blocks[source_state:source_state + 1]
                remaining_blocks = np.delete(state_blocks, source_state, axis=0)
                reordered_blocks = np.insert(remaining_blocks, destination_state, moved_block, axis=0)

                permutation = list(range(num_states))
                moved_value = permutation.pop(source_state)
                permutation.insert(destination_state, moved_value)
                permutation = np.asarray(permutation, dtype=int)
                state_mapping = np.empty(num_states, dtype=int)
                state_mapping[permutation] = np.arange(num_states, dtype=int)

                columns = np.arange(genes_per_state)
                next_state_columns = (columns % 4 == 1)
                next_state_values = reordered_blocks[:, next_state_columns]
                clipped_values = np.clip(next_state_values, 0, num_states - 1)
                reordered_blocks[:, next_state_columns] = state_mapping[clipped_values]
                genes = reordered_blocks.reshape(-1)

        for i in range(current_genome_length):
            individual[i] = int(genes[i])
        return (individual,)

    toolbox.register("mutate", mutate)  # Uses adaptive rate by default
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def run_ga_conditional_pda(
    maze: Maze,
    num_states: int,
    num_stack_symbols: int,
    population_size: int,
    generations: int,
    seed: Optional[int],
    max_steps_per_individual: int,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    elitism_count: int = 2,
    dynamic_maze: bool = False,
    num_eval_mazes: int = 1,
    step_callback=None,
    num_parallel_jobs: int = 1
) -> Tuple[List[int], float, Dict[str, Any], None]:
    evaluator = Evaluator(maze, max_steps=max_steps_per_individual, n_states=num_states, n_stack_syms=num_stack_symbols)
    toolbox = setup_deap_conditional_pda(num_states, num_stack_symbols, evaluator.score, seed=seed)

    if num_parallel_jobs != 1:
        pool_size = multiprocessing.cpu_count() if num_parallel_jobs < 0 else num_parallel_jobs
        worker_pool = multiprocessing.Pool(processes=pool_size)
        toolbox.register("map", worker_pool.map)
        print(f"Parallel execution enabled: {pool_size} workers")
    else:
        worker_pool = None

    try:
        population = toolbox.population(n=population_size)
        hall_of_fame = tools.HallOfFame(1)
        statistics = tools.Statistics(lambda ind: ind.fitness.values)
        statistics.register("avg", lambda fits: float(np.mean(fits)))
        statistics.register("min", lambda fits: float(np.min(fits)))
        statistics.register("max", lambda fits: float(np.max(fits)))

        genome_length = num_states * 32 * (max(1, num_stack_symbols) + 1)
        print(f"Starting GA: n_states={num_states} pop={population_size} gens={generations} stack_syms={num_stack_symbols} genome_len={genome_length} elitism={elitism_count} dynamic_maze={dynamic_maze} eval_mazes={num_eval_mazes}")

        fitness_values = list(toolbox.map(toolbox.evaluate, population))
        for individual, fitness in zip(population, fitness_values):
            individual.fitness.values = fitness

        if hall_of_fame is not None:
            hall_of_fame.update(population)

        stats_record = statistics.compile(population)
        print(stats_record)

        for generation in range(1, generations + 1):
            if dynamic_maze:
                # Generate multiple mazes for evaluation
                mazes = []
                for i in range(num_eval_mazes):
                    mazes.append(Maze(maze.width, maze.height, seed=random.randint(0, 1_000_000), remove_walls=maze.remove_walls_count))
                evaluator.set_mazes(mazes)
                # Invalidate all fitness values since maze changed
                for ind in population:
                    del ind.fitness.values

            # Elitism: preserve best individuals, they need re-evaluation if maze changed
            elite = tools.selBest(population, elitism_count) if not dynamic_maze else []
            elite = list(map(toolbox.clone, elite))

            selection_count = len(population) - len(elite)
            offspring = toolbox.select(population, selection_count)
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_probability:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_probability:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            individuals_to_evaluate = [ind for ind in offspring if not ind.fitness.valid]
            if dynamic_maze:
                # Re-evaluate entire population on new maze
                individuals_to_evaluate = elite + offspring

            fitness_values = list(toolbox.map(toolbox.evaluate, individuals_to_evaluate))
            for individual, fitness in zip(individuals_to_evaluate, fitness_values):
                individual.fitness.values = fitness

            population[:] = elite + offspring

            # Hall of fame only makes sense for static maze
            if hall_of_fame is not None and not dynamic_maze:
                hall_of_fame.update(population)

            stats_record = statistics.compile(population)
            print(f"Gen {generation}: {stats_record}")

            if step_callback:
                # Always select best from current population based on current fitness
                current_best = tools.selBest(population, 1)[0]
                # Pass the first evaluation maze (or original maze for static mode)
                display_maze = evaluator.maze
                step_callback(generation, stats_record, current_best, display_maze)

    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    # For dynamic maze, use best from final population; for static, use hall of fame
    if dynamic_maze:
        best_individual = tools.selBest(population, 1)[0]
    else:
        best_individual = hall_of_fame[0]
    best_score, best_output = evaluator.score(best_individual)
    return best_individual, best_score, best_output, None

def run_es_conditional_pda(
    maze: Maze,
    num_states: int,
    num_stack_symbols: int,
    parent_population_size: int,
    offspring_size: int,
    generations: int,
    seed: Optional[int],
    max_steps_per_individual: int,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    dynamic_maze: bool = False,
    num_eval_mazes: int = 1,
    step_callback=None,
    num_parallel_jobs: int = 1
) -> Tuple[List[int], float, Dict[str, Any], None]:
    evaluator = Evaluator(maze, max_steps=max_steps_per_individual, n_states=num_states, n_stack_syms=num_stack_symbols)
    toolbox = setup_deap_conditional_pda(num_states, num_stack_symbols, evaluator.score, seed=seed)

    if num_parallel_jobs != 1:
        pool_size = multiprocessing.cpu_count() if num_parallel_jobs < 0 else num_parallel_jobs
        worker_pool = multiprocessing.Pool(processes=pool_size)
        toolbox.register("map", worker_pool.map)
        print(f"Parallel execution enabled: {pool_size} workers")
    else:
        worker_pool = None
    
    try:
        population = toolbox.population(n=parent_population_size)
        hall_of_fame = tools.HallOfFame(1)
        statistics = tools.Statistics(lambda ind: ind.fitness.values)
        statistics.register("avg", lambda fits: float(np.mean(fits)))
        statistics.register("min", lambda fits: float(np.min(fits)))
        statistics.register("max", lambda fits: float(np.max(fits)))

        print(f"Starting ES: mu={parent_population_size} lambda={offspring_size} gens={generations} stack_syms={num_stack_symbols} dynamic_maze={dynamic_maze} eval_mazes={num_eval_mazes}")

        fitness_values = list(toolbox.map(toolbox.evaluate, population))
        for individual, fitness in zip(population, fitness_values):
            individual.fitness.values = fitness

        if hall_of_fame is not None:
            hall_of_fame.update(population)

        stats_record = statistics.compile(population)
        print(stats_record)

        for generation in range(1, generations + 1):
            if dynamic_maze:
                # Generate multiple mazes for evaluation
                mazes = []
                for i in range(num_eval_mazes):
                    mazes.append(Maze(maze.width, maze.height, seed=random.randint(0, 1_000_000), remove_walls=maze.remove_walls_count))
                evaluator.set_mazes(mazes)
                for individual in population:
                    del individual.fitness.values

            individuals_to_evaluate = [ind for ind in population if not ind.fitness.valid]
            fitness_values = list(toolbox.map(toolbox.evaluate, individuals_to_evaluate))
            for individual, fitness in zip(individuals_to_evaluate, fitness_values):
                individual.fitness.values = fitness
                individual.fitness.values = fitness

            offspring = algorithms.varOr(population, toolbox, lambda_=offspring_size, cxpb=crossover_probability, mutpb=mutation_probability)

            individuals_to_evaluate = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = list(toolbox.map(toolbox.evaluate, individuals_to_evaluate))
            for individual, fitness in zip(individuals_to_evaluate, fitness_values):
                individual.fitness.values = fitness

            population[:] = toolbox.select(population + offspring, parent_population_size)

            # Hall of fame only makes sense for static maze
            if hall_of_fame is not None and not dynamic_maze:
                hall_of_fame.update(population)

            stats_record = statistics.compile(population)
            print(f"Gen {generation}: {stats_record}")

            if step_callback:
                # Always select best from current population based on current fitness
                current_best = tools.selBest(population, 1)[0]
                # Pass the first evaluation maze (or original maze for static mode)
                display_maze = evaluator.maze
                step_callback(generation, stats_record, current_best, display_maze)

    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    # For dynamic maze, use best from final population; for static, use hall of fame
    if dynamic_maze:
        best_individual = tools.selBest(population, 1)[0]
    else:
        best_individual = hall_of_fame[0]
    best_score, best_output = evaluator.score(best_individual)
    return best_individual, best_score, best_output, None
