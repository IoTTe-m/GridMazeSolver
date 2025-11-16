import random
import argparse
import math
import os
import sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

class Maze:

    def __init__(self, width=21, height=21, seed=None):
        assert width % 2 == 1 and height % 2 == 1, "width and height must be odd"
        self.width = width
        self.height = height
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.grid = np.ones((height, width), dtype=np.int8)
        self._generate()

    def _generate(self):
        H, W = self.height, self.width
        stack = [(1, 1)]
        self.grid[1, 1] = 0
        while stack:
            x, y = stack[-1]
            neighbors = []
            # check four directions (jump by 2 to reach the next cell)
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 1 <= nx < H-1 and 1 <= ny < W-1 and self.grid[nx, ny] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                wx, wy = (x + nx)//2, (y + ny)//2
                self.grid[nx, ny] = 0
                self.grid[wx, wy] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def set_start_goal(self, start=(1,1), goal=None):
        if goal is None:
            goal = (self.height-2, self.width-2)
        self.start = start
        self.goal = goal

    def is_free(self, pos):
        x, y = pos
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x, y] == 0

    def render(self, path_positions=None, savepath=None, figsize=(6,6)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.grid, cmap='gray_r', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if path_positions:
            xs = [p[1] for p in path_positions]
            ys = [p[0] for p in path_positions]
            ax.plot(xs, ys, linewidth=2)
            ax.scatter([self.start[1], self.goal[1]], [self.start[0], self.goal[0]], c='red')
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
            print(f"Saved visual to {savepath}")
        plt.close(fig)

# --- Agent, FSA encoding and simulator ------------------------------
# Actions: 0 = Forward, 1 = Turn Left, 2 = Turn Right
ACTION_TO_CHAR = {0: '^', 1: '<', 2: '>'}

# Direction vectors: 0=N,1=E,2=S,3=W
DIRECTION_VECTORS = [(-1,0), (0,1), (1,0), (0,-1)]

class Simulator:
    def __init__(self, maze: Maze, max_steps=200, n_states=8):
        self.maze = maze
        self.max_steps = max_steps
        self.n_states = n_states
        self.reset()

    def reset(self):
        # position, direction, path history, counters
        self.pos = tuple(self.maze.start)
        self.dir = 1  # start facing East
        self.state = 0  # start in FSA state 0
        self.steps = 0
        self.collisions = 0
        self.path = [self.pos]
        self.reached = False

    def _decode_state(self, genome, state_idx):
        base = state_idx * 4
        return (genome[base + 0], genome[base + 1], genome[base + 2], genome[base + 3])

    def step_fsa(self, genome):
        # sense: is the cell in front free?
        dx, dy = DIRECTION_VECTORS[self.dir]
        look_pos = (self.pos[0] + dx, self.pos[1] + dy)
        front_free = 1 if self.maze.is_free(look_pos) else 0

        # get transitions for current state
        a_block, n_block, a_free, n_free = self._decode_state(genome, self.state)

        # pick based on sensor
        if front_free:
            action, next_state = a_free, n_free
        else:
            action, next_state = a_block, n_block

        # execute action
        if action == 0:  # Forward
            newpos = (self.pos[0] + dx, self.pos[1] + dy)
            if self.maze.is_free(newpos):
                self.pos = newpos
                self.path.append(self.pos)
            else:
                # attempted forward into wall
                self.collisions += 1
        elif action == 1:  # Left turn
            self.dir = (self.dir - 1) % 4
        elif action == 2:  # Right turn
            self.dir = (self.dir + 1) % 4

        # update internal state and counters
        self.state = int(next_state)
        self.steps += 1
        if self.pos == self.maze.goal:
            self.reached = True

    def run_genome(self, genome):
        # basic validation: genome length must match expected
        expected_len = self.n_states * 4
        if len(genome) != expected_len:
            raise ValueError(f'Genome length {len(genome)} does not match expected {expected_len}')

        self.reset()
        for _ in range(self.max_steps):
            if self.reached:
                break
            self.step_fsa(genome)
        return {
            'reached': self.reached,
            'steps': self.steps,
            'collisions': self.collisions,
            'path': list(self.path)
        }

# --- Fitness evaluator ---------------------------------------------
class Evaluator:
    def __init__(self, maze: Maze, max_steps=200, n_states=8):
        self.maze = maze
        self.max_steps = max_steps
        # simulator must know n_states to validate genome shape
        self.sim = Simulator(maze, max_steps=max_steps, n_states=n_states)

    def score(self, genome):
        out = self.sim.run_genome(genome)
        score = 0

        if out['reached']:
            score += 10000000
            score += (self.max_steps - out['steps']) * 10
            score -= out['collisions'] * 20
        else:
            # Reward earlier improvements in Manhattan distance to goal
            distances = []
            for step, pos in enumerate(out['path']):
                manhattan = abs(pos[0] - self.maze.goal[0]) + abs(pos[1] - self.maze.goal[1])
                distances.append((step, manhattan))

            min_dist = float('inf')
            for step, dist in distances:
                if dist < min_dist:
                    min_dist = dist
                    # reward inversely proportional to step (earlier improvements worth more)
                    score += (self.max_steps - step)

            # baseline partial credit for closeness at final position
            cur = out['path'][-1]
            manhattan = abs(cur[0] - self.maze.goal[0]) + abs(cur[1] - self.maze.goal[1])
            score += max(0, 1000 - manhattan * 5)
            score -= out['collisions'] * 10

        return score, out

def setup_deap_fsa(n_states, eval_fn, max_steps, seed=None):
    if seed is not None:
        random.seed(seed)

    # ensure creators exist
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Custom initializer: create an individual by filling each state's fields
    def init_fsa():
        chrom = []
        for s in range(n_states):
            # action_if_blocked: 0..2
            chrom.append(random.randint(0,2))
            # next_if_blocked: 0..n_states-1
            chrom.append(random.randint(0,n_states-1))
            # action_if_free: 0..2
            chrom.append(random.randint(0,2))
            # next_if_free: 0..n_states-1
            chrom.append(random.randint(0,n_states-1))
        return creator.Individual(chrom)

    toolbox.register('individual', init_fsa)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # evaluation wrapper to fit DEAP's API
    def evaluate(ind):
        sc, _ = eval_fn(ind)
        return (sc,)

    toolbox.register('evaluate', evaluate)
    toolbox.register('mate', tools.cxTwoPoint)

    # custom mutation: randomly change either an action (0..2) or a next_state index
    def mutate_fsa(individual, indpb=0.05):
        # indpb = probability of each gene being mutated
        for i in range(len(individual)):
            if random.random() < indpb:
                pos_in_state = i % 4
                if pos_in_state in (0, 2):
                    # action gene
                    individual[i] = random.randint(0,2)
                else:
                    # next_state gene
                    individual[i] = random.randint(0, n_states-1)
        return (individual,)

    toolbox.register('mutate', mutate_fsa, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)

    return toolbox

def run_ga_deap_fsa(maze, n_states=8, pop_size=200, gens=200, cxpb=0.6, mutpb=0.3, seed=None, ind_max_steps=200):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states)
    toolbox = setup_deap_fsa(n_states, evaluator.score, max_steps=ind_max_steps, seed=seed)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', lambda fits: float(np.mean(fits)))
    stats.register('min', lambda fits: float(np.min(fits)))
    stats.register('max', lambda fits: float(np.max(fits)))

    print('Starting GA (FSA): n_states=%d pop=%d gens=%d' % (n_states, pop_size, gens))
    # use eaSimple but with custom crossover/mutation probabilities
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gens, stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, log


def run_es_deap_fsa(maze, n_states=8, mu=100, lam=200, gens=200, cxpb=0.6, mutpb=0.3, seed=None, ind_max_steps=200):
    evaluator = Evaluator(maze, max_steps=ind_max_steps, n_states=n_states)
    toolbox = setup_deap_fsa(n_states, evaluator.score, max_steps=ind_max_steps, seed=seed)

    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', lambda fits: float(np.mean(fits)))
    stats.register('min', lambda fits: float(np.min(fits)))
    stats.register('max', lambda fits: float(np.max(fits)))

    print('Starting ES (mu+lambda) (FSA): mu=%d lambda=%d gens=%d' % (mu, lam, gens))

    def varAnd(population, toolbox, lambda_, cxpb, mutpb):
        offspring = []
        while len(offspring) < lambda_:
            op = random.random()
            if op < cxpb and len(population) >= 2:
                a, b = map(toolbox.clone, random.sample(population, 2))
                toolbox.mate(a, b)
                toolbox.mutate(a)
                toolbox.mutate(b)
                del a.fitness.values
                del b.fitness.values
                offspring.append(a)
                if len(offspring) < lambda_:
                    offspring.append(b)
            else:
                a = toolbox.clone(random.choice(population))
                toolbox.mutate(a)
                del a.fitness.values
                offspring.append(a)
        return offspring

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lam, cxpb, mutpb, ngen=gens, stats=stats, halloffame=hof, verbose=True)
    best = hof[0]
    best_score, best_out = evaluator.score(best)
    return best, best_score, best_out, log

def genome_to_fsa_string(genome, n_states):
    lines = []
    for s in range(n_states):
        base = s*4
        a_block, n_block, a_free, n_free = genome[base:base+4]
        lines.append(f'State {s}: if blocked -> (act={ACTION_TO_CHAR[a_block]}, next={n_block}); if free -> (act={ACTION_TO_CHAR[a_free]}, next={n_free})')
    return '\n'.join(lines)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['ga','es'], default='ga', help='Algorithm to run')
    p.add_argument('--width', type=int, default=21, help='Maze width (odd)')
    p.add_argument('--height', type=int, default=21, help='Maze height (odd)')
    p.add_argument('--n_states', type=int, default=8, help='Number of FSA states')
    p.add_argument('--max_steps', type=int, default=400, help='Max steps per evaluation')
    p.add_argument('--pop', type=int, default=200, help='Population size (GA) or mu (ES)')
    p.add_argument('--gens', type=int, default=200, help='Generations to run')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out', type=str, default='best_path.png', help='Output filename for best path image')
    return p.parse_args()


def main():
    args = parse_args()
    if args.width %2 ==0 or args.height %2 ==0:
        print('Width and height should be odd; adjusting by +1')
        args.width += args.width%2==0
        args.height += args.height%2==0

    maze = Maze(width=args.width, height=args.height, seed=args.seed)
    maze.set_start_goal(start=(1,1), goal=(args.height-2, args.width-2))

    print(f'Maze size: {args.width}x{args.height}. Start={maze.start} Goal={maze.goal}')

    if args.algo == 'ga':
        best, best_score, best_out, log = run_ga_deap_fsa(maze, n_states=args.n_states, pop_size=args.pop, gens=args.gens, seed=args.seed, ind_max_steps=args.max_steps)
    elif args.algo == 'es':
        best, best_score, best_out, log = run_es_deap_fsa(maze, n_states=args.n_states, mu=args.pop, lam=args.pop*2, gens=args.gens, seed=args.seed, ind_max_steps=args.max_steps)
    else:
        raise ValueError('Unknown algorithm')

    print('\n=== BEST SOLUTION ===')
    print('Score:', best_score)
    print('Reached goal:', best_out['reached'])
    print('Steps taken:', best_out['steps'])
    print('Collisions:', best_out['collisions'])
    print('\nFSA description:')
    print(genome_to_fsa_string(best, args.n_states))

    # Save visualization of the best path and maze
    maze.render(path_positions=best_out['path'], savepath=args.out)
    maze.render(savepath='maze.png')

    print('Done.')

if __name__ == '__main__':
    main()

# todo:
# add logging
# some way to save/load the agent idk