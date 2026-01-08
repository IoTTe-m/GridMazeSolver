import argparse

import tkinter as tk

from .gui import SolverGUI
from .maze import Maze
from .ga import run_ga_conditional_pda, run_es_conditional_pda
from .genome import genome_to_conditional_pda_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve conditional PDA agents to solve a grid maze (DEAP required)")
    parser.add_argument("--algo", choices=["ga", "es"], default="ga", help="Algorithm to run (ga or es)")
    parser.add_argument("--width", type=int, default=21, help="Maze width (odd)")
    parser.add_argument("--height", type=int, default=21, help="Maze height (odd)")
    parser.add_argument("--n_states", type=int, default=8, help="Number of control states in the PDA")
    parser.add_argument("--n_stack_syms", type=int, default=4, help="Number of possible stack symbols (positive int)")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per evaluation")
    parser.add_argument("--pop", type=int, default=200, help="Population size (GA) or mu (ES)")
    parser.add_argument("--gens", type=int, default=200, help="Generations / iterations to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dynamic-maze", action="store_true", help="Regenerate maze every generation")
    parser.add_argument("--eval-mazes", type=int, default=1, help="Number of mazes to evaluate each agent on (for dynamic maze mode)")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI interface")
    parser.add_argument("--remove_walls", type=int, default=0, help="Remove this many random interior walls after maze generation")
    parser.add_argument("--out", type=str, default="best_path.png", help="Output filename for best path image")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs (default: 1, use -1 for all cores)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.gui:
        root = tk.Tk()
        _ = SolverGUI(root)
        root.mainloop()
        return

    # ensure odd sizes
    if args.width % 2 == 0:
        args.width += 1
    if args.height % 2 == 0:
        args.height += 1

    maze = Maze(width=args.width, height=args.height, seed=args.seed, remove_walls=args.remove_walls)
    maze.set_start_goal(start=(1, 1), goal=(args.height - 2, args.width - 2))

    print(f"Maze size: {args.width}x{args.height}. Start={maze.start} Goal={maze.goal}")
    print(f"Conditional PDA config: n_states={args.n_states} stack_symbols={args.n_stack_syms} max_steps={args.max_steps} remove_walls={args.remove_walls} dynamic_maze={args.dynamic_maze} eval_mazes={args.eval_mazes} jobs={args.jobs}")

    if args.algo == "ga":
        best_genome, best_score, best_result, log = run_ga_conditional_pda(
            maze,
            num_states=args.n_states,
            num_stack_symbols=args.n_stack_syms,
            population_size=args.pop,
            generations=args.gens,
            seed=args.seed,
            max_steps_per_individual=args.max_steps,
            dynamic_maze=args.dynamic_maze,
            num_eval_mazes=args.eval_mazes,
            num_parallel_jobs=args.jobs
        )
    else:
        best_genome, best_score, best_result, log = run_es_conditional_pda(
            maze,
            num_states=args.n_states,
            num_stack_symbols=args.n_stack_syms,
            parent_population_size=args.pop,
            offspring_size=args.pop * 2,
            generations=args.gens,
            seed=args.seed,
            max_steps_per_individual=args.max_steps,
            dynamic_maze=args.dynamic_maze,
            num_eval_mazes=args.eval_mazes,
            num_parallel_jobs=args.jobs
        )

    print("\n=== BEST SOLUTION ===")
    print("Score:", best_score)
    print("Reached goal:", best_result["reached"])
    print("Steps taken:", best_result["steps"])
    print("Collisions:", best_result["collisions"])
    print("Stack underflow:", best_result.get("stack_underflow", 0))
    print("\nConditional PDA description:")
    print(genome_to_conditional_pda_string(best_genome, num_states=args.n_states, num_stack_symbols=args.n_stack_syms))

    maze.render(path_positions=best_result["path"], savepath=args.out)
    maze.render(savepath="maze.png")

    print("Done.")

if __name__ == "__main__":
    main()
