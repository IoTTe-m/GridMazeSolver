import argparse
import sys
import matplotlib.pyplot as plt

from .maze import Maze
from .ga import run_ga_conditional_pda, run_es_conditional_pda
from .genome import genome_to_conditional_pda_string

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
    p.add_argument("--dynamic-maze", action="store_true", help="Regenerate maze every generation")
    p.add_argument("--gui", action="store_true", help="Launch the GUI interface")
    p.add_argument("--remove_walls", type=int, default=0, help="Remove this many random interior walls after maze generation")
    p.add_argument("--out", type=str, default="best_path.png", help="Output filename for best path image")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    if args.gui:
        import tkinter as tk
        from .gui import SolverGUI
        root = tk.Tk()
        app = SolverGUI(root)
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
    print(f"Conditional PDA config: n_states={args.n_states} stack_symbols={args.n_stack_syms} max_steps={args.max_steps} remove_walls={args.remove_walls} dynamic_maze={args.dynamic_maze}")

    if args.algo == "ga":
        best, best_score, best_out, log = run_ga_conditional_pda(
            maze, n_states=args.n_states, n_stack_syms=args.n_stack_syms, pop_size=args.pop, gens=args.gens, seed=args.seed, ind_max_steps=args.max_steps, dynamic_maze=args.dynamic_maze
        )
    else:
        best, best_score, best_out, log = run_es_conditional_pda(
            maze, n_states=args.n_states, n_stack_syms=args.n_stack_syms, mu=args.pop, lam=args.pop * 2, gens=args.gens, seed=args.seed, ind_max_steps=args.max_steps, dynamic_maze=args.dynamic_maze
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
