import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any, Optional
import numpy as np

from .maze import Maze
from .ga import run_ga_conditional_pda, run_es_conditional_pda
from .simulator import Simulator
from .evaluator import Evaluator
from .genome import genome_to_conditional_pda_string

class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid Maze Solver GUI")
        self.root.geometry("1400x900")

        self.running = False
        self.closing = False # Flag to prevent callbacks after close
        self._after_id = None
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.stats_history = {"avg": [], "min": [], "max": []}
        self.generations = []
        
        self.setup_ui()
        self.setup_plots()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.check_queue()

    def setup_ui(self):
        # Layout: Left panel (controls), Right panel (Graphs + Maze)
        # Using simple Frames side-by-side instead of PanedWindow to avoid geometry issues
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_container, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        right_frame = ttk.Frame(main_container, width=900)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Sub-frames for Right Panel ---
        self.fig, (self.ax_fitness, self.ax_maze) = plt.subplots(2, 1, figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Controls (Left Panel) ---
        ttk.Label(left_frame, text="Maze Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
        
        self.var_width = self.create_input(left_frame, "Width:", "21")
        self.var_height = self.create_input(left_frame, "Height:", "21")
        self.var_remove_walls = self.create_input(left_frame, "Remove Walls:", "0")
        
        self.var_dynamic = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Dynamic Maze (Generalization)", variable=self.var_dynamic).pack(anchor="w", pady=5)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Evolution Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)

        self.var_algo = tk.StringVar(value="ga")
        radio_frame = ttk.Frame(left_frame)
        radio_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(radio_frame, text="GA (Genetic Algorithm)", variable=self.var_algo, value="ga").pack(anchor="w")
        ttk.Radiobutton(radio_frame, text="ES (Evolution Strategy)", variable=self.var_algo, value="es").pack(anchor="w")

        self.var_pop = self.create_input(left_frame, "Population:", "50")
        self.var_gens = self.create_input(left_frame, "Generations:", "50")
        self.var_max_steps = self.create_input(left_frame, "Max Steps:", "200")
        self.var_n_states = self.create_input(left_frame, "N States:", "8")
        self.var_stack_syms = self.create_input(left_frame, "Stack Syms:", "4")
        self.var_seed = self.create_input(left_frame, "Seed (Optional):", "")

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.btn_run = ttk.Button(left_frame, text="Run Simulation", command=self.on_run)
        self.btn_run.pack(fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(left_frame, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

    def setup_plots(self):
        # Initialize plots with empty data
        self.ax_fitness.set_title("Fitness over Generations")
        self.ax_fitness.set_xlabel("Generation")
        self.ax_fitness.set_ylabel("Fitness")
        self.line_max, = self.ax_fitness.plot([], [], label="Max (Raw)", color="green", alpha=0.3)
        self.line_avg, = self.ax_fitness.plot([], [], label="Avg (Raw)", color="blue", alpha=0.3)
        self.line_max_smooth, = self.ax_fitness.plot([], [], label="Max (Smooth)", color="green", linewidth=2)
        self.line_avg_smooth, = self.ax_fitness.plot([], [], label="Avg (Smooth)", color="blue", linewidth=2)
        self.ax_fitness.legend(loc="upper left")
        
        self.ax_maze.set_xticks([])
        self.ax_maze.set_yticks([])
        # We will create the imshow object later when we have the maze
        self.img_maze = None
        self.line_path, = self.ax_maze.plot([], [], linewidth=2, color="blue", alpha=0.7)
        self.scat_start = None
        self.scat_goal = None

    def create_input(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        ttk.Entry(frame, textvariable=var, width=15).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return var

    def on_run(self):
        if self.running:
            return
        
        try:
            width = int(self.var_width.get())
            height = int(self.var_height.get())
            if width % 2 == 0: width += 1
            if height % 2 == 0: height += 1
            
            params = {
                "width": width,
                "height": height,
                "remove_walls": int(self.var_remove_walls.get()),
                "dynamic_maze": self.var_dynamic.get(),
                "algo": self.var_algo.get(),
                "pop": int(self.var_pop.get()),
                "gens": int(self.var_gens.get()),
                "max_steps": int(self.var_max_steps.get()),
                "n_states": int(self.var_n_states.get()),
                "n_stack_syms": int(self.var_stack_syms.get()),
                "seed": int(self.var_seed.get()) if self.var_seed.get().strip() else None
            }
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your inputs: {e}")
            return

        self.running = True
        self.stop_event.clear()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set("Running...")
        
        # Reset data
        self.stats_history = {"avg": [], "min": [], "max": []}
        self.generations = []
        
        # Clear/Reset lines
        self.line_max.set_data([], [])
        self.line_avg.set_data([], [])
        self.line_max_smooth.set_data([], [])
        self.line_avg_smooth.set_data([], [])
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        
        self.line_path.set_data([], [])
        if self.img_maze:
            self.img_maze.set_data(np.zeros((params["height"], params["width"]))) # Placeholder
            # Force redraw of maze grid if maze size changes? 
            # imshow extent depends on data shape. set_data works if shape is same or compatible?
            # Actually imshow wraps an AxesImage.
            # If dimensions change, set_data might handle it but extent needs update?
            # Easiest is to clear ax_maze if dims change.
            # For now assuming user sets dims once. If they change, we might need full clear.
            # Let's simple clear ax_maze on run if we want to be safe, but we wanted to optimize.
            # Let's keep optimization but assume same size or handle resize.
            pass

        self.canvas.draw()
        
        threading.Thread(target=self.run_solver_thread, args=(params,), daemon=True).start()

    def on_stop(self):
        if self.running:
            self.stop_event.set()
            self.status_var.set("Stopping...")

    def run_solver_thread(self, params):
        try:
            maze = Maze(
                width=params["width"],
                height=params["height"],
                seed=params["seed"],
                remove_walls=params["remove_walls"]
            )
            
            # Initial maze render
            self.queue.put(("maze_update", (maze, [])))

            def step_callback(gen, record, best_ind):
                if self.stop_event.is_set():
                    raise InterruptedError("Stopped by user")
                    
                import copy
                # Deepcopy maze to avoid race condition with the background thread
                # regenerating the maze while the GUI is rendering it.
                maze_copy = copy.deepcopy(maze)
                
                self.queue.put(("stats", (gen, record)))
                self.queue.put(("best_ind", (maze_copy, best_ind, params)))

            if params["algo"] == "ga":
                run_ga_conditional_pda(
                    maze, n_states=params["n_states"], n_stack_syms=params["n_stack_syms"],
                    pop_size=params["pop"], gens=params["gens"], seed=params["seed"],
                    ind_max_steps=params["max_steps"], dynamic_maze=params["dynamic_maze"],
                    step_callback=step_callback
                )
            else:
                run_es_conditional_pda(
                    maze, n_states=params["n_states"], n_stack_syms=params["n_stack_syms"],
                    mu=params["pop"], lam=params["pop"]*2, gens=params["gens"], seed=params["seed"],
                    ind_max_steps=params["max_steps"], dynamic_maze=params["dynamic_maze"],
                    step_callback=step_callback
                )
                
            self.queue.put(("done", None))

        except InterruptedError:
            self.queue.put(("status", "Stopped."))
        except Exception as e:
            self.queue.put(("error", str(e)))
        finally:
            self.queue.put(("cleanup", None))

    def check_queue(self):
        if self.closing:
            return

        try:
            while True:
                msg, data = self.queue.get_nowait()
                if msg == "stats":
                    gen, record = data
                    self.generations.append(gen)
                    self.stats_history["avg"].append(record["avg"])
                    self.stats_history["min"].append(record["min"])
                    self.stats_history["max"].append(record["max"])
                    self.update_fitness_graph()
                elif msg == "best_ind":
                    maze_obj, best_ind, params = data
                    self.render_maze_with_best(maze_obj, best_ind, params)
                elif msg == "maze_update":
                    maze_obj, path = data
                    self.render_maze(maze_obj, path) 
                elif msg == "status":
                    self.status_var.set(data)
                elif msg == "done":
                    self.status_var.set("Completed.")
                elif msg == "error":
                    messagebox.showerror("Error", data)
                    self.status_var.set("Error occurred.")
                elif msg == "cleanup":
                    self.running = False
                    self.btn_run.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        finally:
            if not self.closing:
                # 100ms is standard, but for faster updates we can go lower (e.g. 50ms)
                # But GUI might lag if thread floods queue. 100ms is safe.
                self._after_id = self.root.after(100, self.check_queue)

    def update_fitness_graph(self):
        # Update data instead of clearing
        self.line_max.set_data(self.generations, self.stats_history["max"])
        self.line_avg.set_data(self.generations, self.stats_history["avg"])
        
        # Calculate smoothing
        def smooth(data, alpha=0.1):
            if not data: return []
            s = [data[0]]
            for x in data[1:]:
                s.append(alpha * x + (1 - alpha) * s[-1])
            return s
            
        smooth_max = smooth(self.stats_history["max"])
        smooth_avg = smooth(self.stats_history["avg"])
        
        self.line_max_smooth.set_data(self.generations, smooth_max)
        self.line_avg_smooth.set_data(self.generations, smooth_avg)
        
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        self.canvas.draw_idle()  

    def render_maze_with_best(self, maze_obj, best_ind, params):
        from .simulator import Simulator
        sim = Simulator(maze_obj, max_steps=params["max_steps"], n_states=params["n_states"], n_stack_syms=params["n_stack_syms"])
        out = sim.run_genome(best_ind)
        self.render_maze(maze_obj, out["path"])

    def render_maze(self, maze_obj, path):
        # Update image data
        if self.img_maze is None:
             self.img_maze = self.ax_maze.imshow(maze_obj.grid, cmap="gray_r", interpolation="nearest")
             self.scat_start = self.ax_maze.scatter([maze_obj.start[1]], [maze_obj.start[0]], c="green", label="Start", s=100)
             self.scat_goal = self.ax_maze.scatter([maze_obj.goal[1]], [maze_obj.goal[0]], c="red", label="Goal", s=100)
        else:
             self.img_maze.set_data(maze_obj.grid)
             # Update extent and limits for new dimensions
             h, w = maze_obj.grid.shape
             self.img_maze.set_extent((-0.5, w-0.5, h-0.5, -0.5))
             self.ax_maze.set_xlim(-0.5, w-0.5)
             self.ax_maze.set_ylim(h-0.5, -0.5)
             
             # Update scatter (optional, if start/goal change)
             if self.scat_start:
                 self.scat_start.set_offsets(np.c_[[maze_obj.start[1]], [maze_obj.start[0]]])
             if self.scat_goal:
                 self.scat_goal.set_offsets(np.c_[[maze_obj.goal[1]], [maze_obj.goal[0]]])

        if path:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            self.line_path.set_data(xs, ys)
        else:
            self.line_path.set_data([], [])
            
        self.canvas.draw_idle()

    def on_close(self):
        self.closing = True
        if self._after_id:
            try:
                self.root.after_cancel(self._after_id)
            except ValueError:
                pass
        self.on_stop()
        self.root.destroy()
        plt.close('all') # Close all matplotlib figures
        import sys; sys.exit(0) # Forcefully kill the process
