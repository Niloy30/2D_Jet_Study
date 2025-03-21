"""
Filename: main.py
Author: Niloy Barua
Date: 2025-02-17
Version: 2.0
Description: Surface tracking using frame differencing.
"""

# %% Package Imports

import os

from free_surface_animation import create_free_surface_animation
from plot_surface_dynamics import plot_surface_dynamics
from tracking_surface import process_frames

# %% Choose experiment

experiment_number = "20250317_124335"
conversion_factor = 20 / 100  # mm/pixel
FPS = 7000

# %% File Paths

experiment_path = rf"C:\Users\niloy\Desktop\Experiments\03172025\{experiment_number}"


results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"

if not os.path.exists(results_path):
    os.makedirs(results_path)


# %% Main Processes

print("Starting Main Processes")

edges = process_frames(
    experiment_path,
    save_edges=True,
    save_path=results_path,
)

npy_file = rf"{results_path}\free_surface_data.npy"

# %%

plot_surface_dynamics(
    experiment_number,
    npy_file,
    save_plot=True,
    show_plot=False,
    save_path=results_path,
    scaling=conversion_factor,
    FPS=FPS,
)

# %%
create_free_surface_animation(
    npy_file,
    experiment_path,
    results_path,
    fps=20,
    show_animation=True,
)

# detect_circles(
#     experiment_path, save_obstacle_data=True, save_path=results_path, show=False
# )

print("Done")
# %%
