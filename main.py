"""
Filename: main.py
Author: Niloy Barua
Date: 2025-02-17
Version: 2.0
Description: Surface tracking using frame differencing.
"""

# %% Package Imports

import os

from calibration import get_scaling, get_transformation_matrix
from plot_surface_dynamics import plot_surface_dynamics
from tracking_surface import process_frames

# %% Choose experiment

experiment_number = "20250317_110024"

FPS = 7000

# %% File Paths

calibration_grid = (
    r"E:\FDL\2D Jet Study Experiments\2025-03-17\192.168.0.10_C001H001S0004.bmp"
)

experiment_path = rf"E:\FDL\2D Jet Study Experiments\2025-03-17\{experiment_number}"


results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"

if not os.path.exists(results_path):
    os.makedirs(results_path)


# %% Main Processes

print("Starting Main Processes")


conversion_factor = get_scaling(calibration_grid)  # mm/pixel

M_rot, output_size, rotation_angle, corners = get_transformation_matrix(
    calibration_grid, (13, 18)
)

edges = process_frames(
    experiment_path,
    M_rot,
    output_size,
    rotation_angle,
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

# # %% broken right now
# create_free_surface_animation(
#     npy_file,
#     experiment_path,
#     results_path,
#     fps=20,
#     show_animation=True,
# )

# detect_circles(
#     experiment_path, save_obstacle_data=True, save_path=results_path, show=False
# )

print("Done")
# %%
