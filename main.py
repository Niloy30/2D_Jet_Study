"""
Filename: main.py
Author: Niloy Barua
Date: 2025-02-17
Version: 2.0
Description: Surface tracking using frame differencing.
"""

# %% Package Imports

import os

import numpy as np

from calibration import get_scaling, get_transformation_matrix
from free_surface_animation import create_free_surface_animation
from obstacle_detection import detect_circles
from plot_surface_dynamics import plot_surface_dynamics
from surface_perturbation_growth import analyze_steepness_vs_time
from tracking_surface import process_frames

# %% Choose experiment

experiment_number = "20250508_113617"
day = "05082025"

FPS = 7000

# %% File Paths

calibration_grid = (
    r"C:\Users\niloy\Desktop\Experiments\05082025\192.168.0.10_C001H001S0003.bmp"
)


experiment_path = rf"C:\Users\niloy\Desktop\Experiments\{day}\{experiment_number}"


results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"

if not os.path.exists(results_path):
    os.makedirs(results_path)


# %% Main Processes

print("Starting Main Processes")

pattern_size = (13, 20)
conversion_factor = get_scaling(calibration_grid, pattern_size)  # mm/pixel
np.save(rf"{results_path}\conversion_factor.npy", conversion_factor)


# %%
M_rot, output_size, rotation_angle, corners = get_transformation_matrix(
    calibration_grid, pattern_size
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

# %% broken right now
try: 
    create_free_surface_animation(
        npy_file,
        experiment_path,
        results_path,
        fps=20,
        show_animation=True,
    )
except:
    pass
# %%

if not os.path.exists(rf"{results_path}\obstacle_data.txt"):
    detect_circles(
        experiment_path,
        save_obstacle_data=True,
        save_path=results_path,
        show=True
    )

analyze_steepness_vs_time(results_path, FPS)

print("Done")
# %%
