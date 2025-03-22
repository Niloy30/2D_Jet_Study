"""
Filename: main_parallel.py
Author: Niloy Barua
Date: 2025-03-22
Version: 3.0
Description: Parallelized surface tracking using frame differencing.
"""

# %% Package Imports

import os
import multiprocessing
from calibration import get_scaling, get_transformation_matrix
from plot_surface_dynamics import plot_surface_dynamics
from tracking_surface import process_frames

# %% Constants

BASE_EXPERIMENT_DIR = r"E:\FDL\2D Jet Study Experiments\2025-03-17"
RESULTS_BASE_DIR = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"
CALIBRATION_GRID = (
    r"E:\FDL\2D Jet Study Experiments\2025-03-17\192.168.0.10_C001H001S0004.bmp"
)
FPS = 7000

# Get scaling factor from calibration grid
conversion_factor = get_scaling("Sample Frames/calibration_grid.bmp")  # mm/pixel


def process_experiment(experiment_number):
    """
    Processes a single experiment: applies transformations, tracks surface, and plots results.

    :param experiment_number: Name of the experiment folder.
    """
    experiment_path = os.path.join(BASE_EXPERIMENT_DIR, experiment_number)
    results_path = os.path.join(RESULTS_BASE_DIR, experiment_number)

    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)

    print(f"Processing {experiment_number}...")

    # Get transformation matrix
    M_rot, output_size, rotation_angle, corners = get_transformation_matrix(
        CALIBRATION_GRID, (13, 18)
    )

    # Track the free surface
    edges = process_frames(
        experiment_path,
        M_rot,
        output_size,
        rotation_angle,
        save_edges=True,
        save_path=results_path,
    )

    npy_file = os.path.join(results_path, "free_surface_data.npy")

    # Plot surface dynamics
    plot_surface_dynamics(
        experiment_number,
        npy_file,
        save_plot=True,
        show_plot=False,
        save_path=results_path,
        scaling=conversion_factor,
        FPS=FPS,
    )

    print(f"Finished processing {experiment_number}.")


def process_all_experiments():
    """
    Processes all experiments in the base directory using multiprocessing.
    """
    # Get list of all experiment folders
    experiment_folders = [
        d
        for d in os.listdir(BASE_EXPERIMENT_DIR)
        if os.path.isdir(os.path.join(BASE_EXPERIMENT_DIR, d))
    ]

    if not experiment_folders:
        print("No experiment folders found.")
        return

    num_workers = min(multiprocessing.cpu_count(), len(experiment_folders))
    print(
        f"Processing {len(experiment_folders)} experiments using {num_workers} cores..."
    )

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_experiment, experiment_folders)

    print("All experiments processed.")


if __name__ == "__main__":
    process_all_experiments()
