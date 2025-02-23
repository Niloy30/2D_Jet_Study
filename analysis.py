# %% Import Packages

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


# %% Choose parameters
plot_frame = True
frame_number = 1
experiment_number = "20250211_083548"

# %% File Paths

experiment_path = rf"E:\FDL\2D Jet Study Experiments\2025-02-11\{experiment_number}"


results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"

obstacle_path = rf"{results_path}\obstacle_data.npy"
free_surface_data = rf"{results_path}\free_surface_data.npy"

# %% Load Data

obstacle = np.load(obstacle_path)  # Load the saved array
free_surface_all = np.load(free_surface_data)


# Get Obstacle values (only for single obstacle)
x = obstacle[0][0][0]  # Convert to scalar
y = obstacle[0][0][1]  # Convert to scalar
R = obstacle[0][0][2]  # Convert to scalar

# %% Plotting a frame

if plot_frame:

    free_surface = free_surface_all[:, :, frame_number]

    fig, ax = plt.subplots()
    ax.plot(free_surface[0], free_surface[1], "b")

    # Create and add the circle patch
    circle = Circle((x, y), R, color="k", fill=True)
    ax.plot(x, y, "xr")
    ax.add_patch(circle)

    ax.set_ylim(0, 320)
    ax.set_xlim(0, 512)
    ax.set_aspect("equal")  # Ensure the circle isn't distorted

    plt.show()

# %%
initial_free_surface = free_surface = free_surface_all[:, :, 0]
H0 = initial_free_surface[1, 0]
r = R / H0

# %%
