# %% 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.signal import savgol_filter

from obstacle_detection import detect_circles

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=False)
# %% 

experiment_number = "20250430_125933"
experiment_path = rf"E:\FDL\2D Jet Study Experiments\04302025\{experiment_number}"
results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"
free_surface_data = rf"{results_path}\free_surface_data.npy"


free_surface_all = np.load(free_surface_data)
conversion_factor = np.load(rf"{results_path}\conversion_factor.npy")

circle = detect_circles(
    r"C:\Users\niloy\Desktop\Experiments\04302025\20250430_125933",
    save_obstacle_data=True,
    save_path=results_path,
    show=False,
)

x_circle = circle[0][0]*conversion_factor
y_circle = circle[0][1]*conversion_factor
R_circle = circle[0][2]*conversion_factor

plt.show()
plt.xlim(0,1024*conversion_factor)
plt.ylim(0,1024*conversion_factor)

frame = 175
x_surface = free_surface_all[0, :, frame]*conversion_factor
y_surface = free_surface_all[1, :, frame]*conversion_factor
y_surface_init = np.mean(free_surface_all[1, :, 10]*conversion_factor)
y_surface_max = y_surface_init + 65

idx = np.abs(x_surface - x_circle).argmin()
central_height = y_surface[idx]
quarter_span = np.max(x_surface)/4

# Define the bounds
left_mask = (x_surface >= x_circle - quarter_span) & (x_surface <= x_circle)
right_mask = (x_surface >= x_circle) & (x_surface <= x_circle + quarter_span)

# Find minimum values
min_left = y_surface[left_mask].min()
min_right = y_surface[right_mask].min()

# Find x-locations of those minima
idx_left_all = np.where(left_mask)[0]
idx_left = idx_left_all[np.argmin(y_surface[left_mask])]
x_min_left = x_surface[idx_left]
y_min_left = y_surface[idx_left]

idx_right_all = np.where(right_mask)[0]
idx_right = idx_right_all[np.argmin(y_surface[right_mask])]
x_min_right = x_surface[idx_right]
y_min_right = y_surface[idx_right]

s_L = abs((central_height-y_min_left)/((x_circle-x_min_left)))
s_R = abs((central_height-y_min_right)/((x_circle-x_min_right)))
h = central_height - (y_min_right+y_min_left)/2
w = abs(x_min_left - x_min_right)
#s = (s_L+s_R)/2
s = h/w

plt.plot(x_surface,y_surface )
plt.hlines(np.mean(y_surface), np.min(x_surface), np.max(x_surface), linestyles="--", colors="k")
plt.hlines(y_surface_init, np.min(x_surface), np.max(x_surface), linestyles="-", colors="r",)
plt.hlines(y_surface_max, np.min(x_surface), np.max(x_surface), linestyles="-", colors="b")
plt.scatter(x_circle, central_height, color = "r")
plt.scatter(x_min_left, y_min_left, color = "orange")
plt.scatter(x_min_right, y_min_right, color = "pink")
plt.gca().set_aspect('equal')
# Create a circle
ax = plt.gca()

obstacle = Circle((x_circle, y_circle), R_circle, edgecolor='blue', facecolor='none')  # Change colors as needed
ax.add_patch(obstacle)
plt.title(rf" $s$ = {s:.2g}")
plt.show()

# %%


# --- Load data ---
experiment_number = "20250430_125933"
experiment_path = rf"E:\FDL\2D Jet Study Experiments\04302025\{experiment_number}"
results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"
free_surface_data = rf"{results_path}\free_surface_data.npy"

free_surface_all = np.load(free_surface_data)
conversion_factor = np.load(rf"{results_path}\conversion_factor.npy")

# --- Detect circle (obstacle) ---
circle = detect_circles(
    r"C:\Users\niloy\Desktop\Experiments\04302025\20250430_125933",
    save_obstacle_data=True,
    save_path=results_path,
    show=False,
)

x_circle = circle[0][0] * conversion_factor
y_circle = circle[0][1] * conversion_factor
R_circle = circle[0][2] * conversion_factor

# --- Parameters ---
n_frames = free_surface_all.shape[2]
quarter_span = (np.max(free_surface_all[0, :, 0]) * conversion_factor) / 4
y_surface_init = np.mean(free_surface_all[1, :, 10] * conversion_factor)

# --- Compute s for each frame ---
s_values = []

for frame in range(n_frames):
    x_surface = free_surface_all[0, :, frame] * conversion_factor
    y_surface = free_surface_all[1, :, frame] * conversion_factor
    y_surface =  savgol_filter(y_surface, window_length=11, polyorder=5)
    idx = np.abs(x_surface - x_circle).argmin()
    central_height = y_surface[idx]

    # Define the bounds
    left_mask = (x_surface >= x_circle - quarter_span) & (x_surface <= x_circle)
    right_mask = (x_surface >= x_circle) & (x_surface <= x_circle + quarter_span)

    # Skip frame if masks are empty (to avoid crashes)
    if not left_mask.any() or not right_mask.any():
        s_values.append(np.nan)
        continue

    # Find min heights
    min_left = y_surface[left_mask].min()
    min_right = y_surface[right_mask].min()

    # Indices of minima
    idx_left_all = np.where(left_mask)[0]
    idx_left = idx_left_all[np.argmin(y_surface[left_mask])]
    x_min_left = x_surface[idx_left]
    y_min_left = y_surface[idx_left]

    idx_right_all = np.where(right_mask)[0]
    idx_right = idx_right_all[np.argmin(y_surface[right_mask])]
    x_min_right = x_surface[idx_right]
    y_min_right = y_surface[idx_right]

    # Compute steepness
    s_L = abs((central_height - y_min_left) / (x_circle - x_min_left))
    s_R = abs((central_height - y_min_right) / (x_min_right - x_circle))
    h = central_height - (y_min_right+y_min_left)/2
    w = abs(x_min_left - x_min_right)
    #s = ((s_L + s_R) / 2)/2
    s = h/w

    s_values.append(s)

# --- Plotting s vs. frame ---
frames = np.arange(n_frames)
s_values = np.array(s_values)
#s_smooth = savgol_filter(s_values, window_length=11, polyorder=3)
plt.figure(figsize=(10, 4))
plt.plot(frames, s_values, label='Steepness $s$', color='blue')
plt.xlabel('Frame Number')
plt.ylabel('Steepness $s$')
plt.title('Steepness vs. Frame Number')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
