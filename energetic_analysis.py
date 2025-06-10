# %%

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def energy_gravity(x, eta, rho = 1000, g = 80):
    Eg = 1/2 * rho * g * np.trapezoid(eta**2,x)
    return Eg

def energy_surface_tension(x,eta,gamma = 0.072):
    deta_dx = np.gradient(eta, x)
    integrand = (1+deta_dx**2)**0.5 -1
    Est = gamma * np.trapezoid(integrand,x)
    return Est


g = 9.81
FPS = 7000

experiment_number = "20250516_115020"
results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"

conversion_factor = np.load(rf"{results_path}\conversion_factor.npy")
free_surface_all = np.load(rf"{results_path}\free_surface_data.npy")
steepness_data = np.load(rf"{results_path}\steepness_data.npy")
critical_points = np.load(rf"{results_path}\steepness_critical_points.npy")
circle = np.loadtxt(rf"{results_path}\obstacle_data.txt")

x_circle, y_circle, r_circle = circle * conversion_factor
y_circle = 1024 * conversion_factor - y_circle

X = free_surface_all[0, :, :] * conversion_factor - x_circle
Y = free_surface_all[1, :, :] * conversion_factor - y_circle
T = np.arange(0, X.shape[1], 1) / FPS
s_smooth = steepness_data[0]

Frame_max = np.size(T) - 100
cmap = matplotlib.colormaps["Blues"]
norm = mcolors.Normalize(vmin=0, vmax=Frame_max - 1)
colors = [cmap(norm(i)) for i in range(Frame_max)]

plt.figure(figsize=(10, 6))
frame = 20
while frame < Frame_max:
    smooth_surface = savgol_filter(Y[:, frame], window_length=10, polyorder=2)
    plt.plot(X[:, frame], smooth_surface, color=colors[frame])

    # Calculate far-field mask (±10 mm from center excluded)
    exclusion_half_width = 10  # mm
    x_vals = X[:, frame]
    mask_far_field = np.abs(x_vals) > exclusion_half_width

    # Compute far-field mean height
    far_field_mean = np.mean(Y[mask_far_field, frame])

    # Red dashed line for far-field mean
    red_cmap = matplotlib.colormaps["Reds"]
    red_color = red_cmap(norm(frame))
    plt.axhline(y=far_field_mean, color=red_color, linestyle="--", linewidth=1.2)

    # Plot critical points
    left_min = critical_points[0, frame] - [x_circle, y_circle]
    center = critical_points[1, frame] - [x_circle, y_circle]
    right_min = critical_points[2, frame] - [x_circle, y_circle]

    plt.scatter(left_min[0], left_min[1], color="blue", marker="o", s=30, label="Local Min" if frame == 0 else "")
    plt.scatter(center[0], center[1], color="red", marker="o", s=40, label="Central Max" if frame == 0 else "")
    plt.scatter(right_min[0], right_min[1], color="blue", marker="o", s=30)

    frame += 10


lim = 1024 * conversion_factor
circle = plt.Circle(
    (0, 0), r_circle, color="red", fill=False
)
ax = plt.gca()
ax.add_patch(circle)
ax.set_aspect("equal")
plt.title("Free Surface Evolution")
plt.xlabel("$x/H_0$ (mm)")
plt.ylabel("$y/H_0$ (mm)")
plt.tight_layout()
plt.tight_layout()
plt.savefig(rf"{results_path}\free_surface_evolution.pdf")
plt.show()
# %% 
rho = 1000  # kg/m^3
g = 9.81    # m/s^2
gamma = 0.072  # Surface tension coefficient (N/m)

# Convert X and Y from mm to meters
X_m = X / 1000  # shape: (n_points, n_frames)
Y_m = Y / 1000

# Define exclusion zone width (in meters now)
exclusion_half_width = 0.010  # 10 mm → 0.010 m

# Get x values (assumed fixed in time)
x_vals = X_m[:, 0]

# Create masks
mask_far_field = np.abs(x_vals) > exclusion_half_width

# Initialize energy arrays
Eg_total = np.zeros(Y.shape[1])
Eg_far = np.zeros(Y.shape[1])
Eg_perturbation = np.zeros(Y.shape[1])
Est_perturbation = np.zeros(Y.shape[1])  # Surface tension energy perturbation

# Compute energies frame-by-frame
for i in range(Y.shape[1]):
    eta_total = Y_m[:, i] - np.mean(Y_m[mask_far_field, i])
    eta_far = Y_m[mask_far_field, i] - np.mean(Y_m[mask_far_field, i])

    Eg_total[i] = energy_gravity(X_m[:, i], eta_total, rho=rho, g=g)
    Eg_far[i] = energy_gravity(X_m[mask_far_field, i], eta_far, rho=rho, g=g)
    Eg_perturbation[i] = Eg_total[i] - Eg_far[i]

    # Surface tension energy for total and far field
    Est_total = energy_surface_tension(X_m[:, i], eta_total, gamma=gamma)
    Est_far = energy_surface_tension(X_m[mask_far_field, i], eta_far, gamma=gamma)
    Est_perturbation[i] = Est_total - Est_far

# Smooth energies for better plotting
Eg_perturbation = savgol_filter(Eg_perturbation, window_length=20, polyorder=2)
Est_perturbation = savgol_filter(Est_perturbation, window_length=20, polyorder=2)

# Plot both energies
plt.figure(figsize=(8, 5))
plt.plot(T, Eg_perturbation, label="Perturbation Gravitational Energy")
plt.plot(T, Est_perturbation, label="Perturbation Surface Tension Energy")
plt.title("Isolated Energies of Perturbation")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J/m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(rf"{results_path}\isolated_energies_perturbation_meters.pdf")
plt.show()


# %%

# %%
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 6))

line, = ax.plot([], [], lw=2)
left_pt = ax.scatter([], [], color="blue", s=30, label="Local Min")
center_pt = ax.scatter([], [], color="red", s=40, label="Central Max")
right_pt = ax.scatter([], [], color="blue", s=30)
mean_line, = ax.plot([], [], color="red", linestyle="--", lw=1.2, label="Far-field Mean")
circle_patch = plt.Circle((0, 0), r_circle, color="red", fill=False)

ax.add_patch(circle_patch)
ax.set_xlim(X[:, 0].min(), X[:, 0].max())
ax.set_ylim(Y.min(), Y.max())
ax.set_aspect("equal")
ax.set_title("Free Surface Evolution (Animated)")
ax.set_xlabel("$x/H_0$ (mm)")
ax.set_ylabel("$y/H_0$ (mm)")
ax.legend(loc="lower right")

def init():
    line.set_data([], [])
    mean_line.set_data([], [])
    return line, left_pt, center_pt, right_pt, mean_line


def animate(frame):
    smooth_surface = savgol_filter(Y[:, frame], window_length=10, polyorder=2)
    x_vals = X[:, frame]
    line.set_data(x_vals, smooth_surface)

    # Far-field mean
    mask_far_field = np.abs(x_vals) > 10  # mm
    far_field_mean = np.mean(Y[mask_far_field, frame])
    mean_line.set_data(x_vals, np.full_like(x_vals, far_field_mean))

    # Critical points
    left_min = critical_points[0, frame] - [x_circle, y_circle]
    center = critical_points[1, frame] - [x_circle, y_circle]
    right_min = critical_points[2, frame] - [x_circle, y_circle]

    left_pt.set_offsets([left_min])
    center_pt.set_offsets([center])
    right_pt.set_offsets([right_min])

    return line, left_pt, center_pt, right_pt, mean_line



plt.rcParams['animation.ffmpeg_path'] = r"C:\ffmpeg\bin\ffmpeg.exe"

# Create animation
ani = animation.FuncAnimation(
    fig, animate, frames=range(0, Frame_max, 1), init_func=init,
    blit=True, interval=40
)

# Create ffmpeg writer without dpi
ffmpeg_writer = animation.FFMpegWriter(fps=15)

# Save animation specifying dpi here
ani.save("free_surface_animation.mp4", writer=ffmpeg_writer, dpi=200)

# plt.show()  # Uncomment this line to view inline instead of saving
