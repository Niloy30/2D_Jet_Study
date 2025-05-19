import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

# Dummy setup — replace with your actual data arrays
n_frames = 100
x_data = np.linspace(0, 2*np.pi, 100)
y_data_over_time = [np.sin(x_data + t * 0.1) for t in range(n_frames)]
y_surface_init = 0.0
y_surface_max = 1.0
x_circle = 3.0
y_circle = 0.5
R_circle = 0.25

# Dummy fixed points — replace with your real data
central_height = 0.5
x_min_left = 1.0
y_min_left = -0.5
x_min_right = 5.0
y_min_right = -0.4

# Setup
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Static circle obstacle
obstacle = Circle((x_circle, y_circle), R_circle, edgecolor='blue', facecolor='none')
ax.add_patch(obstacle)

# Plot elements to update
line, = ax.plot([], [], lw=2)
sc_central, = ax.plot([], [], 'ro')     # central height
sc_left, = ax.plot([], [], 'o', color='orange')  # min left
sc_right, = ax.plot([], [], 'o', color='pink')   # min right

# Horizontal lines
hline_mean = LineCollection([], linestyles="--", colors="k")
hline_init = LineCollection([], linestyles="-", colors="r")
hline_max = LineCollection([], linestyles="-", colors="b")
ax.add_collection(hline_mean)
ax.add_collection(hline_init)
ax.add_collection(hline_max)

# Set fixed x-limits
ax.set_xlim(np.min(x_data), np.max(x_data))
ax.set_ylim(-1.5, 1.5)

def update(frame):
    x_surface = np.array(x_data)
    y_surface = np.array(y_data_over_time[frame])

    # Main line
    line.set_data(x_surface, y_surface)

    # Horizontal lines
    x_min = float(np.min(x_surface))
    x_max = float(np.max(x_surface))
    y_mean = float(np.mean(y_surface))
    hline_mean.set_segments([[(x_min, y_mean), (x_max, y_mean)]])
    hline_init.set_segments([[(x_min, y_surface_init), (x_max, y_surface_init)]])
    hline_max.set_segments([[(x_min, y_surface_max), (x_max, y_surface_max)]])

    # Scatter points — wrap scalars in lists
    sc_central.set_data([x_circle], [central_height])
    sc_left.set_data([x_min_left], [y_min_left])
    sc_right.set_data([x_min_right], [y_min_right])

    ax.set_title(rf"Steepness $s$ = {frame/10:.2g}")

    return line, hline_mean, hline_init, hline_max, sc_central, sc_left, sc_right

# Create animation
ani = FuncAnimation(fig, update, frames=n_frames, interval=100)

# Save animation
ani.save('steepness_animation.gif', writer='pillow', fps=15)
