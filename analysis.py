import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

experiment = "20250211_082312"
obstacle_path = rf"./obstacle_data/{experiment}000001_obstacle.npy"
free_surface_data = rf"./free_surface_data/{experiment}_free_surface.npy"

obstacle = np.load(obstacle_path)  # Load the saved array
free_surface_all = np.load(free_surface_data)

frame_number = 200
free_surface = free_surface_all[:, :, frame_number]

fig, ax = plt.subplots()
ax.plot(free_surface[0], free_surface[1])

# Ensure obstacle values are scalars
x = obstacle[0][0][0]  # Convert to scalar
y = 312 - obstacle[0][0][1]  # Convert to scalar
r = obstacle[0][0][2]  # Convert to scalar

# Create and add the circle patch
circle = Circle((x, y), r, color="r", fill=False)
ax.add_patch(circle)

ax.set_ylim(0, 320)
ax.set_xlim(0, 512)
ax.set_aspect("equal")  # Ensure the circle isn't distorted

plt.show()
