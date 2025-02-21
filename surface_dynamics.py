import matplotlib.pyplot as plt
import numpy as np

# Load the saved array
edges = np.load("20250211_083548.npy")  # Ensure the file is in the same directory

# Extract y-values (edges[1] contains y-values)
y_values = edges[1, :, :]  # Shape: (200, time_frames)

# Compute the average y-value at each time frame
avg_y = np.mean(y_values, axis=0)  # Shape: (time_frames,)

# Generate frame numbers as x-axis
frame_numbers = np.arange(avg_y.shape[0])

# Plot the results
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")

plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, avg_y, marker="o", linestyle="-", color="b", label="Avg Y")
plt.xlabel("Frame Number")
plt.ylabel("Average Y Position")
plt.title(f"Average Y Position Over Time \n Shot {20250211_083548}")
plt.legend()
plt.grid(True)
plt.savefig("20250211_083548.png", dpi=300, bbox_inches="tight")
