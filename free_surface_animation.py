import os

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Load the saved edges array
edges = np.load("20250211_083548.npy")

# Get list of frame paths
frame_folder = r"C:\Users\niloy\Desktop\Shots 02112025\20250211_083548"
frames = [f for f in os.listdir(frame_folder) if f.lower().endswith(".bmp")]
frame_paths = [os.path.join(frame_folder, f) for f in frames]

# Load the first frame to get its height
sample_img = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
image_height = sample_img.shape[0]  # Height of the image

# Set up the figure and axis
fig, ax = plt.subplots()

# Display the first frame to initialize the plot
img = cv2.imread(frame_paths[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct coloring
image_display = ax.imshow(img)

# Initialize the spline plot
(spline_plot,) = ax.plot([], [], "r-", linewidth=2)

# Hide axis labels and tick marks
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])


# Function to update animation
def update(i):
    img = cv2.imread(frame_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_display.set_data(img)  # Update background image

    x = edges[0, :, i]
    y = edges[1, :, i]

    if not np.isnan(x).all() and not np.isnan(y).all():  # Avoid plotting empty frames
        y = image_height - y  # Invert y-axis
        spline_plot.set_data(x, y)
    else:
        spline_plot.set_data([], [])  # Hide the spline if there's no data

    ax.set_title(f"Frame {i}")
    return image_display, spline_plot


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(frame_paths) - 1, interval=50)

# Save animation as a .gif file
ani.save("edge_detection_animation.gif", writer="pillow", fps=20)

# Show animation
plt.show()
