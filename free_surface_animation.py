import os

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def create_free_surface_animation(
    edges_file, experiment_path, save_path, fps=20, show_animation=False
):
    """
    Creates an animation overlaying detected edges on image frames and saves it as "free_surface_animation.gif".

    :param edges_file: Path to the .npy file containing edge data.
    :param experiment_path: Path to the directory containing multiple experiment folders.
    :param experiment_number: The specific experiment folder (as a string or number).
    :param save_path: Directory where the output GIF will be saved.
    :param fps: Frames per second for the animation.
    """

    # Construct the experiment folder path
    experiment_folder = os.path.join(experiment_path)

    if not os.path.exists(experiment_folder):
        raise FileNotFoundError(
            f"Experiment folder '{experiment_folder}' does not exist."
        )

    # Ensure save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the saved edges array
    edges = np.load(edges_file)

    # Get list of frame paths
    frames = [f for f in os.listdir(experiment_folder) if f.lower().endswith(".bmp")]
    frame_paths = [
        os.path.join(experiment_folder, f) for f in sorted(frames)
    ]  # Sort to ensure correct order

    if not frame_paths:
        raise FileNotFoundError(f"No BMP frames found in '{experiment_folder}'.")

    # Load the first frame to get its height
    sample_img = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    image_height = sample_img.shape[0]  # Height of the image

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Display the first frame to initialize the plot
    img = cv2.imread(frame_paths[0])
    img = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # Convert BGR to RGB for correct coloring
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

        if (
            not np.isnan(x).all() and not np.isnan(y).all()
        ):  # Avoid plotting empty frames
            y = image_height - y  # Invert y-axis
            spline_plot.set_data(x, y)
        else:
            spline_plot.set_data([], [])  # Hide the spline if there's no data

        ax.set_title(f"Frame {i}")
        return image_display, spline_plot

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_paths) - 1, interval=1000 / fps
    )

    # Define output file path
    output_gif = os.path.join(save_path, "free_surface_animation.gif")

    # Save animation as a .gif file
    ani.save(output_gif, writer="pillow", fps=fps)

    # Show animation
    if show_animation:
        plt.show()

    # print(f"Animation saved at: {output_gif}")


# Example usage:
# create_free_surface_animation("free_surface_data.npy", "C:/Users/niloy/Experiments", "002", "C:/Users/niloy/Animations")
