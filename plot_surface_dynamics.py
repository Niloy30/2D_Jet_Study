import matplotlib.pyplot as plt
import numpy as np


def plot_surface_dynamics(
    experiment_number,
    npy_file,
    save_plot=True,
    show_plot=False,
    save_path="surface_dynamics.pdf",
    scaling=1,
    FPS=1,
):
    """
    Plots the surface dynamics from a saved numpy array.

    Parameters:
        npy_file (str): Path to the saved numpy file containing edge data.
        save_plot (bool): Whether to save the plot as an image file.
        plot_filename (str): Name of the file to save the plot.
    """
    edges = np.load(npy_file)  # Load the saved array
    y_values = edges[1, :, :] * scaling  # Extract y-values
    avg_y = np.mean(y_values, axis=0)  # Compute average y-value per frame
    time = np.arange(avg_y.shape[0]) / FPS  # Generate frame numbers

    # Plotting parameters
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    plt.rc("lines", linewidth=2)

    plt.figure(figsize=(10, 5))
    plt.plot(time, avg_y, marker=".", linestyle="--", color="b", label="Avg Y")
    plt.xlabel("Time")
    plt.ylabel("Average Y Position")
    plt.title(f"Shot {experiment_number}")
    plt.grid(True)

    if save_plot:
        plt.savefig(
            rf"{save_path}\free_surface_plot.pdf",
            dpi=300,
            format="pdf",
            bbox_inches="tight",
        )
    if show_plot:
        plt.show()
