"""
Filename: main.py
Author: Niloy Barua
Date: 2025-02-17
Version: 1.2
Description: Surface tracking using frame differencing.
"""

from surface_dynamics import plot_surface_dynamics
from tracking_surface import process_frames

edges = process_frames(
    r"C:\Users\niloy\Desktop\Shots 02112025\20250211_083548",
    save_edges=True,
    save_path="20250211_083548.npy",
)

plot_surface_dynamics(
    "20250211_083548.npy",
    save_plot=True,
    plot_filename="20250211_083548.png",
    scaling=0.2,
    FPS=8000,
)
