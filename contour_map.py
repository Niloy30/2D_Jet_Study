import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Set plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=False)

# Define Excel file path
results_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"
excel_path = rf"{results_path}\Lambda_Results.xlsx"

# Load the data from Excel
df = pd.read_excel(excel_path, engine="openpyxl")

# Extract data columns
pressure = df["Pressure"].values
fill_height = df["Fill Height"].values
a_star = df["Lambda"].values  # Lambda is represented as 'a_star' in the plot

# Create a grid for contour plotting
fill_height_grid, pressure_grid = np.meshgrid(
    np.linspace(min(fill_height), max(fill_height), 100),
    np.linspace(min(pressure), max(pressure), 100),
)

# Interpolate Lambda (a*) values onto the grid
a_star_grid = griddata(
    (fill_height, pressure), a_star, (fill_height_grid, pressure_grid), method="cubic"
)

# Plot the contour map
plt.figure(figsize=(8, 6))
contour = plt.contourf(
    fill_height_grid, pressure_grid, a_star_grid, levels=15, cmap="winter"
)
plt.colorbar(contour, label=r"$a^*$")
plt.xlabel("Fill Height [mm]")
plt.ylabel("Pressure [PSI]")
plt.title(r"$a^*  = 2+ \frac{g}{a}$")
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig(rf"{results_path}\contour_map.pdf", format="pdf", bbox_inches="tight")
