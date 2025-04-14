# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Given data
pressure = np.array(
    [
        100,
        100,
        100,
        100,
        200,
        200,
        200,
        200,
        300,
        300,
        300,
        300,
        147.5,
        148.6,
        149.4,
        149.8,
        250.9,
        247.1,
        253.0,
        251.3,
    ]
)

fill_height = np.array(
    [0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 30, 20]
)
a_star = np.array(
    [
        2.526260366,
        2.522390046,
        2.569854359,
        2.526462488,
        2.222829655,
        2.217011251,
        2.240027843,
        2.243527927,
        2.146130797,
        2.155889174,
        2.165263892,
        2.172467716,
        2.28703174,
        2.270299534,
        2.274993977,
        2.298420286,
        2.169357162,
        2.171387105,
        2.187002362,
        2.191428411,
    ]
)

a = 9.81 / (a_star / 2 - 1)
# Create grid for contour plot
p_grid, f_grid = np.meshgrid(
    np.linspace(min(pressure), max(pressure), 100),
    np.linspace(min(fill_height), max(fill_height), 100),
)

a_grid = griddata((pressure, fill_height), a, (p_grid, f_grid), method="cubic")

# %%
# Create contour plot
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

plt.figure(figsize=(10, 5))
contour = plt.contourf(p_grid, f_grid, a_grid, levels=100, cmap="winter")
plt.scatter(pressure, fill_height, color="r")
plt.colorbar(contour, label="$a$ [m/s$^2$]")
plt.xlabel("Pressure [PSI]")
plt.ylabel("Fill Height [mm]")
plt.savefig(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\a.pdf"
)
plt.show()

# %%
