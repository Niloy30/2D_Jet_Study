import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Given data
pressure = np.array([100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300])
fill_height = np.array([0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30])
data = np.array(
    [
        2.263130183,
        2.261195023,
        2.284927179,
        2.263231244,
        2.111414828,
        2.108505626,
        2.120013922,
        2.121763963,
        2.073065398,
        2.077944587,
        2.082631946,
        2.086233858,
    ]
)

a_star = 2 * ((data - 2) + 1)
a = 9.81 / (a_star / 2 - 1)
# Create grid for contour plot
p_grid, f_grid = np.meshgrid(
    np.linspace(min(pressure), max(pressure), 100),
    np.linspace(min(fill_height), max(fill_height), 100),
)

a_grid = griddata((pressure, fill_height), a_star, (p_grid, f_grid), method="cubic")

# Create contour plot
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

plt.figure(figsize=(10, 5))
contour = plt.contourf(p_grid, f_grid, a_grid, levels=20, cmap="winter")
plt.colorbar(contour, label="$a^*$ [m/s$^2$]")
plt.xlabel("Pressure [PSI]")
plt.ylabel("Fill Height [mm]")
plt.show()
