# %%
import numpy as np
import matplotlib.pyplot as plt
from calibration import get_scaling
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, fftfreq


calibration_grid = (
    r"E:\FDL\2D Jet Study Experiments\2025-03-17\192.168.0.10_C001H001S0004.bmp"
)
conversion_factor = get_scaling(calibration_grid)  # mm/pixel
# %%
experiment_number = "20250317_124335"
experiment_path = rf"E:\FDL\2D Jet Study Experiments\2025-03-17\{experiment_number}"
results_path = rf"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\{experiment_number}"
free_surface_data = rf"{results_path}\free_surface_data.npy"


free_surface_all = np.load(free_surface_data)


X = free_surface_all[0, :, :]
Y = free_surface_all[1, :, :]
T = np.arange(0, X.shape[1], 1)

Y_average = np.mean(Y, axis=0)

Y_average_m = (Y_average - np.min(Y_average)) * conversion_factor / 1000
T_s = T / 7000
initial_frame_number = np.argmin(Y_average)

# %%
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=False)

plt.plot(T_s, Y_average_m, ".")
# %%
t = T_s
y = Y_average_m
y_vs_t = CubicSpline(t, y)  # Fit cubic spline to (t, y)
U_vs_t = y_vs_t.derivative()(t)  # Evaluate derivative at t points
a_vs_t = y_vs_t.derivative(2)(t)  # Evaluate derivative at t points

y_vs_t = CubicSpline(t, y)  # Fit cubic spline to (t, y)

coeffs = np.polyfit(t, y, 2)  # Returns [a, b, c] for ax² + bx + c
quadratic_fit = np.poly1d(coeffs)  # Convert to a polynomial function

# Generate smooth x values for plotting
t_quad = np.linspace(min(t), max(t), 100)
y_quad = quadratic_fit(t_quad)

plt.figure(figsize=(10, 5))
plt.plot(t, y, marker=".", linestyle="--", color="b", label="Raw Data")
plt.plot(t, y_vs_t(t), "r", label="Cubic Spline")
plt.plot(
    t_quad,
    y_quad,
    label=f"Quadratic Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}",
    color="green",
)
plt.xlabel("$t$ [-]")
plt.ylabel("$y$ ")
plt.title(f"Shot {experiment_number}")
plt.legend()
plt.grid()

# %%
a = coeffs[0] * 2
g = 9.81

L = 2 + g / a

print(L)
