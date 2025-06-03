import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Define the x range
x = np.linspace(-5, 5, 1000)


# Define three Gaussian functions
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))


# Parameters for the three Gaussians
A_side = -3.5  # Side amplitude
mean_side = 1  # Side mean (controls spacing w)
std_side = 1  # Side width

A_center = -2 * A_side  # Center amplitude (controls h)
std_center = std_side  # Center sharpness
params = [
    (A_side, -mean_side, std_side),
    (A_center, 0, std_center),
    (A_side, mean_side, std_side),
]
# Generate each Gaussian
gaussians = [gaussian(x, amp, mean, std) for amp, mean, std in params]

# Sum the Gaussians
sum_wave = sum(gaussians)

# --- Find local maxima and minima ---
# Maxima
peaks, _ = find_peaks(sum_wave)
# Minima (invert the signal)
inv_peaks, _ = find_peaks(-sum_wave)

# Get x and y values for top 1 max and 2 minima near center
# Sort peaks/minima by proximity to x=0
max_idx = peaks[np.argmin(np.abs(x[peaks]))]
minima_idxs = inv_peaks[np.argsort(np.abs(x[inv_peaks]))[:2]]

# Extract coordinates of extrema
x_max = x[max_idx]
y_max = sum_wave[max_idx]

x_min1, x_min2 = x[minima_idxs[0]], x[minima_idxs[1]]
y_min1, y_min2 = sum_wave[minima_idxs[0]], sum_wave[minima_idxs[1]]

# Vertical distance h: max - average(minima)
h = y_max - (y_min1 + y_min2) / 2

# Horizontal distance w: between minima
w = abs(x_min2 - x_min1)

s = h / w

# Print results
print(f"Vertical distance h = {h:.4f}")
print(f"Horizontal distance w = {w:.4f}")
print(f"s = {s}")
# --- Plotting ---
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.figure(figsize=(10, 6))
plt.axis("equal")
plt.plot(x, sum_wave, label="Sum", linewidth=2, color="black")

# Plot max
plt.plot(x[max_idx], sum_wave[max_idx], "ro", label="Central Max")

# Plot minima
for i, idx in enumerate(minima_idxs):
    plt.plot(x[idx], sum_wave[idx], "bo", label=f"Min {i+1}" if i == 0 else None)

plt.xlabel("x")
# plt.title(f"Steepness $s = {s:.1f}$")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.axis("off")
plt.show()
