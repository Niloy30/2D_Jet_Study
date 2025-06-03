# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
# === Step 1: Load Excel File and Filter Rows ===
results_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"
excel_path = rf"{results_path}\results.xlsx"
df = pd.read_excel(excel_path)

# Filter where m == 12.46
filtered_df = df[df["m"] == 12.459999999999999]
experiment_numbers = filtered_df[r"Experiment Number"].astype(str).tolist()
h_values = filtered_df["h"].tolist()

# Sort both lists by h
sorted_pairs = sorted(zip(h_values, experiment_numbers))
h_values_sorted, experiment_numbers_sorted = zip(*sorted_pairs)

# === Step 2: Prepare to Load Steepness Data and Plot ===
fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.cm.winter  # Blue to red colormap
colors = cmap(np.linspace(0, 1, len(h_values_sorted)))

for i, (exp_num, h) in enumerate(zip(experiment_numbers_sorted, h_values_sorted)):
    if not exp_num == "20250508_165511":
        steepness_path = os.path.join(results_path, exp_num, "steepness_data.npy")

        if not os.path.exists(steepness_path):
            print(f"Warning: File not found for experiment {exp_num}: {steepness_path}")
            continue

        # Load steepness data
        steepness_data = np.load(steepness_path)
        s_smooth = steepness_data[0]
        s_smooth = savgol_filter(s_smooth, window_length=100, polyorder=5)
        upper = steepness_data[1]
        lower = steepness_data[2]
        T = np.arange(0, len(s_smooth), 1) / 7000

        # Plotting
        ax.plot(T, s_smooth, label=f"h = {h:.2f}", linewidth=2, color=colors[i])
        # ax.fill_between(T, lower, upper, alpha=0.4, color=colors[i])

# === Step 3: Final Plot Formatting ===
plt.hlines(
    0.2,
    0,
    0.06,
    linestyles="--",
    colors="k",
)
plt.xlabel("$t$ [s]")
plt.ylabel("Steepness $s$")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlim(0, 0.05)
plt.ylim(0, 0.45)
plt.tight_layout()

plt.annotate(
    "Jetting Threshold",
    xy=(0.0125, 0.2),
    xytext=(0.005, 0.23),
    textcoords="data",
    fontsize=12,
    ha="center",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", linewidth=1.5),
)

plt.show()

# %%
