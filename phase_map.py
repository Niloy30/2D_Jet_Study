# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Load data from Excel
file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results merged.xlsx"
df = pd.read_excel(file_path)

# Extract relevant data
h = df["h"]
m = df["m"]
phase = df["Phase"]
who = df["Who"]

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

# Prepare colors and alpha values
colors = ["blue" if t == "Gravity Wave" else "red" for t in phase]
alphas = [
    0.25 if person == "Rubert" else 0.7 for person in who
]  # 50% for Rubert, default 70% for others

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each point individually to apply per-point alpha
for xi, yi, ci, ai in zip(h, m, colors, alphas):
    plt.scatter(xi, yi, s=50, c=ci, alpha=ai, edgecolors="k")

# Beautify plot
plt.xlabel("Height (h)")
plt.ylabel("m")
plt.title("Surface Features by Height, m, Diameter, and Type")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Show the legend manually
red_patch = mpatches.Patch(color="red", label="Jet")
blue_patch = mpatches.Patch(color="blue", label="Gravity Wave")
plt.legend(handles=[blue_patch, red_patch])

plt.show()


# %%
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load data from Excel
# file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results merged.xlsx"
# df = pd.read_excel(file_path)

# # Extract relevant data
# h = df["h"]
# phase = df["Phase"]

# # Plotting parameters
# plt.rc("text", usetex=True)
# plt.rc("font", family="serif", size=14)
# plt.rc("lines", linewidth=2)

# # Prepare colors
# colors = ["blue" if t == "Gravity Wave" else "red" for t in phase]

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(10, 2))  # Short height for 1D effect

# # Plot all points at y=0
# ax.scatter(h, [0] * len(h), s=50, c=colors, alpha=0.7, edgecolors="k")

# # Beautify plot
# ax.set_xlabel("Height (h)")
# ax.set_yticks([])  # Remove y-axis ticks
# ax.set_ylim(-1, 1)  # Add some vertical space so points are visible
# ax.set_title("Distribution of Surface Features by Height")
# ax.grid(True, axis="x", linestyle="--", alpha=0.5)

# # Show the legend manually
# red_patch = mpatches.Patch(color="red", label="Jet")
# blue_patch = mpatches.Patch(color="blue", label="Gravity Wave")
# ax.legend(handles=[blue_patch, red_patch])

# plt.tight_layout()
# plt.show()


# %%

# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load data from Excel
file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results merged.xlsx"
df = pd.read_excel(file_path)

# Extract relevant data
h = df["h"]
m = df["m"]
phase = df["Phase"]
who = df["Who"]

# --- Prepare classification data ---
X = np.column_stack((h, m))  # Features: h and m
y = LabelEncoder().fit_transform(phase)  # Labels: Gravity Wave = 0, Jet = 1

# Train SVC model
weights = np.where(df["m"] > 24, 0.025, 1.0)  # Customize threshold and weight

clf = SVC(kernel="rbf", C=72,gamma= 2e-4)

clf.fit(X, y,sample_weight=weights)

# --- Plotting ---
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

colors = ["blue" if t == "Gravity Wave" else "red" for t in phase]
alphas = [0.7 if person == "Rubert" else 0.7 for person in who]

plt.figure(figsize=(10, 6))

# Plot each point
for xi, yi, ci, ai in zip(h, m, colors, alphas):
    plt.scatter(xi, yi, s=50, c=ci, alpha=ai, edgecolors="k")

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                     np.linspace(*ylim, num=200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, colors="k", levels=[0], alpha=0.8, linestyles=["-"])
#plt.contour(xx, yy, Z, colors="k", levels=[-1, 1], alpha=0.3, linestyles=["--"])


# Labels, title, legend
plt.xlabel(r"Fill Level: $h = \frac{H_0}{R}-1$")
plt.ylabel(r"Spacing: $m = \frac{M}{2R}$")
plt.grid(True, linestyle="--", alpha=0.5)
plt.vlines(1.3,0,26,linestyles='--', color = 'k')
plt.ylim(0,26)


plt.tight_layout()

red_patch = mpatches.Patch(color="red", label="Jet")
blue_patch = mpatches.Patch(color="blue", label="Gravity Wave")
plt.legend(handles=[blue_patch, red_patch])
# Experimental annotation
plt.annotate(
    'Experimental\n boundary',
    xy=(0.75,13),
    xytext=(0.4, 10),
    textcoords='data',
    fontsize=12,
    ha='center',
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.3",
        linewidth=1.5
    )
)

# Analytical annotation
plt.annotate(
    'Analytical\n boundary',
    xy=(1.3,13),
    xytext=(1.6, 10),
    textcoords='data',
    fontsize=12,
    ha='center',
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=0.3",
        linewidth=1.5
    )
)

plt.show()

# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load data from Excel
file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results_model.xlsx"
df = pd.read_excel(file_path)

# Extract relevant data
h = df["h"]
m = df["m"]
phase = df["Phase"]
who = df["Who"]


#clf.fit(X, y)
# --- Plotting ---
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

colors = ["blue" if t == "Gravity Wave" else "red" for t in phase]
alphas = [0.7 if person == "Rubert" else 0.7 for person in who]

plt.figure(figsize=(10, 6))

# Plot each point
for xi, yi, ci, ai in zip(h, m, colors, alphas):
    plt.scatter(xi, yi, s=50, c=ci, alpha=ai, edgecolors="k")


# Labels, title, legend
plt.xlabel("Height (h)")
plt.ylabel("m")
plt.title("Surface Features by Height, m, Diameter, and Type")
plt.grid(True, linestyle="--", alpha=0.5)
plt.vlines(1.3,0,26)
plt.ylim(0,26)
plt.tight_layout()

red_patch = mpatches.Patch(color="red", label="Jet")
blue_patch = mpatches.Patch(color="blue", label="Gravity Wave")
plt.legend(handles=[blue_patch, red_patch])

plt.show()
