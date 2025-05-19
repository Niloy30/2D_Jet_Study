# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Load data from Excel
file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results merged.xlsx"
df = pd.read_excel(file_path)

# Extract relevant data
h = df['h']
m = df['m']
phase = df['Phase']
who = df['Who']

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

# Prepare colors and alpha values
colors = ['blue' if t == 'Gravity Wave' else 'red' for t in phase]
alphas = [0.25 if person == 'Rubert' else 0.7 for person in who]  # 50% for Rubert, default 70% for others

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each point individually to apply per-point alpha
for xi, yi, ci, ai in zip(h, m, colors, alphas):
    plt.scatter(xi, yi, s=50, c=ci, alpha=ai, edgecolors='k')

# Beautify plot
plt.xlabel('Height (h)')
plt.ylabel('m')
plt.title('Surface Features by Height, m, Diameter, and Type')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the legend manually
red_patch = mpatches.Patch(color='red', label='Jet')
blue_patch = mpatches.Patch(color='blue', label='Gravity Wave')
plt.legend(handles=[blue_patch, red_patch])

plt.show()


# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Load data from Excel
file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results merged.xlsx"
df = pd.read_excel(file_path)

# Extract relevant data
h = df['h']
phase = df['Phase']

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

# Prepare colors
colors = ['blue' if t == 'Gravity Wave' else 'red' for t in phase]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 2))  # Short height for 1D effect

# Plot all points at y=0
ax.scatter(h, [0]*len(h), s=50, c=colors, alpha=0.7, edgecolors='k')

# Beautify plot
ax.set_xlabel('Height (h)')
ax.set_yticks([])  # Remove y-axis ticks
ax.set_ylim(-1, 1)  # Add some vertical space so points are visible
ax.set_title('Distribution of Surface Features by Height')
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# Show the legend manually
red_patch = mpatches.Patch(color='red', label='Jet')
blue_patch = mpatches.Patch(color='blue', label='Gravity Wave')
ax.legend(handles=[blue_patch, red_patch])

plt.tight_layout()
plt.show()


# %%
