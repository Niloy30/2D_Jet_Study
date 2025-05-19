import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Load data from Excel
file_path = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results\results.xlsx"
df = pd.read_excel(file_path)

# Extract relevant data
h = df['h']
m = df['m']  # <-- Y-axis variable
Fr = df['Fr']
diameter = df['Diameter']
phase = df['Phase']

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

# Prepare colors and sizes
colors = ['blue' if t == 'Gravity Wave' else 'red' for t in phase]
sizes = [d * 15 for d in diameter]  # scale marker size

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(h, m, s=sizes, c=colors, alpha=0.7, edgecolors='k')

# Beautify plot
plt.xlabel('Height (h)')
plt.ylabel('m')  # Add y-axis label
plt.title('Surface Features by Height, m, Diameter, and Type')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the legend manually
red_patch = mpatches.Patch(color='red', label='Jet')
blue_patch = mpatches.Patch(color='blue', label='Gravity Wave')
plt.legend(handles=[blue_patch, red_patch])

plt.show()
