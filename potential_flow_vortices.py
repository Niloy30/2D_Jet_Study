# %%
import matplotlib.pyplot as plt
import numpy as np

# Parameters
a = 70  # m/s^2
b = 0.01  # m

# Parameters
t = 0.02
Gamma = a**2 * t**3  # Vortex strength (positive for right vortex, negative for left)
b = 0.01  # Distance between vortices (m)
U_inf = a * t  # Background flow velocity (m/s)
clip_speed = 5  # Maximum velocity to display (m/s)

# Vortex positions
x1, y1 = -b / 2, 0
x2, y2 = +b / 2, 0

# Grid
x = np.linspace(-0.05, 0.05, 300)
y = np.linspace(-0.05, 0.05, 300)
X, Y = np.meshgrid(x, y)

# Velocity field and magnitude
U = np.zeros_like(X)
V = np.zeros_like(Y)
Speed = np.zeros_like(X)

# Compute velocity components and magnitude
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_ = X[i, j]
        y_ = Y[i, j]

        # Velocity from both vortices
        r1_sq = (x_ - x1) ** 2 + (y_ - y1) ** 2
        r2_sq = (x_ - x2) ** 2 + (y_ - y2) ** 2

        u1 = (Gamma / (2 * np.pi)) * (y_ - y1) / r1_sq
        v1 = -(Gamma / (2 * np.pi)) * (x_ - x1) / r1_sq
        u2 = -(Gamma / (2 * np.pi)) * (y_ - y2) / r2_sq
        v2 = (Gamma / (2 * np.pi)) * (x_ - x2) / r2_sq

        u = u1 + u2
        v = v1 + v2 + U_inf

        U[i, j] = u
        V[i, j] = v
        Speed[i, j] = np.sqrt(u**2 + v**2)

# Clip velocity magnitude
Speed = np.clip(Speed, 0, clip_speed)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Filled contour plot of speed
contour = ax.contourf(
    X, Y, Speed, levels=np.linspace(0, clip_speed, 100), cmap="Blues", extend="max"
)

# Streamlines
ax.streamplot(X, Y, U, V, density=1.5, color="w", linewidth=0.25, arrowsize=0.75)


# Vortex positions
ax.plot([x1, x2], [y1, y2], "wo", label="Vortices")

# Colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("Velocity Magnitude (m/s)")
cbar.ax.set_yticks([0, 1, 2, 3, 4, 5])

# Labels and formatting
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
plt.tight_layout()
plt.show()


# Velocity at (0, s)
def velocity_at(x, y):
    r1_sq = (x - x1) ** 2 + (y - y1) ** 2
    r2_sq = (x - x2) ** 2 + (y - y2) ** 2
    u1 = (Gamma / (2 * np.pi)) * (y - y1) / r1_sq
    v1 = -(Gamma / (2 * np.pi)) * (x - x1) / r1_sq
    u2 = -(Gamma / (2 * np.pi)) * (y - y2) / r2_sq
    v2 = (Gamma / (2 * np.pi)) * (x - x2) / r2_sq
    return u1 + u2, v1 + v2 + U_inf


s = 0.02
u_s, v_s = velocity_at(0, s)


# %%
t = np.linspace(0.015, 0.05, 500)  # time array (avoid zero)
# Compute induced velocity uy(t)
s = a * t**2 / 2 + b / 2
uy = (-2 / np.pi) * a**2 * b * t**3 / (b**2 + a**2 * t**4)

ratio = uy / (a * t)
# Numerical derivative d(uy)/dt
duy_dt = np.gradient(uy, t)
du_dt_analytical = (
    2
    / np.pi
    * a**2
    / b
    * b**2
    * t**2
    * (a**2 * t**4 - 3 * b**2)
    / (a**2 * t**4 + b**2) ** 2
)

t_crit = ((b**2 / a**2) * (12 + np.sqrt(132)) / 2) ** 0.25
dudt_at_tcrit = (
    2
    / np.pi
    * a**2
    / b
    * b**2
    * t_crit**2
    * (a**2 * t_crit**4 - 3 * b**2)
    / (a**2 * t_crit**4 + b**2) ** 2
)


# Plot both uy and its derivative

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

plt.plot(t, ratio, "blue")
plt.ylabel(r"Induced Velocity Fraction $\frac{u_y}{U}$")
plt.xlabel("$t$ $[s]$")
plt.tight_layout()
plt.savefig(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\01 - Publications\Periodicity effects on Jet formation of accelerated free surface\fig\v_frac.pdf",
    dpi=300,
    format="pdf",
    bbox_inches="tight",
)
plt.show()

plt.plot(t, duy_dt / (a + 9.81), "blue")
plt.ylabel(r"Acceleration Fraction $\frac{a_{\mathrm{v}}}{a_{\mathrm{fs} + g}}$")
plt.xlabel("$t$ $[s]$")
plt.tight_layout()
plt.savefig(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\01 - Publications\Periodicity effects on Jet formation of accelerated free surface\fig\a_frac.pdf",
    dpi=300,
    format="pdf",
    bbox_inches="tight",
)
plt.show()

# %%
g = 9.81
a_max = 1100

a = np.linspace(0, a_max, a_max)
C = 6 + np.sqrt(33)
dudt_at_tcrit = (2 * a / np.pi) * np.sqrt(C) * (C - 3) / (C + 1) ** 2
ratio = dudt_at_tcrit / (a + g)
plt.plot(a, ratio, "blue")
max_ratio = (2 / np.pi) * np.sqrt(C) * (C - 3) / (C + 1) ** 2
plt.hlines(max_ratio, 0, a_max, linestyles="--", color="k")
plt.show()
