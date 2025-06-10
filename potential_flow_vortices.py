# %%
import matplotlib.pyplot as plt
import numpy as np

# Parameters
a = 80  # m/s^2
b = 0.01  # m
t = np.linspace(0.015, 0.05, 500)  # time array (avoid zero)

# Compute induced velocity uy(t)
s = a * t**2 / 2 + b / 2
uy = 2 / np.pi * a**2 * t**3 * -b * (b**2 + a**2 * t**4) ** (-1)
ratio = uy / (a * t)
# Numerical derivative d(uy)/dt
duy_dt = np.gradient(uy, t)

# Plot both uy and its derivative

# Plotting parameters
plt.rc("text", usetex=True)

plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t, ratio, label=r"$u_y(t)/U_{\infty}(t)$", color="blue")
plt.xlabel("Time (s)")
plt.ylabel(r"Fraction of Induced vertical velocity $\frac{u_y}{U_{\infty}}$")
plt.title("Induced vertical velocity")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, duy_dt / a, color="red", label=r"$\frac{du_y}{dt}/a$")
plt.xlabel("Time (s)")
plt.ylabel(r"Fraction of $F/m$ $\frac{du_y}{dt}/a$")
plt.title("Numerical time derivative of $u_y$")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Parameters
t = 0.02
Gamma = a**2*t**3    # Vortex strength (positive for right vortex, negative for left)
b = 0.01       # Distance between vortices (m)
U_inf = a*t     # Background flow velocity (m/s)
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
        r1_sq = (x_ - x1)**2 + (y_ - y1)**2
        r2_sq = (x_ - x2)**2 + (y_ - y2)**2

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
contour = ax.contourf(X, Y, Speed, levels=np.linspace(0, clip_speed, 100), cmap="Blues", extend="max")

# Streamlines
ax.streamplot(X, Y, U, V, density=1.5, color="w", linewidth=0.25, arrowsize=0.75)


# Vortex positions
ax.plot([x1, x2], [y1, y2], "wo", label="Vortices")

# Colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("Velocity Magnitude (m/s)")
cbar.ax.set_yticks([0, 1, 2, 3, 4])

# Labels and formatting
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Velocity at (0, s)
def velocity_at(x, y):
    r1_sq = (x - x1)**2 + (y - y1)**2
    r2_sq = (x - x2)**2 + (y - y2)**2
    u1 = (Gamma / (2 * np.pi)) * (y - y1) / r1_sq
    v1 = -(Gamma / (2 * np.pi)) * (x - x1) / r1_sq
    u2 = -(Gamma / (2 * np.pi)) * (y - y2) / r2_sq
    v2 = (Gamma / (2 * np.pi)) * (x - x2) / r2_sq
    return u1 + u2, v1 + v2 + U_inf

s = 0.02
u_s, v_s = velocity_at(0, s)

