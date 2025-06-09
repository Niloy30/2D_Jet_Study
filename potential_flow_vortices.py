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
a = 80  # m/s^2
b = 0.01  # m
k = 1.0  # adjustable
t = np.linspace(0.001, 0.05, 500)

# Compute Gamma*
Gamma_star = 3 / (1 + (a * t**2) / (4 * k) + (b**2) / (4 * k * a * t**2))

# Plot
plt.plot(t, Gamma_star, label=r"$\Gamma^*(t)$")
plt.axhline(3, color="gray", linestyle="--", label="Asymptote at 3")
plt.xlabel("Time (s)")
plt.ylabel(r"$\Gamma^*(t)$")
plt.title("Asymptotically Saturating $\Gamma^*(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%

import matplotlib.pyplot as plt
import numpy as np

# Parameters
Gamma = 1.0  # Vortex strength (positive for right vortex, negative for left)
b = 0.01  # Distance between vortices (m)
U_inf = 10  # Background flow velocity (m/s)

# Vortex positions
x1, y1 = -b / 2, 0  # Left vortex (negative circulation)
x2, y2 = +b / 2, 0  # Right vortex (positive circulation)


def velocity_at(x, y):
    """
    Returns the velocity vector (u, v) at point (x, y) due to:
    - Two point vortices
    - Uniform flow in +y direction
    """
    # Left vortex (-Gamma)
    r1_sq = (x - x1) ** 2 + (y - y1) ** 2
    u1 = (Gamma / (2 * np.pi)) * (y - y1) / r1_sq
    v1 = -(Gamma / (2 * np.pi)) * (x - x1) / r1_sq

    # Right vortex (+Gamma)
    r2_sq = (x - x2) ** 2 + (y - y2) ** 2
    u2 = -(Gamma / (2 * np.pi)) * (y - y2) / r2_sq
    v2 = (Gamma / (2 * np.pi)) * (x - x2) / r2_sq

    # Uniform flow
    u_uniform = 0
    v_uniform = U_inf

    # Total velocity
    u = u1 + u2 + u_uniform
    v = v1 + v2 + v_uniform
    return u, v


# Generate a grid for streamlines
x = np.linspace(-0.05, 0.05, 300)
y = np.linspace(-0.05, 0.05, 300)
X, Y = np.meshgrid(x, y)

# Compute velocity field
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u, v = velocity_at(X[i, j], Y[i, j])
        U[i, j] = u
        V[i, j] = v

# Compute streamfunction numerically
fig, ax = plt.subplots(figsize=(8, 6))
strm = ax.streamplot(X, Y, U, V, density=2, linewidth=1, arrowsize=1, arrowstyle="->")
ax.plot([x1, x2], [y1, y2], "ro", label="Vortices")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Streamlines of Two Vortices with Uniform +y Flow")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# Example: Get velocity at (0, s)
s = 0.02  # height above vortices
u_s, v_s = velocity_at(0, s)
print(f"Velocity at (0, {s:.3f} m): u = {u_s:.6f} m/s, v = {v_s:.6f} m/s")
