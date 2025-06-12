# %%
import matplotlib.pyplot as plt
import numpy as np

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)

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
# cbar = plt.colorbar(contour, ax=ax)
# cbar.set_label("Velocity Magnitude (m/s)")
# cbar.ax.set_yticks([0, 1, 2, 3, 4, 5])

# Labels and formatting
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.ylim(-0.02,0.02)
plt.xlim(-0.02,0.02)
plt.tight_layout()
plt.savefig(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\01 - Publications\Periodicity effects on Jet formation of accelerated free surface\fig\potential_flow_solution.pdf",
    dpi=300,
    format="pdf",
    bbox_inches="tight",
)
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

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
U = 1.0        # Uniform flow speed
kappa = 0.00005  # Dipole strength
D = 0.01       # Distance between vortices
Gamma = 0.04  # Vortex circulation

# Vortex locations
xv1, yv1 = -D/2, 3*D/4  # Left vortex (clockwise)
xv2, yv2 = +D/2, 3*D/4  # Right vortex (counterclockwise)

# Grid
x = np.linspace(-0.03, 0.03, 300)
y = np.linspace(-0.03, 0.03, 300)
X, Y = np.meshgrid(x, y)

# Uniform flow (in +y direction)
psi_uni = U * X

# Dipole at origin
psi_dip = -kappa * X / (2 * np.pi * (X**2 + Y**2 + 1e-12))

# Vortices
psi_vort1 = (-Gamma / (2 * np.pi)) * np.log(np.sqrt((X-xv1)**2 + (Y-yv1)**2) + 1e-12)
psi_vort2 = (+Gamma / (2 * np.pi)) * np.log(np.sqrt((X-xv2)**2 + (Y-yv2)**2) + 1e-12)

# Total streamfunction
psi = psi_uni + psi_dip + psi_vort1 + psi_vort2

# Velocity components
u = np.gradient(psi, y, axis=0)  # d(psi)/dy
v = -np.gradient(psi, x, axis=1) # -d(psi)/dx

# Find a stagnation point numerically (near the origin)
def vel_mag(p):
    xi, yi = p
    # Uniform flow
    u_uni, v_uni = 0, U
    # Dipole
    r2 = xi**2 + yi**2 + 1e-12
    u_dip = -kappa / (2*np.pi) * ( (xi**2 - yi**2) / r2**2 )
    v_dip = -kappa / (2*np.pi) * ( 2*xi*yi / r2**2 )
    # Vortices
    r1sq = (xi-xv1)**2 + (yi-yv1)**2 + 1e-12
    r2sq = (xi-xv2)**2 + (yi-yv2)**2 + 1e-12
    u_vort1 = ( -Gamma / (2*np.pi) ) * ( yi-yv1 ) / r1sq
    v_vort1 = ( +Gamma / (2*np.pi) ) * ( xi-xv1 ) / r1sq
    u_vort2 = ( +Gamma / (2*np.pi) ) * ( yi-yv2 ) / r2sq
    v_vort2 = ( -Gamma / (2*np.pi) ) * ( xi-xv2 ) / r2sq
    u_tot = u_uni + u_dip + u_vort1 + u_vort2
    v_tot = v_uni + v_dip + v_vort1 + v_vort2
    return u_tot**2 + v_tot**2

# Initial guess near the origin
res = minimize(vel_mag, [0, 0])
x_stag, y_stag = res.x

# Find psi at stagnation point
psi_stag = U*x_stag - kappa*x_stag/(2*np.pi*(x_stag**2+y_stag**2+1e-12)) \
           + (-Gamma/(2*np.pi))*np.log(np.sqrt((x_stag-xv1)**2 + (y_stag-yv1)**2)+1e-12) \
           + (Gamma/(2*np.pi))*np.log(np.sqrt((x_stag-xv2)**2 + (y_stag-yv2)**2)+1e-12)

# Plot
plt.figure(figsize=(8,6))
plt.contourf(X, Y, np.sqrt(u**2 + v**2), levels=100, cmap='Blues')
plt.streamplot(X, Y, u, v, color='k', density=1.5, linewidth=0.7)
plt.plot([xv1, xv2], [yv1, yv2], 'ro', label='Vortices')
plt.plot(0, 0, 'ko', label='Dipole')
plt.plot(x_stag, y_stag, 'go', label='Stagnation point')
# Plot stagnation streamline
plt.contour(X, Y, psi, levels=[psi_stag], colors='lime', linewidths=2, linestyles='--', label='Stagnation streamline')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('Superposed Potential Flow: Uniform + Dipole + Two Vortices\nStagnation Streamline in Green')
plt.axis('equal')
plt.tight_layout()
plt.show()
