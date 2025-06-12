# %%
import matplotlib.pyplot as plt
import numpy as np

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
# Scaling factor
C = 4


def gaussian(x, mu, sig):
    return C / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def wave(x, mu, sig):
    factor = 2
    return (
        gaussian(x, mu, sig)
        - 0.5 * gaussian(x, mu - factor * sig, factor * sig)
        - 0.5 * gaussian(x, mu + factor * sig, factor * sig)
    )


def gravitational_energy(x, eta, rho=1000, g=80):
    Ep = 0.5 * rho * g * np.trapezoid(eta**2, x)
    return Ep


def energy_surface_tension(x, eta, gamma=0.072):
    deta_dx = np.gradient(eta, x)
    integrand = (1 + deta_dx**2) ** 0.5 - 1
    Est = gamma * np.trapezoid(integrand, x)
    return Est


def compute_steepness(x, eta):
    x_max = x[np.argmin(np.abs(x))]
    eta_max = eta[np.argmin(np.abs(x))]

    eta_min_0 = np.min(eta)
    aux = eta - eta_min_0 - 0.01 * (eta_max - eta_min_0)
    idx = np.where((aux < 0) & (x > x_max))[0][0]
    x_min = x[idx]
    eta_min_10 = eta[idx]

    s = (eta_max - eta_min_10) / (2 * (x_min - x_max))
    return s


# Domain and range of sigma
x = np.linspace(-15, 15, 1000)
sigmas = np.linspace(0.75, 2, 100)

Eg_list = []
Est_list = []
s_list = []

for sig in sigmas:
    eta = wave(x, 0, sig)

    x_m = x / 1000  # mm → m
    eta_m = eta / 1000  # mm → m

    Eg = gravitational_energy(x_m, eta_m)
    Est = energy_surface_tension(x_m, eta_m)
    s = compute_steepness(x, eta)

    Eg_list.append(Eg)
    Est_list.append(Est)
    s_list.append(s)

# Primary plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Eg and Est vs s on the left y-axis
ax1.plot(
    s_list,
    np.array(Eg_list) * 1000,
    label=r"Gravitational Energy ($E_{\mathbf{g}}$)",
    color="black",
)
ax1.plot(
    s_list,
    np.array(Est_list) * 1000,
    label=r"Surface Tension Energy ($E_{\mathbf{st}}$)",
    color="red",
)
ax1.set_xlabel("Steepness (s)")
ax1.set_ylabel("Energy (mJ)")
ax1.tick_params(axis="y")
ax1.grid(False)

# Secondary axis for Est/Eg ratio as a fraction
ax2 = ax1.twinx()
energy_ratio = [est / eg for est, eg in zip(Est_list, Eg_list)]
ax2.plot(
    s_list,
    energy_ratio,
    label=r"$E_{\mathbf{st}}$ / $E_{\mathbf{g}}$",
    color="blue",
    linestyle="--",
)
ax2.set_ylabel(
    r"Surface Tension / Gravity Energy ($E_{\mathbf{st}}$ / $E_{\mathbf{g}}$)",
    color="blue",
)
ax2.tick_params(axis="y", labelcolor="blue")

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

plt.savefig(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\01 - Publications\Periodicity effects on Jet formation of accelerated free surface\fig\E_ratio.pdf",
    dpi=300,
    format="pdf",
    bbox_inches="tight",
)
plt.tight_layout()
plt.show()

# %%
x = np.linspace(-10, 10, 1000)

mu = 0
sig = 1

Gaussian_1 = gaussian(x, 0, 1)

factor = 2
Gaussian_2 = -0.5 * gaussian(x, mu + factor * sig, factor * sig)
Gaussian_3 = -0.5 * gaussian(x, mu - factor * sig, factor * sig)
wave_1 = Gaussian_1 + Gaussian_2 + Gaussian_3
wave_2 = wave(x, 0, 1)
plt.plot(x, Gaussian_1, "--", color="k", linewidth="1")
plt.plot(x, Gaussian_2, "--", color="k", linewidth="1")
plt.plot(x, Gaussian_3, "--", color="k", linewidth="1")

plt.plot(x, wave_1, "-", color="b", linewidth="2")
plt.ylabel(r"$\eta$ [mm]")
plt.xlabel(r"$x$ [mm]")
plt.savefig(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\01 - Publications\Periodicity effects on Jet formation of accelerated free surface\fig\Gaussian_Superposition.pdf",
    dpi=300,
    format="pdf",
    bbox_inches="tight",
)

plt.tight_layout()
