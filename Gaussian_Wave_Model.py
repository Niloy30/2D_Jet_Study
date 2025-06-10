

# %%
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Scaling factor
C = 4

# Define color manually since 'colors' dictionary is undefined
CYAN_FILL = '#aeeeee'  # Light cyan color

def gaussian(x, mu, sig):
    return C / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def gaussian_der(x, mu, sig):
    return - (x - mu) / sig * gaussian(x, mu, sig)

def wave(x, mu, sig):
    factor = 2
    return (
        gaussian(x, mu, sig)
        - 0.5 * gaussian(x, mu - factor * sig, factor * sig)
        - 0.5 * gaussian(x, mu + factor * sig, factor * sig)
    )

def wave_der(x, mu, sig):
    return (
        gaussian_der(x, mu, sig)
        - 0.5 * gaussian_der(x, mu - 2 * sig, 2 * sig)
        - 0.5 * gaussian_der(x, mu + 2 * sig, 2 * sig)
    )

def multi_wave(x, m, sig, n_obstacles):
    superposed = wave(x, 0, sig)
    for factor in range(n_obstacles // 2):
        superposed += wave(x, (factor + 1) * m, sig) + wave(x, (-factor - 1) * m, sig)
    return superposed

def multi_wave_der(x, m, sig, n_obstacles):
    superposed = wave_der(x, 0, sig)
    for factor in range(n_obstacles // 2):
        superposed += wave_der(x, (factor + 1) * m, sig) + wave_der(x, (-factor - 1) * m, sig)
    return superposed

def gravitational_energy(x, m, sig, n_obstacles, rho=1000, g=9.81):
    fn = lambda x: multi_wave(np.array([x]), m, sig, n_obstacles) ** 2
    Ep = 0.5 * rho * g * quad(fn, -m/2, m/2)[0]
    return Ep

def surface_tension_energy(x, m, sig, n_obstacles, T=0.072):
    fn = lambda x_val: np.sqrt(1 + multi_wave_der(np.array([x_val]), m, sig, n_obstacles) ** 2) - 1
    Ep = T * quad(fn, -m/2, m/2)[0]
    return Ep

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

def plot_single_wave(ax, x, mu_wave, std, line_color, label):
    y = wave(x, mu_wave, std)
    ax.plot(x, y, color=line_color, lw=2.5, label=label)
    ax.plot(x, gaussian(x, mu_wave, std), ls='--', c='k', alpha=0.5)
    ax.plot(x, -0.5 * gaussian(x, mu_wave - 2 * std, 2 * std), ls='--', c='k', alpha=0.5)
    ax.plot(x, -0.5 * gaussian(x, mu_wave + 2 * std, 2 * std), ls='--', c='k', alpha=0.5)
    ax.fill_between(x, y1=-1, y2=y, color=CYAN_FILL, alpha=0.3)

def plot_multiple_wave(ax, x, m, std, n_obstacles, line_color, label, fill=True, ls='-'):
    superposed = multi_wave(x, m, std, n_obstacles)
    ax.plot(x, superposed, ls=ls, c=line_color, label=label)
    if fill:
        ax.fill_between(x, y1=-1, y2=superposed, color=CYAN_FILL, alpha=0.3)
    return superposed

# Plot multiple Gaussian wave
x = np.linspace(-100, 100, 1000)
sig_0 = 1
m = 32
n_obstacles = 11
superposed0 = multi_wave(x, m, sig_0, n_obstacles)
s0 = compute_steepness(x,superposed0)
Ep_0 = gravitational_energy(x, m, sig_0, n_obstacles, rho=1000, g=9.81)
sig = 1.01
superposed = multi_wave(x, m, sig, n_obstacles)
s = compute_steepness(x,superposed)
Ep = gravitational_energy(x, m, sig, n_obstacles, rho=1000, g=9.81) + surface_tension_energy(x, m, sig, n_obstacles, T=0.072)

print(Ep_0 - Ep)

plt.figure(figsize=(10, 5))
plt.ylim(-0.5,1.5)
plt.xlim(-15,15)
plt.plot(x, superposed0, color='cyan', lw=1, label=f's = {s0:.4f}', ls = "--")
plt.plot(x, superposed, color='blue', lw=1, label=f's = {s:.4f}', ls = "-")
plt.fill_between(x, -1, superposed, color='cyan', alpha=0.3)
plt.vlines(m/2,-0.5,1.5,'k',ls="--")
plt.vlines(-m/2,-0.5,1.5,'k',ls="--")
plt.title(f'M = {m}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

