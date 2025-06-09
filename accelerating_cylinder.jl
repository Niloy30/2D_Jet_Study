using WaterLily
using CUDA
using Plots

cID = "2DCircle"

function circle(n,m;Re=250,U=1,mem=Array)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->sqrt(sum(abs2, x .- center)) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem)
end

sim = circle(3*2^6, 2^7; mem=CuArray)

WaterLily.logger(cID)
using Logging; disable_logging(Logging.Debug)

# Grid spacing in physical units (assuming unit square domain length L=1)
Lx, Ly = 1.0, 1.0
nx, ny = sim.nx, sim.ny
dx, dy = Lx / nx, Ly / ny

tmax = 10.0
dt = sim.Δt
nt = Int(floor(tmax / dt))

Gamma = zeros(Float64, nt)
time = zeros(Float64, nt)

for i in 1:nt
    step!(sim)
    omega_host = Array(sim.ω)  # copy from GPU to CPU
    Gamma[i] = sum(omega_host) * dx * dy
    time[i] = sim.t
end

plot(time, Gamma,
     xlabel="Time",
     ylabel="Vortex Circulation Γ",
     title="Vortex Circulation vs Time",
     lw=2,
     legend=false)


