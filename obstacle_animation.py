# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(6, 2))  # Match this to second animation
ax.set_aspect('equal')
ax.set_xlim(-10, 10)  # Match x-range
ax.set_ylim(-2, 2)    # Match y-range
ax.axis('off')

# --- Static vertical lines ---
ax.plot([-3, -3], [-1.5, 1.5], color='black', linewidth=2)
ax.plot([3, 3], [-1.5, 1.5], color='black', linewidth=2)

# --- Easing Function ---
def ease_in_out(t):
    # cubic ease in/out
    return 3 * t**2 - 2 * t**3

# --- Animation Logic ---
def get_circle_positions(frame):
    if frame < 30:
        return [0]  # stage 1
    elif frame < 60:
        return [-2, 0, 2]  # stage 2
    elif frame < 90:
        # Smooth transition from [-2, 0, 2] to [-1, 0, 1]
        t = (frame - 60) / 30
        eased_t = ease_in_out(t)
        return [-(2 - eased_t), 0, (2 - eased_t)]
    elif frame < 120:
        return [-2, -1, 0, 1, 2]  # stage 3
    else:
        return [-2, -1, 0, 1, 2]  # final static

def update(frame):
    # Remove previous circles only (leave lines)
    for patch in ax.patches[:]:
        patch.remove()

    positions = get_circle_positions(frame)

    for x in positions:
        circle = Circle((x, 0), radius=0.3, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(circle)

    return ax.patches

# --- Animate with higher frame rate ---
ani = FuncAnimation(
    fig, update,
    frames=150,
    interval=50,
    blit=False
)

plt.show()

ani.save(r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\02 - Conferences\01 - CSME 2025\circle_animation.gif", fps=30, dpi=500)

# %%

fig, ax = plt.subplots(figsize=(6, 2))  # Match figure size explicitly
ax.set_aspect('equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-2, 2)
ax.axis('off')

# --- Static vertical symmetry lines ---
ax.plot([-3, -3], [-2, 2], color='black', linewidth=2)
ax.plot([3, 3], [-2, 2], color='black', linewidth=2)

# --- Constants ---
outer_positions = [-6, 6]  # Outside the bounding lines
radius_stages = [0.2, 0.4, 0.6]  # Increasing radii
pause_frames = 15
stage_frames = len(radius_stages) * pause_frames

# --- Radius Selector ---
def get_radius(frame):
    stage = frame // pause_frames
    if stage >= len(radius_stages):
        return radius_stages[-1]
    return radius_stages[stage]

# --- Animation Update ---
def update(frame):
    # Clear dynamic patches
    for patch in ax.patches[:]:
        patch.remove()

    # Radius for all obstacles
    radius = get_radius(frame)

    # Center obstacle (solid)
    center_circle = Circle((0, 0), radius=radius, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(center_circle)

    # Outer mirrored obstacles (dashed)
    for x in outer_positions:
        outer_circle = Circle((x, 0), radius=radius, edgecolor='black',
                              facecolor='none', linewidth=2, linestyle='dashed')
        ax.add_patch(outer_circle)

    return ax.patches

# --- Animate ---
total_frames = stage_frames + pause_frames
ani = FuncAnimation(
    fig, update,
    frames=total_frames,
    interval=200,
    blit=False
)

plt.show()

# --- Save ---
ani.save(
    r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\02 - Conferences\01 - CSME 2025\mirrored_obstacles_v2.gif",
    fps=30,
    dpi=500
)


# %%
