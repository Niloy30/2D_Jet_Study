import os
import shutil

RESULTS_BASE_DIR = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"
START = "20250516_144647"
END = "20250516_150846"

obstacle_data_src = os.path.join(RESULTS_BASE_DIR, START, "obstacle_data.txt")

# Ensure the obstacle data file exists
if not os.path.exists(obstacle_data_src):
    raise FileNotFoundError(f"Source file not found: {obstacle_data_src}")

# Get all subdirectories in RESULTS_BASE_DIR
all_dirs = [
    d
    for d in os.listdir(RESULTS_BASE_DIR)
    if os.path.isdir(os.path.join(RESULTS_BASE_DIR, d)) and START <= d <= END
]

# Sort and skip the first one (START folder)
matching_dirs = sorted(all_dirs)
dirs_to_copy = matching_dirs[1:]  # Skip the first folder

# Output and copy
print(
    "Copying obstacle_data.txt into the following directories (skipping the START folder):"
)
for directory in dirs_to_copy:
    target_path = os.path.join(RESULTS_BASE_DIR, directory, "obstacle_data.txt")

    try:
        # If destination file exists, remove it first
        if os.path.exists(target_path):
            os.remove(target_path)

        shutil.copy2(obstacle_data_src, target_path)
        print(f"- {directory} ✔")
    except PermissionError as e:
        print(f"- {directory} ❌ Skipped (PermissionError: {e})")
    except Exception as e:
        print(f"- {directory} ❌ Skipped (Unexpected error: {e})")
