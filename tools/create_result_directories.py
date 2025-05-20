import os

BASE_EXPERIMENT_DIR = r"C:\Users\niloy\Desktop\Experiments\05132025"
RESULTS_BASE_DIR = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"

# List all items in the base experiment directory
for experiment_number in os.listdir(BASE_EXPERIMENT_DIR):
    experiment_path = os.path.join(BASE_EXPERIMENT_DIR, experiment_number)

    # Check if it's a directory
    if os.path.isdir(experiment_path):
        results_path = os.path.join(RESULTS_BASE_DIR, experiment_number)

        # Only create the folder if it doesn't already exist
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            print(f"Created: {results_path}")
        else:
            print(f"Already exists: {results_path}")
