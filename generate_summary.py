import os
import shutil

# Base directories
base_dir = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"
summary_dir = os.path.join(base_dir, "Summary")

# Ensure the summary directory exists
os.makedirs(summary_dir, exist_ok=True)

# Loop through subdirectories
for subdir in os.listdir(base_dir):
    full_subdir_path = os.path.join(base_dir, subdir)

    # Check if it's a directory and starts with the date
    if os.path.isdir(full_subdir_path) and subdir.startswith("20250516"):
        result_pdf_path = os.path.join(full_subdir_path, "result.pdf")

        # Check if result.pdf exists
        if os.path.isfile(result_pdf_path):
            # Create destination path with directory name as filename
            dest_pdf_path = os.path.join(summary_dir, f"{subdir}.pdf")
            shutil.copyfile(result_pdf_path, dest_pdf_path)
            print(f"Copied {result_pdf_path} to {dest_pdf_path}")
        else:
            print(f"'result.pdf' not found in {full_subdir_path}")
