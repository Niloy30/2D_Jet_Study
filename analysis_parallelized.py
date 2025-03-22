import os
from functools import partial
from multiprocessing import Manager, Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from calibration import get_scaling

# Define global constants
calibration_grid = (
    r"E:\FDL\2D Jet Study Experiments\2025-03-17\192.168.0.10_C001H001S0004.bmp"
)
conversion_factor = get_scaling(calibration_grid)  # mm/pixel
results_dir = r"C:\Users\niloy\Google Drive\School Stuff\M.SC Mechanical Engineering\01 - Fluid Dynamics Lab\03 - PDA\01 - 2D Surface Perturbations\Results"


# Function to process a single experiment
def process_experiment(experiment_number, conversion_factor, progress_queue):
    try:
        results_path = os.path.join(results_dir, experiment_number)
        free_surface_data = os.path.join(results_path, "free_surface_data.npy")

        # Load data
        free_surface_all = np.load(free_surface_data)
        X = free_surface_all[0, :, :]
        Y = free_surface_all[1, :, :]
        T = np.arange(0, X.shape[1], 1)

        # Process data
        Y_average = np.mean(Y, axis=0)
        Y_average_m = (Y_average - np.min(Y_average)) * conversion_factor / 1000
        T_s = T / 7000

        # Fit quadratic curve
        coeffs = np.polyfit(T_s, Y_average_m, 2)
        a = coeffs[0] * 2
        g = 9.81
        Lambda = 2 + g / a

        # Update progress
        progress_queue.put(1)

        return experiment_number, Lambda

    except Exception as e:
        print(f"Error processing {experiment_number}: {e}")
        return experiment_number, None


# Main function to parallelize
def main():
    # Get list of experiments from the results directory
    experiment_numbers = [
        name
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    ]

    # Use multiprocessing with a progress bar
    manager = Manager()
    progress_queue = manager.Queue()

    # Use `partial` to pass extra arguments
    process_with_args = partial(
        process_experiment,
        conversion_factor=conversion_factor,
        progress_queue=progress_queue,
    )

    pool = Pool(cpu_count())

    results = []
    with tqdm(total=len(experiment_numbers), desc="Processing Experiments") as pbar:
        for result in pool.imap_unordered(process_with_args, experiment_numbers):
            results.append(result)
            pbar.update(progress_queue.get())

    pool.close()
    pool.join()

    # Save results to Excel
    df = pd.DataFrame(results, columns=["Experiment Number", "Lambda"])
    output_path = os.path.join(results_dir, "Lambda_Results.xlsx")
    df.to_excel(output_path, engine="openpyxl", index=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
