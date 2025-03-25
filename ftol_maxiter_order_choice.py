import numpy as np
import pandas as pd
from scipy.signal import lfilter, unit_impulse
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from myfunctions_tf import ComplexFilterOptimizer
from tqdm import tqdm

def main():
    orders = [60 ,80]
    ftols = [1e-1, 1e-2, 1e-5]
    maxiters = [100, 300, 500]

    data_path_base = "/home/stud/casperc/bhome/Project3_DL_sigpros/generated_data/2025_03_18-01_17_48-standard_amplitude_npz"
    TE_lw_folders = os.listdir(data_path_base)
    results = [] # Store results
    total_iterations = len(orders) * len(ftols) * len(maxiters) * len(TE_lw_folders)
    with tqdm(total=total_iterations) as pbar:
        for order in orders:
            for ftol in ftols:
                for maxiter in maxiters:
                    error_signals = []
                    times = []
                    for TE_lw in TE_lw_folders:
                        path = os.path.join(data_path_base, TE_lw)
                        for file in os.listdir(path)[:16]:
                            if file.endswith(".npz"):
                                path_to_file = os.path.join(path, file)
                                data = np.load(path_to_file)
                                input_signal = data["augmented_ifft"]
                                complex_optimizer = ComplexFilterOptimizer(input_signal, order, options={'ftol':ftol, 'maxiter':maxiter})
                                start = time.perf_counter()
                                b_opt, a_opt = complex_optimizer.optimize()
                                end = time.perf_counter()
                                estimated_signal = lfilter(b_opt, a_opt, unit_impulse(len(input_signal)))
                                error_signal = np.linalg.norm(input_signal - estimated_signal) ** 2

                                error_signals.append(error_signal)
                                times.append(end-start)
                    # Calculate mean and std of error signals and times
                    error_mean = np.mean(error_signals)
                    error_std = np.std(error_signals)
                    time_mean = np.mean(times)
                    time_std = np.std(times)
                    results.append({
                        'order': order,
                        'ftol': ftol,
                        'maxiter': maxiter,
                        'error_mean': error_mean,
                        'error_std': error_std,
                        'time_mean': time_mean,
                        'time_std': time_std
                    })
                    pbar.update(1)
        # Convert results to a Pandas DataFrame for easier analysis
        results_df = pd.DataFrame(results)

        # Save the results to a CSV file
        output_file = "/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/optimization_results_3.csv"
        results_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    main()