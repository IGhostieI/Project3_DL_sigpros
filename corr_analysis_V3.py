import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tick
from scipy.interpolate import interp1d
import re
from myfunctions_tf import BPF_ppm, load_from_directory, multi_channel_cnn, ComplexFilterOptimizer, input_ouput_key_sorting, model_input_output_prep, load_mrui_txt_data, postprocess_predicted_data, read_LCModel_table
import pandas as pd
import pickle
from scipy.stats import shapiro, spearmanr, kendalltau, pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

def resample_input_data(metadata, FID, ppm):
    # Extract sampling interval and calculate ppm range
    sampling_interval = float(metadata["SamplingInterval"]) * 1E-3
    sampling_frequency = 1 / sampling_interval
    ppm_band_width = sampling_frequency * 1E6 / float(metadata["TransmitterFrequency"])
    
    # Calculate the original ppm range
    data_ppm = np.linspace(4.7 - ppm_band_width / 2, 4.7 + ppm_band_width / 2, int(metadata["PointsInDataset"]))
    
    # Check if the data's ppm range is sufficient
    if data_ppm.min()>ppm.min() and data_ppm.max()<ppm.max():
        raise ValueError(f"Data's ppm range is contained in the desired ppm range. Data ppm: {data_ppm[0]} to {data_ppm[-1]}, Desired ppm: {ppm[0]} to {ppm[-1]} and cannot be resampled.")
    
    # Extract the desired ppm range
    ppm_where = np.where((data_ppm >= ppm.min()) & (data_ppm <= ppm.max()))[0]
    FID_extracted = FID[ppm_where]
    data_ppm_extracted = data_ppm[ppm_where]
    
    # Interpolate to match the desired ppm length
    if len(data_ppm_extracted) != len(ppm):
        target_ppm = np.linspace(ppm.min(), ppm.max(), len(ppm))
        interpolator = interp1d(data_ppm_extracted, FID_extracted, kind="linear", fill_value="extrapolate")
        FID_interpolated = interpolator(target_ppm)
    else:
        FID_interpolated = FID_extracted
    
    return FID_interpolated

def process_and_save(
    path, 
    model_configuration, 
    ppm, 
    passband_GABA=None, 
    passband_full=None, 
    margin=0, 
    ftype="butter", 
    pass_band_gain=1.0, 
    model_weights_path=None
):
    # Load data
    FID, freq, metadata = load_mrui_txt_data(path)
    norm_freq_fft = np.fft.fftshift(np.fft.fft(FID)) / np.max(np.fft.fftshift(np.fft.fft(FID)))
    norm_FID = np.fft.ifft(norm_freq_fft)
    norm_FID = resample_input_data(metadata, norm_FID, ppm)
    norm_freq_fft = np.fft.fft(norm_FID)
    if len(norm_FID) != 2048:
        print(f"Warning: Dataset has {len(norm_FID)} points instead of 2048.")
        
    

    # Optimize filter coefficients
    if "_ab" in model_configuration or model_configuration == "ab":
        optim = ComplexFilterOptimizer(input_signal=norm_FID, order=60, options={'method': 'SLSQP', 'ftol': 1e-3, 'maxiter': 50})
        b_opt, a_opt = optim.optimize()
        a_opt = np.pad(a_opt, (0, 2048 - len(a_opt)))
        b_opt = np.pad(b_opt, (0, 2048 - len(b_opt)))

    # Apply band-pass filters if passbands are provided
    if "gaba" in model_configuration:
        filtered_GABA, _, _ = BPF_ppm(fft_signal=norm_freq_fft, ppm_range=ppm, ppm_pass=passband_GABA, margin=margin, ftype=ftype, gain_passband=pass_band_gain)

    if "full" in model_configuration:
        filtered_full, _, _ = BPF_ppm(fft_signal=norm_freq_fft, ppm_range=ppm, ppm_pass=passband_full, margin=margin, ftype=ftype, gain_passband=pass_band_gain)

    # Prepare input for the model
    if model_configuration == "FID":
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag]).T, axis=0)
    elif model_configuration == "FID_gaba" and filtered_GABA is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, filtered_GABA.real, filtered_GABA.imag]).T, axis=0)
    elif model_configuration == "FID_full_filt" and filtered_full is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, filtered_full.real, filtered_full.imag]).T, axis=0)
    elif model_configuration == "FID_ab_gaba" and filtered_GABA is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real, b_opt.imag, filtered_GABA.real, filtered_GABA.imag]).T, axis=0)
    elif model_configuration == "FID_ab_full_filt" and filtered_full is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real, b_opt.imag, filtered_full.real, filtered_full.imag]).T, axis=0)
    elif model_configuration == "FID_ab":
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real, b_opt.imag]).T, axis=0)
    elif model_configuration == "ab":
        in_vivo_input = np.expand_dims(np.array([a_opt.real, a_opt.imag, b_opt.real, b_opt.imag]).T, axis=0)
    else:
        raise ValueError(f"Unknown or incompatible model configuration: {model_configuration}")

    # Load model
    input_keys, output_keys = input_ouput_key_sorting(model_weights_path)
    model = multi_channel_cnn(input_shape=(2048, 2 * len(input_keys)), output_shape=(2048, 2 * len(output_keys)), num_blocks=20, filter1=32, kernel_size=16, print_summary=False)
    model.build(input_shape=(None, 2048, 2 * len(input_keys)))
    model.load_weights(model_weights_path)

    # Predict and post-process output
    in_vivo_output = np.squeeze(model.predict(in_vivo_input), axis=0)
    in_vivo_output_dict = postprocess_predicted_data(in_vivo_output, output_keys)

    # Return or save the processed data
    return in_vivo_output_dict

# Format p-values with scientific notation for small values
def format_p_value(p):
    if p < 0.001:
        return f"{p:.2e}"  # Scientific notation for very small p-values
    else:
        return f"{p:.3f}"  # Standard format for larger p-values

# File paths
raw_data_base = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/correlation_analysis"


# Configuration
ppm = np.linspace(-3.121, 12.528, 2048)
passband_GABA = np.array([2.1, 2.7])  # ppm
passband_full = np.array([1.6, 4.2])  # ppm
margin = 1
ftype = "butter"
pass_band_gain = 2

DO_PREDICTIONS = False  # Set to False to load from saved pickle files
slurm = True  # Set to True if running on SLURM cluster'
global_normalization = False  # Set to True if you want to normalize the data globally

# Weight dictionary
weight_dict = {
    "ab":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_05_12-16_18_06_['a', 'b']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",

    "FID":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_23-21_37_38_['augmented_ifft']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_ab":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_23-21_37_44_['augmented_ifft', 'a', 'b']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_ab_full_filt":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_24-09_56_20_['augmented_ifft', 'a', 'b', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_full_filt":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_24-10_52_49_['augmented_ifft', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
}
loaded_metabolites = ["NAA", "Glu+Gln", "PCh+GPC", "Glu", "Gln", "NAA+NAAG"]
target_metabolites = ["NAA", "PCh+GPC", "Gln"]
config_order = ["FID", "FID_ab", "FID_full_filt", "FID_ab_full_filt", "ab"]
config_nice_names = {"FID": "FID", "FID_ab": "FID + (a,b)", 
                   "FID_full_filt": "FID + full filter", 
                   "FID_ab_full_filt": "FID + (a,b) + full filter", 
                   "ab": "(a,b)"}

if not slurm:
    available_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(available_devices[6], 'GPU')
    print(f"visible devices: {tf.config.get_visible_devices()}")
    available_devices = tf.config.list_physical_devices('GPU')
    print(f"visible devices: {tf.config.get_visible_devices()}")
    
if DO_PREDICTIONS:
    for config, weight_path in weight_dict.items():
        # Set up input and output keys
        input_keys, output_keys = input_ouput_key_sorting(weight_path)
        model = multi_channel_cnn(
            input_shape=(2048, 2 * len(input_keys)),
            output_shape=(2048, 2 * len(output_keys)),
            num_blocks=20,
            filter1=32,
            kernel_size=16,
            print_summary=False
        )
        model.build(input_shape=(None, 2048, 2 * len(input_keys)))
        model.load_weights(weight_path)

        # Initialize data dictionary
        data_dict = {}
        for folder in os.listdir(raw_data_base):
            folder_path = os.path.join(raw_data_base, folder)
            if not os.path.isdir(folder_path):
                print(f"Skipping {folder_path} as it is not a directory.")
                continue
            data_dict[folder] = {"lcm": {}, "pred": {}}

            for file in os.listdir(folder_path):
                
                if file.endswith(".txt") and "S" in file and "HLSVD" not in file:
                    file_path = os.path.join(folder_path, file)
                    table_file = os.path.join(folder_path, file.replace(".txt", ".table"))
                    if os.path.exists(table_file) and os.path.exists(file_path):
                        LCModel_metabolites_dict = read_LCModel_table(table_file, loaded_metabolites)
                        if all(metabolite["%SD"] <= 50 for metabolite in LCModel_metabolites_dict.values()):
                            in_vivo_output_dict = process_and_save(
                                file_path,
                                model_configuration=config,
                                ppm=ppm,
                                passband_GABA=passband_GABA,
                                passband_full=passband_full,
                                margin=margin,
                                ftype=ftype,
                                pass_band_gain=pass_band_gain,
                                model_weights_path=weight_path
                            )
                            subject_id = file.split(".")[0]
                            data_dict[folder]["lcm"][subject_id] = LCModel_metabolites_dict
                            data_dict[folder]["pred"][subject_id] = in_vivo_output_dict

            # Remove empty folders
            if not data_dict[folder]["lcm"] and not data_dict[folder]["pred"]:
                print(f"Removing empty folder: {folder}")
                del data_dict[folder]
        
        # Save dictionary
        with open(f"concentrations/{config}.pickle", "wb") as f:
            pickle.dump(data_dict, f)

os.makedirs("correlations", exist_ok=True)
# Load saved pickle files

ppm_crop = np.where((ppm >= 0.2) & (ppm <= 4.2))[0]
            
# Before starting any plots, first load all data and find global max values
all_configs_data = {}
global_max_concentration_pred = float('-inf')
global_max_concentration_lcm = float('-inf')
font_size_offset = 9
# First pass: load all data and find global maximums
print("Finding global maximum values across all configurations...")
for config in weight_dict.keys():
    # Load saved pickle file
    with open(f"concentrations/{config}.pickle", "rb") as f:
        data_dict = pickle.load(f)
        all_configs_data[config] = data_dict
    if global_normalization:
        # Find max values for this configuration
        for folder, data in data_dict.items():
            for subject, lcm_data in data["lcm"].items():
                pred_data = data["pred"].get(subject, {})
                
                # Update global max concentration for lcm
                for met, values in lcm_data.items():
                    global_max_concentration_lcm = max(global_max_concentration_lcm, values["/Cr+PCr"])
                    
                    # Update global max concentration for pred
                    if met == "PCh+GPC":
                        global_max_concentration_pred = max(global_max_concentration_pred, 
                            np.sum(pred_data["PCho"][ppm_crop].real + pred_data["GPC"][ppm_crop].real) / 
                            np.sum(pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real))
                    elif met == "NAA+NAAG":
                        global_max_concentration_pred = max(global_max_concentration_pred, 
                            np.sum(pred_data["NAA"][ppm_crop].real + pred_data["NAAG"][ppm_crop].real) / 
                            np.sum(pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real))
                    elif met == "Cr+PCr":
                        global_max_concentration_pred = max(global_max_concentration_pred, 
                            np.sum(pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real) / 
                            np.sum(pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real))
                    elif met == "Glu+Gln":
                        global_max_concentration_pred = max(global_max_concentration_pred, 
                            np.sum(pred_data["Glu"][ppm_crop].real + pred_data["Gln"][ppm_crop].real) / 
                            np.sum(pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real))
                    else:
                        if met in pred_data:
                            global_max_concentration_pred = max(global_max_concentration_pred, 
                                np.sum(pred_data[met][ppm_crop].real) / 
                                np.sum(pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real))
    else:
        global_max_concentration_lcm = 1
        global_max_concentration_pred = 1
    print(f"Global max LCM value: {global_max_concentration_lcm:.4f}")
    print(f"Global max prediction value: {global_max_concentration_pred:.4f}")

# Second pass: process each configuration using the global maximums
for config, data_dict in all_configs_data.items():
    print(f"Processing configuration: {config}")
    
    # Normalize using global maximums
    normalized_data = {}
    
    for folder, data in data_dict.items():
        normalized_data[folder] = {"lcm": {}, "pred": {}}
        for subject, lcm_data in data["lcm"].items():
            pred_data = data["pred"].get(subject, {})
            normalized_data[folder]["lcm"][subject] = {}
            normalized_data[folder]["pred"][subject] = {}

            for met in lcm_data.keys():
                # Normalize LCM data using global maximum
                normalized_data[folder]["lcm"][subject][met] = {
                    "/Cr+PCr": lcm_data[met]["/Cr+PCr"] / global_max_concentration_lcm
                }

                # Normalize predicted data using global maximum
                if met == "Glu+Gln":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["Glu"][ppm_crop].real + pred_data["Gln"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real) / global_max_concentration_pred
                elif met == "Cr+PCr":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred
                elif met == "NAA+NAAG":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["NAA"][ppm_crop].real + pred_data["NAAG"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred
                elif met == "PCh+GPC":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["PCho"][ppm_crop].real + pred_data["GPC"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred
                else:
                    if met in pred_data.keys():
                        normalized_data[folder]["pred"][subject][met] = np.sum(
                            pred_data[met][ppm_crop].real)/np.sum(
                            pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred

    # Scatter plot
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{config}_scatter", exist_ok=True)
    
    # Initialize lists to collect results for all metabolites
    all_results = []
    all_formatted_results = []
    
    # Create figure with larger font size
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size':23+font_size_offset,
                         'lines.linewidth': 6
                         })
    SCATTER_SIZE = 500
    SCATTER_OPACITY = 0.6
    plt.subplots_adjust(hspace=0.15, wspace=0.2, top=0.88)  # Make room at the top
    
    for i, met in enumerate(target_metabolites, start=1):
        lcm_all = []
        pred_all = []

        for folder, data in normalized_data.items():
            lcm_values = [v[met]["/Cr+PCr"] for v in data["lcm"].values() if met in v]
            pred_values = [v[met] for v in data["pred"].values() if met in v]

            # Append to global lists
            lcm_all.extend(lcm_values)
            pred_all.extend(pred_values)

        # Convert to NumPy arrays
        lcm_all = np.array(lcm_all)
        pred_all = np.array(pred_all)

        # Perform Shapiro-Wilk test for normality
        lcm_shapiro_stat, lcm_shapiro_p = shapiro(lcm_all)
        pred_shapiro_stat, pred_shapiro_p = shapiro(pred_all)
        
        # Handle potential constant array warnings
        if np.var(lcm_all) > 0 and np.var(pred_all) > 0:
            spearman_corr, spearman_p = spearmanr(lcm_all, pred_all)
            tau, kendall_p = kendalltau(lcm_all, pred_all)
            pearson_corr, pearson_p = pearsonr(lcm_all, pred_all)
        else:
            spearman_corr = spearman_p = tau = kendall_p = pearson_corr = pearson_p = float('nan')
            print(f"Warning: Constant input detected for {met}. Correlation coefficients set to NaN.")
        
        # Add result for this metabolite to the list
        all_results.append({
            "Metabolite": met,
            "LCM Shapiro Stat": lcm_shapiro_stat,
            "LCM Shapiro P": lcm_shapiro_p,
            "Pred Shapiro Stat": pred_shapiro_stat, 
            "Pred Shapiro P": pred_shapiro_p,
            "Kendall Tau": tau,
            "Kendall P": kendall_p,
            "Spearman Rho": spearman_corr,
            "Spearman P": spearman_p,
            "Pearson R": pearson_corr,
            "Pearson P": pearson_p
        })
        
        # Add formatted result for this metabolite
        all_formatted_results.append({
            "Metabolite": met,
            "LCM Shapiro": f"{lcm_shapiro_stat:.3f} ({format_p_value(lcm_shapiro_p)})",
            "Pred Shapiro": f"{pred_shapiro_stat:.3f} ({format_p_value(pred_shapiro_p)})",
            "Kendall τ": f"{tau:.3f} ({format_p_value(kendall_p)})",
            "Spearman ρ": f"{spearman_corr:.3f} ({format_p_value(spearman_p)})",
            "Pearson r": f"{pearson_corr:.3f} ({format_p_value(pearson_p)})"
        })
        
        # Scatter plot
        plt.subplot(2, 2, i)

        # Scatter data points with increased size and opacity
        for folder, data in normalized_data.items():
            lcm_values = [v[met]["/Cr+PCr"] for v in data["lcm"].values() if met in v]
            pred_values = [v[met] for v in data["pred"].values() if met in v]
            plt.scatter(lcm_values, pred_values, label=folder, s=SCATTER_SIZE, alpha=SCATTER_OPACITY)  # Increased size and opacity

        # Add Pearson line (linear regression)
        if len(lcm_all) > 1:  # Ensure we have enough points
            """ slope, intercept = np.polyfit(lcm_all, pred_all, 1)
            x_line = np.linspace(0, 3, 300)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r-', label='Pearson')  # Simplified label """
            
            # Add LOWESS curve for Spearman visualization
            try:
                # Generate LOWESS fit with frac=0.6 (adjust as needed)
                lowess_result = lowess(pred_all, lcm_all, frac=0.6)
                plt.plot(lowess_result[:, 0], lowess_result[:, 1], color="#8B0000", label='Spearman', zorder=10)  # Simplified label

            except Exception as e:
                print(f"Could not generate LOWESS curve for {met}: {e}")
            try:
                # Generate another LOWESS fit with different parameters to represent Kendall's Tau
                # Using a higher frac value creates a smoother curve
                kendall_lowess_result = lowess(pred_all, lcm_all, frac=0.75)  # Higher frac for more smoothing
                plt.plot(kendall_lowess_result[:, 0], kendall_lowess_result[:, 1], color="#FF8C00",  label='Kendall', linestyle="--",zorder=10)  # Blue dashed line
            except Exception as e:
                print(f"Could not generate Kendall visualization for {met}: {e}")
        # Create a single annotation with all statistics
        stats_text = (f'$\\mathbf{{{met}}}$\n'
                    f'Spearman ρ = {spearman_corr:.2f} (p={format_p_value(spearman_p)})\n'
                    f'Kendall τ = {tau:.2f} (p={format_p_value(kendall_p)})')
        # Determine where data is clustered
        x_mean = np.mean(lcm_all)
        y_mean = np.mean(pred_all)
        if global_normalization:
            # Adjust limits based on global max values
            x_lim = (-0.05, 1.05)
            y_lim = (-0.05, 1.05)
        else:
            x_lim = (-0.05, 3.0)
            y_lim = (-0.05, 3.0)
        x_mid = (x_lim[0] + x_lim[1]) / 2
        y_mid = (y_lim[0] + y_lim[1]) / 2
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        # Choose position based on data distribution
        # Determine position based just on top/bottom data distribution
        if y_mean > y_mid:
            # Data mostly in upper half, place annotations in bottom
            x_pos, y_pos = 0.05, 0.05
        else:
            # Data mostly in lower half, place annotations in top
            x_pos, y_pos = 0.05, 0.95

        # Apply positioning with the consolidated text
        plt.annotate(stats_text, xy=(x_pos, y_pos), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.95, ec="gray"),
                    ha="left", va='top' if y_pos > 0.5 else 'bottom', fontsize=20+font_size_offset)
        # Improved axis labels with bold fonts
        if i%2:
            ax = plt.gca()  # Get current axis
            ax.text(-0.15, 0.5, 'DL', fontsize=25+font_size_offset, fontweight='bold',
                   transform=ax.transAxes, rotation=90, ha='center', va='center')
        if i>2:
            ax = plt.gca()  # Get current axis
            ax.text(0.5, -0.1, 'LCModel', fontsize=25+font_size_offset, fontweight='bold',
                   transform=ax.transAxes, ha='center', va='center')
        
        # Add minor ticks
        ax = plt.gca()  # Get current axis
        ax.xaxis.set_major_locator(tick.MultipleLocator(0.5))  # Major ticks every 0.5 on x-axis
        ax.yaxis.set_major_locator(tick.MultipleLocator(0.5))  # Major ticks every 0.5 on y-axis
        ax.xaxis.set_minor_locator(tick.MultipleLocator(0.1))  # Minor ticks every 0.1
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))  # Minor ticks every 0.1

        # Make the minor ticks more visible but less prominent than major ticks
        ax.tick_params(which='major', length=14, width=4, labelsize=18+font_size_offset)
        ax.tick_params(which='minor', length=7, width=3, color='gray')

        # Add grid for better readability
        ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

    # Get legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels and label not in ['Pearson', 'LOWESS']:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Add the Pearson and LOWESS lines to the legend if they exist
    for handle, label in zip(handles, labels):
        if label in ['Pearson', 'Spearman', 'Kendall'] and label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Create a grid-style legend at the top of the figure
    plt.figlegend(unique_handles, unique_labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.98),  # Position at the top center
                ncol=4,  # Adjust the number of columns as needed
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=21+font_size_offset)
    
    # Add a title showing which configuration this is
    plt.suptitle(f"Configuration: {config}", fontsize=27+font_size_offset, y=0.995)
    
    # Save with tight layout
    plt.savefig(f"{output_dir}/{config}_scatter.png", bbox_inches='tight')
    plt.close()
    
    print(f"Total number of subjects for {config}: {len(lcm_all)}")

    # Create DataFrames from the collected results
    results_df = pd.DataFrame(all_results)
    formatted_df = pd.DataFrame(all_formatted_results)
    
    # Save the raw data for further analysis
    results_df.to_csv(f"correlations/{config}_correlation_results_raw.csv", index=False)
    
    # Save the formatted results for display
    formatted_df.to_csv(f"correlations/{config}_correlation_results.csv", index=False)

# Before creating the combined figure, determine the range for each metabolite
metabolite_ranges = {}
print("Determining consistent axis ranges for each metabolite...")

for met in target_metabolites:
    all_lcm_values = []
    all_pred_values = []
    
    # Collect all values across all configurations
    for config in config_order:
        if config not in all_configs_data:
            continue
            
        data_dict = all_configs_data[config]
        for folder, data in data_dict.items():
            for subject, lcm_data in data["lcm"].items():
                if met in lcm_data:
                    # Normalize in the same way as the plotting code
                    lcm_value = lcm_data[met]["/Cr+PCr"] / global_max_concentration_lcm
                    all_lcm_values.append(lcm_value)
                    
                    # Get corresponding pred value
                    pred_data = data["pred"].get(subject, {})
                    if met == "Glu+Gln":
                        pred_value = np.sum(pred_data["Glu"][ppm_crop].real + pred_data["Gln"][ppm_crop].real)/np.sum(
                            pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real) / global_max_concentration_pred
                    elif met == "NAA+NAAG":
                        pred_value = np.sum(pred_data["NAA"][ppm_crop].real + pred_data["NAAG"][ppm_crop].real)/np.sum(
                            pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real) / global_max_concentration_pred
                    elif met == "PCh+GPC":
                        pred_value = np.sum(pred_data["PCho"][ppm_crop].real + pred_data["GPC"][ppm_crop].real)/np.sum(
                            pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real) / global_max_concentration_pred
                    elif met in pred_data:
                        pred_value = np.sum(pred_data[met][ppm_crop].real)/np.sum(
                            pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real) / global_max_concentration_pred
                    else:
                        continue
                        
                    all_pred_values.append(pred_value)
    
    # Calculate range with some padding
    if all_lcm_values and all_pred_values:
        x_min, x_max = min(all_lcm_values), max(all_lcm_values)
        y_min, y_max = min(all_pred_values), max(all_pred_values)
        
        # Add 5% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_min = max(0, x_min - 0.05 * x_range)  # Don't go below 0
        x_max = x_max + 0.05 * x_range
        y_min = max(0, y_min - 0.05 * y_range)  # Don't go below 0
        y_max = y_max + 0.05 * y_range
        
        metabolite_ranges[met] = {
            'x_lim': (x_min, x_max),
            'y_lim': (y_min, y_max)
        }
        print(f"Range for {met}: x={metabolite_ranges[met]['x_lim']}, y={metabolite_ranges[met]['y_lim']}")
    else:
        # Fallback to default
        metabolite_ranges[met] = {
            'x_lim': (-0.05, 3.0),
            'y_lim': (-0.05, 3.0)
        }
        print(f"No data found for {met}, using default range")

# After the individual configuration plots, create a combined figure
print("Creating combined figure with all configurations...")

# Set up figure and axes
fig, axs = plt.subplots(5, 3, figsize=(30, 50))
plt.rcParams.update({'font.size': 18+font_size_offset, 'lines.linewidth': 4})

# Add metabolite titles to the first row using set_title
for col_idx, met in enumerate(target_metabolites):
    axs[0, col_idx].set_title(met, fontsize=28+font_size_offset, fontweight='bold', pad=15)

# First create the figure with subplots and do initial layout adjustments
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.04)

# Calculate subplot grid center AFTER initial layout
subplot_top = axs[0,0].get_position().y1
subplot_bottom = axs[-1,0].get_position().y0
subplot_center_y = (subplot_top + subplot_bottom) / 2

# Calculate horizontal center of subplot grid
subplot_left = axs[0,0].get_position().x0
subplot_right = axs[0,-1].get_position().x1
subplot_center_x = (subplot_left + subplot_right) / 2

# Add DL label centered on the subplot grid vertically
fig.text(0.02, subplot_center_y, 'DL', fontsize=28+font_size_offset, fontweight='bold', 
        rotation='vertical', va='center', ha='center')

# Add LCModel label centered on the subplot grid horizontally
fig.text(subplot_center_x, 0.02, 'LCModel', fontsize=28+font_size_offset, fontweight='bold',
        va='center', ha='center')

# Process all configurations and metabolites
for row_idx, config in enumerate(config_order):
    if config not in all_configs_data:
        print(f"Warning: Configuration {config} not found, skipping")
        continue
    
    # Add configuration label as y-axis label for each row
    config_name = config_nice_names[config]
    axs[row_idx, 0].set_ylabel(config_name, fontsize=24+font_size_offset, 
                              fontweight='bold', labelpad=10)
    
    print(f"Adding row {row_idx+1}: {config}")
    normalized_data = {}
    data_dict = all_configs_data[config]
    
    # Normalize the data in the same way as before
    for folder, data in data_dict.items():
        normalized_data[folder] = {"lcm": {}, "pred": {}}
        for subject, lcm_data in data["lcm"].items():
            pred_data = data["pred"].get(subject, {})
            normalized_data[folder]["lcm"][subject] = {}
            normalized_data[folder]["pred"][subject] = {}

            for met in lcm_data.keys():
                # Normalize LCM data using global maximum
                normalized_data[folder]["lcm"][subject][met] = {
                    "/Cr+PCr": lcm_data[met]["/Cr+PCr"] / global_max_concentration_lcm
                }

                # Normalize predicted data using global maximum (same as before)
                if met == "Glu+Gln":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["Glu"][ppm_crop].real + pred_data["Gln"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real) / global_max_concentration_pred
                elif met == "Cr+PCr":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["Cr"][ppm_crop].real + pred_data["PCr"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred
                elif met == "NAA+NAAG":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["NAA"][ppm_crop].real + pred_data["NAAG"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred
                elif met == "PCh+GPC":
                    normalized_data[folder]["pred"][subject][met] = np.sum(
                        pred_data["PCho"][ppm_crop].real + pred_data["GPC"][ppm_crop].real)/np.sum(
                        pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred
                else:
                    if met in pred_data.keys():
                        normalized_data[folder]["pred"][subject][met] = np.sum(
                            pred_data[met][ppm_crop].real)/np.sum(
                            pred_data["Cr"][ppm_crop].real+pred_data["PCr"][ppm_crop].real)/global_max_concentration_pred

    # Process each metabolite for this configuration
    for col_idx, met in enumerate(target_metabolites):
        # Use the axes array directly
        ax = axs[row_idx, col_idx]
        
        # Collect data points
        lcm_all = []
        pred_all = []
        
        for folder, data in normalized_data.items():
            lcm_values = [v[met]["/Cr+PCr"] for v in data["lcm"].values() if met in v]
            pred_values = [v[met] for v in data["pred"].values() if met in v]
            
            # Append to global lists
            lcm_all.extend(lcm_values)
            pred_all.extend(pred_values)
            
            # Scatter data points
            ax.scatter(lcm_values, pred_values, label=folder, s=200, alpha=0.6)
        
        # Convert to NumPy arrays
        lcm_all = np.array(lcm_all)
        pred_all = np.array(pred_all)
        
        # Calculate correlation statistics
        if np.var(lcm_all) > 0 and np.var(pred_all) > 0:
            spearman_corr, spearman_p = spearmanr(lcm_all, pred_all)
            tau, kendall_p = kendalltau(lcm_all, pred_all)
        else:
            spearman_corr = spearman_p = tau = kendall_p = float('nan')
            
        # Add LOWESS curves if possible
        if len(lcm_all) > 1:
            try:
                # Spearman visualization
                lowess_result = lowess(pred_all, lcm_all, frac=0.6)
                ax.plot(lowess_result[:, 0], lowess_result[:, 1], 
                         color="#8B0000", label='Spearman', zorder=10)
                
                # Kendall visualization
                kendall_lowess_result = lowess(pred_all, lcm_all, frac=0.75)
                ax.plot(kendall_lowess_result[:, 0], kendall_lowess_result[:, 1], 
                         color="#FF8C00", label='Kendall', linestyle="--", zorder=10)
            except Exception as e:
                print(f"Error generating LOWESS for {config}-{met}: {e}")
        
        # Set consistent axis limits for each metabolite across all configurations
        ax.set_xlim(metabolite_ranges[met]['x_lim'])
        ax.set_ylim(metabolite_ranges[met]['y_lim'])
        
        # Add statistics annotation
        stats_text = (f'ρ = {spearman_corr:.2f} (p={format_p_value(spearman_p)})\n'
                     f'τ = {tau:.2f} (p={format_p_value(kendall_p)})')
        
        # Determine annotation position based on the metabolite's range
        y_mean = np.mean(pred_all)
        y_mid = np.mean(metabolite_ranges[met]['y_lim'])
        if y_mean > y_mid:
            y_pos = 0.05  # Bottom
        else:
            y_pos = 0.95  # Top
            
        ax.annotate(stats_text, xy=(0.05, y_pos), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.95, ec="gray"),
                    ha="left", va='top' if y_pos > 0.5 else 'bottom', fontsize=16+font_size_offset)
        
        # Set more dynamic tick intervals to ensure 3-4 ticks per axis
        x_min, x_max = metabolite_ranges[met]['x_lim']
        y_min, y_max = metabolite_ranges[met]['y_lim']
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Calculate tick spacing to give approximately 3-4 ticks
        x_major = round(x_range / 4, 2)  # Try to get 4 ticks
        y_major = round(y_range / 4, 2)  # Try to get 4 ticks
        
        # Minimum tick spacing to prevent overcrowding
        if x_major < 0.1:
            x_major = 0.1
        if y_major < 0.1:
            y_major = 0.1
            
        # Special handling for PCh+GPC and Glu+Gln to add more x-ticks
        if met == "PCh+GPC":
            # Force more x-ticks for PCh+GPC
            x_major = 0.1 if x_range < 0.5 else 0.2
        elif met == "Glu+Gln":
            # Force more x-ticks for Glu+Gln
            x_major = 0.25 if x_range < 1.5 else 0.5
            
        # Apply the tick locators
        ax.xaxis.set_major_locator(tick.MultipleLocator(x_major))
        ax.yaxis.set_major_locator(tick.MultipleLocator(y_major))
        ax.xaxis.set_minor_locator(tick.MultipleLocator(x_major/2))
        ax.yaxis.set_minor_locator(tick.MultipleLocator(y_major/2))
        
        # Style the ticks
        ax.tick_params(which='major', length=10, width=2, labelsize=16+font_size_offset)
        ax.tick_params(which='minor', length=5, width=1, color='gray')
        
        # Add clean grid
        ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

# Get legend handles and labels from the last subplot
handles, labels = plt.gca().get_legend_handles_labels()

# Remove duplicates while preserving order
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

# Create a common legend with more space at the top
plt.figlegend(unique_handles, unique_labels,
             loc='upper center',
             bbox_to_anchor=(0.5, 0.995),  # Positioned higher for more space
             ncol=5,
             frameon=True,
             fancybox=True,
             shadow=True,
             fontsize=18+font_size_offset)

# Save the improved figure
plt.savefig(f"{output_dir}/all_scatter_improved.png", bbox_inches='tight', dpi=300)
plt.close()

print("Combined figure saved as all_scatter_improved.png")