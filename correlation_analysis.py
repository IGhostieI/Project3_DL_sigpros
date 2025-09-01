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
    model_configureation, 
    ppm, 
    passband_GABA=None, 
    passband_full=None, 
    margin=0, 
    ftype="bandpass", 
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
    if "_ab" in model_configureation:
        optim = ComplexFilterOptimizer(input_signal=norm_FID, order=60, options={'method': 'SLSQP', 'ftol': 1e-3, 'maxiter': 50})
        b_opt, a_opt = optim.optimize()
        a_opt = np.pad(a_opt, (0, 2048 - len(a_opt)))
        b_opt = np.pad(b_opt, (0, 2048 - len(b_opt)))

    # Apply band-pass filters if passbands are provided
    if "gaba" in model_configureation:
        filtered_GABA, _, _ = BPF_ppm(fft_signal=norm_freq_fft, ppm_range=ppm, ppm_pass=passband_GABA, margin=margin, ftype=ftype, gain_passband=pass_band_gain)

    if "full" in model_configureation:
        filtered_full, _, _ = BPF_ppm(fft_signal=norm_freq_fft, ppm_range=ppm, ppm_pass=passband_full, margin=margin, ftype=ftype, gain_passband=pass_band_gain)

    # Prepare input for the model
    if model_configureation == "FID":
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag]).T, axis=0)
    elif model_configureation == "FID_gaba" and filtered_GABA is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, filtered_GABA.real, filtered_GABA.imag]).T, axis=0)
    elif model_configureation == "FID_full_filt" and filtered_full is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, filtered_full.real, filtered_full.imag]).T, axis=0)
    elif model_configureation == "FID_ab_gaba" and filtered_GABA is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real, b_opt.imag, filtered_GABA.real, filtered_GABA.imag]).T, axis=0)
    elif model_configureation == "FID_ab_full_filt" and filtered_full is not None:
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real, b_opt.imag, filtered_full.real, filtered_full.imag]).T, axis=0)
    elif model_configureation == "FID_ab":
        in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real, b_opt.imag]).T, axis=0)
    else:
        raise ValueError(f"Unknown or incompatible model configuration: {model_configureation}")

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

# Helper function to extract folder type and subject identifier
def extract_folder_and_subject(file_path):
    # Extract folder name and file name
    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    # Extract subject identifier (e.g., S04) from the file name
    subject_match = re.search(r"(S\d+)", file_name)
    subject_id = subject_match.group(1) if subject_match else None

    # Extract folder type (e.g., G*_P, P*_P, or fmrs_pain)
    folder_type = None
    if "fmrs_pain" in file_path:
        folder_type = "fmrs_pain"
    elif re.search(r"P\d+_P", folder_name):
        folder_type = folder_name
    elif re.search(r"G\d+_P", folder_name):
        folder_type = folder_name
    else:
        # Handle cases where folder type is in the file name (e.g., G8_P_S03.table)
        folder_match = re.search(r"(G\d+_P|P\d+_P)", file_name)
        folder_type = folder_match.group(1) if folder_match else None

    # Validate that both folder type and subject ID are present
    if folder_type and subject_id:
        return folder_type, subject_id
    else:
        return None, None

slurm = True
LOAD_PREDICTIONS_FROM_DIR = True

if not slurm:
    available_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(available_devices[6], 'GPU')
    print(f"visible devices: {tf.config.get_visible_devices()}")
    available_devices = tf.config.list_physical_devices('GPU')
    print(f"visible devices: {tf.config.get_visible_devices()}")
# File paths
raw_data_base = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/mrui_files"

lcmodel_base = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/Osprey"

os.makedirs("concentrations", exist_ok=True)

# Model weights 
weight_dict = {
    "FID":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_23-21_37_38_['augmented_ifft']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_ab":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_23-21_37_44_['augmented_ifft', 'a', 'b']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_ab_gaba":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_23-21_40_40_['augmented_ifft', 'a', 'b', 'filtered_GABA']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_ab_full_filt":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_24-09_56_20_['augmented_ifft', 'a', 'b', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    
    "FID_full_filt":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_24-10_52_49_['augmented_ifft', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5",
    "FID_gaba":"/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_24-10_54_09_['augmented_ifft', 'filtered_GABA']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
}
# Get P*_P folders and fmrs_pain in mrui_files/

# Get full paths to all S*.txt and HLSVD.txt files for Philips (fmrs_pain and P*_P) and GE
philips_files_fmrs_pain = [
    os.path.join(raw_data_base, folder, file)
    for folder in os.listdir(raw_data_base)
    if folder == "fmrs_pain"
    for file in os.listdir(os.path.join(raw_data_base, folder))
    if file.endswith(".txt") and ("S" in file) and "HLSVD" not in file
]

philips_files_other = [
    os.path.join(raw_data_base, folder, file)
    for folder in os.listdir(raw_data_base)
    if folder.startswith("P") and folder.endswith("_P")
    for file in os.listdir(os.path.join(raw_data_base, folder))
    if file.endswith(".txt") and ("S" in file) and "HLSVD" not in file
]

philips_files_HLSVD = [
    os.path.join(raw_data_base, folder, file)
    for folder in os.listdir(raw_data_base)
    if folder == "fmrs_pain" or (folder.startswith("P") and folder.endswith("_P"))
    for file in os.listdir(os.path.join(raw_data_base, folder))
    if file.endswith(".txt") and ("S" in file and "HLSVD" in file)
]

ge_files = [
    os.path.join(raw_data_base, folder, file)
    for folder in os.listdir(raw_data_base)
    if folder.startswith("G") and folder.endswith("_P")
    for file in os.listdir(os.path.join(raw_data_base, folder))
    if file.endswith(".txt") and "S" in file and "HLSVD" not in file
]

ge_files_HLSVD = [
    os.path.join(raw_data_base, folder, file)
    for folder in os.listdir(raw_data_base)
    if folder.startswith("G") and folder.endswith("_P")
    for file in os.listdir(os.path.join(raw_data_base, folder))
    if file.endswith(".txt") and ("S" in file and "HLSVD" in file)
]

# Get .table files for Philips and GE
philips_table_files_other = [
    os.path.join(lcmodel_base, "Philips", "LCMoutput", file)
    for file in os.listdir(os.path.join(lcmodel_base, "Philips", "LCMoutput"))
    if file.endswith(".table")
]
# Separate .table files for fmrs_pain and other Philips datasets
philips_table_files_fmrs_pain = [
    os.path.join(lcmodel_base, "fmrs_pain", "LCMoutput", file)
    for file in os.listdir(os.path.join(lcmodel_base, "fmrs_pain", "LCMoutput"))
    if file.endswith(".table")
]
ge_table_files = [
    os.path.join(lcmodel_base, "GE", "LCMoutput", file)
    for file in os.listdir(os.path.join(lcmodel_base, "GE", "LCMoutput"))
    if file.endswith(".table")
]



# Pair fmrs_pain files
philips_pairs_fmrs_pain = [
    (fmrs_file, table_file)
    for fmrs_file in philips_files_fmrs_pain
    for table_file in philips_table_files_fmrs_pain
    if extract_folder_and_subject(fmrs_file) == extract_folder_and_subject(table_file)
]

# Pair other Philips files
philips_pairs_other = [
    (philips_file, table_file)
    for philips_file in philips_files_other
    for table_file in philips_table_files_other
    if extract_folder_and_subject(philips_file) == extract_folder_and_subject(table_file)
]

ge_pairs = [
    (ge_file, table_file)
    for ge_file in ge_files
    for table_file in ge_table_files
    if extract_folder_and_subject(ge_file) == extract_folder_and_subject(table_file)
]

ppm = np.linspace(-3.121, 12.528, 2048)
passband_GABA =  np.array([2.1, 2.7]) # ppm
passband_full = np.array([1.6, 4.2]) # ppm
margin = 1
ftype = "butter"
pass_band_gain = 2

metabolite_list = ["NAA", "NAAG","Cr", "PCr", "GABA", "Gln","Glu", "NAA+NAAG", "Cr+PCr", "PCh+GPC", "Glu+Gln"]

if not LOAD_PREDICTIONS_FROM_DIR:
    for config in list(weight_dict.keys())[:]:
        # Load model
        input_keys, output_keys = input_ouput_key_sorting(weight_dict[config])
        model = multi_channel_cnn(input_shape = (2048,2*len(input_keys)), output_shape=(2048, 2*len(output_keys) ), num_blocks=20, filter1=32, kernel_size=16, print_summary=False)
        model.build(input_shape=(None, 2048, 2*len(input_keys)))
        model.load_weights(weight_dict[config])
    # Process Philips files
        exclusion_counts_ge = {}
        GE_concentration_dict = {}
        GE_shown_exclusion = False
        for ge_file, table_file in ge_pairs[:]:
            # Read LCModel table
            LCModel_metabolites_dict = read_LCModel_table(table_file, metabolite_list)
            SD_list = [(f"{key}: {LCModel_metabolites_dict[key]['%SD']}", key, LCModel_metabolites_dict[key]['%SD']) for key in LCModel_metabolites_dict.keys()]
            SDs = ', '.join(item[0] for item in SD_list)
            
            if any(item[2] > 50 for item in SD_list):
                folder, subject = extract_folder_and_subject(ge_file)
                # Increment exclusion count for the folder
                if folder not in exclusion_counts_ge:
                    exclusion_counts_ge[folder] = []
                exclusion_counts_ge[folder].append(subject)
            else:
                # Load and process data
                in_vivo_output_dict = process_and_save(
                    ge_file, 
                    model_configureation=config, 
                    ppm=ppm, 
                    passband_GABA=passband_GABA, 
                    passband_full=passband_full, 
                    margin=margin, 
                    ftype=ftype, 
                    pass_band_gain=pass_band_gain, 
                    model_weights_path=weight_dict[config]
                )
                # Save the processed data
                folder, subject = extract_folder_and_subject(ge_file)
                if folder not in GE_concentration_dict:
                    GE_concentration_dict[folder] = {}
                if subject not in GE_concentration_dict[folder]:
                    GE_concentration_dict[folder][subject] = {}
                GE_concentration_dict[folder][subject]["pred"] = in_vivo_output_dict
                GE_concentration_dict[folder][subject]["lcm"] = LCModel_metabolites_dict

        if not GE_shown_exclusion:
            # Print summary of exclusions
            print("Exclusion Summary, GE:")
            for folder, count in exclusion_counts_ge.items():
                print(f"{folder}: {len(count)} subjects excluded due to %SD > 50")
            GE_shown_exclusion = True
        # Save the GE concentration dictionary to a file
        with open(f"concentrations/GE_concentration_dict_{config}.pickle", "wb") as f:
            pickle.dump(GE_concentration_dict, f)

        
        # Process Philips files
        exclusion_counts_philips_fmrs_pain = {}
        Philips_fmrs_pain_concentration_dict = {}
        Philips_fmrs_pain_shown_exclusion = False
        for philips_file, table_file in philips_pairs_fmrs_pain:
            # Read LCModel table
            LCModel_metabolites_dict = read_LCModel_table(table_file, metabolite_list)
            SD_list = [(f"{key}: {LCModel_metabolites_dict[key]['%SD']}", key, LCModel_metabolites_dict[key]['%SD']) for key in LCModel_metabolites_dict.keys()]
            SDs = ', '.join(item[0] for item in SD_list)
            
            if any(item[2] > 50 for item in SD_list):
                folder, subject = extract_folder_and_subject(philips_file)
                # Increment exclusion count for the folder
                if folder not in exclusion_counts_philips_fmrs_pain:
                    exclusion_counts_philips_fmrs_pain[folder] = []
                exclusion_counts_philips_fmrs_pain[folder].append(subject)
            else:
                # Load and process data
                in_vivo_output_dict = process_and_save(
                    philips_file, 
                    model_configureation=config, 
                    ppm=ppm, 
                    passband_GABA=passband_GABA, 
                    passband_full=passband_full, 
                    margin=margin, 
                    ftype=ftype, 
                    pass_band_gain=pass_band_gain, 
                    model_weights_path=weight_dict[config]
                )
                # Save the processed data
                folder, subject = extract_folder_and_subject(philips_file)
                if folder not in Philips_fmrs_pain_concentration_dict:
                    Philips_fmrs_pain_concentration_dict[folder] = {}
                if subject not in Philips_fmrs_pain_concentration_dict[folder]:
                    Philips_fmrs_pain_concentration_dict[folder][subject] = {}
                Philips_fmrs_pain_concentration_dict[folder][subject]["pred"] = in_vivo_output_dict
                Philips_fmrs_pain_concentration_dict[folder][subject]["lcm"] = LCModel_metabolites_dict
                
        if not Philips_fmrs_pain_shown_exclusion:
            # Print summary of exclusions
            print("Exclusion Summary, fmrs_pain:")
            for folder, count in exclusion_counts_philips_fmrs_pain.items():
                print(f"{folder}: {len(count)} subjects excluded due to %SD > 50")
            Philips_fmrs_pain_shown_exclusion = True
        # Save the Philips concentration dictionary to a file
        with open(f"concentrations/Philips_fmrs_pain_concentration_dict_{config}.pickle", "wb") as f:
            pickle.dump(Philips_fmrs_pain_concentration_dict, f)
        # Process Philips files
        exclusion_counts_philips_other = {}
        Philips_other_concentration_dict = {}
        Philips_other_shown_exclusion = False
        for philips_file, table_file in philips_pairs_other:
            # Read LCModel table
            LCModel_metabolites_dict = read_LCModel_table(table_file, metabolite_list)
            SD_list = [(f"{key}: {LCModel_metabolites_dict[key]['%SD']}", key, LCModel_metabolites_dict[key]['%SD']) for key in LCModel_metabolites_dict.keys()]
            SDs = ', '.join(item[0] for item in SD_list)
            
            if any(item[2] > 50 for item in SD_list):
                folder, subject = extract_folder_and_subject(philips_file)
                # Increment exclusion count for the folder
                if folder not in exclusion_counts_philips_other:
                    exclusion_counts_philips_other[folder] = []
                exclusion_counts_philips_other[folder].append(subject)
            else:
                # Load and process data
                in_vivo_output_dict = process_and_save(
                    philips_file, 
                    model_configureation=config, 
                    ppm=ppm, 
                    passband_GABA=passband_GABA, 
                    passband_full=passband_full, 
                    margin=margin, 
                    ftype=ftype, 
                    pass_band_gain=pass_band_gain, 
                    model_weights_path=weight_dict[config]
                )
                # Save the processed data
                folder, subject = extract_folder_and_subject(philips_file)
                if folder not in Philips_other_concentration_dict:
                    Philips_other_concentration_dict[folder] = {}
                if subject not in Philips_other_concentration_dict[folder]:
                    Philips_other_concentration_dict[folder][subject] = {}
                Philips_other_concentration_dict[folder][subject]["pred"] = in_vivo_output_dict
                Philips_other_concentration_dict[folder][subject]["lcm"] = LCModel_metabolites_dict
        if not Philips_other_shown_exclusion:
            # Print summary of exclusions
            print("Exclusion Summary, Philips:")
            for folder, count in exclusion_counts_philips_other.items():
                print(f"{folder}: {len(count)} subjects excluded due to %SD > 50")
            Philips_other_shown_exclusion = True
        # Save the Philips concentration dictionary to a file
        with open(f"concentrations/Philips_other_concentration_dict_{config}.pickle", "wb") as f:
            pickle.dump(Philips_other_concentration_dict, f)
else:
    # Load the saved concentration dictionaries
    with open("concentrations/Philips_other_concentration_dict_FID.pickle", "rb") as f:
        Philips_other_concentration_dict = pickle.load(f)
    with open("concentrations/GE_concentration_dict_FID.pickle", "rb") as f:
        GE_concentration_dict = pickle.load(f)   
    with open("concentrations/Philips_fmrs_pain_concentration_dict_FID.pickle", "rb") as f:
        Philips_fmrs_pain_concentration_dict = pickle.load(f)
        
def process_concentrations(concentration_dict, Cr_ref, Cr_mean, Cr_pred_mean, lcm_metabolites):
    """Process concentrations for a given dictionary."""
    metabolites_lcm = {met: [] for met in lcm_metabolites}
    metabolites_pred = {met: [] for met in lcm_metabolites}
    

    for folder, subjects in concentration_dict.items():
        for subject, data in subjects.items():
            if "pred" in data and "lcm" in data:
                for met in metabolites_lcm.keys():
                    metabolites_lcm[met].append(data["lcm"][met]["conc"]* Cr_ref / Cr_mean)
                    if met == "PCh":
                        met = "PCho"
                    elif met == "NAA+NAAG":
                        metabolites_pred[met].append(np.sum(data["pred"]["NAA"].real + data["pred"]["NAAG"].real) * Cr_ref / Cr_pred_mean)
                    elif met == "Cr+PCr":
                        metabolites_pred[met].append(np.sum(data["pred"]["Cr"].real + data["pred"]["PCr"].real) * Cr_ref / Cr_pred_mean)
                        continue
                    elif met == "Glu+Gln":
                        metabolites_pred[met].append(np.sum(data["pred"]["Glu"].real + data["pred"]["Gln"].real) * Cr_ref / Cr_pred_mean)
                        continue
                    elif met == "PCh+GPC":
                        metabolites_pred[met].append(np.sum(data["pred"]["GPC"].real + data["pred"]["PCho"].real) * Cr_ref / Cr_pred_mean)
                        continue
                    else:
                        metabolites_pred[met].append(np.sum(data["pred"][met].real) * Cr_ref / Cr_pred_mean)

    return metabolites_pred, metabolites_lcm

def plot_concentrations(metabolites_pred, metabolites_lcm, label_prefix, colors, output_dir="figures"):
    """Plot predicted vs LCModel concentrations for each metabolite."""
    os.makedirs(output_dir, exist_ok=True)
    for met, color in zip(metabolites_pred.keys(), colors):
        plt.figure(figsize=(10, 6))
        plt.scatter(metabolites_pred[met], metabolites_lcm[met], label=f"{met} {label_prefix}", alpha=0.5, color=color)
        plt.xlabel(f"Predicted {met} Concentration")
        plt.ylabel(f"LCModel {met} Concentration")
        plt.legend()
        plt.savefig(f"{output_dir}/{met}_{label_prefix}.png")
        plt.close()

# Calculate mean Cr concentrations
def calculate_mean_cr(concentration_dict):
    Cr = [data["lcm"]["Cr"]["conc"] for folder, subjects in concentration_dict.items() for subject, data in subjects.items() if "pred" in data and "lcm" in data]
    Cr_pred = [np.sum(data["pred"]["Cr"].real) for folder, subjects in concentration_dict.items() for subject, data in subjects.items() if "pred" in data and "lcm" in data]
    return np.mean(Cr), np.mean(Cr_pred)

# Calculate mean Cr concentrations for each dataset
Cr_Philips, Cr_pred_Philips = calculate_mean_cr(Philips_other_concentration_dict)
Cr_GE, Cr_pred_GE = calculate_mean_cr(GE_concentration_dict)
Cr_fmrs_pain, Cr_pred_fmrs_pain = calculate_mean_cr(Philips_fmrs_pain_concentration_dict)

# Process concentrations for each dataset
Philips_pred, Philips_lcm = process_concentrations(Philips_other_concentration_dict, Cr_Philips, Cr_Philips, Cr_pred_Philips, metabolite_list)
GE_pred, GE_lcm = process_concentrations(GE_concentration_dict, Cr_Philips, Cr_GE, Cr_pred_GE, metabolite_list)
fmrs_pain_pred, fmrs_pain_lcm = process_concentrations(Philips_fmrs_pain_concentration_dict, Cr_Philips, Cr_fmrs_pain, Cr_pred_fmrs_pain, metabolite_list)

# Plot concentrations
plot_concentrations(Philips_pred, Philips_lcm, "Philips", ["blue"] * 7)
plot_concentrations(GE_pred, GE_lcm, "GE", ["red"] * 7)
plot_concentrations(fmrs_pain_pred, fmrs_pain_lcm, "fmrs_pain", ["green"] * 7)

""" # Calculate the mean Cr concentration from the GE and Philips lcm

Cr_Philips = []
Cr_GE = []
Cr_fmrs_pain = []
Cr_pred_Philips = []
Cr_pred_GE = []
Cr_pred_fmrs_pain = []

# Process Philips concentrations
for folder, subjects in Philips_other_concentration_dict.items():
    for subject, data in subjects.items():
        if "pred" in data and "lcm" in data:
            Cr_Philips.append(data["lcm"]["Cr"]["conc"])
            Cr_pred_Philips.append(data["pred"]["Cr"].real)

# Process GE concentrations
for folder, subjects in GE_concentration_dict.items():
    for subject, data in subjects.items():
        if "pred" in data and "lcm" in data:
            Cr_GE.append(data["lcm"]["Cr"]["conc"])
            Cr_pred_GE.append(data["pred"]["Cr"].real)
# Process Philips fmrs_pain concentrations
for folder, subjects in Philips_fmrs_pain_concentration_dict.items():
    for subject, data in subjects.items():
        if "pred" in data and "lcm" in data:
            Cr_fmrs_pain.append(data["lcm"]["Cr"]["conc"])
            Cr_pred_fmrs_pain.append(data["pred"]["Cr"].real)
# Calculate mean Cr concentrations
Cr_Philips = np.array(Cr_Philips).mean()
Cr_GE = np.array(Cr_GE).mean()
Cr_fmrs_pain = np.array(Cr_fmrs_pain).mean()
Cr_pred_Philips = np.array(Cr_pred_Philips).mean()
Cr_pred_GE = np.array(Cr_pred_GE).mean()
Cr_pred_fmrs_pain = np.array(Cr_pred_fmrs_pain).mean()

# Calculate concentrations for Philips, GE, and Philips fmrs_pain
# Create arrays for each metabolite
NAA_array_Philips_pred = []
NAAG_array_Philips_pred = []
GPC_array_Philips_pred = []
PCh_array_Philips_pred = []
GABA_array_Philips_pred = []
Gln_array_Philips_pred = []
Glu_array_Philips_pred = []

NAA_array_Philips_lcm = []
NAAG_array_Philips_lcm = []
GPC_array_Philips_lcm = []
PCh_array_Philips_lcm = []
GABA_array_Philips_lcm = []
Gln_array_Philips_lcm = []
Glu_array_Philips_lcm = []

# Process Philips data
for folder, subjects in Philips_other_concentration_dict.items():
    for subject, data in subjects.items():
        if "pred" in data and "lcm" in data:
            NAA_array_Philips_lcm.append(data["lcm"]["NAA"]["conc"])
            NAAG_array_Philips_lcm.append(data["lcm"]["NAAG"]["conc"])
            GPC_array_Philips_lcm.append(data["lcm"]["GPC"]["conc"])
            PCh_array_Philips_lcm.append(data["lcm"]["PCh"]["conc"])
            GABA_array_Philips_lcm.append(data["lcm"]["GABA"]["conc"])
            Gln_array_Philips_lcm.append(data["lcm"]["Gln"]["conc"])
            Glu_array_Philips_lcm.append(data["lcm"]["Glu"]["conc"])
            NAA_array_Philips_pred.append(np.sum(data["pred"]["NAA"].real) * Cr_Philips / Cr_pred_Philips)
            NAAG_array_Philips_pred.append(np.sum(data["pred"]["NAAG"].real) * Cr_Philips / Cr_pred_Philips)
            GPC_array_Philips_pred.append(np.sum(data["pred"]["GPC"].real) * Cr_Philips / Cr_pred_Philips)
            PCh_array_Philips_pred.append(np.sum(data["pred"]["PCho"].real) * Cr_Philips / Cr_pred_Philips)
            GABA_array_Philips_pred.append(np.sum(data["pred"]["GABA"].real) * Cr_Philips / Cr_pred_Philips)
            Gln_array_Philips_pred.append(np.sum(data["pred"]["Gln"].real) * Cr_Philips / Cr_pred_Philips)
            Glu_array_Philips_pred.append(np.sum(data["pred"]["Glu"].real) * Cr_Philips / Cr_pred_Philips)

# Repeat similar steps for GE and Philips fmrs_pain
# Process GE data
NAA_array_GE_pred = []
NAAG_array_GE_pred = []
GPC_array_GE_pred = []
PCh_array_GE_pred = []
GABA_array_GE_pred = []
Gln_array_GE_pred = []
Glu_array_GE_pred = []

NAA_array_GE_lcm = []
NAAG_array_GE_lcm = []
GPC_array_GE_lcm = []
PCh_array_GE_lcm = []
GABA_array_GE_lcm = []
Gln_array_GE_lcm = []
Glu_array_GE_lcm = []

for folder, subjects in GE_concentration_dict.items():
    for subject, data in subjects.items():
        if "pred" in data and "lcm" in data:
            NAA_array_GE_lcm.append(data["lcm"]["NAA"]["conc"] * Cr_Philips / Cr_GE)
            NAAG_array_GE_lcm.append(data["lcm"]["NAAG"]["conc"] * Cr_Philips / Cr_GE)
            GPC_array_GE_lcm.append(data["lcm"]["GPC"]["conc"] * Cr_Philips / Cr_GE)
            PCh_array_GE_lcm.append(data["lcm"]["PCh"]["conc"] * Cr_Philips / Cr_GE)
            GABA_array_GE_lcm.append(data["lcm"]["GABA"]["conc"] * Cr_Philips / Cr_GE)
            Gln_array_GE_lcm.append(data["lcm"]["Gln"]["conc"] * Cr_Philips / Cr_GE)
            Glu_array_GE_lcm.append(data["lcm"]["Glu"]["conc"] * Cr_Philips / Cr_GE)
            NAA_array_GE_pred.append(np.sum(data["pred"]["NAA"].real) * Cr_Philips / Cr_pred_GE)
            NAAG_array_GE_pred.append(np.sum(data["pred"]["NAAG"].real) * Cr_Philips / Cr_pred_GE)
            GPC_array_GE_pred.append(np.sum(data["pred"]["GPC"].real) * Cr_Philips / Cr_pred_GE)
            PCh_array_GE_pred.append(np.sum(data["pred"]["PCho"].real) * Cr_Philips / Cr_pred_GE)
            GABA_array_GE_pred.append(np.sum(data["pred"]["GABA"].real) * Cr_Philips / Cr_pred_GE)
            Gln_array_GE_pred.append(np.sum(data["pred"]["Gln"].real) * Cr_Philips / Cr_pred_GE)
            Glu_array_GE_pred.append(np.sum(data["pred"]["Glu"].real) * Cr_Philips / Cr_pred_GE)

# Process Philips fmrs_pain data
NAA_array_fmrs_pain_pred = []
NAAG_array_fmrs_pain_pred = []
GPC_array_fmrs_pain_pred = []
PCh_array_fmrs_pain_pred = []
GABA_array_fmrs_pain_pred = []
Gln_array_fmrs_pain_pred = []
Glu_array_fmrs_pain_pred = []

NAA_array_fmrs_pain_lcm = []
NAAG_array_fmrs_pain_lcm = []
GPC_array_fmrs_pain_lcm = []
PCh_array_fmrs_pain_lcm = []
GABA_array_fmrs_pain_lcm = []
Gln_array_fmrs_pain_lcm = []
Glu_array_fmrs_pain_lcm = []

for folder, subjects in Philips_fmrs_pain_concentration_dict.items():
    for subject, data in subjects.items():
        if "pred" in data and "lcm" in data:
            NAA_array_fmrs_pain_lcm.append(data["lcm"]["NAA"]["conc"])
            NAAG_array_fmrs_pain_lcm.append(data["lcm"]["NAAG"]["conc"])
            GPC_array_fmrs_pain_lcm.append(data["lcm"]["GPC"]["conc"])
            PCh_array_fmrs_pain_lcm.append(data["lcm"]["PCh"]["conc"])
            GABA_array_fmrs_pain_lcm.append(data["lcm"]["GABA"]["conc"])
            Gln_array_fmrs_pain_lcm.append(data["lcm"]["Gln"]["conc"])
            Glu_array_fmrs_pain_lcm.append(data["lcm"]["Glu"]["conc"])
            NAA_array_fmrs_pain_pred.append(np.sum(data["pred"]["NAA"].real) * Cr_Philips / Cr_pred_fmrs_pain)
            NAAG_array_fmrs_pain_pred.append(np.sum(data["pred"]["NAAG"].real) * Cr_Philips / Cr_pred_fmrs_pain)
            GPC_array_fmrs_pain_pred.append(np.sum(data["pred"]["GPC"].real) * Cr_Philips / Cr_pred_fmrs_pain)
            PCh_array_fmrs_pain_pred.append(np.sum(data["pred"]["PCho"].real) * Cr_Philips / Cr_pred_fmrs_pain)
            GABA_array_fmrs_pain_pred.append(np.sum(data["pred"]["GABA"].real) * Cr_Philips / Cr_pred_fmrs_pain)
            Gln_array_fmrs_pain_pred.append(np.sum(data["pred"]["Gln"].real) * Cr_Philips / Cr_pred_fmrs_pain)
            Glu_array_fmrs_pain_pred.append(np.sum(data["pred"]["Glu"].real) * Cr_Philips / Cr_pred_fmrs_pain)


# Test plotting
plt.figure(figsize=(10, 6))
plt.scatter(NAA_array_Philips_pred, NAA_array_Philips_lcm, label="NAA Philips", alpha=0.5, color="blue")
plt.scatter(NAA_array_GE_pred, NAA_array_GE_lcm, label="NAA GE", alpha=0.5, color="red")
plt.scatter(NAA_array_fmrs_pain_pred, NAA_array_fmrs_pain_lcm, label="NAA fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted NAA Concentration")
plt.ylabel("LCModel NAA Concentration")
plt.legend()
plt.savefig("figures/NAA_Philips.png") 

plt.figure(figsize=(10, 6))
plt.scatter(NAAG_array_Philips_pred, NAAG_array_Philips_lcm, label="NAAG Philips", alpha=0.5, color="blue")
plt.scatter(NAAG_array_GE_pred, NAAG_array_GE_lcm, label="NAAG GE", alpha=0.5, color="red")
plt.scatter(NAAG_array_fmrs_pain_pred, NAAG_array_fmrs_pain_lcm, label="NAAG fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted NAAG Concentration")
plt.ylabel("LCModel NAAG Concentration")
plt.legend()
plt.savefig("figures/NAAG_Philips.png")
plt.figure(figsize=(10, 6))
plt.scatter(GPC_array_Philips_pred, GPC_array_Philips_lcm, label="GPC Philips", alpha=0.5, color="blue")
plt.scatter(GPC_array_GE_pred, GPC_array_GE_lcm, label="GPC GE", alpha=0.5, color="red")
plt.scatter(GPC_array_fmrs_pain_pred, GPC_array_fmrs_pain_lcm, label="GPC fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted GPC Concentration")
plt.ylabel("LCModel GPC Concentration")
plt.legend()
plt.savefig("figures/GPC_Philips.png")
plt.figure(figsize=(10, 6))
plt.scatter(PCh_array_Philips_pred, PCh_array_Philips_lcm, label="PCh Philips", alpha=0.5, color="blue")
plt.scatter(PCh_array_GE_pred, PCh_array_GE_lcm, label="PCh GE", alpha=0.5, color="red")
plt.scatter(PCh_array_fmrs_pain_pred, PCh_array_fmrs_pain_lcm, label="PCh fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted PCh Concentration")
plt.ylabel("LCModel PCh Concentration")
plt.legend()
plt.savefig("figures/PCh_Philips.png")
plt.figure(figsize=(10, 6))
plt.scatter(GABA_array_Philips_pred, GABA_array_Philips_lcm, label="GABA Philips", alpha=0.5, color="blue")
plt.scatter(GABA_array_GE_pred, GABA_array_GE_lcm, label="GABA GE", alpha=0.5, color="red")
plt.scatter(GABA_array_fmrs_pain_pred, GABA_array_fmrs_pain_lcm, label="GABA fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted GABA Concentration")
plt.ylabel("LCModel GABA Concentration")
plt.legend()
plt.savefig("figures/GABA_Philips.png")
plt.figure(figsize=(10, 6))
plt.scatter(Gln_array_Philips_pred, Gln_array_Philips_lcm, label="Gln Philips", alpha=0.5, color="blue")
plt.scatter(Gln_array_GE_pred, Gln_array_GE_lcm, label="Gln GE", alpha=0.5, color="red")
plt.scatter(Gln_array_fmrs_pain_pred, Gln_array_fmrs_pain_lcm, label="Gln fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted Gln Concentration")
plt.ylabel("LCModel Gln Concentration")
plt.legend()
plt.savefig("figures/Gln_Philips.png")
plt.figure(figsize=(10, 6))
plt.scatter(Glu_array_Philips_pred, Glu_array_Philips_lcm, label="Glu Philips", alpha=0.5, color="blue")
plt.scatter(Glu_array_GE_pred, Glu_array_GE_lcm, label="Glu GE", alpha=0.5, color="red")
plt.scatter(Glu_array_fmrs_pain_pred, Glu_array_fmrs_pain_lcm, label="Glu fmrs_pain", alpha=0.5, color="green")
plt.xlabel("Predicted Glu Concentration")
plt.ylabel("LCModel Glu Concentration")
plt.legend()
plt.savefig("figures/Glu_Philips.png") """       