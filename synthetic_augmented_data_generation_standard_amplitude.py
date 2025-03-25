import os
import numpy as np
import os
import numpy as np
from scipy.io import savemat, loadmat
import scipy.signal as signal
import scipy.stats as stats
from scipy.fft import ifft, fftshift, fft
import time
from tqdm import tqdm
from myfunctions_tf import create_baseline, MM_baseline, read_complex_raw, scale_complex_data, add_gauss_noise_from_wanted_SNR, frequency_shift_jitter, BPF_ppm, ComplexFilterOptimizer, load_mrui_txt_data

def main():
    # Define base paths for raw files and save folder
    linewidths = [3, 5, 7, 10]
    TEs = [20, 30, 40]
    base_path = os.getcwd()
    num_spectra = 2500
    
    ppm_range = (12.528, -3.121)
    ppm_jitter_range = (-0.03, 0.03)
    now = time.strftime("%Y_%m_%d-%H_%M_%S")
    optimization = True
    if optimization:
        iir_order = 60
        optimizer_options = {'method':'SLSQP', 'ftol':1e-3, 'maxiter':50}
    
    BPF_filtering = True
    if BPF_filtering:
        passband_GABA =  np.array([2.1, 2.7]) # ppm
        passband_full = np.array([1.6, 4.2]) # ppm
        margin = 1
        ftype = "butter"
        pass_band_gain = 2
    # Define the format to save the data in, and the folder name to save it in
    save_format = "npz" # "mat" or "npz"

    save_folder = f"generated_data/{now}-standard_amplitude_{save_format}"
    os.makedirs(os.path.join(base_path, save_folder), exist_ok=True)
    for TE in TEs:
        for linewidth in linewidths:
            raw_files_path = f"/home/stud/casperc/bhome/Project3_DL_sigpros/DL_basis_sets/TE{TE}_lw{linewidth:02d}"
            TE_lw = f"TE{TE}_lw{linewidth:02d}"
            # Ensure the save folder exists
            print(f"Creating folder {save_folder}/{TE_lw}")
            os.makedirs(os.path.join(base_path, save_folder, TE_lw), exist_ok=True)

            # List all raw files in the specified directory
            raw_files = os.listdir(os.path.join(base_path, raw_files_path))

            # Define metabolite concentration ranges
            metabolite_concentrations = {
                "Ala": (0.1, 1.5), "Asp": (1.0, 2.0), "Cr": (4.0, 7.0), "GABA": (2.0, 3.0), "Glc": (1.0, 2.0),
                "Gln": (2.0, 5.0), "Glu": (4.0, 6.0), "GPC": (0.5, 2.0), "GSH": (1.5, 3.0), "Lac": (0.2, 1.0),
                "mI": (4.0, 7.0), "NAA": (7.5, 10), "NAAG": (0.5, 2.5), "PCho": (0.5, 2.0), "PCr": (1.0, 3), "Tau": (2.0, 5.0), "Gly":(1.0, 2.0) #Check the values
            }
            raw_files = [file for file in raw_files if ".txt" in file and file.removesuffix(".txt") in metabolite_concentrations.keys()]
            # metabolite_concentrations_brain_cancer = {}
            # metabolite_concentrations_prostate = {}

            # Start timer
            start = time.perf_counter()
           
            # Loop over the number of synthetic spectra to generate
            print(f"Generating {num_spectra} synthetic spectra")
            
            start_val = 0
            for i in tqdm(range(start_val+1,start_val+num_spectra+1)):
                metabolite_spectra = {}
                metabolite_FID = {}
                # Initialize/reset arrays to store real and imaginary parts of spectra
                total_real = np.zeros(2048)
                total_imag = np.zeros(2048)
                total_real_before_scaling = np.zeros(2048)
                total_imag_before_scaling = np.zeros(2048)
                
                # Loop over each raw file
                for raw_file in raw_files:
                    # Read the real and imaginary parts of the raw spectrum
                    _, metabolite_raw, _ = load_mrui_txt_data(os.path.join(base_path, raw_files_path, raw_file))
                    metabolite_raw = fftshift(metabolite_raw)
                    real = metabolite_raw.real
                    imag = metabolite_raw.imag
                    # Calculate the area under the curve (AUC) of the metabolite spectrum
                    AUC_metabolite = np.sum(abs(np.sqrt(real**2 + imag**2)))
                    # Get the current metabolite name from the file name
                    current_metabolite = raw_file.replace(".txt", "")
                    if current_metabolite != "Cr":
                        real, imag = frequency_shift_jitter(complex_data=(real, imag), ppm_range=ppm_range, ppm_jitter_range=ppm_jitter_range)
                    # Normalize the spectrum and scale it by a random factor within the metabolite concentration range
                    normalize_factor = np.sum(np.sqrt(real**2 + imag**2))
                    scaling_factor = np.random.uniform(metabolite_concentrations[current_metabolite][0], metabolite_concentrations[current_metabolite][1])
                    ref_real, ref_imag = scale_complex_data(real, imag, scaling_factor / normalize_factor)
                    metabolite_spectra[current_metabolite] = ref_real +  1j*ref_imag
                    current_metabolite_ifft = f"{current_metabolite}_ifft"
                    metabolite_FID[current_metabolite_ifft] = ifft(fftshift(metabolite_spectra[current_metabolite])) # Restore the FID of the metabolite, reverse the fftshift to get correct FID
                    
                    
                    # Store the real and imaginary parts of the spectra
                    total_real_before_scaling += real
                    total_imag_before_scaling += imag
                    total_real += ref_real
                    total_imag += ref_imag
                    
                # Add Gaussian noise to the spectra
                SNR = np.random.uniform(10, 50)
                total_real_augmented, total_imag_augmented = add_gauss_noise_from_wanted_SNR(total_real, total_imag, SNR)

                # Create a baseline and add it to the augmented spectra
                input_range = np.linspace(-3.121, 12.528, 2048)
                n_gaussians = np.random.randint(5, 10)
                STD_RANGE = (0.5, 10)
                AMPLITUDE_RANGE = (-1/100000, 1/100000)
                
                baseline_real, _, _  = MM_baseline(ppm_range=input_range, max_signal=np.max(total_real_augmented), tolerance=0.1, jitter_range=(-0.1, 0.1))
                baseline_imag= create_baseline(input_range, n_gaussians=n_gaussians, std_range=STD_RANGE, amplitude_range=AMPLITUDE_RANGE)
                scaling_factor = np.random.uniform(0, 1) # Scale the baseline by a random factor to decrease the importance of the baseline and give the model more freedom to learn the metabolite spectra
                baseline_real*=scaling_factor
                baseline_imag*=scaling_factor
                total_real_augmented += baseline_real
                total_imag_augmented += baseline_imag

                complex_augmented = total_real_augmented + total_imag_augmented*1j
                complex_augmented_ifft = ifft(fftshift(complex_augmented)) # Restore the FID of the augmented spectrum, reverse the fftshift to get correct FID
            
                original = total_real + 1j*total_imag
                augmented = total_real_augmented + 1j*total_imag_augmented
                baseline = baseline_real + 1j*baseline_imag
                augmented_ifft = complex_augmented_ifft.real + 1j*complex_augmented_ifft.imag
                
                # Scale all the different spectra to the same range, based on the real part of the augmented spectrum
                max_val = np.max(total_real_augmented)
                min_val = 0# np.min(total_real_augmented)
                for key in metabolite_spectra.keys():
                    metabolite_spectra[key] = (metabolite_spectra[key] - min_val) / (max_val - min_val)
                    FID_key = f"{key}_ifft"
                    metabolite_FID[FID_key] = (metabolite_FID[FID_key] - min_val) / (max_val - min_val)
                original = (original - min_val) / (max_val - min_val)
                augmented = (augmented - min_val) / (max_val - min_val)
                baseline = (baseline - min_val) / (max_val - min_val)
                augmented_ifft = (augmented_ifft - min_val) / (max_val - min_val)
                
                original_ifft = np.fft.ifft(fftshift(original)) # Restore the FID of the original spectrum, reverse the fftshift to get correct FID
                if optimization:
                    # Estimate the poles and zeros of the filter
                    
                    optimizer = ComplexFilterOptimizer(input_signal=augmented_ifft, order=iir_order, options=optimizer_options)
                    b, a = optimizer.optimize()
                else:
                    b = np.zeros(2048)
                    a = np.zeros(2048)
                    optimizer_options = None
                    iir_order = 0
                    
                if BPF_filtering:
                    # Apply BPF to the augmented spectrum
                    ppm_range = np.array([12.528, -3.121])
                    
                    filtered_GABA, fft_filtered_GABA, impulse_response_GABA = BPF_ppm(augmented, ppm_range=ppm_range, ppm_pass=passband_GABA, margin=margin, gain_passband=pass_band_gain, ftype=ftype)
                    filtered_full, fft_filtered_full, impulse_response_full = BPF_ppm(augmented, ppm_range=ppm_range, ppm_pass=passband_full, margin=margin, gain_passband=pass_band_gain, ftype=ftype)
                else:
                    filtered_GABA = augmented
                    filtered_full = augmented
                    fft_filtered_GABA = fftshift(fft(filtered_GABA))
                    fft_filtered_full = fftshift(fft(filtered_full))
                    impulse_response = np.zeros(2048)
                    passband_GABA = np.zeros(2)
                    passband_full = np.zeros(2)
                    margin = 0
                    ftype = None
                metadata_list = [SNR, n_gaussians, STD_RANGE, AMPLITUDE_RANGE, max_val, min_val, optimizer_options, passband_GABA, passband_full, margin,ftype, ppm_range, linewidth]
                metadata_keys = ["SNR", "n_gaussians", "STD_RANGE", "AMPLITUDE_RANGE", "max_val", "min_val", "optimizer_options", "passband_GABA", "passband_full", "margin", "ftype", "ppm_range", "linewidth"]   
                # Save the generated spectra to a dictionary, and add the metabolite spectra
                
                # Format the index for the file name
                i_str = str(i).zfill(str(num_spectra).__len__())
                
                if save_format == "mat":
                    savedict = {
                        "original": original, 
                        "augmented": augmented, 
                        "baseline": baseline, 
                        "metadata": metadata_list, 
                        "metadata_keys": metadata_keys,
                    "augmented_ifft":augmented_ifft,
                        "a":a,
                        "b":b,
                        "iir_order": iir_order,
                        "original_ifft": original_ifft
                    }
                    for key in metabolite_spectra.keys():
                        savedict[key] = metabolite_spectra[key]
                    # Uncomment the following line to save the data to a .mat file
                    for key in metabolite_FID.keys():
                        savedict[key] = metabolite_FID[key]
                    savemat(os.path.join(base_path, save_folder, f"{i_str}_data.{save_format}"), savedict)
                    
                elif save_format == "npz":
                    metadata = {metadata_keys[i]: metadata_list[i] for i in range(len(metadata_list))}
                    savedict = {
                        "original": original, 
                        "augmented": augmented, 
                        "baseline": baseline, 
                        "metadata": str(metadata.items()),
                        "augmented_ifft":augmented_ifft,
                        "a":a,
                        "b":b, 
                        "original_ifft": original_ifft,
                        "iir_order": iir_order,
                        "filter_impulse_response_GABA": impulse_response_GABA,
                        "filter_impulse_response_full": impulse_response_full,
                        "filtered_GABA": filtered_GABA,
                        "filtered_full": filtered_full,
                        "fft_filtered_GABA": fft_filtered_GABA,
                        "fft_filtered_full": fft_filtered_full,
                        }
                    for key in metabolite_spectra.keys():
                        savedict[key] = metabolite_spectra[key]
                    for key in metabolite_FID.keys(): 
                        savedict[key] = metabolite_FID[key]
                    np.savez(os.path.join(base_path, save_folder, TE_lw, f"TE{TE}_lw{linewidth:02d}_{i_str}_data.{save_format}"), **savedict)
                
            

            # End timer and print elapsed time
            end = time.perf_counter()
            print(f"Time elapsed: {(end-start)//3600} h {((end-start)%3600)//60} m {round((end-start)%60, 2)} s")

            # load a sample data if load is true
            load = False
            if load:
                load_file = os.listdir(os.path.join(base_path, save_folder))[-1]
                print(f"Loading file {load_file}")
                test_data = loadmat(os.path.join(base_path, save_folder, load_file), struct_as_record=False)
                print(test_data.keys())
                print(f'metabolite_spectra keys: {type(test_data["Ala"])}')
                print(f'shape of metabolite spectra within: {test_data["Ala"].shape}')
                print(f'shape of original data: {test_data["original"].shape}')
                print(f'Max value of augmented: {np.max(test_data["augmented", "augmented_ifft"])}, Min value of augmented: {np.min(test_data["augmented"])}')
                print(test_data["metadata"])

if __name__ == "__main__":
    main()