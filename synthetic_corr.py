import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tick

import re
from myfunctions_tf import BPF_ppm, load_from_directory, multi_channel_cnn, ComplexFilterOptimizer, input_ouput_key_sorting, model_input_output_prep, load_mrui_txt_data, make_current_time_directory, load_NPZ_from_list

import pandas as pd

FID_gaba_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_24-10_54_09_['augmented_ifft', 'filtered_GABA']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_full_filt_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_24-10_52_49_['augmented_ifft', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_ab_gaba_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_23-21_40_40_['augmented_ifft', 'a', 'b', 'filtered_GABA']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_ab_full_filt_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_24-09_56_20_['augmented_ifft', 'a', 'b', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_ab_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_23-21_37_44_['augmented_ifft', 'a', 'b']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/2025_03_23-21_37_38_['augmented_ifft']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"

model_weights_paths = {
    "FID_gaba": FID_gaba_weights_path,
    "FID_full_filt": FID_full_filt_weights_path,
    "FID_ab_gaba": FID_ab_gaba_weights_path,
    "FID_ab_full_filt": FID_ab_full_filt_weights_path,
    "FID_ab": FID_ab_weights_path,
    "FID": FID_weights_path
}

# Load synthetic data
## TE20
base_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/generated_data/train"
data_list = sorted(os.listdir(base_path))
TEs = [20, 30, 40]
lws = [3, 5, 10]
data_dict = {}
ppm_full = np.linspace(-3.121, 12.528, 2048)
ppm_where = np.where((ppm_full >= 0.2) & (ppm_full <= 4.2))[0]
ppm = ppm_full[ppm_where]

passband_GABA =  np.array([2.1, 2.7]) # ppm
passband_full = np.array([1.6, 4.2]) # ppm
margin = 1
ftype = "butter"
pass_band_gain = 2

data_dict = {}
for TE in TEs:
    for lw in lws:
        pattern = f"TE{TE}_lw{lw:02d}.*"
        files = [data for data in data_list if re.match(pattern, data)][:1000]
        data_list = []
        for file in files:
            data_list.append(os.path.join(base_path, file))
    input_data, output_data = load_NPZ_from_list(data_list, ppm_range=ppm, ppm_where=ppm_where)
for key in data_dict.keys():
    optim = ComplexFilterOptimizer(input_signal=data_dict[key]["augmented_ifft"], order=60, options={'method':'SLSQP', 'ftol':1e-3, 'maxiter':50})
    b_opt, a_opt = optim.optimize()
    data_dict[key]["a"] = a_opt
    data_dict[key]["b"] = b_opt
    filtered_GABA, fft_filtered_GABA, impulse_response_GABA = BPF_ppm(fft_signal=data_dict[key]["augmented"], ppm_range=ppm_full, ppm_pass=passband_GABA, margin=margin, ftype=ftype, gain_passband=pass_band_gain)
    filtered_full, fft_filtered_full, impulse_response_full = BPF_ppm(fft_signal=data_dict[key]["augmented"], ppm_range=ppm_full, ppm_pass=passband_full, margin=margin, ftype=ftype, gain_passband=pass_band_gain)
    data_dict[key]["filtered_GABA"] = filtered_GABA
    data_dict[key]["filtered_full"] = filtered_full
    data_dict[key]["impulse_response_GABA"] = impulse_response_GABA
    data_dict[key]["impulse_response_full"] = impulse_response_full
    data_dict[key]["fft_filtered_GABA"] = fft_filtered_GABA
    data_dict[key]["fft_filtered_full"] = fft_filtered_full

time_dir = make_current_time_directory(main_folder_path="/home/stud/casperc/bhome/Project3_DL_sigpros/figures", data_description=f"model_eval", make_logs_dir=False)
