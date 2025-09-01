import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tick

import re
from myfunctions_tf import BPF_ppm, load_from_directory, multi_channel_cnn, ComplexFilterOptimizer, input_ouput_key_sorting, model_input_output_prep, load_mrui_txt_data, postprocess_predicted_data

path_water = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/mrui_files/G8_P/S01.txt"
path_HLSVD = "/home/stud/casperc/bhome/Project3_DL_sigpros/in_vivo_data/mrui_files/G8_P/S01.txt"
paths = [path_HLSVD, path_water]
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

ppm = np.linspace(-3.121, 12.528, 2048)
ppm_where = np.where((ppm >= 0.2) & (ppm <= 4.2))

passband_GABA =  np.array([2.1, 2.7]) # ppm
passband_full = np.array([1.6, 4.2]) # ppm
margin = 1
ftype = "butter"
pass_band_gain = 2

num_col = (len(paths)*len(model_weights_paths.keys()))
print(num_col)
plt.figure(figsize=(num_col*15, 45))
for model_configureation in model_weights_paths.keys():
    for path in paths:
        FID, freq, metadata = load_mrui_txt_data(path)
        norm_freq = freq / np.max(freq)
        norm_FID_iift = np.fft.ifft(norm_freq)
        norm_freq_fft = np.fft.fftshift(np.fft.fft(FID))/np.max(np.fft.fftshift(np.fft.fft(FID)))
        norm_FID = np.fft.ifft(norm_freq_fft)
        if "HLSVD" in path:
            i = 0
        else:
            i = 1
        
        optim = ComplexFilterOptimizer(input_signal=norm_FID, order=60, options={'method':'SLSQP', 'ftol':1e-3, 'maxiter':50})
        b_opt, a_opt = optim.optimize()
        a_opt, b_opt = np.pad(a_opt, (0,2048-len(a_opt))), np.pad(b_opt, (0,2048-len(b_opt)))
        filtered_GABA, fft_filtered_GABA, impulse_response_GABA = BPF_ppm(fft_signal=norm_freq_fft, ppm_range=ppm, ppm_pass=passband_GABA, margin=margin, ftype=ftype, gain_passband=pass_band_gain)
        filtered_full, fft_filtered_full, impulse_response_full = BPF_ppm(fft_signal=norm_freq_fft, ppm_range=ppm, ppm_pass=passband_full, margin=margin, ftype=ftype, gain_passband=pass_band_gain)
        print(model_configureation)
        if model_configureation == "FID":
            in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag]).T, axis=0)
            j=1
        elif model_configureation == "FID_gaba":
            in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, filtered_GABA.real, filtered_GABA.imag]).T, axis=0)
            j=2
        elif model_configureation == "FID_full_filt":
            in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, filtered_full.real, filtered_full.imag]).T, axis=0)
            j=3
        elif model_configureation == "FID_ab_gaba":
            in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real,
                                                    b_opt.imag, norm_FID.real, norm_FID.imag]).T, axis=0)
            j=4
        elif model_configureation == "FID_ab_full_filt":
            in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real,
                                                    b_opt.imag, filtered_full.real, filtered_full.imag]).T, axis=0)
            j=5
        elif model_configureation == "FID_ab":
            in_vivo_input = np.expand_dims(np.array([norm_FID.real, norm_FID.imag, a_opt.real, a_opt.imag, b_opt.real,
                                                    b_opt.imag]).T, axis=0)
            j=6
        
        
        available_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(available_devices[6], 'GPU')
        print(f"visible devices: {tf.config.get_visible_devices()}")
        input_keys, output_keys = input_ouput_key_sorting(model_weights_paths[model_configureation])
        model = multi_channel_cnn(input_shape = (2048,2*len(input_keys)), output_shape=(2048, 2*len(output_keys) ), num_blocks=20, filter1=32, kernel_size=16, print_summary=False)
        model.build(input_shape=(None, 2048, 2*len(input_keys)))
        model.load_weights(model_weights_paths[model_configureation])
        
        
        in_vivo_output = np.squeeze(model.predict(in_vivo_input), axis=0)
        in_vivo_output_dict = postprocess_predicted_data(in_vivo_output, output_keys)
        #print(f"j: {j}, i: {i}, 2*j-1+i = {2*j-1+i}")
        plt.subplot(3, num_col, 2*j-1+i)
        plt.plot(ppm, norm_freq.real, label="MR spectrum", color='black')
        plt.plot(ppm, in_vivo_output_dict["original"].real, label="predicted", color='magenta')
        plt.title(f"{model_configureation} {'- HLSVD' if i == 0 else ''}")
        plt.yticks([])
        plt.xticks(np.arange(-3, 12, 1), [str(num) for num in np.arange(-3,12)], fontsize=24)
        plt.xlim(12.5, -3.2)
        plt.ylim(-0.2, 1)
        # Set major and minor ticks
        plt.gca().xaxis.set_major_locator(tick.MultipleLocator(0.5))
        plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(0.1))
        plt.tick_params(axis='x', which='major', length=10, width=2, direction='in', labelsize=24)
        plt.tick_params(axis='x', which='minor', length=5, width=2, direction='in', labelsize=24)
        plt.gca().invert_xaxis()
        # Remove the box around the plot
        ax = plt.gca()
        ax.spines['top'].set_visible(False)    # Remove the top border
        ax.spines['right'].set_visible(False)  # Remove the right border
        ax.spines['left'].set_visible(False)   # Remove the left border
        ax.spines['bottom'].set_visible(True) # Keep the x-axis
        print(f"j: {j}, i: {i}, 2*j-1+i+num_col = {2*j-1+i+num_col}")
        plt.subplot(3, num_col, 2*j-1+i+num_col)
        plt.plot(ppm[ppm_where], norm_freq[ppm_where].real, label="MR spectrum", color='black')
        plt.plot(ppm[ppm_where], in_vivo_output_dict["original"][ppm_where].real, '--', label="predicted", color='magenta')
        plt.yticks([])

        plt.xticks(np.arange(0, 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        plt.xlim(4.2, 0)
        plt.ylim(-0.2, 1)
        # Set major and minor ticks
        plt.gca().xaxis.set_major_locator(tick.MultipleLocator(0.5))
        plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(0.1))
        
        plt.tick_params(axis='x', which='major', length=10, width=2, direction='in', labelsize=24)
        plt.tick_params(axis='x', which='minor', length=5, width=2, direction='in', labelsize=24)
        plt.gca().invert_xaxis()
        # Remove the box around the plot
        ax = plt.gca()
        ax.spines['top'].set_visible(False)    # Remove the top border
        ax.spines['right'].set_visible(False)  # Remove the right border
        ax.spines['left'].set_visible(False)   # Remove the left border
        ax.spines['bottom'].set_visible(True) # Keep the x-axis
        #print(f"j: {j}, i: {i}, 2*j-1+i+2*num_col = {2*j-1+i+2*num_col}")
        plt.subplot(3, num_col, 2*j-1+i+2*num_col)
        plt.plot(ppm[ppm_where], norm_freq[ppm_where].real, label="MR spectrum", color='black')
        plt.plot(ppm[ppm_where], in_vivo_output_dict["baseline"][ppm_where].real, label="predicted baseline",color='blue')
        plt.plot(ppm[ppm_where], in_vivo_output_dict["original"][ppm_where].real+in_vivo_output_dict["baseline"][ppm_where].real, '--', label="predicted original + baseline", color='magenta')
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])

        plt.xticks(np.arange(0, 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        plt.xlim(4.2, 0)    
        plt.ylim(-0.2, 1)
        # Set major and minor ticks
        plt.gca().xaxis.set_major_locator(tick.MultipleLocator(0.5))
        plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(0.1))
        
        plt.tick_params(axis='x', which='major', length=10, width=2, direction='in', labelsize=24)
        plt.tick_params(axis='x', which='minor', length=5, width=2, direction='in', labelsize=24)
        plt.gca().invert_xaxis()
        # Remove the box around the plot
        ax = plt.gca()
        ax.spines['top'].set_visible(False)    # Remove the top border
        ax.spines['right'].set_visible(False)  # Remove the right border
        ax.spines['left'].set_visible(False)   # Remove the left border
        ax.spines['bottom'].set_visible(True) # Keep the x-axis
        
plt.savefig("figures/test_all_models_GE.png")