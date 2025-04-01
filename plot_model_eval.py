import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tick
import re
from myfunctions_tf import BPF_ppm, load_from_directory, multi_channel_cnn, ComplexFilterOptimizer, input_ouput_key_sorting, model_input_output_prep, load_mrui_txt_data, make_current_time_directory


font = {'family' : 'serif',
            'weight':'normal',
            'size'   : 36}
matplotlib.rc('font', **font)
plt.rcParams['lines.linewidth'] = 3


available_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(available_devices[6], 'GPU')
print(f"visible devices: {tf.config.get_visible_devices()}")

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


for TE in TEs:
    for lw in lws:
        pattern = f"TE{TE}_lw{lw:02d}.*"
        file = next((data for data in data_list if re.match(pattern, data)), None)
        NPZ_data = np.load(os.path.join(base_path, file))
        data_dict[f"TE{TE}_lw{lw}"] = dict(NPZ_data)
        print(f"Loaded {file}")

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

for model_condfiguration in model_weights_paths.keys():
    input_keys, output_keys = input_ouput_key_sorting(model_weights_paths[model_condfiguration])
    # compile model
    model = multi_channel_cnn(input_shape = (2048,2*len(input_keys)), output_shape=(2048, 2*len(output_keys) ), num_blocks=20, filter1=32, kernel_size=16, print_summary=False)

    model.build(input_shape=(None, 2048, 2))
    model.load_weights(model_weights_paths[model_condfiguration])

    """ input_data, output_data = load_from_directory(path="/home/stud/casperc/bhome/Project3_DL_sigpros/generated_data/train_2", input_keys=input_keys, output_keys=output_keys, num_signals=1)
    output = model_FID.predict(input_data)
    output_dict = dict(zip(output_keys, output.T)) 
    plt.figure()
    plt.plot(ppm, output_dict["original"].__abs__(), label="Augmented") """
    # Prepare input and output for model

    print(data_dict.keys())

    
    model_condfiguration_path = os.path.join(time_dir, model_condfiguration)
    os.makedirs(model_condfiguration_path, exist_ok=True)
    print(f"########## {model_condfiguration} ##########")
    for key in list(data_dict.keys()):
        print(key)
        #os.makedirs(os.path.join(model_condfiguration_path, key), exist_ok=True)

        input_data, _ = model_input_output_prep(data_dict[key], input_keys, output_keys=output_keys)
        #print(data_dict[key].keys())
        #print(f"input data shape: {input_data.shape}")
        output = model.predict(input_data)
        # Process the output to match the format of output_keys
        output_dict = {}
        for i, output_key in enumerate(output_keys):
            real_part = output[:, :, 2 * i]  # Extract real part
            imag_part = output[:, :, 2 * i + 1]  # Extract imaginary part
            output_dict[output_key] = (real_part + 1j * imag_part).squeeze().T  # Combine into complex numbers
            
            """for k in output_dict.keys():
            print(k)
            plt.figure()
            plt.plot(ppm, output_dict[k].real[ppm_where], label=k)	
            plt.plot(ppm, data_dict[key][k].real[ppm_where], label="GT", linestyle="--")
            plt.legend()
            plt.xlabel("Chemical shift [ppm]")
            plt.title(f"Real part of {k}")
            plt.gca().invert_xaxis()
            plt.title(f"Real part of {k}")
            plt.savefig(os.path.join(os.getcwd(), time_dir, key, f"{key}_{k}_real.png"), dpi=200) """
            
        # plot data
        plt.figure(figsize=(15, 45))
        plt.subplot(3,1,1)
        plt.plot(ppm, data_dict[key]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[key]["original"].real[ppm_where], label="GT-Full", color='magenta', linestyle="--")
        plt.plot(ppm, output_dict["original"].real[ppm_where], label="DL-Full", color='cyan', linestyle="--")
        plt.legend(fontsize=24)
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        plt.title(f"{model_condfiguration}-{key} Synthetic")
        plt.gca().invert_xaxis()
        
        plt.subplot(3,1,2)
        plt.plot(ppm, data_dict[key]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[key]["NAA"].real[ppm_where], label="GT-NAA", color='magenta', linestyle="--")
        plt.plot(ppm, output_dict["NAA"].real[ppm_where], label="DL-NAA", color='cyan', linestyle="--")
        plt.legend(fontsize=24)
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        #plt.title(f"Real part of {key} NAA")
        plt.gca().invert_xaxis()
        
        plt.subplot(3,1,3)
        plt.plot(ppm, data_dict[key]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[key]["GABA"].real[ppm_where], label="GT-GABA", color='magenta', linestyle="--")
        plt.plot(ppm, output_dict["GABA"].real[ppm_where], label="DL-GABA",color='cyan', linestyle="--")
        plt.legend(fontsize=24)
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        plt.gca().invert_xaxis()
        plt.title(f"{key} GABA")
        plt.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.05)
        plt.savefig(os.path.join(os.getcwd(), model_condfiguration_path, f"{key}_real.png"), dpi=500)
        plt.close()




""" plt.figure(figsize=(15, 15))
plt.plot(ppm, input_signal[ppm_where], label="input", color='black')
plt.plot(ppm, validation_output[0,:,0][ppm_where]+0.3, label="GT", color='magenta', linestyle="--")
plt.legend(fontsize=24)
plt.xlabel("Chemical shift [ppm]")
plt.yticks([])
plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)

# Set major and minor ticks
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.gca().invert_xaxis()



plt.legend()
plt.xlabel("Chemical Shift [ppm]")
#plt.title(f"Syntheic data")

plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.15)

plt.savefig(f"test_susynth_GT_.png", dpi=500)
plt.close()

plt.figure(figsize=(15, 15))
plt.title("Synthetic data")
plt.plot(ppm, input_signal[ppm_where], label="input", color='black')
plt.plot(ppm, fit[:,0][ppm_where]+0.3, label="DL Model", color='cyan', linestyle="--")
plt.legend(fontsize=24)
plt.xlabel("Chemical shift [ppm]")
plt.yticks([])
plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)

# Set major and minor ticks
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.gca().invert_xaxis()



plt.legend()
plt.xlabel("Chemical Shift [ppm]")
plt.title(f"Syntheic data")

plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.15)

plt.savefig(f"test_susynth_.png", dpi=500)
plt.close() """