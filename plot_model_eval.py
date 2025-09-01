import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tick
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.colors as mcolors

import re
from myfunctions_tf import BPF_ppm, load_from_directory, multi_channel_cnn, ComplexFilterOptimizer, input_ouput_key_sorting, model_input_output_prep, load_mrui_txt_data, make_current_time_directory, MultiColorLegendHandler

# Function to adjust brightness of a color
def adjust_brightness(color, factor=1.2):
    rgb = mcolors.to_rgb(color)  # Convert to RGB
    adjusted_rgb = tuple(min(1, c * factor) for c in rgb)  # Scale each channel
    return adjusted_rgb

font = {'family' : 'serif',
            'weight':'normal',
            'size'   : 36}
matplotlib.rc('font', **font)
plt.rcParams['lines.linewidth'] = 3


available_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(available_devices[6], 'GPU')
print(f"visible devices: {tf.config.get_visible_devices()}")

FID_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_23-21_37_38_['augmented_ifft']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_ab_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_23-21_37_44_['augmented_ifft', 'a', 'b']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_full_filt_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_24-10_52_49_['augmented_ifft', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
FID_ab_full_filt_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_03_24-09_56_20_['augmented_ifft', 'a', 'b', 'filtered_full']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"
ab_weights_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/tf_experiments/final/2025_05_12-16_18_06_['a', 'b']_['original', 'baseline', 'NAA', 'NAAG', 'Cr', 'PCr', 'PCho', 'GPC', 'GABA', 'Gln', 'Glu']_ResNet1D_RMSprop_Huber/checkpoint.hdf5"

model_weights_paths = {
    "FID_full_filt": FID_full_filt_weights_path,
    "FID_ab_full_filt": FID_ab_full_filt_weights_path,
    "FID_ab": FID_ab_weights_path,
    "FID": FID_weights_path,
    "ab": ab_weights_path
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
    filtered_full, fft_filtered_full, impulse_response_full = BPF_ppm(fft_signal=data_dict[key]["augmented"], ppm_range=ppm_full, ppm_pass=passband_full, margin=margin, ftype=ftype, gain_passband=pass_band_gain)
    data_dict[key]["filtered_full"] = filtered_full
    data_dict[key]["impulse_response_full"] = impulse_response_full
    data_dict[key]["fft_filtered_full"] = fft_filtered_full

time_dir = make_current_time_directory(main_folder_path="/home/stud/casperc/bhome/Project3_DL_sigpros/figures", data_description=f"model_eval", make_logs_dir=False)

# Define colors for metabolites
metabolite_colors = {
    "origignal": "#ff7f0e", # Orange
    "Glu": "#008080",  # Teal
    "Gln": "#2ca02c", # Green
    "Cr": "#d62728", # Red
    "NAA": "#9467bd", # Purple
    "NAAG": "#8c564b" # Brown
}

# Generate lighter/darker colors for predictions
prediction_colors = {met: adjust_brightness(color, factor=0.7) for met, color in metabolite_colors.items()}

for TE_lw in list(data_dict.keys()):
    print(TE_lw)
    plt.figure(figsize=(40, 15))
    for model_condfiguration in model_weights_paths.keys():
        if model_condfiguration == "FID":
            subplot_id = 1
            title = "FID"
        elif model_condfiguration == "FID_ab":
            subplot_id = 2
            title = "FID + a,b"
        elif model_condfiguration == "FID_full_filt":
            subplot_id = 3
            title = "FID + filtered FID"
        elif model_condfiguration == "FID_ab_full_filt":
            subplot_id = 4
            title = "FID + a,b + filtered FID"
        elif model_condfiguration == "ab":
            subplot_id = 5
            title = "a,b"
        else:
            print(f"Model configuration {model_condfiguration} not included in the plot.")
            continue
        # compile model and prepare
        input_keys, output_keys = input_ouput_key_sorting(model_weights_paths[model_condfiguration])
        model = multi_channel_cnn(input_shape = (2048,2*len(input_keys)), output_shape=(2048, 2*len(output_keys) ), num_blocks=20, filter1=32, kernel_size=16, print_summary=False)
        model.build(input_shape=(None, 2048, 2))
        model.load_weights(model_weights_paths[model_condfiguration])
        
        # Prepare input and output for model
        input_data, _ = model_input_output_prep(data_dict[TE_lw], input_keys, output_keys=output_keys)
        output = model.predict(input_data)
        # Process the output to match the format of output_keys
        output_dict = {}
        for i, output_key in enumerate(output_keys):
            real_part = output[:, :, 2 * i]
            imag_part = output[:, :, 2 * i + 1]
            output_dict[output_key] = (real_part + 1j * imag_part).squeeze().T
        # plot data
        plt.subplot(1, 5, subplot_id)
        
        # Create dummy handles for the legend
        handles = [
            Line2D([], [], label='GT'),
            Line2D([], [], label='Predicted'),
            Line2D([], [], label='MR spectrum', color='black', linewidth=5)
            ]
        plt.legend(handles=handles, loc='upper left', fontsize=24, handler_map={handles[2]: HandlerLine2D(), handles[0]: MultiColorLegendHandler(colormap='rainbow', num_segments=4, linestyle='-', linewidth=5), handles[1]: MultiColorLegendHandler(colormap='rainbow', num_segments=3,linewidth=5, linestyle=(0, (2, 4)))}, borderaxespad=2.0)
        plt.plot(ppm, data_dict[TE_lw]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[TE_lw]["original"].real[ppm_where], linestyle="-", color=metabolite_colors["origignal"])
        plt.plot(ppm, output_dict["original"].real[ppm_where], linestyle="--", color=prediction_colors["origignal"])
        metabolite_names = ["Glu", "Gln", "NAAG", "NAA", "Cr"]
        position = -0.1
        for factor, met in enumerate(metabolite_names, start=1):
            y_pos = position * factor
            plt.plot(ppm, data_dict[TE_lw][met].real[ppm_where]+y_pos, linestyle="-", color=metabolite_colors[met])
            plt.plot(ppm, output_dict[met].real[ppm_where]+y_pos, linestyle="--", color=prediction_colors[met])
            if subplot_id == 1:   
                plt.text(4.3, y_pos, met, fontsize=32, verticalalignment='center', horizontalalignment='right')  # Adjust x=4.3 to position text outside the plot
            
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0, 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=32)
        plt.ylim(-0.55, 1)
        # Set major and minor ticks
        plt.gca().xaxis.set_major_locator(tick.MultipleLocator(0.5))
        plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(0.1))
        
        plt.tick_params(axis='x', which='major', length=10, width=2, direction='in', labelsize=32)
        plt.tick_params(axis='x', which='minor', length=5, width=2, direction='in', labelsize=32)
        plt.gca().invert_xaxis()
        # Remove the box around the plot
        ax = plt.gca()
        ax.spines['top'].set_visible(False)    # Remove the top border
        ax.spines['right'].set_visible(False)  # Remove the right border
        ax.spines['left'].set_visible(False)   # Remove the left border
        ax.spines['bottom'].set_visible(True) # Keep the x-axis
        plt.title(f"{title}")
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.1, hspace=0.05, wspace=0.025)
    plt.savefig(os.path.join(os.getcwd(), time_dir, f"{TE_lw}.png"), dpi=300)
""" for id_model, model_condfiguration in enumerate(model_weights_paths.keys(), start=1):
    input_keys, output_keys = input_ouput_key_sorting(model_weights_paths[model_condfiguration])
    # compile model
    model = multi_channel_cnn(input_shape = (2048,2*len(input_keys)), output_shape=(2048, 2*len(output_keys) ), num_blocks=20, filter1=32, kernel_size=16, print_summary=False)

    model.build(input_shape=(None, 2048, 2))
    model.load_weights(model_weights_paths[model_condfiguration])

    # Prepare input and output for model

    print(data_dict.keys())

    
    model_condfiguration_path = os.path.join(time_dir, model_condfiguration)
    os.makedirs(model_condfiguration_path, exist_ok=True)
    print(f"########## {model_condfiguration} ##########")
    for TE_lw in list(data_dict.keys()):
        print(TE_lw)
        #os.makedirs(os.path.join(model_condfiguration_path, TE_lw), exist_ok=True)

        input_data, _ = model_input_output_prep(data_dict[TE_lw], input_keys, output_keys=output_keys)
        #print(data_dict[key].keys())
        #print(f"input data shape: {input_data.shape}")
        output = model.predict(input_data)
        # Process the output to match the format of output_keys
        output_dict = {}
        for i, output_key in enumerate(output_keys):
            real_part = output[:, :, 2 * i]  # Extract real part
            imag_part = output[:, :, 2 * i + 1]  # Extract imaginary part
            output_dict[output_key] = (real_part + 1j * imag_part).squeeze().T  # Combine into complex numbers
            

            
        # plot data
        plt.figure(figsize=(15, 45))
        plt.subplot(3,1,1)
        plt.plot(ppm, data_dict[TE_lw]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[TE_lw]["original"].real[ppm_where], label="GT-Full", color='magenta', linestyle="--")
        plt.plot(ppm, output_dict["original"].real[ppm_where], label="DL-Full", color='cyan', linestyle="--")
        plt.legend(fontsize=24)
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        plt.title(f"{model_condfiguration}-{TE_lw} Synthetic")
        plt.gca().invert_xaxis()
        
        plt.subplot(3,1,2)
        plt.plot(ppm, data_dict[TE_lw]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[TE_lw]["NAA"].real[ppm_where], label="GT-NAA", color='magenta', linestyle="--")
        plt.plot(ppm, output_dict["NAA"].real[ppm_where], label="DL-NAA", color='cyan', linestyle="--")
        plt.legend(fontsize=24)
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        #plt.title(f"Real part of {TE_lw} NAA")
        plt.gca().invert_xaxis()
        
        plt.subplot(3,1,3)
        plt.plot(ppm, data_dict[TE_lw]["augmented"].real[ppm_where], label="MR spectrum", color='black')
        plt.plot(ppm, data_dict[TE_lw]["GABA"].real[ppm_where], label="GT-GABA", color='magenta', linestyle="--")
        plt.plot(ppm, output_dict["GABA"].real[ppm_where], label="DL-GABA",color='cyan', linestyle="--")
        plt.legend(fontsize=24)
        plt.xlabel("Chemical shift [ppm]")
        plt.yticks([])
        plt.xticks(np.arange(0., 4.2, 0.5), ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'], fontsize=24)
        plt.gca().invert_xaxis()
        plt.title(f"{TE_lw} GABA")
        plt.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.05)
        plt.savefig(os.path.join(os.getcwd(), model_condfiguration_path, f"{TE_lw}_real.png"), dpi=500)
        plt.close() """