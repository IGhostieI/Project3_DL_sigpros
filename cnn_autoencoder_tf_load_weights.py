import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging,

from matplotlib.style import available
import numpy as np
from typing import Tuple, List
import scipy.stats as stats

from myfunctions_tf import CNN_DenoiseFit_Functional, CNN_Autoencoder_Functional, load_from_directory, make_current_time_directory, read_LCModel_coord, load_mrui_txt_data, ResNet1D, BPF_ppm, ComplexFilterOptimizer, input_ouput_key_sorting

from datetime import datetime
import scipy.signal as signal 
from scipy.fft import fft, fftshift, ifft
import matplotlib
import matplotlib.pyplot as plt
import time
import re

font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 24}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (18,10)

import tensorflow as tf
tf.keras.backend.clear_session()
print("###########################")
print("TensorFlow version:", tf.__version__)
print("###########################")


    

def main():
    available_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(available_devices[0], 'GPU')
    print(f"visible devices: {tf.config.get_visible_devices()}")
    weights = "/home/stud/casperc/bhome/Project1(DeepFit)/tf_experiments/Final_Experiments/2025_01_21-23_43_54_['augmented_ifft']_['original', 'baseline', 'NAA', 'Cr', 'PCho', 'GABA', 'Glu', 'Gln']_ResNet1D/checkpoint.hdf5"
    test_data = True
    save_path = "/home/stud/casperc/bhome/Project1(DeepFit)/tf_load_weights_output"
    folder_description = "amp_normalized"
    if test_data:
        folder_description += "_test_data"
    else:
        folder_description += "_sine_wave"
    save_folder = make_current_time_directory(save_path, folder_description, make_logs_dir=False)
    
    # Check if the weights file exists
    if not os.path.exists(weights):
        print(f"Error: The weights file at {weights} does not exist.")
        return
    # Load the data
    ppm = np.linspace(12.528, -3.121, 2048)
    if test_data:
        data_path = "/home/stud/casperc/bhome/Project1(DeepFit)/SUBJECTS/GLUPI_S01/S01_mrui_HLSVD.txt"
        input_complex, freq, _ = load_mrui_txt_data(data_path)
        max_val = np.max(freq)
        
        input_complex_scaled = input_complex / max_val
    
    else:
        sine_wave = np.sin(2 * np.pi * 1 * ppm)
        input_real = sine_wave.copy()
        input_imag = sine_wave.copy()
        input_complex = input_real+ 1j*input_imag
        max_val = np.max(input_complex.real)
        
        
        input_complex_scaled = input_complex / max_val
    
    # load and build the model
    model = ResNet1D(num_blocks=20, kernel_size=8, input_shape=(2048,2), output_shape=(2048,16))
    
    model.build(input_shape=(None, 2048, 2))
    model.load_weights(weights)
    
    ppm_range = np.array([-3.121, 12.528])
    passband =  np.array([2.1, 2.7]) # ppm
    stopband =  np.array([1.6, 3.2]) # ppm (Change to fix the stopband)
    ftype = "butter"
    pass_band_gain = 2
    filtered_signal, fft_filtered_signal, impulse_response = BPF_ppm(input_complex_scaled, ppm_range=ppm_range, ppm_pass=passband, ppm_stop=stopband, gain_passband=pass_band_gain, ftype=ftype)
    
    iir_order = 60
    start = time.perf_counter()
    optimizer_options = {'method':'SLSQP', 'ftol':1e-6, 'maxiter':1000}
    
    optimizer = ComplexFilterOptimizer(input_signal=input_complex_scaled, order=iir_order, options=optimizer_options)
    print("Optimizing the filter")
    b, a = optimizer.optimize()
    estimate = signal.lfilter(b, a, input_complex_scaled)
    
    end = time.perf_counter()
    print(f"Time taken to optimize the filter: {end-start:.2f} seconds")
    a, b = np.pad(a, (0,2048-len(a))), np.pad(b, (0,2048-len(b)))
    
    model_input= np.array([input_complex_scaled.real, input_complex_scaled.imag]).T #, input_complex_scaled.imag, a.real, a.imag, b.real, b.imag, filtered_signal.real, filtered_signal.imag
    print(model_input.shape)
    
    predicted = np.squeeze(model.predict(tf.expand_dims(model_input, axis=0)), axis=0)
    print(f"prediction shape: {predicted.shape}")
    
    # 'augmented_ifft', 'a', 'b', 'filtered_signal'
    #"original", "baseline", "NAA","Cr", "PCho", "GABA","Glu","Gln" 
    
    input_keys, output_keys = input_ouput_key_sorting(weights)
    
    output = {}
    for i, key in enumerate(output_keys):
        output[f"{key}_real"]=predicted[:,2*i]
        output[f"{key}_imag"]=predicted[:,2*i+1] 
        
    ppm = np.linspace(-3.121, 12.528,  2048)
    plt.figure(figsize=(20, 200))
    plt.subplot(10,1,1)
    plt.plot(input_complex_scaled.real, label="Input Real", color="black")
    plt.plot(estimate.real, label="Estimated Real", linestyle="--", color="red")
    plt.title("Input and signal processing estimated signal time domain")
    plt.legend()
    plt.subplot(10,1,2)
    plt.plot(fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black")#ppm, 
    plt.plot(fftshift(fft(estimate)).real, label="Estimated Real", linestyle="--", color="red")#ppm, 
    plt.title("Input and signal processing estimated signal frequency domain")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.subplot(10,1,3)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).__abs__(), label="|Input|", color="black")
    plt.plot(ppm, fftshift(fft(filtered_signal)).__abs__(), label="|Filtered|", color="red")
    
    plt.title("Filtered signal and impulse response")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.subplot(10,1,4)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, output["baseline_real"], label="estimated baseline", linestyle="--", color="magenta")
    plt.plot(ppm, output["original_real"], label="Predicted Real", linestyle="-", color="cyan")
    plt.title("Full spectrrum fitted signal with baseline estimation")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.subplot(10,1,5)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real-output["baseline_real"], label="Input with estimated baseline correction", linestyle="--", color="magenta")
    plt.plot(ppm, output["NAA_real"], label="NAA", linestyle="-", color="cyan")
    plt.legend()
    plt.title("NAA fitting")
    plt.gca().invert_xaxis()
    plt.subplot(10,1,6)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real-output["baseline_real"], label="Input with estimated baseline correction", linestyle="--", color="magenta")
    plt.plot(ppm, output["Cr_real"], label="Cr", linestyle="-", color="cyan")
    plt.legend()
    plt.title("Cr fitting")
    plt.gca().invert_xaxis()
    plt.subplot(10,1,7)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real-output["baseline_real"], label="Input with estimated baseline correction", linestyle="--", color="magenta")
    plt.plot(ppm, output["PCho_real"], label="PCho", linestyle="-", color="cyan")
    plt.legend()
    plt.title("PCho fitting")
    plt.gca().invert_xaxis()
    plt.subplot(10,1,8)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real-output["baseline_real"], label="Input with estimated baseline correction", linestyle="--", color="magenta")
    plt.plot(ppm, output["GABA_real"], label="GABA", linestyle="-", color="cyan")
    plt.legend()
    plt.title("GABA fitting")
    plt.gca().invert_xaxis()
    plt.subplot(10,1,9)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real-output["baseline_real"], label="Input with estimated baseline correction", linestyle="--", color="magenta")
    plt.plot(ppm, output["Glu_real"], label="Glu", linestyle="-", color="cyan")
    plt.legend()
    plt.title("Glu fitting")
    plt.gca().invert_xaxis()
    plt.subplot(10,1,10)
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real, label="Input Real", color="black", linestyle="--")
    plt.plot(ppm, fftshift(fft(input_complex_scaled)).real-output["baseline_real"], label="Input with estimated baseline correction", linestyle="--", color="magenta")
    plt.plot(ppm, output["Gln_real"], label="Gln", linestyle="-", color="cyan")
    plt.legend()
    plt.title("Gln fitting")
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(save_folder, "output.png"))
    plt.close()

if __name__ == "__main__":
    main()