from myfunctions_tf import load_mrui_txt_data, BPF_ppm
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import os 
from typing import Union



def main():
    path = "/home/stud/casperc/bhome/Project1(DeepFit)/SUBJECTS/GLUPI_S01/S01_mrui.txt"
    if path.endswith(".npz"):
        data = np.load(path)
        fft_signal = data["augmented"][:2048] + 1j * data["augmented"][2048:]
        ifft_signal = data["augmented_ifft"][:2048] + 1j * data["augmented_ifft"][2048:]
    elif path.endswith(".txt"):
        sig_real, sig_imag, fft_real, fft_imag, metadata = load_mrui_txt_data(path)
        fft_signal = fft_real + 1j * fft_imag
        ifft_signal = sig_real + 1j * sig_imag
    else:
        print("File format not supported")
        
    ppm_range = np.array([12.528, -3.121])
    ppm_pass = np.array([2.1, 2.9])
    ppm_stop = np.array([1.5, 4])
    ppm = np.linspace(ppm_range.max(), ppm_range.min(), 2048)
    filtered_signal, fft_filtered_signal, complex_response = BPF_ppm(fft_signal, ppm_range, ppm_pass, ppm_stop)
    
    
    # Set the general font size
    plt.rcParams.update({'font.size': 20, 'lines.linewidth': 2})
    plt.figure(figsize=(40, 30))
    # Plot time domain
    plt.subplot(3, 3, 1)
    plt.title(f"Time Domain")
    plt.plot(np.real(ifft_signal[:200]), label="Original (Real)", linewidth=2, color="black")

    plt.legend()
    plt.subplot(3, 3, 2)
    plt.title(f"Frequency Domain (Real)")
    plt.plot(ppm, fft_signal.real, label="Original", linewidth=2, color="black")
    plt.plot(ppm, complex_response.real, label="Filter Real", linewidth=2,color="magenta")
    plt.gca().invert_xaxis()
    plt.legend()


    plt.subplot(3, 3, 3)
    plt.title(f"Frequency Domain (Imag)")
    plt.plot(ppm,fft_signal.imag, label="Original",linewidth=2, color="black")
    plt.plot(ppm,complex_response.imag, label="Filter Imag", linewidth=2, color="blue")
    plt.gca().invert_xaxis()
    plt.legend()


    plt.subplot(3, 3, 4)
    plt.plot(np.real(filtered_signal[:100]), label="Filtered (Real)", linestyle="--",color="magenta", linewidth=2)
    plt.plot(np.imag(filtered_signal[:100]), label="Filtered (Imag)", linestyle="--",color="blue", linewidth=2)
    plt.plot(filtered_signal[:100].__abs__(), label="Filtered abs", linestyle="-",color="green", linewidth=2)
    plt.legend()


    plt.subplot(3, 3, 5)
    plt.plot(ppm,fft_filtered_signal.real, label="Real part filtered", linestyle="-", color="magenta", linewidth=2)
    plt.plot(ppm,complex_response.real, label="Filter Real", linestyle="--", color="green", linewidth=2)
    plt.gca().invert_xaxis()
    plt.legend()


    plt.subplot(3, 3, 6)
    plt.plot(ppm,fft_filtered_signal.imag, label="Filtered Imag", linestyle="-",color="blue", linewidth=2)
    plt.plot(ppm,complex_response.imag, label="Filter Imag", linestyle="--",color="green", linewidth=2)
    plt.xlabel("Chemical shift (ppm)")
    plt.gca().invert_xaxis()
    plt.legend()



    plt.subplot(3, 3, 8)
    plt.title(f"Frequency Domain (Magnitude)")
    plt.plot(ppm,np.abs(fft_signal), label="Original abs", linestyle="-",linewidth=2, color="black")
    plt.plot(ppm,fft_filtered_signal.__abs__(), label="Filtered abs", linestyle="--",linewidth=1, color="red")
    #plt.plot(ppm,complex_response.__abs__(), label="Filter abs", linestyle="--")
    plt.gca().invert_xaxis()
    plt.xlabel("Chemical shift (ppm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_BPF_filtering.png")
    
if __name__ == "__main__":
    main()