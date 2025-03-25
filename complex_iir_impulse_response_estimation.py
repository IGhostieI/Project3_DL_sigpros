import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter, dimpulse, tf2zpk, unit_impulse
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.fft import fft, fftshift

from myfunctions_tf import ComplexFilterOptimizer

# Load complex signal
path = "/home/stud/casperc/bhome/Project3_DL_sigpros/generated_data/2025_03_18-01_10_31-standard_amplitude_npz/TE40_lw07/TE40_lw07_03_data.npz"
data = np.load(path)
input_signal = data["augmented_ifft"]

orders = np.arange(20, 61, 20) 

# Set up the plot
plt.rcParams.update({'lines.linewidth': 2
                     ,'font.size': 20})
fig = plt.figure(figsize=(4*10,len(orders)*10))
gs = GridSpec(2*len(orders), 7, figure=fig)
for i, order in enumerate(orders, 1):
    complex_optimizer = ComplexFilterOptimizer(input_signal, order, options={'ftol':1e-3, 'maxiter':50})
    start = time.perf_counter()
    b_opt, a_opt = complex_optimizer.optimize()
    end = time.perf_counter()
    z, p, k = tf2zpk(b_opt, a_opt)
    estimated_signal = lfilter(b_opt, a_opt, unit_impulse(len(input_signal)))
    error_signal = np.linalg.norm(input_signal - estimated_signal) ** 2
    print(f"Optimization time: {end-start}, number of coefficients: {order}, error: {np.sum(error_signal)}")
    
    # Plot real part of the signal
    ax_real = fig.add_subplot(gs[2*(i-1), 0])
    ax_real.set_title(f"Real Part (Order: {order})")
    ax_real.plot(input_signal.real[:200], label="Original")
    ax_real.plot(estimated_signal.real[:200], label="Estimated", linestyle="--")
    ax_real.legend()

    # Plot imaginary part of the signal
    ax_imag = fig.add_subplot(gs[2*(i-1)+1, 0])
    ax_imag.set_title(f"Imaginary Part (Order: {order})")
    ax_imag.plot(input_signal.imag[:200], label="Original")
    ax_imag.plot(estimated_signal.imag[:200], label="Estimated", linestyle="--")
    ax_imag.legend()

    # Plot frequency domain (Real)
    ax_phase = fig.add_subplot(gs[2*(i-1):2*i, 1:3])
    ax_phase.set_title(f"Frequency Domain (Real)")
    ax_phase.plot(fftshift(fft(input_signal).real), label="Original")
    ax_phase.plot(fftshift(fft(estimated_signal).real), label="Estimated", linestyle="--")
    ax_phase.legend()
    # Plot frequency domain (iMAG)
    ax_phase = fig.add_subplot(gs[2*(i-1):2*i, 3:5])
    ax_phase.set_title(f"Frequency Domain (Imag)")
    ax_phase.plot(fftshift(fft(input_signal).imag), label="Original")
    ax_phase.plot(fftshift(fft(estimated_signal).imag), label="Estimated", linestyle="--")
    ax_phase.legend()

    # Plot pole-zero plot
    ax_pz = fig.add_subplot(gs[2*(i-1):2*i, 5:7])
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    ax_pz.add_artist(unit_circle)
    ax_pz.set_title(f"Pole-Zero Plot (Coeffs: {order})\nGain: {k:.2e}")
    ax_pz.scatter(np.real(z), np.imag(z), marker='x', color='red', label='Zeros')
    ax_pz.scatter(np.real(p), np.imag(p), marker='o', color='blue', label='Poles')
    legend = ax_pz.legend()
plt.tight_layout()
plt.savefig("test_minimize_sys_2.png") 
