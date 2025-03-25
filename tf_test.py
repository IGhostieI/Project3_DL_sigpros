import os

import numpy as np
from typing import Tuple, List
import scipy.special
import scipy.stats as stats
import scipy
import scipy.signal as signal
from scipy.fft import fft, ifft, fftshift
from scipy.optimize import minimize
import matplotlib 
import matplotlib.pyplot as plt
import re #Regular expressions library

import tensorflow as tf

# test if tensorflow is working
def test_tf():
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print("###########################")
    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", len(gpus))
    print("Available CPUs:", len(cpus))
    
    if gpus:
        try:
            # Create a simple tensor and perform a calculation on GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
            print("Matrix multiplication result:", c.numpy())
            print("Computation performed on:", c.device)
        except RuntimeError as e:
            print("GPU error:", e)
    else:
        print("No GPU available. Using CPU.")
    
    print("###########################")
    print("TensorFlow is working")
    return

def main():
    test_tf()

if __name__ == "__main__":
    main()