import os

import numpy as np
from typing import Tuple, List
import scipy.stats as stats
import scipy
import scipy.signal as signal
from scipy.fft import fft, ifft, fftshift
from scipy.optimize import minimize
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib import rc
import re #Regular expressions library


font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 24}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (18,10)
plt.rcParams['lines.linewidth'] = 2  # Set the desired linewidth value

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv1DTranspose, MaxPooling1D, Input, UpSampling1D, BatchNormalization, Activation, Reshape
from tensorflow.keras import Model

from datetime import datetime
import random

def huber_cosine_loss(y_true, y_pred, delta=1.0, alpha=0.5):
    """
    Custom loss function combining Huber loss and cosine similarity.
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Ground truth values
    y_pred : tf.Tensor
        Predicted values
    delta : float, optional
        Threshold for Huber loss (default: 1.0)
    alpha : float, optional
        Weight parameter between Huber (alpha) and cosine loss (1-alpha)
        alpha=1.0 means pure Huber loss, alpha=0.0 means pure cosine loss
        
    Returns:
    --------
    tf.Tensor
        Weighted combination of Huber loss and cosine similarity loss
    """
    # Compute Huber loss
    huber = tf.keras.losses.Huber(delta=delta)(y_true, y_pred)
    
    # Reshape tensors for cosine similarity if needed
    # Assuming inputs are [batch_size, time_points, channels]
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    # Compute cosine similarity
    # Using the negative of cosine_similarity since we want to minimize loss
    # (cosine_similarity measures similarity, higher is better)
    norm_true = tf.nn.l2_normalize(y_true_flat, axis=1)
    norm_pred = tf.nn.l2_normalize(y_pred_flat, axis=1)
    cosine_sim = tf.reduce_sum(norm_true * norm_pred, axis=1)
    cosine_loss = 1.0 - cosine_sim  # Convert to loss (0 is best)
    
    # Reduce to scalar if needed
    cosine_loss = tf.reduce_mean(cosine_loss)
    
    # Linear combination
    combined_loss = alpha * huber + (1.0 - alpha) * cosine_loss
    
    return combined_loss

# Create a Keras-compatible loss function class
class HuberCosineLoss(tf.keras.losses.Loss):
    def __init__(self, delta=1.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        return huber_cosine_loss(y_true, y_pred, self.delta, self.alpha)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'alpha': self.alpha
        })
        return config


def read_LCModel_table(file_path: str, metabolites: List) -> dict:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    metabolite_dict = {}
    
    for metabolite in metabolites:
        for line in lines:
            # Use regex to check for exact metabolite match at beginning of string or after space
            pattern = r'(^|\s)' + re.escape(metabolite) + r'(\s|$)'
            if re.search(pattern, line) and ("/" not in line and "=" not in line):
                metabolite = metabolite.strip()
                metabolite_dict[metabolite] = {}
                # Use regex to handle cases with no spaces between values and metabolite names
                match = re.match(r"([\d.Ee+-]+)\s*([\d%]+)\s*([\d.Ee+-]+)\s*([\w+]+)", line.strip())
                if match:
                    metabolite_dict[metabolite]["conc"] = float(match.group(1))
                    metabolite_dict[metabolite]["%SD"] = float(match.group(2).strip("%"))
                    metabolite_dict[metabolite]["/Cr+PCr"] = float(match.group(3))
    return metabolite_dict

def postprocess_predicted_data(predicted_data: np.ndarray, output_keys: List[str]) -> dict:
    output_dict = {}
    for i, key in enumerate(output_keys):
        output_dict[f"{key}"] = predicted_data[:, 2*i]+ 1j*predicted_data[:, 2*i+1]
    return output_dict

class MultiColorLegendHandler(HandlerBase):
    """
    A custom legend handler for multicolored lines in the legend.
    """
    def __init__(self, colormap='viridis', num_segments=9, linestyle='-', linewidth=2):
        super().__init__()
        self.colormap = plt.get_cmap(colormap)
        self.num_segments = num_segments
        self.linestyle = linestyle
        self.linewidth = linewidth

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        segment_width = width / self.num_segments
        segments = []
        colors = [self.colormap(i / (self.num_segments - 1)) for i in range(self.num_segments)]

        for i in range(self.num_segments):
            start_x = x0 + i * segment_width
            end_x = start_x + segment_width
            # Ensure the last segment ends exactly at the total width
            if i == self.num_segments - 1:
                end_x = x0 + width
            segments.append(Line2D(
                [start_x, end_x],
                [y0 + height / 2, y0 + height / 2],
                color=colors[i],
                linestyle=self.linestyle,
                linewidth=self.linewidth,
                transform=trans
            ))
        return segments

def extract_bracket_content(string: str) -> List[str]:
    pattern = re.compile(r'\[([^\]]+)\]')
    matches = pattern.findall(string)
    return matches

def input_ouput_key_sorting(bracket_string:str) -> Tuple[List[str], List[str]]:
    input_output = extract_bracket_content(bracket_string)
    input_keys = input_output[0]
    output_keys = input_output[1]
    input_keys = input_keys.replace("'", "").split(", ")
    output_keys = output_keys.replace("'", "").split(", ")
    return input_keys, output_keys

def flatten_list(x):
    return [item for sublist in x for item in sublist]

def read_complex_raw(filename:str, shift_fft:bool=True)->Tuple[np.array, np.array]: # Check .RAW header file and numbers
    with open(filename, "r") as coord:
        text = coord.read().split("\n")
        data = text[text.index(" $END")+1:]
        data = [line.lstrip() for line in data]
        data = [line for line in data if line!=""]
        data = np.array([line.split() for line in data], dtype=float)
        real_data = data[:,0]
        imaginary_data = data[:,1]
        if shift_fft:
            real_data = fftshift(real_data)
            imaginary_data = fftshift(imaginary_data)
    return real_data, imaginary_data

def model_input_output_prep(data_dict, input_keys, output_keys=None, complex_data=True):
    input_data = []
    output_data = []
    for input_key in input_keys:
        loaded_input = data_dict[input_key]
        if input_key == "a" or input_key == "b":
            loaded_input = np.pad(loaded_input, (0, 2048-len(loaded_input)), mode="constant", constant_values=0)
        real_part_input = loaded_input.real
        imag_part_input = loaded_input.imag
        if complex_data:
            input_data.append(np.array(real_part_input).T)
            input_data.append(np.array(imag_part_input).T)
        else:
            input_data.append(np.array(real_part_input).T)
    
    for output_key in output_keys:
        try :
            loaded_output = data_dict[output_key]
        except:
            print(f"Output key {output_key} not found")
            continue
        real_part_output = loaded_output.real
        imag_part_output = loaded_output.imag
        if complex_data:
            output_data.append(np.array(real_part_output).T)
            output_data.append  (np.array(imag_part_output).T)
        else:
            output_data.append(real_part_output)
    """ input_data.append(np.array(input_data).T)  # Transpose to get (2048, 2*len(input_keys))
    output_data.append(np.array(output_data).T)  # Transpose to get (2048, 2*len(output_keys)) """
    input_data = np.array(input_data).T
    output_data = np.array(output_data).T
    print(f"Loaded input shape: {input_data.shape}, Loaded output shape: {output_data.shape}")
    
    return np.expand_dims(input_data, axis=0), np.expand_dims(output_data, axis=0)



def load_from_directory(path:str, num_signals:int=-1, input_keys:List[str]=["augmented"], output_keys:List[str]=["original"], complex_data:bool=True, data_format:str="npz") -> Tuple[np.array, np.array]:
    files = [f for f in os.listdir(path) if f.endswith(f'.{data_format}')]
    input_data = []
    output_data = []
    
    if num_signals > 0 and num_signals < len(files):
        files = files[:num_signals]
    
    for file in files:
        file_input_data = []
        file_output_data = []
        
        if data_format == "mat":
            loaded_data = scipy.io.loadmat(os.path.join(path, file))
        elif data_format == "npz":
            loaded_data = np.load(os.path.join(path, file), allow_pickle=True)
        
        for input_key in input_keys:
            loaded_input = loaded_data[input_key]
            if input_key == "a" or input_key == "b":
                loaded_input = np.pad(loaded_input, (0, 2048-len(loaded_input)), mode="constant", constant_values=0)
            real_part_input = loaded_input.real
            imag_part_input = loaded_input.imag
            if complex_data:
                file_input_data.extend([real_part_input, imag_part_input])
            else:
                file_input_data.append(real_part_input)
 
        for output_key in output_keys:
            try :
                loaded_output = loaded_data[output_key]
            except:
                print(f"Output key {output_key} not found in file {file}")
                continue
            real_part_output = loaded_output.real
            imag_part_output = loaded_output.imag
            if complex_data:
                file_output_data.extend([real_part_output, imag_part_output])
            else:
                file_output_data.append(real_part_output)
        
        input_data.append(np.array(file_input_data).T)  # Transpose to get (2048, 2*len(input_keys))
        output_data.append(np.array(file_output_data).T)  # Transpose to get (2048, 2*len(output_keys))
    
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    print(f"Loaded input shape: {input_data.shape}, Loaded output shape: {output_data.shape}")
    
    return input_data, output_data

def load_NPZ_from_list(path_list, input_keys:List[str]=["augmented"], output_keys:List[str]=["original"], complex_data:bool=True, data_format:str="npz") -> Tuple[np.array, np.array]:
    input_data = []
    output_data = []
    
    
    for path in path_list:
        file_input_data = []
        file_output_data = []
        
        if data_format == "mat":
            loaded_data = scipy.io.loadmat(path)
        elif data_format == "npz":
            loaded_data = np.load(path, allow_pickle=True)
        
        for input_key in input_keys:
            loaded_input = loaded_data[input_key]
            if input_key == "a" or input_key == "b":
                loaded_input = np.pad(loaded_input, (0, 2048-len(loaded_input)), mode="constant", constant_values=0)
            real_part_input = loaded_input.real
            imag_part_input = loaded_input.imag
            if complex_data:
                file_input_data.extend([real_part_input, imag_part_input])
            else:
                file_input_data.append(real_part_input)
        
        for output_key in output_keys:
            try :
                loaded_output = loaded_data[output_key]
            except:
                print(f"Output key {output_key} not found in file {path}")
                continue
            real_part_output = loaded_output.real
            imag_part_output = loaded_output.imag
            if complex_data:
                file_output_data.extend([real_part_output, imag_part_output])
            else:
                file_output_data.append(real_part_output)
        
        input_data.append(np.array(file_input_data).T)  # Transpose to get (2048, 2*len(input_keys))
        output_data.append(np.array(file_output_data).T)  # Transpose to get (2048, 2*len(output_keys))
    
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    print(f"Loaded input shape: {input_data.shape}, Loaded output shape: {output_data.shape}")
    
    return input_data, output_data


def load_mrui_txt_data(filename:str)->Tuple[np.array, np.array, dict]:
    """_summary_

    Args:
        filename (str): _description_

    Returns:
        
        sig (np.array): _description_
        fft (np.array): _description_
        metadata (dict): _description_
        
    """
    with open(filename, "r") as mrui:
        text = mrui.read().split("\n")
        data = text[text.index("sig(real)	sig(imag)	fft(real)	fft(imag)")+2:]
        metadata = text[:text.index("sig(real)	sig(imag)	fft(real)	fft(imag)")-4]
        metadata = [line.replace(" ","").split(":") for line in metadata if re.search(r": .+", line)]
        metadata = {line[0]:line[1].lstrip().replace("\n","") for line in metadata}
        
        data = [line.lstrip() for line in data]
        data = [line for line in data if line!=""]
        data = np.array([line.split() for line in data], dtype=float)
        sig_real = data[:,0]
        sig_imag = data[:,1]
        fft_real = data[:,2]
        fft_imag = data[:,3]
    return sig_real+1j*sig_imag, fft_real + 1j*fft_imag, metadata    
        
        
def make_current_time_directory(main_folder_path:str, data_description:str, make_logs_dir:bool=True)->str:
    now = datetime.now()
    current_experiment = now.strftime(f"%Y_%m_%d-%H_%M_%S_{data_description}")
    if not os.path.exists(os.path.join(main_folder_path, current_experiment)):
        if not os.path.exists(main_folder_path):
            os.mkdir(main_folder_path)
    
    path = os.path.join(main_folder_path, current_experiment)
    os.mkdir(path)
    if make_logs_dir:
        os.makedirs(os.path.join(main_folder_path, current_experiment,"logs"), exist_ok=True)
    print(f"\nDirectory Created {path}.")
    
    return path

def scale_complex_data(real_component:np.ndarray, imaginary_component:np.ndarray, scaling_factor:float=1)->Tuple[np.array, np.array]:
    """Linearly scale the real and imaginary components of a complex signal by a given factor "scaling factor".

    Args:
        real_component (np.ndarray): Real component of the signal
        imaginary_component (np.ndarray):Imaginary component of the signal
        scaling_factor (float, optional): Scaling factor to be multiplied with the components. Defaults to 1.

    Returns:
        Tuple[np.array, np.array]: _description_
    """
    return real_component*scaling_factor, imaginary_component*scaling_factor

def add_gauss_noise_from_wanted_SNR(real_component:np.ndarray, imaginary_component:np.ndarray, SNR:float=0.1)->Tuple[np.array, np.array]:
    """Add gaussian noise to a complex signal with a given SNR. The SNR is calculated as: 
    SNR = max(||signal||)/SD, where SD is the standard deviation of the noise. Formula for SNR from Öz et al (2021)
    https://doi.org/10.1002/nbm.4236

    Args:
        real_component (np.ndarray): Real component of the signal
        imaginary_component (np.ndarray): Imaginary component of the signal
        SNR (float, optional): Signal to Noise Ratio. Defaults to 0.1.

    Returns:
        Tuple[np.array, np.array]: _description_
    """
    SD = np.max(abs(np.sqrt(real_component**2+imaginary_component**2)))/SNR
    noise = np.random.normal(0, SD, len(real_component))
    return real_component+noise, imaginary_component+noise

def create_baseline(input_range:np.ndarray, std_range:Tuple[float, float]=(0,1), n_gaussians:int=1, amplitude_range:Tuple[float, float]=(1,1))->np.ndarray:
    n_points = len(input_range)
    baseline = np.zeros(n_points)
    for i in range(n_gaussians):
        # Create the start and end of a segment of the baseline. The segments will be equally spaced. 
        start = 0+i*n_points//n_gaussians 
        end = n_points//n_gaussians+i*n_points//n_gaussians
        
        current_point = (np.random.randint(start, end)) # Current point is placed randomly in the segment
        std = np.random.uniform(std_range[0], std_range[1])
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        gaussian = amplitude*n_points*stats.norm.pdf(input_range, loc=input_range[current_point], scale = std)
        baseline += gaussian
    return baseline


def create_baseline_V2(input_range:np.ndarray, std_range:Tuple[float, float]=(0,1), n_gaussians:int=1, amplitude_range:Tuple[float, float]=(1,1))->np.ndarray:
    n_points = len(input_range)
    baseline = np.zeros(n_points)
    for i in range(n_gaussians):
        # Create the start and end of a segment of the baseline. The segments will be equally spaced. 
        start = 0+i*n_points//n_gaussians 
        end = n_points//n_gaussians+i*n_points//n_gaussians
        
        current_point = (np.random.randint(start, end)) # Current point is placed randomly in the segment
        std = np.random.uniform(std_range[0], std_range[1])
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        
        
        
        gaussian = amplitude*n_points*stats.norm.pdf(input_range, loc=input_range[current_point], scale = std)
        baseline += gaussian
    return baseline

def read_LCModel_coord(filename:str)->Tuple[np.array, np.array, np.array, np.array]:
    """_summary_

    Args:
        filename (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        
        freq (np.array): _description_
        data (np.array): _description_
        fit (np.array): _description_
        baseline (np.array): _description_
    """
    with open(filename, "r") as coord:
        text = coord.read().split("\n")
        # Find the first line that ends with " NY"
        text_array = np.array(text)
        index = np.where(np.char.endswith(text_array, " NY"))[0]
        if len(index) == 0:
            raise ValueError("No line ending with ' NY' found in the file.")
        
        # Extract the line and process it
        line = text_array[index[0]]  # Get the first matching line
        length = int(line.lstrip().split(" ")[0])  # Extract the number at the start of the line
        
        step_size = length // 10 + 1
        start_val = index[0] + 1  # Use the first match and add 1
        end_val = start_val + step_size
        freq = [line.lstrip() for line in text[start_val:end_val]]
        freq = [re.sub(r"\s+", " ", line).split(" ") for line in freq]
        #print(f"freq from: {start_val} - {end_val}")
        #print(f"freq: {freq[-1]}")
        start_val = end_val + 1
        end_val = start_val + step_size
        data = [line.lstrip() for line in text[start_val:end_val]]
        data = [re.sub(r"\s+", " ", line).split(" ") for line in data]
        start_val = end_val + 1
        end_val = start_val + step_size
        fit = [line.lstrip() for line in text[start_val:end_val]]
        fit = [re.sub(r"\s+", " ", line).split(" ") for line in fit]
        start_val = end_val + 1
        end_val = start_val + step_size
        baseline = [line.lstrip() for line in text[start_val:end_val]]
        baseline = [re.sub(r"\s+", " ", line).split(" ") for line in baseline]
        baseline = np.array(flatten_list(baseline), dtype=float)
        
        freq = np.array(flatten_list(freq), dtype=float)
        data = np.array(flatten_list(data), dtype=float)
        fit = np.array(flatten_list(fit), dtype=float)
        
    return freq, data, fit, baseline

def make_metadata_file(filepath:str, data_description:dict):
    with open(os.path.join(filepath, "metadata.txt"), "w") as meta:
        for key, value in data_description.items():
            meta.write(f"{key}: {value}\n")

def frequency_shift_jitter(complex_data:Tuple[np.array, np.array], ppm_range:Tuple[float, float]=(0, 10), ppm_jitter_range:Tuple[float, float]=(-1, 1))->Tuple[np.array, np.array]:
    ppm_shift = np.random.uniform(ppm_jitter_range[0], ppm_jitter_range[1])
    #print(ppm_shift)
    ppm_per_point = (ppm_range[1]-ppm_range[0])/len(complex_data[0])
    index_shift = int(ppm_shift//ppm_per_point)
    
    if index_shift <= 0:
        index_shift = abs(index_shift)
        # Shift the data to the left
        # remove the last index_shift points
        real_data = complex_data[0][index_shift:]
        imag_data = complex_data[1][index_shift:]
        diff_real = np.diff(real_data)[-1]
        diff_imag = np.diff(imag_data)[-1]
        shifted_real =  np.pad(real_data,(0,index_shift),mode="linear_ramp", end_values=(real_data[-1]+diff_real, real_data[-1]+index_shift*diff_real))
        
        shifted_imag = np.pad(imag_data,(0,index_shift),mode="linear_ramp", end_values=(imag_data[-1]+diff_imag, imag_data[-1]+index_shift*diff_imag))
        # Add index_shift points to the beginning of the data that are linearly interpolated from the first two points
    elif index_shift >= 0:
        # shift the data to the right
        # remove the first index_shift points
        
        real_data = complex_data[0][:-index_shift]
        imag_data = complex_data[1][:-index_shift]
        diff_real = np.diff(real_data)[0]
        diff_imag = np.diff(imag_data)[0]
        
        shifted_real = np.pad(real_data,(index_shift,0),mode="linear_ramp", end_values=(real_data[0]-index_shift*diff_real, real_data[0]-diff_real))
        shifted_imag = np.pad(imag_data,(index_shift,0),mode="linear_ramp", end_values=(imag_data[0]-index_shift*diff_imag, imag_data[0]-diff_imag))
        # add index_shift points to the end of the data that are linearly interpolated based on the differntial of the last two points
    
    
    
    return shifted_real, shifted_imag 

def BPF_ppm(fft_signal:np.ndarray[np.complex64], ppm_range:np.ndarray[np.float32, np.float32], ppm_pass:np.ndarray[np.float32, np.float32], margin=0.75, gain_passband:float=1, ftype:str="butter") -> tuple[np.ndarray[np.complex64], np.ndarray[np.complex64], np.ndarray[np.complex64]]:
    """Take a signal in the frequency domain and filter it using a bandpass filter in the frequency domain, return the filtered signal in the time domain

    Args:
        fft_signal (_type_): _description_
        min_ppm (_type_): _description_
        max_ppm (_type_): _description_
        ppm_pass (_type_): _description_
        margin: _description_
        ftype (str, optional): _description_. Defaults to "butter".

    Returns:
        _type_: _description_
    """
    ppm_stop = np.array([ppm_pass[0]-margin, ppm_pass[1]+margin])
    wp = (ppm_pass - ppm_range.min()) / (ppm_range.max() - ppm_range.min())
    ws = (ppm_stop - ppm_range.min()) / (ppm_range.max() - ppm_range.min())
    
    b, a = signal.iirdesign(wp=wp, ws=ws, gpass=1, gstop=60, ftype=ftype)
    
    _, complex_response = signal.freqz(b, a, worN=len(fft_signal))
    complex_response*=gain_passband
    fft_filtered_signal = fft_signal * complex_response # filtering in frequency domain
    filtered_signal = ifft(fft_filtered_signal)
    fft_filtered_signal= fft_filtered_signal.copy()
    complex_response = complex_response
    return filtered_signal, fft_filtered_signal, complex_response

def voigt(x, loc=0, FWHM=1, mixing_factor=0.5, amplitude_scaling=1):
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    gamma = FWHM/2
    normal_part = stats.norm.pdf(x, loc=loc, scale=sigma)
    lorentzian_part = stats.cauchy.pdf(x, loc=loc, scale=gamma)
    
    voigt = (1 - mixing_factor) * normal_part + mixing_factor * lorentzian_part
    scaled_voigt = amplitude_scaling*voigt / np.max(voigt)  
    
    return scaled_voigt

def MM_baseline(ppm_range, max_signal=1, tolerance=0.1, jitter_range=(-0.1, 0.1)):
    ### Hardcoded MM component list ### 
    MM_list = [[{"loc":2.02, "FWHM":0.125, "amplitude_scaling":max_signal/2}, {"loc":2.02+0.125, "FWHM":0.125, "amplitude_scaling":max_signal/4}, {"loc":2.02-0.125, "FWHM":0.125, "amplitude_scaling":max_signal/4},{"loc":2.02-0.25, "FWHM":0.125, "amplitude_scaling":max_signal/6},{"loc":2.02+0.25, "FWHM":0.125, "amplitude_scaling":max_signal/6}], 
           
           [{"loc":3.75, "FWHM":0.5, "amplitude_scaling":max_signal/3},{"loc":3.75+0.5, "FWHM":0.25, "amplitude_scaling":max_signal/6}, {"loc":3.75-0.5, "FWHM":0.25, "amplitude_scaling":max_signal/6}], 
           [{"loc":2.75, "FWHM":1, "amplitude_scaling":max_signal/24}],
           
           [{"loc":1, "FWHM":5, "amplitude_scaling":max_signal/45}]]
    ################################################################################################
    
    mixing_factor = 0.5
    full_baseline = np.zeros_like(ppm_range)
    
    individual_MM_components = []
    indivdiual_MM = []
    
    for MM in MM_list:
        full_MM = np.zeros_like(ppm_range)
        for MM_component in MM:
            mixing_factor = np.random.uniform(0,1)
            amplitude_scaling = np.random.uniform(1-tolerance,1+tolerance)*MM_component["amplitude_scaling"]# +-5% scaling to each individual voigt component
            FWHM = MM_component["FWHM"]*np.random.uniform(1-tolerance,1+tolerance)
            y = voigt(x=ppm_range, FWHM=FWHM, mixing_factor=mixing_factor, loc=MM_component["loc"], amplitude_scaling=amplitude_scaling)
            y, _ = frequency_shift_jitter(complex_data=(y, np.zeros_like(y)), ppm_range=(ppm_range.min(), ppm_range.max()), ppm_jitter_range=(0-0.1, 0.1))
            individual_MM_components.append(y)
            full_MM += y
        full_baseline += full_MM
        indivdiual_MM.append(full_MM)
        
    full_baseline, _ = frequency_shift_jitter(complex_data=(full_baseline, np.zeros_like(full_baseline)), ppm_range=(ppm_range.min(), ppm_range.max()), ppm_jitter_range=jitter_range)
    
    return full_baseline, indivdiual_MM, individual_MM_components

class ComplexFilterOptimizer:
    def __init__(self, input_signal, order, options={'method':'SLSQP', 'ftol':1e-6, 'maxiter':100}):
        self.input_signal = input_signal
        self.order = order
        self.b = np.array([1] + [0] * (order-1), dtype=complex)
        self.a = np.array([1] + [0] * (order-1), dtype=complex)
        self.initial_params = np.concatenate((self.b.real, self.b.imag, self.a.real, self.a.imag))
        self.options = {'method':'SLSQP', 'ftol':1e-6, 'maxiter':100}
        self.options.update(options)
     
    def objective(self, params):
        b_real, b_imag = params[:self.order], params[self.order:2*self.order]
        a_real, a_imag = params[2*self.order:3*self.order], params[3*self.order:]
        b = b_real + 1j * b_imag
        a = a_real + 1j * a_imag
        impulse = signal.unit_impulse(len(self.input_signal))
        y = signal.lfilter(b,a, impulse)
        error = np.sum(np.linalg.norm(self.input_signal - y) ** 2)
        return error

    def optimize(self):
        result = minimize(self.objective, self.initial_params, method=self.options['method'], 
                          options={'ftol':self.options['ftol'], 'maxiter':self.options['maxiter']})
        
        b_real, b_imag = result.x[:self.order], result.x[self.order:2*self.order]
        a_real, a_imag = result.x[2*self.order:3*self.order], result.x[3*self.order:]
        b_opt = b_real + 1j * b_imag
        a_opt = a_real + 1j * a_imag
        return b_opt, a_opt


class CNN_DenoiseFit_Functional(Model):
    def __init__(self, output_channels=2, custom_input_shape=(2048, 2)):
        super(CNN_DenoiseFit_Functional, self).__init__()
        self.output_channels = output_channels
        # Encoder
        filter = 32
        self.conv1 = Conv1D(filter, kernel_size=32, strides = 1, activation='relu', padding='same', input_shape=custom_input_shape)
        self.conv2 = Conv1D(filter, kernel_size=16, strides = 1, activation='relu', padding='same') 
        self.conv3 = Conv1D(filter, kernel_size=8, strides = 2, activation='relu', padding='same') # 2048 -> 1024
        self.conv4 = Conv1D(filter, kernel_size=8, strides=2, activation='relu', padding='same') # 1024 -> 512
        self.maxpool1 = MaxPooling1D(pool_size=8, padding='same') # 512 ->64
       
        
        # Decoder
        self.conv5 = Conv1DTranspose(filter, kernel_size=8, strides=2, activation='relu', padding='same') # 64 -> 128
        self.conv6 = Conv1DTranspose(filter, kernel_size=8, strides=2, activation='relu', padding='same') # 128 -> 256
        self.conv7 = Conv1DTranspose(filter, kernel_size=16, strides=2, activation='relu', padding='same') # 256 -> 512
        self.upsample1 = UpSampling1D(2) # 512 -> 1024
        self.conv8 = Conv1DTranspose(filter, kernel_size=16, strides=2, activation='linear', padding='same') # 1024 -> 2048
        self.final_output = Conv1D(filters=self.output_channels, kernel_size=1, activation='linear', padding='same')
        

        
    def call(self, x,):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) # 2048 -> 1024 
        x = self.conv4(x) # 1024 -> 512
        x = self.maxpool1(x) # 512 -> 64
        x = self.conv5(x) # 64 -> 128
        x = self.conv6(x) # 128 -> 256
        x = self.conv7(x) # 256 -> 512
        x = self.upsample1(x) # 512 -> 1024
        x = self.conv8(x) # 1024 -> 2048
        x = self.final_output(x) 
        return x

            
class CNN_Autoencoder_Functional(Model):
    def __init__(self, custom_input_shape=(2048, 2)):
        super(CNN_Autoencoder_Functional, self).__init__()
        self.custom_input_shape = custom_input_shape
        # defne the layers
        
        # Encoder
        self.input_layer = Input(self.custom_input_shape) 
        self.conv1 = Conv1D(32, 1024, activation='relu', padding='same')
        self.maxpool1 = MaxPooling1D(2, padding='same')
        self.conv2 = Conv1D(64, 512, activation='relu', padding='same')
        self.maxpool2 = MaxPooling1D(2, padding='same')
        self.conv3 = Conv1D(128, 256, activation='relu', padding='same')
        self.maxpool3 = MaxPooling1D(2, padding='same')
        self.conv4 = Conv1D(128, 128, activation='relu', padding='same')
        self.maxpool4 = MaxPooling1D(2, padding='same')
        self.conv5 = Conv1D(256, 64, activation='relu', padding='same')
        self.maxpool5 = MaxPooling1D(2, padding='same')
        self.conv6 = Conv1D(512, 32, activation='relu', padding='same')
        self.maxpool6 = MaxPooling1D(2, padding='same')
        self.conv7 = Conv1D(1024, 16, activation='relu', padding='same')
        self.maxpool7 = MaxPooling1D(2, padding='same')
        self.conv8 = Conv1D(2048, 8, activation='relu', padding='same')
        
        # Decoder
        self.conv9 = Conv1DTranspose(1024, 16, activation='relu', padding='same')
        self.upsample1 = UpSampling1D(2)
        self.conv10 = Conv1DTranspose(512, 32, activation='relu', padding='same')
        self.upsample2 = UpSampling1D(2)
        self.conv11 = Conv1DTranspose(256, 64, activation='relu', padding='same')
        self.upsample3 = UpSampling1D(2)
        self.conv12 = Conv1DTranspose(128, 128, activation='linear', padding='same')
        self.upsample4 = UpSampling1D(2)
        self.conv13 = Conv1DTranspose(128, 256, activation='relu', padding='same')
        self.upsample5 = UpSampling1D(2)
        self.conv14 = Conv1DTranspose(64, 512, activation='relu', padding='same')
        self.upsample6 = UpSampling1D(2)
        self.conv15 = Conv1DTranspose(32, 1024, activation='relu', padding='same')
        self.upsample7 = UpSampling1D(2)
        self.conv16 = Conv1DTranspose(2, 2048, activation='linear', padding='same')
        

        
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x) # 2048 -> 1024
        x = self.conv2(x)
        x = self.maxpool2(x) # 1024 -> 512
        x = self.conv3(x)
        x = self.maxpool3(x) # 512 -> 256
        x = self.conv4(x)
        x = self.maxpool4(x) # 256 -> 128
        x = self.conv5(x)
        x = self.maxpool5(x) # 128 -> 64
        x = self.conv6(x)
        x = self.maxpool6(x) # 64 -> 32
        x = self.conv7(x)
        x = self.maxpool7(x) # 32 -> 16
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.upsample1(x) # 16 -> 32
        x = self.conv10(x)
        x = self.upsample2(x) # 32 -> 64
        x = self.conv11(x)
        x = self.upsample3(x) # 64 -> 128
        x = self.conv12(x)
        x = self.upsample4(x) # 128 -> 256
        x = self.conv13(x)
        x = self.upsample5(x) # 256 -> 512
        x = self.conv14(x)
        x = self.upsample6(x) # 512 -> 1024
        x = self.conv15(x)
        x = self.upsample7(x) # 1024 -> 2048
        x = self.conv16(x)
        return x

class CNN_Autoencoder(Model):
    def __init__(self, input_shape=(1024, 2)):
        super(CNN_Autoencoder, self).__init__()
        self.input_signal = input_shape
        self.encoder = tf.keras.Sequential([
            Conv1D(16, 32, activation='relu', padding='same'), # kernel_initializer='he_normal'
            MaxPooling1D(2, padding='same'),  # 1024 -> 512
            Conv1D(32, 13, activation='relu', padding='same'),
            MaxPooling1D(2, padding='same'),  # 512 -> 256
            Conv1D(128, 5, activation='relu', padding='same'),
            MaxPooling1D(2, padding='same')   # 256 -> 128
        ])
        self.decoder = tf.keras.Sequential([
            Conv1DTranspose(128, 5, activation='relu', padding='same'),
            UpSampling1D(2),  # 128 -> 256
            Conv1DTranspose(32, 13, activation='relu', padding='same'),
            UpSampling1D(2),  # 256 -> 512
            Conv1DTranspose(16, 32, activation='relu', padding='same'),
            UpSampling1D(2),  # 512 -> 1024
            Conv1DTranspose(2, 3, activation='linear', padding='same')  # Output shape (1024, 2)
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VisualizePredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir:str, how_often:int=5, validation_data=None, total_epochs:int=0, input_keys:List[str]=[], output_keys:List[str]=[]):
        super(VisualizePredictionCallback, self).__init__()
        self.output_dir = output_dir
        self.how_often = how_often
        self.validation_data = validation_data
        self.total_epochs = total_epochs
        self.input_keys = input_keys
        self.output_keys = output_keys
        # Set global linewidth and font properties
        plt.rcParams['lines.linewidth'] = 4  # Change the linewidth
        plt.rcParams['font.size'] = 36      # Change the font size
        plt.rcParams['font.family'] = 'serif' # Change the font family
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.how_often == 0 or epoch == 0 or epoch == self.total_epochs-1:
            ppm = np.linspace(-3.121, 12.528, 2048)
            # Fetch the validation data
            val_data = list(self.validation_data.unbatch().as_numpy_iterator())
            sample = random.choice(val_data)
            input_sample, true_output = sample
            #print(f"type of input_sample: {type(input_sample)}")
            #print(f"Input shape: {input_sample.shape}")
            input_real, input_imag = np.split(input_sample, 2, axis=-1)
            true_real, true_imag = np.split(true_output, 2, axis=-1)
            
            # Make a prediction
            predicted = np.squeeze(self.model.predict(tf.expand_dims(input_sample, axis=0)), axis=0)
            predicted_real, predicted_imag = np.split(predicted, 2, axis=-1)
            #print(f"type of predicted: {type(predicted)}")
            #print(f"Predicted shape: {predicted.shape}")
            epoch_string = f"{epoch+1}".zfill(len(str(self.total_epochs)))
            save_path = os.path.join(self.output_dir, f'epoch_{epoch_string}_prediction.png')
            # Plot the prediction and the ground truth
            # calculate the number of plots needed
            num_plots = len(self.input_keys) + len(self.output_keys) + int("augmented_ifft" in self.input_keys)
            plt.figure(figsize=(30, 20*num_plots))
            # Plot input with conditional plotting
            index =1
            
            for input_key in self.input_keys:
                plt.subplot(num_plots, 1, index)
                label = input_key.replace("_", " ")
                if input_key == "a" :
                    a_index = self.input_keys.index("a")
                    a_real_part = np.trim_zeros(input_sample[:,2*(a_index)], "b")
                    a_imaginary_part = np.trim_zeros(input_sample[:,2*(a_index)+1], "b")
                    plt.scatter(a_real_part, a_imaginary_part, label="a", s=250, c="black")
                    plt.xlabel("Real")
                    plt.ylabel("Imaginary")
                    plt.legend()
                    
                elif input_key == "b":
                    b_index = self.input_keys.index("b")
                    b_real_part = np.trim_zeros(input_sample[:,2*(b_index)], "b")
                    b_imaginary_part = np.trim_zeros(input_sample[:,2*(b_index)+1], "b")
                    plt.scatter(b_real_part, b_imaginary_part, label="b", s=250, c="black")
                    plt.xlabel("Real")
                    plt.ylabel("Imaginary")
                    plt.legend()
                    
                elif input_key == "augmented":
                    plt.plot(ppm, input_sample[:,2*(self.input_keys.index(input_key))], label=label, color="black")
                    if "baseline" in self.output_keys:
                        baseline_index = self.input_keys.index("baseline")
                        plt.plot(ppm, true_output[:,2*baseline_index], label="baseline", color="red")
                    plt.xlabel("Chemical shift [ppm]")
                    plt.gca().invert_xaxis()  # Flip the x-axis
                    plt.ylabel("Intensity [a.u.]")
                    plt.legend()
                    
                elif input_key == "augmented_ifft":
                    plt.plot(input_sample[:,2*(self.input_keys.index(input_key))], label=label, color="black")
                    """ if "a" and "b" in self.input_keys:
                        a_index = self.input_keys.index("a")
                        a_real_part = np.trim_zeros(input_sample[:,2*(a_index)], "b")
                        a_imaginary_part = np.trim_zeros(input_sample[:,2*(a_index)+1], "b")
                        b_index = self.input_keys.index("b")
                        b_real_part = np.trim_zeros(input_sample[:,2*(b_index)], "b")
                        b_imaginary_part = np.trim_zeros(input_sample[:,2*(b_index)+1], "b")
                        a = a_real_part + 1j*a_imaginary_part
                        b = b_real_part + 1j*b_imaginary_part
                        print(f"imput signal shape: {input_sample.shape}")
                        impulse = signal.unit_impulse(input_sample.shape[0])
                        ab_estimate = signal.lfilter(b,a, impulse)
                        plt.plot(ab_estimate, label="optimized impulse response", color="red", linestyle="--") """
                    plt.xlabel("Time [idx]")
                    plt.ylabel("Intensity [a.u.]")
                    plt.legend()
                    # plot only the 200 first elements
                    index += 1
                    plt.subplot(num_plots, 1, index)
                    plt.plot(input_sample[:200,2*(self.input_keys.index(input_key))], label=label+" truncated", color="black")
                    """ if "a" and "b" in self.input_keys:
                        plt.plot(ab_estimate[:200], label="Truncated optimized impulse response", color="red", linestyle="--") """
                elif "filtered" in input_key:
                    if "fft" not in input_key:
                        filtered_signal = input_sample[:,2*(self.input_keys.index(input_key))]+1j*input_sample[:,2*(self.input_keys.index(input_key))+1]
                        fft_filtered_signal = fft(filtered_signal)
                        filtered_signal_abs = np.abs(fft_filtered_signal)
                    else:
                        filtered_signal_abs = np.abs(input_sample[:,2*(self.input_keys.index(input_key))]+1j*input_sample[:,2*(self.input_keys.index(input_key))+1])
                        
                    if "augmented" in self.input_keys:
                        augmented_index = self.input_keys.index("augmented")
                        plt.plot(ppm, input_sample[:,2*augmented_index], label="input spectrum", color="black")
                    elif "augmented_ifft" in self.input_keys:
                        
                        augmented_ifft_index = self.input_keys.index("augmented_ifft")
                        augmented_fft_from_ifft = fftshift(fft(input_sample[:,2*augmented_ifft_index]+1j*input_sample[:,2*augmented_ifft_index+1]))
                        plt.plot(ppm, augmented_fft_from_ifft, label="FFT(input)", color="black")
                        
                    plt.plot(ppm, filtered_signal_abs, label=label+" (norm)", color="magenta")
                    plt.xlabel("Chemical shift [ppm]")
                    plt.gca().invert_xaxis()  # Flip the x-axis
                    plt.ylabel("Intensity [a.u.]")
                    plt.legend()
                index += 1
                # plot outputs 
               
            for output_key in self.output_keys:
                plt.subplot(num_plots, 1, index)
                if output_key == "original":
                    if "baseline" in self.output_keys and "augmented" in self.input_keys:
                        baseline_index = self.output_keys.index("baseline")
                        plt.plot(ppm, input_sample[:,2*augmented_index]-predicted[:,2*baseline_index], label=" Estimated baseline-corrected full spectrum", color="black")
                    elif "baseline" in self.output_keys and "augmented" not in self.input_keys and "augmented_ifft" in self.input_keys:
                        augmented_ifft_index = self.input_keys.index("augmented_ifft")
                        augmented_fft_from_ifft = fftshift(fft(input_sample[:,2*augmented_ifft_index]+1j*input_sample[:,2*augmented_ifft_index+1]))
                        baseline_index = self.output_keys.index("baseline")
                        plt.plot(ppm, augmented_fft_from_ifft-true_output[:,2*baseline_index], label="Baseline-corrected full spectrum", color="black")
                    
                    plt.plot(ppm, true_output[:,2*(self.output_keys.index(output_key))], label="Ground truth", color="magenta")
                    plt.plot(ppm, predicted[:,2*(self.output_keys.index(output_key))], label="predicted", color="cyan")
                    plt.xlabel("Chemical shift [ppm]")
                    plt.gca().invert_xaxis()  # Flip the x-axis
                    plt.ylabel("Intensity [a.u.]")
                    plt.title("Full Spectrum fitting")
                    plt.legend()
                elif output_key == "baseline":
                        if "augmented" in self.input_keys:
                            augmented_index = self.input_keys.index("augmented")
                            plt.plot(ppm, input_sample[:,2*augmented_index], label="Full metabolite pectrum", color="black")
                        elif "augmented_ifft" in self.input_keys:
                            augmented_ifft_index = self.input_keys.index("augmented_ifft")
                            augmented_fft_from_ifft = fftshift(fft(input_sample[:,2*augmented_ifft_index]+1j*input_sample[:,2*augmented_ifft_index+1]))
                            plt.plot(ppm, augmented_fft_from_ifft.real, label="FFT(input)", color="black")
                        
                        plt.plot(ppm, true_output[:,2*(self.output_keys.index(output_key))], label="Ground truth", color="magenta")
                        plt.plot(ppm, predicted[:,2*(self.output_keys.index(output_key))], label="predicted",color="cyan")
                        plt.xlabel("Chemical shift [ppm]")
                        plt.gca().invert_xaxis()  # Flip the x-axis
                        plt.ylabel("Intensity [a.u.]")
                        plt.title("Baseline fitting")
                        plt.legend()   
                elif output_key != "metadata" or output_key != "iir_order":
                    
                    if "baseline" in self.output_keys and "augmented" in self.input_keys:
                        baseline_index = self.output_keys.index("baseline")
                        plt.plot(ppm, input_sample[:,2*augmented_index]-predicted[:,2*baseline_index], label="Baseline-corrected full spectrum", color="black")
                    elif "baseline" in self.output_keys and "augmented" not in self.input_keys and "augmented_ifft" in self.input_keys:
                        augmented_ifft_index = self.input_keys.index("augmented_ifft")
                        augmented_fft_from_ifft = fftshift(fft(input_sample[:,2*augmented_ifft_index]+1j*input_sample[:,2*augmented_ifft_index+1]))
                        baseline_index = self.output_keys.index("baseline")
                        plt.plot(ppm, augmented_fft_from_ifft-true_output[:,2*baseline_index], label="Baseline-corrected full spectrum", color="black")
                        
                    plt.plot(ppm, true_output[:,2*(self.output_keys.index(output_key))], label="Ground truth", color="magenta")
                    plt.plot(ppm, predicted[:,2*(self.output_keys.index(output_key))], label="predicted", color="cyan")
                    plt.xlabel("Chemical shift [ppm]")
                    plt.gca().invert_xaxis()  # Flip the x-axis
                    plt.ylabel("Intensity [a.u.]")
                    plt.title(f"{output_key} fitting")
                    plt.legend()
                index += 1
                  
                
            
            # Adjust layout
            plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.03, wspace=0.05, hspace=0.3)            
            # Save the plot
            plt.savefig(save_path)
            plt.close()

# The following functions are inspired by : https://github.com/Sakib1263/TF-1D-2D-ResNetV1-2-SEResNet-ResNeXt-SEResNeXt
###################################################################################################################
def conv_1D_block(x, filters, kernel_size=3, strides=1):
    # 1D Convolutional block + Batch Normalization + ReLU Activation
    x=Conv1D(filters, kernel_size, strides=strides, padding='same')(x) # fix activation and BatchNormalization
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size, strides=1):
    conv = conv_1D_block(x, filters, kernel_size=kernel_size, strides=strides)
    conv = conv_1D_block(conv, filters, kernel_size=kernel_size, strides=strides)
    return conv

def conv_1D_transpose_block(x, filters, kernel_size, strides=1):
    # 1D Convolutional Transpose block + Batch Normalization + ReLU Activation
    x = Conv1DTranspose(filters, kernel_size, strides=strides, padding='same')(x) # fix activation and BatchNormalization
    x = BatchNormalization()(x) # consider removal
    x = Activation('relu')(x)
    return x 

def convT_block(x, filters, kernel_size, strides=1):
    conv = conv_1D_transpose_block(x, filters, kernel_size=kernel_size, strides=strides)
    conv = conv_1D_transpose_block(conv, filters, kernel_size=kernel_size, strides=strides)
    return conv   

def ERB(input_tensor, filter1 = 32, kernel_size=3):# Enhanced Residual Block
        f = filter1
        conv11 = tf.keras.layers.Conv1D(f, kernel_size=1, padding='same', activation='relu')(input_tensor) # kernel_initializer='he_uniform',
        x = tf.keras.layers.Conv1D(f, kernel_size=kernel_size, padding='same', activation='relu')(conv11) # kernel_initializer='he_uniform',
        x = tf.keras.layers.add([conv11, x])
        x = tf.keras.layers.Conv1D(f, kernel_size=1, padding='same', activation='relu')(x) # kernel_initializer='he_uniform',
        x = tf.keras.layers.add([conv11, x])
        return x


""" def ResNet1D(input_shape=(2048, 2), output_shape=(2048, 2), num_blocks=4, filter1=32, kernel_size=3, print_summary=True):
    input = Input(shape=input_shape)
    x = input
    signal_length = input_shape[0]
    for i in range(num_blocks):
        if i< num_blocks//2 and  signal_length>32:
            x = conv_block(x, filter1, kernel_size=kernel_size, strides=2)
            signal_length /= 4
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        elif i < num_blocks//2:
            x = conv_block(x, filter1, kernel_size=kernel_size, strides=1)
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        
        elif i >= num_blocks//2 and signal_length>=32 and signal_length<input_shape[0]:
            x = convT_block(x, filters=filter1, kernel_size=kernel_size, strides=2)
            signal_length *=4
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        elif i > num_blocks//2:
            x = convT_block(x, filters=filter1, kernel_size=kernel_size, strides=1)
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        if i % 2 == 0:
            x = ERB(x, filter1, kernel_size=kernel_size)
            if print_summary:
                print(f"ERB at block {i}, signal length: {signal_length}")
            
    outputs = tf.keras.layers.Conv1D(output_shape[-1], kernel_size=1, padding='same', activation='linear')(x)
    model = Model(inputs=input, outputs=outputs)
    return model """
    
def ResNet1D(input_shape=(2048, 2), output_shape=(2048, 2), num_blocks=4, filter1=32, kernel_size=3, print_summary=True):
    input = Input(shape=input_shape)
    x = input
    signal_length = input_shape[0]
    skip_connection = None

    for i in range(num_blocks):
        if i == 1:  # Store the output of the early convolutional layer
            skip_connection = x

        if i < num_blocks // 2 and signal_length > 32:
            x = conv_block(x, filter1, kernel_size=kernel_size, strides=2)
            signal_length /= 4
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        elif i < num_blocks // 2:
            x = conv_block(x, filter1, kernel_size=kernel_size, strides=1)
            if print_summary:
                print(f"Signal length {i}: {signal_length}")

        elif i >= num_blocks // 2 and signal_length >= 32 and signal_length < input_shape[0]:
            x = convT_block(x, filters=filter1, kernel_size=kernel_size, strides=2)
            signal_length *= 4
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        elif i > num_blocks // 2:
            x = convT_block(x, filters=filter1, kernel_size=kernel_size, strides=1)
            if print_summary:
                print(f"Signal length {i}: {signal_length}")

        if i == num_blocks - 2:  # Add the skip connection to the later transposed convolutional layer
            skip_connection = tf.keras.layers.Conv1D(filters=filter1, kernel_size=1, padding='same')(skip_connection)
            skip_connection = UpSampling1D(size=4)(skip_connection)  # Upsample to match the shape
            x = tf.keras.layers.add([x, skip_connection])

        if i % 2 == 0:
            x = ERB(x, filter1, kernel_size=kernel_size)
            if print_summary:
                print(f"ERB at block {i}, signal length: {signal_length}")

    outputs = tf.keras.layers.Conv1D(output_shape[-1], kernel_size=1, padding='same', activation='linear')(x)
    model = Model(inputs=input, outputs=outputs)
    return model
###################################################################################################################

def multi_channel_cnn(input_shape=(2048, 6), output_shape=(2048, 2), num_blocks=4, filter1=32, kernel_size=3, print_summary=True):
    input = Input(shape=input_shape)
    x = input
    signal_length = input_shape[0]
    skip_connection = None

    for i in range(num_blocks):
        if i == 1:  # Store the output of the early convolutional layer
            skip_connection = x

        if i < num_blocks // 2 and signal_length > 32:
            x = conv_block(x, filter1, kernel_size=kernel_size, strides=2)
            signal_length //= 4
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        elif i < num_blocks // 2:
            x = conv_block(x, filter1, kernel_size=kernel_size, strides=1)
            if print_summary:
                print(f"Signal length {i}: {signal_length}")

        elif i >= num_blocks // 2 and signal_length >= 32 and signal_length < input_shape[0]:
            x = convT_block(x, filters=filter1, kernel_size=kernel_size, strides=2)
            signal_length *= 4
            if print_summary:
                print(f"Signal length {i}: {signal_length}")
        elif i > num_blocks // 2:
            x = convT_block(x, filters=filter1, kernel_size=kernel_size, strides=1)
            if print_summary:
                print(f"Signal length {i}: {signal_length}")

        if i == num_blocks - 2:  # Add the skip connection to the later transposed convolutional layer
            skip_connection = tf.keras.layers.Conv1D(filters=filter1, kernel_size=1, padding='same')(skip_connection)
            skip_connection = UpSampling1D(size=4)(skip_connection)  # Upsample to match the shape
            x = tf.keras.layers.add([x, skip_connection])

        if i % 2 == 0:
            x = ERB(x, filter1, kernel_size=kernel_size)
            if print_summary:
                print(f"ERB at block {i}, signal length: {signal_length}")

    # Flatten the output for LSTM layers
    x = Flatten()(x)
    
    # Reshape the flattened output to 3D shape for LSTM layers
    x = Reshape((signal_length, -1))(x)
    
    # LSTM layers
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer
    x = Dense(output_shape[-1]*output_shape[0], activation='linear')(x)
    x = Reshape((output_shape[0], output_shape[-1]))(x)
    outputs = Conv1D(output_shape[-1], kernel_size=1, padding='same', activation='linear')(x)
    
    model = Model(inputs=input, outputs=outputs)
    return model

def main():
    available_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(available_devices[4], 'GPU')
    print(f"visible devices: {tf.config.get_visible_devices()}")
    # Test of multi_channel_cnn
    dummy_input, _ = load_from_directory("/home/stud/casperc/bhome/Project3_DL_sigpros/generated_data/train", num_signals=1, input_keys= ["augmented_ifft", "a", "b", "filtered_signal"], output_keys=["original", "baseline", "NAA", "Cr", "Cho"])
    model = multi_channel_cnn(input_shape=(2048, 8), output_shape=(2048, 10), num_blocks=20)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()