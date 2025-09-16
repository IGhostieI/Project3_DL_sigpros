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

from myfunctions_tf import load_from_directory, make_current_time_directory, VisualizePredictionCallback, make_metadata_file, multi_channel_cnn_explicit_conv, build_mrs_unet_1d, multi_channel_cnn
from tensorflow.keras.utils import plot_model

from datetime import datetime
import scipy 
import matplotlib
import matplotlib.pyplot as plt
import time

font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 24}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (20, 60)

import tensorflow as tf
tf.keras.backend.clear_session()
print("###########################")
print("TensorFlow version:", tf.__version__)
print("###########################")
slurm = True if 'SLURM_JOB_ID' in os.environ else False
print("SLURM job:", slurm)

class BalancedChannelMSE(tf.keras.losses.Loss):
    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, y_true, y_pred):
        # y_* shape: [B, T, C]
        per_ch_mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=[0,1])  # [C]
        # Scale per channel by its true energy (stop_grad so it doesn't backprop)
        ch_energy = tf.reduce_mean(tf.square(y_true), axis=[0,1])            # [C]
        weights = tf.stop_gradient(1.0 / (ch_energy + self.eps))             # [C]
        # Normalize weights to keep overall loss scale stable
        weights = weights * (tf.cast(tf.shape(weights)[0], weights.dtype) /
                             (tf.reduce_sum(weights) + self.eps))
        return tf.reduce_sum(weights * per_ch_mse)

def simple_cnn_time_to_freq(input_shape=(2048, 2), output_shape=(2048, 2)):
    """
    Simple CNN for time->freq conversion (real/imag channels).
    Suitable for overfitting small samples (e.g., sinusoid to sum of Lorentzians).
    """
    from tensorflow.keras.layers import Input, Conv1D
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Conv1D(32, 7, activation='relu', padding='same')(inputs)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    outputs = Conv1D(output_shape[-1], 1, activation='linear', padding='same')(x)
    model = Model(inputs, outputs, name="simple_cnn_time_to_freq")
    return model

def main():
    available_devices = tf.config.list_physical_devices('GPU')
    print(f"available devices: {available_devices}")

    if available_devices:
        tf.config.set_visible_devices(available_devices[0], 'GPU')  # Use only the first GPU
        print(f"Using GPU: {available_devices[0]}")
    else:
        print("No GPU available, using CPU.")

    print(f"visible devices: {tf.config.get_visible_devices()}")
    if slurm:
        print(f"SLURM JOB ID: {os.environ['SLURM_JOB_ID']}")
    else:
        print("Not running on SLURM")
    metadata = {
        "description": "U-net",
        "input_key": ["augmented_ifft"], # "augmented_ifft","a", "b", "filtered_full", "filtered_GABA"
        "output_key": ["NAA"], # "original",  
        "batch_size": 1,
        "epochs": 100,
        "optimizer": "Adagrad", #SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
        "learning_rate": 1e-5,
        "loss": "Huber",# "MeanSquaredError", "MeanAbsoluteError", "MeanAbsolutePercentageError", "MeanSquaredLogarithmicError", "CosineSimilarity", "KLDivergence", "Poisson", "Huber", "LogCosh"
        "early_stopping": 100,
        "train_data": 1, # -1 means all data - Updated to the actual number of data after loading
        "validation_data": 1, # -1 means all data - Updated to the actual number of data after loading
        "training_time": "Not yet trained",
        "kernel_initializer": "he_normal", # "GlorotUniform", "HeNormal", "LecunNormal", "LecunUniform", "TruncatedNormal", "RandomNormal", "RandomUniform"
        "num_blocks": 5,
        "kernel_size": 5,
        "num_filters": 16,
    }

    # load and prepare data
    train_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/synth_gen/train_2"
    val_path = "/home/stud/casperc/bhome/Project3_DL_sigpros/synth_gen/val_2"

    train_input, train_output = load_from_directory(train_path, num_signals=metadata["train_data"], input_keys=metadata["input_key"], output_keys=metadata["output_key"], complex_data=True)
    validation_input, validation_output = load_from_directory(val_path, num_signals=metadata["validation_data"], input_keys=metadata["input_key"], output_keys=metadata["output_key"], complex_data=True)
    
    metadata["train_data"] = train_input.shape
    metadata["validation_data"] = validation_input.shape
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output)).batch(batch_size=metadata["batch_size"])
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_input, validation_output)).batch(batch_size=metadata["batch_size"])
    
    # create save directory
    os.makedirs(os.path.join(os.getcwd(), "Final_for_paper"), exist_ok=True)
    main_folder_path = os.path.join(os.getcwd(), "Final_for_paper")
    path = make_current_time_directory(main_folder_path=main_folder_path, data_description=f"{metadata['input_key'][:]}_{metadata['output_key'][:]}_Simple_CNN_{metadata['optimizer']}_{metadata['loss']}", make_logs_dir=True)
    log_dir = os.path.join(path, "logs")


    metrics = [
    "mse",                           # Mean Squared Error
    "mae"
    ]
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=metadata["early_stopping"], monitor="val_loss", mode='min', restore_best_weights=True,verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path,"checkpoint.hdf5"), monitor="val_loss", save_best_only=True, verbose=1, save_weights_only=True, mode='min'),
        VisualizePredictionCallback(output_dir=log_dir, how_often=1, validation_data=validation_dataset, total_epochs=metadata["epochs"], input_keys=metadata["input_key"], output_keys=metadata["output_key"])
        ]
    
    # build model
    #model = ResNet1D(num_blocks=20, kernel_size=16, input_shape=(2048,2*len(metadata["input_key"])), output_shape=(2048,2*len(metadata["output_key"])))
    model = build_mrs_unet_1d( in_ch=2*len(metadata["input_key"]), out_ch=2*len(metadata["output_key"]), use_bias=True, name='MRS_UNet1D', filt_down=(16, 32, 64, 80,  96, 128, 256), kernel_sizes = (7, 5, 5, 3, 3, 2, 2))
    #model = simple_cnn_time_to_freq(input_shape=(2048, 2*len(metadata["input_key"])), output_shape=(2048, 2*len(metadata["output_key"])))
    #model.summary(line_length=120)
    #model = multi_channel_cnn(num_blocks=metadata["num_blocks"], kernel_size=metadata["kernel_size"], input_shape=(2048,2*len(metadata["input_key"])), output_shape=(2048,2*len(metadata["output_key"])), filter1=metadata["num_filters"])
    # compile model
    model.compile(optimizer=metadata["optimizer"], loss=metadata["loss"], metrics=metrics)
    # train model
    print("\nTraining Starts....................")
    start = time.perf_counter()
    model.fit(x=train_dataset, epochs=metadata["epochs"], validation_data=validation_dataset, callbacks=callbacks)
    now = time.perf_counter()
    metadata["training_time"] = f"{(now-start)//3600} hours, {((now-start)%3600)//60} minutes and {round(((now-start)%3600)%60, 3)} seconds"
    print(f"Training took {(now-start)//3600} hours, {((now-start)%3600)//60} minutes and {round(((now-start)%3600)%60, 3)} seconds")
    # Generate train loss curves
    x_axis = np.arange(1, len(model.history.history['loss'])+1)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x_axis, model.history.history['loss'], label='train', linewidth=4, color="blue")
    # Generate validation loss curves
    plt.plot(x_axis, model.history.history['val_loss'], label='validation', linewidth=4, linestyle= "--", color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Loss Curves")
    plt.subplot(3,1,2)
    plt.plot(x_axis, model.history.history['loss'], label='train', linewidth=4, color="blue")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Training Loss")
    plt.subplot(3,1,3)
    plt.plot(x_axis, model.history.history['val_loss'], label='validation', linewidth=4, linestyle= "--", color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Validation Loss")
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)
    plt.savefig(os.path.join(path, "loss_curves.png"))
    metadata["epochs"] = len(model.history.history['val_loss'])
    make_metadata_file(filepath=path, data_description=metadata)
    
    
if __name__ == "__main__":
    main()
