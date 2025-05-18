# -*- coding: utf-8 -*-
"""
=========================================================================================
Main Training Script for a Thermodynamics-Informed Deep Learning Material Model
=========================================================================================

Purpose:
-----------------------------------------------------------------------------------------
This script serves as the main execution driver for training and evaluating a
Physics-Informed Deep Learning (PIDL) model. The model is designed to predict the
complex mechanical behavior of materials, specifically short fiber-reinforced
nanoparticle-filled epoxies under various ambient conditions, by enforcing
thermodynamic consistency[cite: 4, 5].

The framework integrates Long Short-Term Memory (LSTM) networks with Feed-Forward
Neural Networks (FFNNs)[cite: 6]. LSTMs are employed to predict the evolution of internal
state variables, which characterize the material's history-dependent behavior and
internal dissipation processes[cite: 6]. Another FFNN is utilized to approximate the
material's free-energy function, defining the thermodynamic state of the system[cite: 7].

This script orchestrates the entire workflow:
1.  **Data Loading and Preprocessing:**
    * Loads experimental data (e.g., from cyclic loading-unloading tests [cite: 9])
        using utility functions from `misc.py`. This data typically includes
        stress-strain curves under various conditions (e.g., temperature, moisture,
        volume fractions of constituents [cite: 10]).
    * The raw inputs from experiments are processed into features suitable for the
        neural network. These can include kinematic measures of strain (related to the
        Right Cauchy-Green tensor $C^{t+1}$ [cite: 127]), timestep information ($\Delta t$ [cite: 128]),
        and parameters representing ambient conditions like moisture content ($w_m^{t+1}$),
        nanoparticle volume fraction ($v_{np}^{t+1}$), fiber volume fraction ($v_f^{t+1}$),
        and temperature ($\Theta^{t+1}$)[cite: 128, 129].
    * Performs crucial preprocessing steps like data normalization (scaling inputs
        and outputs to a range like [-1, 1]) [cite: 137] and structuring data into sequences
        for time-series analysis.

2.  **Hyperparameter Configuration:**
    * Allows specification of neural network architecture parameters (e.g., layer sizes,
        number of internal variables) and training settings (e.g., learning rate,
        number of epochs, batch size).

3.  **Model Initialization and Training:**
    * Initializes the custom PIDL model defined in `DL_model.py`. This model
        encapsulates the LSTMs, FFNNs, and the mechanisms for enforcing thermodynamic
        principles (e.g., non-negative dissipation, stress derivation from free energy).
    * Trains the model using the processed experimental data.

4.  **Evaluation and Postprocessing:**
    * Evaluates the trained model's performance on both training and validation datasets.
    * Saves various outputs, including predicted stress, learned free energy,
        dissipation rates, internal variable evolution, and ground truth data, into
        structured directories for analysis and visualization.

Dependencies:
    - tensorflow: For building and training the neural network.
    - numpy: For numerical operations.
    - matplotlib: For plotting data.
    - os, shutil, pathlib: For file and directory manipulation.
    - misc.py: Contains utility functions for data loading (e.g., `read_dictionaries_exp`)
               and initial processing specific to the experimental data format.
    - DL_model.py: Defines the core PIDL neural network architecture (`DL` class),
                   including LSTMs, FFNNs, and the implementation of thermodynamic
                   consistency.

Adapting this Script for Other Applications:
-----------------------------------------------------------------------------------------
Researchers aiming to adapt this framework for different material systems or problems
(e.g., alloy design, different types of composites) would typically need to:
    1.  Modify data loading and feature extraction in `misc.py` to suit their specific
        input data format and relevant physical parameters.
    2.  Adjust the feature selection, reordering, and concatenation logic in Section 3
        of this script to match their new input features.
    3.  Potentially redesign parts of the neural network architecture or the physics-informed
        constraints within `DL_model.py` to reflect the physics of the new system.
    4.  Re-tune hyperparameters (Section 2 of this script) for optimal performance on the
        new problem.
"""
import sys
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt # For plotting data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow informational messages (1=info, 2=warnings, 3=errors)
from pathlib import Path # For object-oriented path manipulation
from misc import read_dictionaries_exp # Custom function from misc.py for loading experimental data
import shutil # For directory operations (e.g., rmtree)

# Suppress verbose TensorFlow v1 logging if active through tf.compat.v1
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# GPU Configuration:
# Attempt to configure CUDNN paths for GPU usage if nvidia.cudnn is available.
# This is primarily relevant for Linux systems with specific CUDA/cuDNN setups.
try:
    import nvidia.cudnn
    # Get the parent directory of the nvidia.cudnn module
    cudnn_path = Path(nvidia.cudnn.__file__).parent
    # Construct the path to the cuDNN library directory
    cudnn_lib_path = cudnn_path / "lib"
    # Prepend the cuDNN library path to LD_LIBRARY_PATH environment variable
    os.environ["LD_LIBRARY_PATH"] = str(cudnn_lib_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    print("NVIDIA cuDNN path configured using nvidia.cudnn package.", file=sys.stderr)
except ImportError:
    print("nvidia.cudnn module not found. Proceeding without custom cuDNN path modification.", file=sys.stderr)
    pass # If nvidia.cudnn is not installed, continue without modifying paths.

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), file=sys.stderr)

# Configure GPU memory growth.
# This prevents TensorFlow from allocating all GPU memory at startup,
# allowing for more flexible GPU memory usage, especially when sharing GPUs or running multiple models.
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: # Check if any GPUs are detected by TensorFlow
    try:
      # Set memory growth to True for the first detected GPU.
      # This means TensorFlow will only allocate memory as it's needed.
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
      print(f"Memory growth set to True for GPU: {physical_devices[0].name}", file=sys.stderr)
    except RuntimeError as e:
      # This error can occur if memory growth is set after the GPU has already been initialized.
      print(f"Could not set memory growth for GPU: {e}. It might already be initialized.", file=sys.stderr)
      pass
else:
    print("No GPUs detected by TensorFlow. Running on CPU.", file=sys.stderr)

# --- 1. Data Loading Configuration ---
# Define the number of raw features expected from each data file loaded by `read_dictionaries_exp`
# and the number of primary response variables (targets).
# These values are passed to `read_dictionaries_exp` in `misc.py`.
numFeatures = 6  # Number of features constructed by `read_dictionaries_exp` in `misc.py`.
                 # This includes strain, nv, deltaT, temp, zita, and stress (as an input feature).
numResponses = 1 # Number of primary output responses from the data (e.g., stress_11).

# Define root directory for TRAINING experimental data.
# Assumes data is in a subdirectory named 'data_experiments_train' relative to the script's current directory.
root_train_data = os.path.join(os.path.abspath(os.curdir), "data_experiments_train")
# Count the number of .mat files to inform `read_dictionaries_exp`.
# Adjust the pattern if using other file types or naming conventions.
pattern_train = 'epoxy_*_*_*.mat' # File pattern to match experimental data files.
try:
    nr_files_train = len([
        entry for entry in os.listdir(root_train_data)
        if os.path.isfile(os.path.join(root_train_data, entry)) and entry.endswith('.mat') # Ensure it's a .mat file
    ])
except FileNotFoundError:
    print(f"Error: Training data directory not found at {root_train_data}", file=sys.stderr)
    sys.exit(1) # Exit script if data directory is missing

print(f"Loading training data from: {root_train_data} using pattern: {pattern_train}", file=sys.stderr)
# Load training data using the custom function from `misc.py`.
# `input_data_exp` will contain features like strain, environmental conditions, and previous stress.
# `output_data_exp` will contain the target stress.
# `numObservations_exp` is the number of experiment files loaded.
input_data_exp, output_data_exp, numObservations_exp = read_dictionaries_exp(
    root_train_data, pattern_train, nr_files_train, numFeatures, numResponses
)

# Define root directory for VALIDATION experimental data.
root_val_data = os.path.join(os.path.abspath(os.curdir), "data_experiments_validation")
pattern_val = 'epoxy_*_*_*.mat'
try:
    nr_files_val = len([
        entry for entry in os.listdir(root_val_data)
        if os.path.isfile(os.path.join(root_val_data, entry)) and entry.endswith('.mat')
    ])
except FileNotFoundError:
    print(f"Error: Validation data directory not found at {root_val_data}", file=sys.stderr)
    sys.exit(1)

print(f"Loading validation data from: {root_val_data} using pattern: {pattern_val}", file=sys.stderr)
input_data_exp_val, output_data_exp_val, numObservations_exp_val = read_dictionaries_exp(
    root_val_data, pattern_val, nr_files_val, numFeatures, numResponses
)

print(f"Number of training experiments loaded: {numObservations_exp}", file=sys.stderr)
print(f"Number of validation experiments loaded: {numObservations_exp_val}", file=sys.stderr)

# --- 2. Hyperparameter Configuration ---
# These hyperparameters define the neural network architecture (passed to DL_model.py)
# and the training process. They often require tuning for optimal performance on a given dataset.

# Network Architecture Parameters
layer_size = 30             # Number of units (neurons) in general LSTM and Dense hidden layers.
layer_size_fenergy = 30     # Number of units in hidden layers of the FFNN predicting free energy.
internal_variables = 8      # Number of internal state variables the model learns to predict.
                            # These variables capture history-dependent effects. [cite: 6]
print(f'General Layer Size: {layer_size}', file=sys.stderr)
print(f'Free Energy Network Layer Size: {layer_size_fenergy}', file=sys.stderr)
print(f'Number of Internal Variables: {internal_variables}', file=sys.stderr)

# Parameters for Adaptive Loss Weighting (if used by the model in DL_model.py)
# The paper mentions an adaptive rule for weighting the dissipation loss term[cite: 150, 151].
# These specific parameters might relate to such a scheme if implemented.
initiate_adaptive_weight = 999999 # Iteration to start adapting (a very high value might mean it's off or controlled differently).
update_adapt_frequency = 100      # Frequency (in epochs/steps) of adaptive constant updates.
beta_adaptive_weight = 1.0        # Example initial beta value for an adaptive weight.
decay_rate_beta = 0.9             # Decay rate for beta.
decay_steps_beta = 1000           # Steps over which beta decays.
initial_adapt_const_val = 10      # Another example initial value for an adaptive constant.

# Training Settings
learning_rate = 0.0001      # Initial learning rate for the optimizer (e.g., Adam, Nadam).
num_epochs = 5000           # Maximum number of training epochs.
L2_regularization_strength = 1e-6 # Strength of L2 regularization (penalty on large weights to prevent overfitting).
batch_size = 16             # Number of samples processed before the model's weights are updated.
# print_out_interval = 100  # Interval for printing training progress (likely used within DL_model.setTraining)
timesteps = 1000            # Number of time steps to use from each experiment's sequence for training.
                            # This defines the sequence length input to the LSTMs.
timesteps_val = timesteps   # Using the same sequence length for validation.

# --- 3. Data Preparation for Neural Network Input ---
# This section prepares the final input tensors for the neural network.
# It involves selecting relevant timesteps, choosing/reordering features, and structuring
# the data as sequences (inputs at time 't' to predict outputs at 't+1').

# Prepare full experimental output sequences (e.g., stress over time)
# Slicing up to `timesteps_val` (which is `timesteps`)
# Shape: (num_experiments, timesteps, numResponses)
all_y_exp = output_data_exp[:, :timesteps_val, :]

# Prepare full experimental input sequences.
# `input_data_exp` from `misc.py` has features: [strain, nv, deltaT, temp, zita, stress_prev_step_raw]
# This step reorders some of these features and concatenates the (target) stress `all_y_exp`
# to form the final input tensor `all_x_exp`.
# The reordering `[0,2,1,4,3]` selects:
# input_data_exp column 0 (strain) -> becomes feature 0 of the selected 5
# input_data_exp column 2 (deltaT) -> becomes feature 1
# input_data_exp column 1 (nv)     -> becomes feature 2
# input_data_exp column 4 (zita)   -> becomes feature 3
# input_data_exp column 3 (temp)   -> becomes feature 4
# Then, `all_y_exp` (stress) is appended as the 6th feature.
# This makes the input `all_x_exp` = [strain, deltaT, nv, zita, temp, stress_current_or_target]
# The stress here (`all_y_exp`) is effectively the stress at the *current* timestep if we are
# predicting the *next* timestep's stress. This is crucial for recurrent models.
selected_feature_indices_from_input_data_exp = [0, 2, 1, 4, 3] # Indices for reordering
all_x_exp = np.concatenate(
    (input_data_exp[:, :timesteps, selected_feature_indices_from_input_data_exp], all_y_exp),
    axis=-1 # Concatenate along the last axis (features)
)
# Shape of all_x_exp: (num_experiments, timesteps, num_reordered_features + numResponses)
# which is (num_experiments, timesteps, 5 + 1 = 6)

# Repeat for validation data
all_y_exp_val = output_data_exp_val[:, :timesteps_val, :]
all_x_exp_val = np.concatenate(
    (input_data_exp_val[:, :timesteps, selected_feature_indices_from_input_data_exp], all_y_exp_val),
    axis=-1
)

# Use these combined tensors as the base for training/validation sets
all_x_for_model = all_x_exp
all_y_for_model_targets = all_y_exp # This is the stress output, matching `timesteps` length

# Separate into training inputs and a specific feature for plotting (e.g., strain)
train_x_sequences = all_x_for_model[:, :timesteps, :] # Model input sequences for training
# `E_grla_train` extracts the first feature from `train_x_sequences` (which is the reordered strain).
# This is used for plotting stress-strain curves.
# Shape: (num_experiments, timesteps, 1)
E_grla_train_plot = np.expand_dims(train_x_sequences[:, :timesteps, 0], axis=-1)

# Corresponding target sequences for training (stress)
train_y_targets_full = all_y_for_model_targets[:, :timesteps, :]

# Repeat for validation
validation_x_sequences = all_x_exp_val[:, :timesteps, :]
E_grla_val_plot = np.expand_dims(validation_x_sequences[:, :timesteps, 0], axis=-1)
validation_y_targets_full = all_y_exp_val[:, :timesteps_val, :]


# For time-series prediction: the model typically predicts y(t+1) based on inputs x(t).
# Thus, the target sequence `train_y_targets_shifted` should start from the second timestep
# of the original target sequence.
# `train_x_sequences` (inputs) run from t=0 to t=N-1 (length N = `timesteps`).
# `train_y_targets_shifted` (targets) will run from original y at t=1 to t=N (length N-1, if N is original length).
# So, train_y_targets_full[:, 1:, :] gives targets from y_1, y_2, ..., y_{N-1}.
# Sequence length becomes `timesteps - 1`.
train_y_targets_shifted = train_y_targets_full[:, 1:, :]
validation_y_targets_shifted = validation_y_targets_full[:, 1:, :]

# `train_x_sequences` (inputs) are NOT shifted here.
# This means: train_x_sequences[sample, time_j, :] is input at time j,
# and it's used to predict train_y_targets_shifted[sample, time_j, :]
# which corresponds to the original stress at time j+1.
# The DL_model.py handles this sequence length difference internally.

# Plot initial training and validation data (e.g., Stress vs. Strain)
# This visualization helps to verify data loading and understand its characteristics.
# Adapt plot labels if features represent different physical quantities.
print("Plotting initial training data (Stress vs. Key Input Feature)...", file=sys.stderr)
for i in range(train_x_sequences.shape[0]): # Iterate over each experiment
    plt.figure(figsize=(8, 6))
    # Plotting E_grla (strain) from t=1 up to `timesteps-1` against train_y (stress) which is already shifted (length `timesteps-1`).
    # E_grla_train_plot[i, 1:timesteps, 0] ensures matching sequence lengths for plotting.
    plt.plot(E_grla_train_plot[i, 1:timesteps, 0], train_y_targets_shifted[i, :, 0])

    # Extract metadata for plot titles from the first timestep (t=0) of `train_x_sequences`.
    # These indices correspond to the feature order in `train_x_sequences`:
    # Feature 2: NV-like parameter (original input_data_exp col 1)
    # Feature 3: Zita-like parameter (original input_data_exp col 4)
    # Feature 4: Temp-like parameter (original input_data_exp col 3)
    # Example: 'Zita' might represent moisture state, 'NV' nanoparticle content, 'Temp' temperature.
    # These are specific to the experimental setup of the reference paper.
    title_str = (
        f'Train Exp {i}: Param_NV-like = {train_x_sequences[i,0,2]:.2f}, '
        f'Param_Zita-like = {train_x_sequences[i,0,3]:.2f}, '
        f'Param_Temp-like = {train_x_sequences[i,0,4]:.2f}'
    )
    plt.title(title_str)
    plt.xlabel("Key Input Feature for Plot (e.g., Strain)")
    plt.ylabel("Target Output (e.g., Stress)")
    plt.grid(True)
    plt.pause(0.05) # Pause to allow plot to render; adjust as needed.

print("Plotting initial validation data...", file=sys.stderr)
for i in range(validation_x_sequences.shape[0]):
    plt.figure(figsize=(8, 6))
    plt.plot(E_grla_val_plot[i, 1:timesteps, 0], validation_y_targets_shifted[i, :, 0])
    title_str = (
        f'Validation Exp {i}: Param_NV-like = {validation_x_sequences[i,0,2]:.2f}, '
        f'Param_Zita-like = {validation_x_sequences[i,0,3]:.2f}, '
        f'Param_Temp-like = {validation_x_sequences[i,0,4]:.2f}'
    )
    plt.title(title_str)
    plt.xlabel("Key Input Feature for Plot (e.g., Strain)")
    plt.ylabel("Target Output (e.g., Stress)")
    plt.grid(True)
    plt.pause(0.05)
# Consider `plt.show()` if plots should remain open, or `plt.savefig()` to save them.

# Determine the number of input features for the neural network.
# `train_x_sequences` has shape (num_experiments, timesteps, num_final_features)
num_final_input_features = train_x_sequences.shape[2]
print(f"Number of features per timestep for NN input: {num_final_input_features}", file=sys.stderr) # Should be 6

# --- 4. Data Normalization ---
# Normalize input and output data to the range [-1, 1]. This is crucial for stable NN training.
# Min-Max normalization formula used: normalized = (value - min) / (max - min) * 2 - 1
# Or, as implemented: scaled_value = (value - offset_m) / scale_s * 2 - 1
# where offset_m is min_value, and scale_s is (max_value - min_value).
# The paper uses mf = (f_max + f_min)/2 and sf = (f_max - f_min)/2, then scaled = (x - mf)/sf.
# The script uses (x - min_val) / (max_val - min_val) * 2 - 1, which is equivalent.

print("Normalizing training and validation data...", file=sys.stderr)

# Normalization of INPUT features for the model (`all_x_for_model`).
# Calculate min/max scaling factors based on the ENTIRE training input dataset (`all_x_for_model`).
# These factors will then be applied to both training and validation sets.
# max_values_x / min_values_x will have shape (num_final_input_features,).
max_values_x = np.max(np.max(all_x_for_model, axis=1), axis=0) # Max across all experiments and timesteps for each feature
min_values_x = np.min(np.min(all_x_for_model, axis=1), axis=0) # Min for each feature

# `s_all_inputs` is the range (max - min) for each input feature.
s_all_inputs = max_values_x - min_values_x
# Prevent division by zero if a feature is constant (range is zero) by setting its scale to 1.0.
s_all_inputs[s_all_inputs == 0] = 1.0
# `m_all_inputs` is the offset (min value) for each input feature.
m_all_inputs = min_values_x

# Apply normalization to the input sequences for training and validation.
# `train_x_sequences` and `validation_x_sequences` are used here.
normalized_train_input = (train_x_sequences - m_all_inputs) / s_all_inputs * 2.0 - 1.0
normalized_validation_input = (validation_x_sequences - m_all_inputs) / s_all_inputs * 2.0 - 1.0

# Store individual scaling components for clarity, and pass `m_all_inputs`, `s_all_inputs` to the model.
# These names (s_c11, s_dt, etc.) reflect the order of features in `all_x_for_model`:
# [strain, deltaT, nv, zita, temp, stress_from_all_y_exp]
s_c11_strain_like, s_dt_timestep, s_vp_nv_like, s_zita_like, s_temp_like, s_sig11_prev_stress = s_all_inputs
m_c11_strain_like, m_dt_timestep, m_vp_nv_like, m_zita_like, m_temp_like, m_sig11_prev_stress = m_all_inputs
max_c_strain_like = max_values_x[0] # Max value of the first feature (strain-like)

# Normalization of OUTPUT targets (e.g., stress: `all_y_for_model_targets`).
# Calculate min/max scaling factors based on the ENTIRE training output dataset.
max_values_y = np.max(np.max(all_y_for_model_targets, axis=1), axis=0) # Max for each response variable
min_values_y = np.min(np.min(all_y_for_model_targets, axis=1), axis=0) # Min for each response

s_out_targets = max_values_y - min_values_y # Range for each response
s_out_targets[s_out_targets == 0] = 1.0 # Prevent division by zero
m_out_targets = min_values_y # Offset for each response

# Apply normalization to the SHIFTED target sequences for training and validation.
normalized_train_output = (train_y_targets_shifted - m_out_targets) / s_out_targets * 2.0 - 1.0
normalized_validation_output = (validation_y_targets_shifted - m_out_targets) / s_out_targets * 2.0 - 1.0

# Scaling factors for the primary target (assuming single response, e.g., stress).
s_sig11_target = s_out_targets[0]
m_sig11_target = m_out_targets[0]
max_sig11_target_val = max_values_y[0]

# `max_psi` is a heuristic for scaling the free energy (Psi) back to physical units, if Psi is predicted.
# This calculation is specific to the model in the reference paper.
max_psi_heuristic = max_sig11_target_val * max_c_strain_like

# Convert NumPy arrays to TensorFlow tensors for use with the TensorFlow model.
train_x_tf = tf.convert_to_tensor(normalized_train_input, dtype=tf.float32)
train_y_tf = tf.convert_to_tensor(normalized_train_output, dtype=tf.float32)
# Unnormalized versions can be useful for debugging or for specific calculations
# that might occur inside the model if it needs access to original scales.
# train_x_unnormalized_tf = tf.convert_to_tensor(train_x_sequences, dtype=tf.float32)

val_x_tf = tf.convert_to_tensor(normalized_validation_input, dtype=tf.float32)
val_y_tf = tf.convert_to_tensor(normalized_validation_output, dtype=tf.float32)
# val_x_unnormalized_tf = tf.convert_to_tensor(validation_x_sequences, dtype=tf.float32)


# --- 5. Create Folders for Postprocessing ---
# Define a base directory for all experiment outputs to keep them organized.
# Output directories are created (or cleared and recreated) for each run.
dir_base_results = "./experiment_outputs_pinn" # Changed name for clarity
if not os.path.exists(dir_base_results):
    os.makedirs(dir_base_results)

# Define specific subdirectories for different types of saved data.
output_dirs_config = {
    'ground_truth_train': os.path.join(dir_base_results, 'ground_truth_train_stress_unnorm'),
    'predictions_train': os.path.join(dir_base_results, 'predictions_train_all_norm'),
    'model_checkpoints': os.path.join(dir_base_results, 'model_checkpoints_keras'), # For Keras model.save or custom checkpoints
    'predictions_validation': os.path.join(dir_base_results, 'predictions_validation_all_norm'),
    'ground_truth_validation': os.path.join(dir_base_results, 'ground_truth_validation_stress_unnorm'),
    'input_features_train_unnorm': os.path.join(dir_base_results, 'input_features_train_unnorm'),
    'input_features_validation_unnorm': os.path.join(dir_base_results, 'input_features_validation_unnorm'),
    'key_feature_plot_train_unnorm': os.path.join(dir_base_results, 'key_feature_plot_train_unnorm'), # e.g., strain
    'key_feature_plot_validation_unnorm': os.path.join(dir_base_results, 'key_feature_plot_validation_unnorm'),
    'model_weights_final': os.path.join(dir_base_results, 'model_weights_final') # For model.save_weights
}
# The original script used names like 'stress_exact', 'final_predictions', etc.
# which are kept here but mapped to more descriptive internal keys.
# The paths are constructed to be inside `dir_base_results`.

# Original directory names from the script:
# dir_stress_exact_train_orig = './stress_exact'
# dir_final_predictions_train_orig= './final_predictions'
# dir_checkpoints_orig = './checkpoints' (standard Keras checkpoints during training)
# dir_final_validation_pred_orig = './final_validation'
# dir_stress_exact_validation_orig = './stress_exact_validation'
# dir_input_train_orig = './input'
# dir_input_validation_orig = './input_validation'
# dir_strain_train_orig = './strain' (this is E_grla_train_plot)
# dir_strain_validation_orig = './strain_validation' (this is E_grla_val_plot)
# dir_checkpoints_v2_orig = './checkpoints_v2' (Purpose unclear, maybe for different save format)
# dir_weights_orig = './weights' (Final weights after training)

# Mapping to original directory names for output saving compatibility:
compat_output_dirs = {
    'stress_exact_train_orig': './stress_exact',
    'final_predictions_train_orig': './final_predictions',
    'checkpoints_training_orig': './checkpoints', # Keras default in DL_model.py might use this
    'final_validation_pred_orig': './final_validation',
    'stress_exact_validation_orig': './stress_exact_validation',
    'input_train_unnorm_orig': './input',
    'input_validation_unnorm_orig': './input_validation',
    'strain_plot_train_unnorm_orig': './strain',
    'strain_plot_validation_unnorm_orig': './strain_validation',
    'checkpoints_v2_orig': './checkpoints_v2', # Keeping this, though its exact use might need review from DL_model
    'model_weights_final_orig': './weights' # For final ThermoANN.save_weights
}

# Create directories, removing them if they already exist to ensure a clean run.
# Using the original directory names for this step.
for original_dir_path in compat_output_dirs.values():
    if os.path.exists(original_dir_path):
        shutil.rmtree(original_dir_path) # Removes the directory and its contents
    os.makedirs(original_dir_path) # Creates the directory
print(f"Output directories created/cleared.", file=sys.stderr)


# --- 6. Model Initialization and Training ---
# The custom PIDL model is defined in `DL_model.py`.
from DL_model import DL # Import the custom Deep Learning model class.

# Verbosity flags for training process
silent_training_stages = False # If True, suppresses some print statements during model setup/training stages.
# silent_model_summary = True    # If True, would suppress model summary if DL class were to print one.

# Initialize the PIDL model (`ThermoANN`).
# Pass scaling factors, network architecture parameters, and the heuristic `max_psi_heuristic`.
# The DL class constructor in `DL_model.py` will use these to set up the model.
ThermoANN = DL(
    s_all=s_all_inputs,         # Scaling factors (ranges) for input features.
    m_all=m_all_inputs,         # Offset values (mins) for input features.
    s_out=s_out_targets,        # Scaling factors for output targets.
    m_out=m_out_targets,        # Offset values for output targets.
    layer_size=layer_size,
    internal_variables=internal_variables,
    layer_size_fenergy=layer_size_fenergy,
    max_psi=max_psi_heuristic,
    training_silent=silent_training_stages
)

# --- Optional: Build model or load weights ---
# If the model is not built in its constructor or on first call, build it explicitly:
# input_shape_for_build = (None, train_x_tf.shape[1], train_x_tf.shape[2]) # (batch_size, timesteps, features)
# ThermoANN.build(input_shape_for_build)

# To load pre-trained weights (e.g., for continuing training or for inference):
# path_to_weights = os.path.join(compat_output_dirs['model_weights_final_orig'], 'ThermoTANN_weights')
# if tf.train.latest_checkpoint(os.path.dirname(path_to_weights)): # Check if checkpoint files exist
#     print(f"Loading pre-trained weights from {path_to_weights}", file=sys.stderr)
#     ThermoANN.load_weights(path_to_weights) # .expect_partial() might be needed if architecture changed slightly
# else:
#     print("No pre-trained weights found or specified. Training from scratch.", file=sys.stderr)

# The model compilation (optimizer, loss) is handled within ThermoANN.setTraining() in DL_model.py.

if not silent_training_stages: print("\n... Starting model training process via setTraining method.", file=sys.stderr)
# Train the model using the custom `setTraining` method defined in `DL_model.py`.
# This method encapsulates the Keras `fit` call and associated setup.
historyTraining = ThermoANN.setTraining(
    # TANN=ThermoANN, # First argument in original was the model itself, often handled by `self`
    normalized_train_input=train_x_tf,      # Normalized training input sequences
    normalized_train_output=train_y_tf,     # Normalized training target sequences (shifted)
    learning_rate_schedule=learning_rate,   # Initial learning rate (schedule created in setTraining)
    num_epochs=num_epochs,
    batch_size_training=batch_size,
    normalized_val_input=val_x_tf,          # Normalized validation input sequences
    normalized_val_output=val_y_tf,         # Normalized validation target sequences (shifted)
    L2_reg=L2_regularization_strength       # Pass L2 strength if used by setTraining to apply to layers
)

if not silent_training_stages: print("\n... Saving final model weights after training.", file=sys.stderr)
# Save the trained model weights to the path specified in `compat_output_dirs`.
final_weights_path = os.path.join(compat_output_dirs['model_weights_final_orig'], 'ThermoTANN_weights')
ThermoANN.save_weights(final_weights_path, save_format='tf') # TensorFlow format (folder with assets, variables, .pb)
print(f"\n... Training completed. Final model weights saved to {final_weights_path}", file=sys.stderr)


# --- 7. Postprocessing: Evaluation and Saving Predictions ---

# Evaluate the model on the TRAINING data (using normalized data).
# The `evaluate` method is standard Keras, called on the compiled model.
print("\n... Evaluating model on training data (normalized).", file=sys.stderr)
# The DL_model.py's setTraining method already compiled ThermoANN.
eval_results_train = ThermoANN.evaluate(train_x_tf, train_y_tf, batch_size=batch_size, verbose=0)
if isinstance(eval_results_train, list): # If multiple metrics (loss + other metrics)
    print(f"Training data evaluation - Loss (MAE + Physics): {eval_results_train[0]:.4f}, Other Metrics: {eval_results_train[1:]}", file=sys.stderr)
else: # Single loss value
    print(f"Training data evaluation - Loss (MAE + Physics): {eval_results_train:.4f}", file=sys.stderr)

# Obtain detailed predictions and internal states from the model for the TRAINING set.
# `obtain_output` is a custom method in `DL_model.py`.
# It returns: unnormalized_stress, psi_sequence, dissipation_rate, z_i_sequence
print("\n... Obtaining predictions and internal states for TRAINING data.", file=sys.stderr)
stress_predict_unnorm_train, psi_sequence_train, dissipation_rate_train, z_internal_vars_train = ThermoANN.obtain_output(
    train_x_tf, # Normalized input
    train_y_tf  # Normalized true output (may or may not be used by obtain_output)
)

# Convert predicted TensorFlow tensors to NumPy arrays for saving.
stress_predict_train_np = stress_predict_unnorm_train.numpy() # This is UNNORMALIZED stress
psi_train_np = psi_sequence_train.numpy()                     # Free energy (likely normalized or scaled by max_psi)
dissipation_rate_train_np = dissipation_rate_train.numpy()    # Dissipation (likely physical scale due to max_psi)
z_train_np = z_internal_vars_train.numpy()                    # Internal variables (likely normalized scale)

print("\n... Saving detailed outputs for TRAINING data to text files.", file=sys.stderr)
# Iterate over each experiment/sample in the training set to save its data.
for i in range(train_x_sequences.shape[0]):
    # Define filenames using original directory structure for compatibility.
    # Saving various predicted (and some ground truth) quantities.
    # Suffixes like '_norm' or '_unnorm' are added for clarity on the state of saved data.

    # Predictions (from ThermoANN.obtain_output)
    # NOTE: psi, dissipation, z might be normalized or scaled. Check DL_model.py for exact output scale.
    # The `stress_predict_train_np` is explicitly unnormalized by `obtain_output`.
    file_psi_train = os.path.join(compat_output_dirs['final_predictions_train_orig'], f"fnergy_{i}.txt") # Free energy
    file_z_train = os.path.join(compat_output_dirs['final_predictions_train_orig'], f"zi_{i}.txt")         # Internal variables
    file_diss_train = os.path.join(compat_output_dirs['final_predictions_train_orig'], f"diss_{i}.txt")    # Dissipation rate
    file_stress_pred_train = os.path.join(compat_output_dirs['final_predictions_train_orig'], f"stress_pred_unnorm_{i}.txt") # UNNORMALIZED predicted stress

    # Ground truth and original inputs (UNNORMALIZED)
    file_stress_exact_train = os.path.join(compat_output_dirs['stress_exact_train_orig'], f"stress_unnorm_{i}.txt") # Unnormalized actual stress
    file_input_features_train = os.path.join(compat_output_dirs['input_train_unnorm_orig'], f"input_unnorm_{i}.txt") # Unnormalized input features to NN
    file_strain_plot_train = os.path.join(compat_output_dirs['strain_plot_train_unnorm_orig'], f"strain_unnorm_{i}.txt") # Unnormalized strain feature for plotting

    # Save data. Reshape to 2D array (timesteps, features_per_timestep) before saving.
    np.savetxt(file_psi_train, psi_train_np[i].reshape(psi_train_np.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_z_train, z_train_np[i].reshape(z_train_np.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_diss_train, dissipation_rate_train_np[i].reshape(dissipation_rate_train_np.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_stress_pred_train, stress_predict_train_np[i].reshape(stress_predict_train_np.shape[1], -1), delimiter=',', fmt='%f')

    # Save unnormalized ground truth target (shifted stress sequence)
    np.savetxt(file_stress_exact_train, train_y_targets_shifted[i].reshape(train_y_targets_shifted.shape[1], -1), delimiter=',', fmt='%f')
    # Save unnormalized input sequences fed to the model
    np.savetxt(file_input_features_train, train_x_sequences[i].reshape(train_x_sequences.shape[1], -1), delimiter=',', fmt='%f')
    # Save unnormalized strain feature used for plotting
    np.savetxt(file_strain_plot_train, E_grla_train_plot[i].reshape(E_grla_train_plot.shape[1], -1), delimiter=',', fmt='%f')
print("\n... Outputs for TRAINING data saved.", file=sys.stderr)


# Repeat for VALIDATION data
print("\n... Evaluating model on validation data (normalized).", file=sys.stderr)
eval_results_val = ThermoANN.evaluate(val_x_tf, val_y_tf, batch_size=batch_size, verbose=0)
if isinstance(eval_results_val, list):
    print(f"Validation data evaluation - Loss (MAE + Physics): {eval_results_val[0]:.4f}, Other Metrics: {eval_results_val[1:]}", file=sys.stderr)
else:
    print(f"Validation data evaluation - Loss (MAE + Physics): {eval_results_val:.4f}", file=sys.stderr)

print("\n... Obtaining predictions and internal states for VALIDATION data.", file=sys.stderr)
stress_predict_unnorm_val, psi_sequence_val, dissipation_rate_val, z_internal_vars_val = ThermoANN.obtain_output(
    val_x_tf, val_y_tf
)

stress_predict_val_np = stress_predict_unnorm_val.numpy()
psi_val_np = psi_sequence_val.numpy()
dissipation_rate_val_np = dissipation_rate_val.numpy()
z_val_np = z_internal_vars_val.numpy()

print("\n... Saving detailed outputs for VALIDATION data to text files.", file=sys.stderr)
for i in range(validation_x_sequences.shape[0]):
    file_psi_val = os.path.join(compat_output_dirs['final_validation_pred_orig'], f"fnergy_{i}.txt")
    file_z_val = os.path.join(compat_output_dirs['final_validation_pred_orig'], f"zi_{i}.txt")
    file_diss_val = os.path.join(compat_output_dirs['final_validation_pred_orig'], f"diss_{i}.txt")
    file_stress_pred_val = os.path.join(compat_output_dirs['final_validation_pred_orig'], f"stress_pred_unnorm_{i}.txt")

    file_stress_exact_val = os.path.join(compat_output_dirs['stress_exact_validation_orig'], f"stress_unnorm_{i}.txt")
    file_input_features_val = os.path.join(compat_output_dirs['input_validation_unnorm_orig'], f"input_unnorm_{i}.txt")
    file_strain_plot_val = os.path.join(compat_output_dirs['strain_plot_validation_unnorm_orig'], f"strain_unnorm_{i}.txt")

    np.savetxt(file_psi_val, psi_val_np[i].reshape(psi_val_np.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_z_val, z_val_np[i].reshape(z_val_np.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_diss_val, dissipation_rate_val_np[i].reshape(dissipation_rate_val_np.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_stress_pred_val, stress_predict_val_np[i].reshape(stress_predict_val_np.shape[1], -1), delimiter=',', fmt='%f')

    np.savetxt(file_stress_exact_val, validation_y_targets_shifted[i].reshape(validation_y_targets_shifted.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_input_features_val, validation_x_sequences[i].reshape(validation_x_sequences.shape[1], -1), delimiter=',', fmt='%f')
    np.savetxt(file_strain_plot_val, E_grla_val_plot[i].reshape(E_grla_val_plot.shape[1], -1), delimiter=',', fmt='%f')
print("\n... Outputs for VALIDATION data saved.", file=sys.stderr)
print(f"\n--- Script execution finished. All results saved in respective subdirectories (e.g., ./final_predictions, ./weights). ---", file=sys.stderr)