# -*- coding: utf-8 -*-
"""
=========================================================================================
Miscellaneous Utility Functions for Data Loading and Preprocessing
=========================================================================================

Purpose:
-----------------------------------------------------------------------------------------
This module provides a collection of helper functions primarily designed for loading,
preprocessing, and transforming experimental and/or synthetic data for use with the
thermodynamics-informed deep learning material model. The functions handle:

1.  **Data Ingestion:** Reading data from .mat (MATLAB) files, which are assumed to
    contain time-series data such as stress, strain, and time vectors from material tests,
    or synthetic data with a similar structure.
2.  **Data Resampling:** Resizing time-series arrays to a uniform sequence length using
    interpolation. This is crucial for preparing data for recurrent neural networks (RNNs)
    like LSTMs that typically expect fixed-length sequences.
3.  **Feature Engineering (Experiment-Specific):**
    * Extracting specific parameters (e.g., nanoparticle volume fraction, temperature,
        moisture/conditioning indicators) directly from filenames based on a predefined
        naming convention. This is highly specific to the dataset organization of the
        original research.
    * Constructing input feature arrays by combining time-series data (e.g., strain,
        previous step's stress, timestep size) with scalar parameters (e.g., temperature).
4.  **Feature Engineering (Continuum Mechanics - Advanced/Alternative Path):**
    * The `preprocess_input` function performs complex calculations based on continuum
        mechanics to derive features like kinematic invariants of the Right Cauchy-Green
        deformation tensor ($C$) and their derivatives. This is intended for models
        that explicitly use these invariants as inputs, as discussed in theoretical
        sections of the associated research paper[cite: 93, 133]. However, the main
        data loading path used by `Main_ML.py` (`read_dictionaries_exp`) prepares a
        different set of features.

Associated Research:
-----------------------------------------------------------------------------------------
These utilities support the data pipeline for the model described in the paper:
"A thermodynamically consistent physics-informed deep learning material model for
 short fiber/polymer nanocomposites" (Comput. Methods Appl. Mech. Engrg., 2024)[cite: 1, 2].

Notes for Adaptation:
-----------------------------------------------------------------------------------------
Researchers looking to adapt this codebase for different material systems (e.g., alloys),
experimental setups, or data formats will likely need to make significant modifications,
particularly to:
-   `getData_exp()`: Adjust for different keys in .mat files or rewrite entirely if using
    other data formats (e.g., CSV, text).
-   `read_dictionaries_exp()`:
    -   The filename parsing logic for extracting parameters like 'nv', 'zita', 'temp'
        is hardcoded and dataset-specific. This must be adapted or replaced.
    -   The construction of the `input_array` (feature vector) needs to be tailored to
        the relevant input features for the new problem.
-   `preprocess_input()`: This function, with its detailed continuum mechanics derivations,
    is highly specialized for anisotropic hyperelastic/viscoelastic models. For other
    material classes or physics, it would likely be replaced by domain-specific feature
    engineering routines.
-   Array resizing (`resize_array`) and pre-allocation strategies (`max_sequence_len` in
    `read_dictionaries`) might need adjustment based on the characteristics of the new dataset.
"""
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress verbose TensorFlow logging (3 = errors only)
import scipy.io # For loading .mat (MATLAB) files
from fnmatch import fnmatch # For Unix shell-style wildcard matching of filenames

# `tensorflow.keras.layers` and `keras` were imported in the original file but not used.
# `matplotlib.pyplot` was imported in the original file but not used.
# `scipy.ndimage.interpolation` is deprecated in newer SciPy; `scipy.ndimage.zoom` is the direct replacement.
from scipy.ndimage import zoom as ndimage_zoom # Using an alias for clarity

def resize_array(x_original, target_length=1000):
    """
    Resizes a 1D array (extracted from the first column if input is 2D) to a specified
    target length using cubic spline interpolation (default for `zoom`).

    This function is used to ensure all time-series sequences have a uniform length
    before being fed into a neural network (especially LSTMs).

    Args:
        x_original (np.ndarray): The input array. Expected to be 1D or 2D. If 2D,
                                 only the first column (x_original[:,0]) is processed.
        target_length (int, optional): The desired sequence length after resizing.
                                       Defaults to 1000, which is a common sequence length
                                       used in the associated study.

    Returns:
        np.ndarray: The resized array, reshaped to [target_length, 1].

    Note for Adaptation:
        - The choice of `target_length` (1000) is specific to this project's setup.
          It should be chosen based on the characteristic duration of phenomena in new datasets
          or computational constraints.
        - Cubic interpolation is the default for `ndimage_zoom`. Other interpolation orders
          can be specified if needed (e.g., order=1 for linear).
    """
    if x_original.ndim == 1:
        x_data_1d = x_original
    elif x_original.ndim >= 2:
        x_data_1d = x_original[:, 0] # Process only the first column if 2D or higher
    else:
        raise ValueError("Input array x_original must be at least 1-dimensional.")

    if len(x_data_1d) == 0:
        # Handle empty array: return an array of zeros with the target shape.
        # This prevents errors if an empty data file or column is encountered.
        # print(f"Warning: Attempting to resize an empty array. Returning zeros.", file=sys.stderr)
        return np.zeros((target_length, 1))

    # Calculate the zoom factor required to achieve the target length.
    zoom_factor = target_length / len(x_data_1d)

    # Perform the resizing using SciPy's ndimage.zoom function.
    # `ndimage_zoom` performs interpolation. Default order is 3 (cubic spline).
    x_resized_1d = ndimage_zoom(x_data_1d, zoom_factor, order=3)

    # Expand dimensions to make it a column vector [target_length, 1].
    return np.expand_dims(x_resized_1d, axis=-1)


def getData_exp(input_mat_file_path, target_sequence_length=1000):
    """
    Loads and preprocesses data from a single .mat file containing experimental results.
    This function is tailored to the specific structure of .mat files used in the
    original study (e.g., for short fiber/polymer nanocomposites), which typically
    contain stress, strain, and time vector data from mechanical tests.

    Args:
        input_mat_file_path (str): Absolute or relative path to the .mat file.
        target_sequence_length (int, optional): The desired sequence length for the
                                                time-series arrays after resizing.
                                                Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            - stress_res (np.ndarray): Resized stress data, shape [N, 1] where N is
                                       effective length after potential truncation.
            - trueStrain_res (np.ndarray): Resized true strain data (adjusted by +1,
                                           possibly representing stretch ratio $\lambda$),
                                           shape [N, 1].
            - deltaT_res (float): Mean time step of the resized time vector.

    Note for Adaptation:
        - The dictionary keys ('expStress', 'timeVec', 'trueStrain') are specific to the
          .mat files from the original research. These must be changed if your .mat files
          use different variable names.
        - If your experimental data is in a different format (e.g., CSV, TXT, another
          binary format), this entire function will need to be rewritten to parse that format.
        - The operation `trueStrain = (C['trueStrain']... + 1)` suggests that the stored
          'trueStrain' might be logarithmic strain, and adding 1 could be part of a
          conversion to a stretch-like measure, or an offset. This needs to be understood
          based on the raw data definition.
        - The truncation `[:-10]` if `deltaT_res < 0` is a data-specific cleaning step,
          likely to handle minor artifacts at the end of sequences due to interpolation.
          This may not be necessary or may need adjustment for other datasets.
    """
    mat_contents = scipy.io.loadmat(input_mat_file_path)

    # Extract data arrays from the loaded .mat file using expected keys.
    stress_raw = mat_contents['expStress'].astype('float64')
    timeVec_raw = mat_contents['timeVec'].astype('float64')
    # The addition of 1 to 'trueStrain' is a specific transformation.
    # It might convert engineering strain to a stretch-like quantity or handle log strains.
    # For example, if trueStrain is $\ln(L/L_0)$, then exp(trueStrain) = $L/L_0$ (stretch ratio $\lambda$).
    # If it's engineering strain $\epsilon = (L-L_0)/L_0$, then $1+\epsilon = L/L_0 = \lambda$.
    # Assuming it means stretch ratio $\lambda$:
    trueStrain_stretch_ratio_raw = (mat_contents['trueStrain'].astype('float64') + 1.0)

    # Resize the arrays to the target sequence length using the `resize_array` utility.
    stress_res = resize_array(stress_raw, target_length=target_sequence_length)
    timeVec_res = resize_array(timeVec_raw, target_length=target_sequence_length)
    trueStrain_res = resize_array(trueStrain_stretch_ratio_raw, target_length=target_sequence_length)

    # Calculate the mean time step ($\Delta t$) from the resized time vector.
    # `np.diff` calculates the difference between consecutive elements.
    if len(timeVec_res) > 1:
        deltaT_res = np.mean(np.diff(timeVec_res, axis=0))
    else:
        deltaT_res = 0.0 # Or handle as an error/warning if time vector is too short

    # A data cleaning step: if resizing results in a negative average time step
    # (which can happen due to interpolation artifacts if original data has issues or is very short),
    # truncate the last few data points and recalculate deltaT.
    # This is a heuristic fix for potential issues at the end of interpolated sequences.
    if deltaT_res < 0:
        # print(f"Warning: Negative deltaT_res ({deltaT_res:.4e}) detected in {input_mat_file_path}. Truncating data.", file=sys.stderr)
        num_points_to_truncate = 10
        if stress_res.shape[0] > num_points_to_truncate: # Ensure array is long enough to truncate
            stress_res = stress_res[:-num_points_to_truncate]
            timeVec_res = timeVec_res[:-num_points_to_truncate] # Important to also truncate timeVec_res
            trueStrain_res = trueStrain_res[:-num_points_to_truncate]
            if len(timeVec_res) > 1:
                deltaT_res = np.mean(np.diff(timeVec_res, axis=0))
            else:
                deltaT_res = 0.0 # Reset if it becomes too short after truncation
                # print(f"Warning: timeVec_res too short after truncation for {input_mat_file_path}.", file=sys.stderr)
        # else:
            # print(f"Warning: Cannot truncate {num_points_to_truncate} points from sequence of length {stress_res.shape[0]} in {input_mat_file_path}", file=sys.stderr)

    return stress_res, trueStrain_res, deltaT_res


def getData(input_mat_file_path):
    """
    Loads generic 't' (targets/outputs) and 'x' (features/inputs) data from a .mat file.
    This function is simpler than `getData_exp` and might be used for loading synthetic
    data generated from a constitutive model, where 't' could be stress and 'x' could
    be strain and other parameters.

    Args:
        input_mat_file_path (str): Path to the .mat file.

    Returns:
        tuple: A tuple containing:
            - s_data (np.ndarray): Data associated with the key 't' in the .mat file.
            - x_data (np.ndarray): Data associated with the key 'x' in the .mat file.

    Note for Adaptation:
        - This function assumes the .mat file contains variables named 't' and 'x'.
          Modify these keys if your synthetic data uses different variable names.
        - The interpretation of 't' as outputs and 'x' as inputs is based on how
          `read_dictionaries` uses this function.
    """
    mat_contents = scipy.io.loadmat(input_mat_file_path)
    # 's' is often used for stress (output) or target variables.
    s_data = mat_contents['t'].astype('float64') # In `read_dictionaries`, this is treated as outputs.
    # 'x' is often used for strain (input) or feature variables.
    x_data = mat_contents['x'].astype('float64') # In `read_dictionaries`, this is treated as inputs.
    return s_data, x_data


def read_dictionaries(root_path, file_pattern, num_files_to_load,
                      num_features, num_responses, max_sequence_len=10000):
    """
    Reads data from multiple .mat files found by walking a directory tree, using the
    `getData` function. It aggregates the data into NumPy arrays.
    This function seems intended for loading datasets where each file represents one
    experiment or simulation, and the data within files might be structured with
    features in one dimension and timesteps in another.

    Args:
        root_path (str): The root directory to search recursively for .mat files.
        file_pattern (str): A wildcard pattern (e.g., '*.mat') to match filenames.
        num_files_to_load (int): The maximum number of files to load. The function
                                 will stop once this many files are processed.
        num_features (int): The number of input features expected per time step in each file.
                            Used for pre-allocating the `input_array`.
        num_responses (int): The number of output responses expected per time step.
                             Used for pre-allocating the `output_array`.
        max_sequence_len (int, optional): Pre-allocated maximum sequence length for the
                                          NumPy arrays. This should be chosen to be larger
                                          than or equal to the longest sequence in the dataset
                                          to avoid truncation or errors. Defaults to 10000.

    Returns:
        tuple: A tuple containing:
            - input_array (np.ndarray): Aggregated input features from all loaded files.
                                        Shape: [files_actually_read, max_sequence_len, num_features].
            - output_array (np.ndarray): Aggregated output responses from all loaded files.
                                         Shape: [files_actually_read, max_sequence_len, num_responses].
            - k (int): The actual number of files read and processed.

    Note for Adaptation:
        - The dictionaries `input_dict` and `output_dict` are initialized but not used or returned.
          They could be potentially used if a dictionary-based data structure per file were needed.
        - This function assumes `getData(file_path)` returns `(outputs, inputs)`.
        - Data from `.mat` files loaded by `getData` is assumed to be shaped [num_vars, num_timesteps].
          It's then transposed to [num_timesteps, num_vars] before being placed into the
          pre-allocated `input_array` and `output_array`.
        - Pre-allocating with `max_sequence_len` can be memory-intensive if it's much larger
          than typical sequence lengths. An alternative is to append to lists and then convert
          to a NumPy array, possibly with padding for uniform length if required by the model.
    """
    file_counter = 0 # Counter for the number of files read
    # Pre-allocate NumPy arrays for efficiency.
    # Ensure `max_sequence_len` is large enough to accommodate the longest sequence in the dataset.
    input_array = np.zeros((num_files_to_load, max_sequence_len, num_features))
    output_array = np.zeros((num_files_to_load, max_sequence_len, num_responses))

    # The following dictionaries are declared but not populated or used in the current code.
    # They might have been intended for a different data structure or future use.
    # input_dict = {}
    # output_dict = {}

    # Walk through the directory tree starting from `root_path`.
    # `os.walk` yields a 3-tuple (dirpath, dirnames, filenames) for each directory.
    for current_path, _, files_in_current_path in os.walk(root_path):
        for filename in files_in_current_path:
            # Check if the filename matches the specified pattern (e.g., "*.mat").
            if fnmatch(filename, file_pattern):
                if file_counter >= num_files_to_load: # Stop if the desired number of files has been loaded.
                    break
                
                full_file_path = os.path.join(current_path, filename)
                
                # Load data using the `getData` utility, which expects 't' (outputs) and 'x' (inputs).
                # `s_data_from_file` corresponds to 't', treated as outputs.
                # `x_data_from_file` corresponds to 'x', treated as inputs.
                s_data_from_file, x_data_from_file = getData(full_file_path)
                
                # Data from .mat files might be [variables, timesteps]. Transpose to [timesteps, variables].
                # Determine actual sequence length from the loaded data.
                # Assuming inputs 'x' determine the primary sequence length for `input_array`.
                actual_seq_len_inputs = x_data_from_file.shape[1] # Assuming shape [num_vars, timesteps]
                if actual_seq_len_inputs > max_sequence_len:
                    # print(f"Warning: Input sequence in {filename} ({actual_seq_len_inputs}) longer than max_sequence_len ({max_sequence_len}). Truncating.", file=sys.stderr)
                    actual_seq_len_inputs = max_sequence_len
                input_array[file_counter, :actual_seq_len_inputs, :] = x_data_from_file.T[:actual_seq_len_inputs, :]

                actual_seq_len_outputs = s_data_from_file.shape[1] # Assuming shape [num_vars, timesteps]
                if actual_seq_len_outputs > max_sequence_len:
                    # print(f"Warning: Output sequence in {filename} ({actual_seq_len_outputs}) longer than max_sequence_len ({max_sequence_len}). Truncating.", file=sys.stderr)
                    actual_seq_len_outputs = max_sequence_len
                output_array[file_counter, :actual_seq_len_outputs, :] = s_data_from_file.T[:actual_seq_len_outputs, :]
                
                file_counter += 1 # Increment file counter

        if file_counter >= num_files_to_load: # Stop walking directories if enough files are loaded.
            break
            
    if file_counter < num_files_to_load:
        print(f"Warning: Expected to load {num_files_to_load} files, but only read {file_counter}.", file=sys.stderr)
        # Trim the pre-allocated arrays to the actual number of files read.
        input_array = input_array[:file_counter]
        output_array = output_array[:file_counter]

    return input_array, output_array, file_counter


def read_dictionaries_exp(root_path, file_pattern, num_files_to_load,
                          final_num_input_features, # Renamed for clarity (was 'features')
                          num_responses_from_file,  # Was 'responses'
                          target_sequence_length=1000): # Added default, consistent with getData_exp
    """
    Reads and processes experimental data from multiple .mat files using `getData_exp`.
    This function is highly specific to the experimental data and naming conventions of
    the original research[cite: 9, 259]. It extracts primary data (stress, strain, time) from
    .mat files and derives additional scalar features (nv, zita, temp) by parsing
    information encoded in the filenames. These are then assembled into structured
    NumPy arrays for model input and output.

    Args:
        root_path (str): The root directory to search recursively for .mat files.
        file_pattern (str): A wildcard pattern (e.g., 'epoxy_*_*_*.mat') for matching filenames.
        num_files_to_load (int): The maximum number of files to load.
        final_num_input_features (int): The total number of features in the `input_array` to be
                                        constructed per time step (e.g., 6 in the original paper).
        num_responses_from_file (int): The number of response variables extracted directly from
                                       each file by `getData_exp` (e.g., 1 for stress).
        target_sequence_length (int, optional): The sequence length to which data from each
                                                file will be resized. Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            - input_array (np.ndarray): Assembled input features. Shape:
                                        [files_actually_read, target_sequence_length, final_num_input_features].
                                        Features include resized strain, nv, deltaT, temp, zita, and resized stress.
            - output_array (np.ndarray): Assembled output responses (typically resized stress). Shape:
                                         [files_actually_read, target_sequence_length, num_responses_from_file].
            - k (int): The actual number of files read and processed.

    Critical Notes for Adaptation:
        - **Filename Parsing:** The extraction of `nv`, `zita`, `temp` by indexing into the
          filename (`name[6]`, `name[8]`, `name[10]`) is extremely dataset-specific.
          This logic MUST BE MODIFIED if your filenames have a different structure, or if these
          parameters are stored within the data files themselves or need to be generated differently.
          The paper indicates these relate to nanoparticle volume fraction, moisture/conditioning, and temperature[cite: 265, 266, 267].
        - **Feature Construction:** The `input_array` is manually assembled in a specific order:
          [resized_strain, nv, deltaT_resized, temp, zita, resized_stress].
          This order is implicitly expected by the main training script (`Main_ML.py`) when it
          further processes these features. If you change the features or their sources, this
          assembly and the downstream processing in `Main_ML.py` must be updated.
        - **Hardcoded Array Size:** The original code had `np.zeros((nr_files,13700,...))`. This
          has been changed to use `target_sequence_length` for consistency, as `getData_exp`
          already resizes data to this length. If `13700` represented a pre-resize maximum,
          that logic would need to be handled differently (e.g., resizing after loading all raw data).
    """
    file_counter = 0 # Counter for files successfully read and processed

    # Pre-allocate NumPy arrays using the target sequence length.
    # `getData_exp` resizes sequences to `target_sequence_length`.
    input_array = np.zeros((num_files_to_load, target_sequence_length, final_num_input_features))
    output_array = np.zeros((num_files_to_load, target_sequence_length, num_responses_from_file))

    # The dictionaries `input_dict` and `output_dict` were initialized in the original code
    # but not used. They are omitted here for clarity.
    # input_dict = {}
    # output_dict = {}

    for current_path, _, files_in_current_path in os.walk(root_path):
        for filename in files_in_current_path:
            if fnmatch(filename, file_pattern):
                if file_counter >= num_files_to_load:
                    break # Stop if enough files are loaded

                full_file_path = os.path.join(current_path, filename)

                # Load and resize data from the current .mat file.
                # `stress_resized` is the primary output (e.g., stress).
                # `strain_resized` is a primary input (e.g., true strain or stretch).
                # `deltaT_val_resized` is the mean timestep of the resized sequence.
                stress_resized, strain_resized, deltaT_val_resized = getData_exp(
                    full_file_path, target_sequence_length=target_sequence_length
                )

                # Determine the actual sequence length after potential truncation in `getData_exp`.
                # This should ideally be equal to `target_sequence_length`.
                current_actual_seq_len = strain_resized.shape[0]
                if current_actual_seq_len == 0 : # Skip if data loading/resizing failed and returned empty
                    # print(f"Warning: Empty sequence obtained from {filename}. Skipping.", file=sys.stderr)
                    continue


                # ---!!! CRITICAL DATASET-SPECIFIC FILENAME PARSING !!!---
                # The following logic extracts scalar features (nv, zita, temp) by parsing characters
                # at specific positions in the filename. This is highly dependent on the exact
                # naming convention of the original dataset (e.g., 'epoxy_1_1_1_*.mat').
                # This section MUST be adapted or replaced for any other dataset.
                # The paper (Section 4.3, Experiments) describes conditions for training data:
                # Nanoparticle volume fraction $v_{np} \in \{0, 10\}\%$. [cite: 265]
                # Temperature $\Theta \in \{-20, 23, 60\}^{\circ}C$. [cite: 265]
                # Moisture content $w_w \in \{0, 1\}$ (0 for dry, 1 for saturated). [cite: 265]
                # This suggests `name[6]` relates to $v_{np}$, `name[8]` to $w_w$ (zita), `name[10]` to $\Theta$.
                nv, zita, temp = -1, -1, -999 # Default error values
                try:
                    # Example interpretation:
                    # name[6]: '1' -> nv=0 (0% BNP), '2' -> nv=5 (5% BNP used for validation in paper), '3' (else) -> nv=10 (10% BNP)
                    # This mapping needs to align with how these parameters are numerically represented.
                    char_nv = filename[6] # Character at index 6 related to Nanoparticle Volume (NV)
                    if char_nv == '1': nv = 0.0
                    elif char_nv == '2': nv = 5.0 # Paper validation uses 5% BNP [cite: 266]
                    else: nv = 10.0 # Assumed mapping for other values

                    char_zita = filename[8] # Character at index 8 related to Zita (e.g., moisture/condition)
                    if char_zita == '1': zita = 0.0 # e.g., dry state
                    else: zita = 1.0 # e.g., saturated state

                    char_temp = filename[10] # Character at index 10 related to Temperature
                    if char_temp == '1': temp = -20.0
                    elif char_temp == '2': temp = 23.0
                    elif char_temp == '3': temp = 50.0
                    else: temp = 60.0 # Default or fourth category
                except (IndexError, ValueError) as e:
                    print(f"Error parsing filename '{filename}' for features (nv, zita, temp): {e}. Skipping file.", file=sys.stderr)
                    continue # Skip this file if filename parsing fails.
                # ---!!! END OF CRITICAL DATASET-SPECIFIC FILENAME PARSING !!!---

                # Populate the `input_array` for the current file/experiment.
                # The features are assembled in a specific order, assumed by `Main_ML.py`.
                # All scalar features (nv, deltaT_val_resized, temp, zita) are broadcasted
                # across the time dimension (all timesteps in the sequence get the same scalar value).

                # Ensure data fits into pre-allocated array; truncate if current_actual_seq_len > target_sequence_length
                # (though ideally they are equal due to resize_array).
                len_to_fill = min(current_actual_seq_len, target_sequence_length)

                input_array[file_counter, :len_to_fill, 0] = strain_resized[:len_to_fill, 0]  # Feature 0: Resized Strain
                input_array[file_counter, :len_to_fill, 1] = nv                               # Feature 1: NV (e.g., nanoparticle content)
                input_array[file_counter, :len_to_fill, 2] = deltaT_val_resized               # Feature 2: Resized $\Delta t$
                input_array[file_counter, :len_to_fill, 3] = temp                             # Feature 3: Temperature
                input_array[file_counter, :len_to_fill, 4] = zita                             # Feature 4: Zita (e.g., moisture state)
                input_array[file_counter, :len_to_fill, 5] = stress_resized[:len_to_fill, 0] # Feature 5: Resized Stress (used as an input feature)

                # Populate the `output_array` (primary target variable, typically stress).
                output_array[file_counter, :len_to_fill, 0] = stress_resized[:len_to_fill, 0]

                file_counter += 1 # Increment counter for successfully processed files

        if file_counter >= num_files_to_load: # Stop walking if enough files loaded
            break

    if file_counter < num_files_to_load:
        print(f"Warning: Expected to load {num_files_to_load} experimental files, but only read and processed {file_counter}.", file=sys.stderr)
        # Trim the pre-allocated arrays to the actual number of files successfully processed.
        input_array = input_array[:file_counter]
        output_array = output_array[:file_counter]

    return input_array, output_array, file_counter


def preprocess_input(X_batch_raw, num_samples, num_timesteps,
                     num_output_features_X_new, num_output_features_X_post):
    """
    Performs advanced feature engineering based on continuum mechanics principles.
    This function calculates kinematic invariants (e.g., $I_1, I_2, I_3, I_4, I_5$) from
    a deformation gradient $F$ and structural tensors (representing fiber orientations).
    It also computes derivatives of the Right Cauchy-Green tensor ($C = F^T F$)
    with respect to these invariants. These calculations are relevant for formulating
    anisotropic hyperelastic free energy functions.

    **WARNING: This function is highly specialized for the anisotropic hyperelastic
    material model detailed in the theoretical sections of the associated research paper
    (e.g., Section 2.2 and Appendix [cite: 91, 376]). It is VERY UNLIKELY to be directly
    applicable to other problems (like alloy design) or different material models
    without complete replacement or substantial modification.**

    The input `X_batch_raw` is assumed to have a specific flat structure per timestep,
    containing components of $F$, structural tensors, and other scalar parameters.

    Args:
        X_batch_raw (np.ndarray): Raw input data batch. Expected shape:
                                  [num_samples, num_timesteps, num_raw_flat_features].
                                  The internal structure of the last dimension is critical.
        num_samples (int): Number of samples (experiments/simulations) in the batch.
        num_timesteps (int): Number of timesteps per sample.
        num_output_features_X_new (int): Number of features for the primary processed
                                         NN input array `X_new`.
        num_output_features_X_post (int): Number of features for the auxiliary processed
                                          NN input array `X_post` (containing derivatives).

    Returns:
        tuple: A tuple containing:
            - X_new (np.ndarray): Processed primary input features for the NN, typically
                                  containing invariants and scalar parameters.
                                  Shape: [num_samples, num_timesteps, num_output_features_X_new].
            - X_post (np.ndarray): Processed auxiliary input features, typically derivatives
                                   of $C$ w.r.t. invariants.
                                   Shape: [num_samples, num_timesteps, num_output_features_X_post].
            - X_E_output (np.ndarray): Green-Lagrange strain tensor components (vectorized,
                                       upper triangular part). Shape: [num_samples, num_timesteps, 6].

    Assumed Structure of `current_input` (a single timestep from `X_batch_raw[k,i,:]`):
    (This interpretation is based on the original code's indexing and typical continuum mechanics inputs)
        - current_input[0]: $\Delta t$ (timestep size)
        - current_input[1]: $v_p$ (a scalar parameter, e.g., related to volume fraction, scaled by 100)
        - current_input[2]: $v_f$ (another scalar parameter, not used in final concatenation)
        - current_input[3]: $\zeta$ (another scalar parameter, 'zita')
        - current_input[4:13]: Components of Deformation Gradient $F$ (reshaped to 3x3)
        - current_input[13:22]: Components of a structural tensor $a_0 \otimes a_0$ (or $M_0=a_0 \otimes a_0$)
                                 (reshaped to 3x3), representing a fiber direction.
    """
    print("--------------------------------------------------------------------------------", file=sys.stderr)
    print("WARNING: Executing `preprocess_input` function from `misc.py`.", file=sys.stderr)
    print("This function implements highly specific continuum mechanics calculations for polymer composites.", file=sys.stderr)
    print("It is likely NOT suitable for other material systems (e.g., alloys) without complete replacement.", file=sys.stderr)
    print("Ensure the input `X_batch_raw` matches the expected complex structure if you intend to use this.", file=sys.stderr)
    print("--------------------------------------------------------------------------------", file=sys.stderr)


    # Pre-allocate output arrays
    X_new = np.zeros((num_samples, num_timesteps, num_output_features_X_new))
    X_post = np.zeros((num_samples, num_timesteps, num_output_features_X_post))
    X_E_output = np.zeros((num_samples, num_timesteps, 6)) # For storing 6 unique components of symmetric E_grla

    I_3x3 = np.eye(3) # 3x3 Identity matrix

    for k_sample_idx in range(num_samples):
        for i_timestep_idx in range(num_timesteps):
            current_raw_input_vector = X_batch_raw[k_sample_idx, i_timestep_idx, :]

            # --- Extract and Reshape Inputs based on Assumed Structure ---
            delta_t = np.reshape(current_raw_input_vector[0], (1,)) * 2.0 # Scaled $\Delta t$
            vp_param = np.reshape(current_raw_input_vector[1] * 100.0, (1,)) # Scaled parameter 'vp'
            # vf_param = np.reshape(current_raw_input_vector[2], (1,)) # Parameter 'vf', not used below
            zita_param = np.reshape(current_raw_input_vector[3], (1,))   # Parameter 'zita'

            F_tensor = np.reshape(current_raw_input_vector[4:13], (3, 3)) # Deformation Gradient $F$
            
            # Structural tensor $M_0 = a_0 \otimes a_0$ for primary fiber direction (example)
            # The original code uses `a0_t` for this, which can be confusing.
            # Let's assume current_raw_input_vector[13:22] represents $M_0$.
            # However, the invariant calculations below use fixed $a_1, a_2, a_3$ vectors.
            # This part of the original code seems to have a disconnect or assumes fixed fiber directions
            # not directly taken from current_raw_input_vector[13:22] for I4, I5 calculations.
            # For now, proceed with fixed fiber directions as in the original invariant calculation logic.
            # M0_tensor = np.reshape(current_raw_input_vector[13:22], (3,3))

            # Skip calculation if F is zero (e.g., padding in sequences)
            if not np.any(F_tensor):
                continue

            # --- Kinematic Calculations (Continuum Mechanics) ---
            # J_det = np.linalg.det(F_tensor) # Jacobian of deformation
            # F_bar = (J_det**(-1.0/3.0)) * F_tensor # Isochoric part of F (if J_det > 0)
            # For simplicity or specific model assumptions, the original uses F directly as F_bar:
            F_bar = F_tensor
            C_bar = F_bar.T @ F_bar # Right Cauchy-Green tensor $C = F^T F$ (or $\bar{C} = \bar{F}^T \bar{F}$)

            E_grla_tensor = 0.5 * (C_bar - I_3x3) # Green-Lagrange strain tensor $E = \frac{1}{2}(C-I)$

            # --- Calculate Invariants of C_bar ---
            # Isotropic invariants (classical invariants of C)
            Ibar1_val = np.expand_dims(np.trace(C_bar), axis=0) # $I_1 = tr(C)$
            
            # $I_2 = \frac{1}{2}[ (tr(C))^2 - tr(C^2) ] = tr(cof(C))$
            # cofC_bar = np.linalg.det(C_bar) * np.linalg.inv(C_bar).T # Cofactor of C_bar
            # Ibar2_val = np.expand_dims(np.trace(cofC_bar), axis=0)
            # Simpler way for I2 if using principal stretches or eigenvalues, or direct formula:
            Ibar2_val = np.expand_dims(0.5 * (np.trace(C_bar)**2 - np.trace(C_bar @ C_bar)), axis=0)


            Ibar3_val = np.reshape(np.linalg.det(C_bar), (1,)) # $I_3 = det(C)$

            # Anisotropic invariants (related to fiber directions)
            # The original code iterates j = 0,1,2 for three fixed orthogonal directions a1, a2, a3
            # Let's define these fixed directions.
            # These should ideally come from the material's structural tensors (e.g. M0_tensor if it defined fibers).
            # The original script seems to use hardcoded orthogonal fiber directions for these specific invariants.
            
            fiber_dir_1 = np.array([1.0, 0.0, 0.0]) # Example: along x-axis
            M1_tensor = np.outer(fiber_dir_1, fiber_dir_1) # $M_1 = a_1 \otimes a_1$
            Ibar4_M1 = np.expand_dims(np.trace(C_bar @ M1_tensor), axis=0) # $I_4 = C : M_1 = a_1 \cdot C a_1$
            # $I_5 = C^2 : M_1 = a_1 \cdot C^2 a_1$. Original code used tr(cof(C_bar) @ M1_tensor)
            # The definition of I5 varies. Paper uses $a_0 \cdot C^2 a_0$. Let's use $C \cdot M_1 \cdot C$.
            # Or simpler: $a_1 \cdot (C \cdot C) a_1$. The tr(cof(C_bar)@A1) is unusual.
            # Let's stick to what the code implies for derivation if possible or paper.
            # The paper Appendix A.5 lists dI5/dC for I5 = a0.C^2.a0, which is tr(C A C + C A C)/2
            # However, the original python script calculation for Ibar5 used tr(cofCbar*A1).
            # This is very specific. We'll use the definition most consistent with the derivatives used later.
            # The derivatives dCdIbar5 etc. are symbolic.
            # For now, use the simpler definition if derivatives are general.
            # Let's use I5 = tr(C_bar @ M1_tensor @ C_bar @ M1_tensor) for some anisotropic models or just I_5 = a1.C^2.a1
            Ibar5_M1 = np.expand_dims(fiber_dir_1 @ C_bar @ C_bar @ fiber_dir_1, axis=0) # $a_1 \cdot C^2 a_1$
            Inv_for_dir1 = np.concatenate((Ibar4_M1, Ibar5_M1), axis=0)

            fiber_dir_2 = np.array([0.0, 1.0, 0.0]) # Example: along y-axis
            M2_tensor = np.outer(fiber_dir_2, fiber_dir_2)
            Ibar4_M2 = np.expand_dims(np.trace(C_bar @ M2_tensor), axis=0)
            Ibar5_M2 = np.expand_dims(fiber_dir_2 @ C_bar @ C_bar @ fiber_dir_2, axis=0)
            Inv_for_dir2 = np.concatenate((Ibar4_M2, Ibar5_M2), axis=0)
            
            # (The original code calculates Inv_third for fiber_dir_3 but doesn't use it in `new_input`)

            # --- Derivatives of C_bar w.r.t Invariants (Symbolic for use in some constitutive laws) ---
            # These are standard expressions for isotropic hyperelasticity.
            # $\partial I_1 / \partial C = I$
            dCdIbar1_tensor = I_3x3
            # $\partial I_2 / \partial C = I_1 I - C$
            dCdIbar2_tensor = Ibar1_val[0] * I_3x3 - C_bar
            # $\partial I_3 / \partial C = I_3 C^{-T} = det(C) C^{-1}$ (since C is symmetric)
            dCdIbar3_tensor = Ibar3_val[0] * np.linalg.inv(C_bar) # if det(C_bar) != 0 else np.zeros((3,3))

            # For anisotropic terms (derivatives w.r.t $I_4(M_1)$, $I_5(M_1)$, $I_4(M_2)$, $I_5(M_2)$):
            # $\partial I_4(M_1) / \partial C = M_1$
            dCdIbar4_M1_tensor = M1_tensor
            # $\partial I_5(M_1) / \partial C = M_1 C + C M_1$ (for $I_5 = a_1 \cdot C^2 a_1 = C: (a_1 \otimes C a_1)$... no, this is $a_1 \cdot C M_1 C a_1$ is not $I_5$)
            # For $I_5 = tr(C M C)$ with $M=a \otimes a$, then $\partial I_5 / \partial C = M C M$.
            # For $I_5 = a_1 \cdot C^2 a_1 = C : (C(a_1 \otimes a_1) + (a_1 \otimes a_1)C)/2 $, this is tricky.
            # Paper Appendix A.5 has for $I_5 = a_0 \cdot C a_0$: $\partial I_5 / \partial C = a_0 \otimes C a_0 + a_0 C \otimes a_0$. This matches original $dCdIbar5$.
            # The variable Ibar5 was $a_0 \cdot C^2 a_0$. $\partial (a_0 \cdot C^2 a_0) / \partial C = a_0 \otimes (C a_0) + (C a_0) \otimes a_0$
            dCdIbar5_M1_tensor = np.outer(fiber_dir_1, C_bar @ fiber_dir_1) + np.outer(C_bar @ fiber_dir_1, fiber_dir_1)


            dCdIbar4_M2_tensor = M2_tensor # (was dCdIbar6 in original)
            dCdIbar5_M2_tensor = np.outer(fiber_dir_2, C_bar @ fiber_dir_2) + np.outer(C_bar @ fiber_dir_2, fiber_dir_2) # (was dCdIbar7 in original)


            # --- Vectorize Symmetric Tensors (upper triangular part) ---
            # For 3x3 symmetric tensor T, components are [T00, T01, T02, T11, T12, T22]
            Cbar_vec6 = C_bar[np.triu_indices(3)]
            E_grla_vec6 = E_grla_tensor[np.triu_indices(3)]
            dCdIbar1_vec6 = dCdIbar1_tensor[np.triu_indices(3)]
            dCdIbar2_vec6 = dCdIbar2_tensor[np.triu_indices(3)]
            dCdIbar3_vec6 = dCdIbar3_tensor[np.triu_indices(3)]
            dCdIbar4_M1_vec6 = dCdIbar4_M1_tensor[np.triu_indices(3)]
            dCdIbar5_M1_vec6 = dCdIbar5_M1_tensor[np.triu_indices(3)]
            dCdIbar4_M2_vec6 = dCdIbar4_M2_tensor[np.triu_indices(3)]
            dCdIbar5_M2_vec6 = dCdIbar5_M2_tensor[np.triu_indices(3)]

            # Temperature feature placeholder (original code had `temp = 0`)
            temp_placeholder_feature = np.reshape(0.0, (1,))

            # --- Assemble Processed Feature Vectors ---
            # `X_new`: primary input features for the NN model
            # Concatenates vectorized C_bar, scalar parameters, and invariants.
            # Order: Cbar_vec6 (6), delta_t (1), vp_param (1), zita_param (1), temp_placeholder (1),
            #        Ibar1 (1), Ibar2 (1), Ibar3 (1), Inv_for_dir1 (2), Inv_for_dir2 (2)
            # Total = 6+1+1+1+1+1+1+1+2+2 = 17 features (matches original if numFeatures=17)
            processed_features_X_new = np.concatenate((
                Cbar_vec6, delta_t, vp_param, zita_param, temp_placeholder_feature,
                Ibar1_val, Ibar2_val, Ibar3_val,
                Inv_for_dir1, Inv_for_dir2
            ), axis=0) # Ensure concatenation along the feature axis (axis=0 for 1D arrays)

            # `X_post`: auxiliary input features (derivatives of C_bar w.r.t. invariants)
            # Order: Cbar_vec6 (6), dCdI1 (6), dCdI2 (6), dCdI3 (6),
            #        dCdI4_M1 (6), dCdI5_M1 (6), dCdI4_M2 (6), dCdI5_M2 (6)
            # Total = 6 + 6*7 = 48 features (matches original if postprocess_features=48)
            processed_features_X_post = np.concatenate((
                Cbar_vec6, dCdIbar1_vec6, dCdIbar2_vec6, dCdIbar3_vec6,
                dCdIbar4_M1_vec6, dCdIbar5_M1_vec6,
                dCdIbar4_M2_vec6, dCdIbar5_M2_vec6
            ), axis=0)

            # Assign to output arrays (ensure correct length)
            X_new[k_sample_idx, i_timestep_idx, :len(processed_features_X_new)] = processed_features_X_new
            X_post[k_sample_idx, i_timestep_idx, :len(processed_features_X_post)] = processed_features_X_post
            X_E_output[k_sample_idx, i_timestep_idx, :] = E_grla_vec6

    return X_new, X_post, X_E_output