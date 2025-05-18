# -*- coding: utf-8 -*-
"""
=========================================================================================
Physics-Informed Deep Learning (PIDL) Model Definition
=========================================================================================

Purpose:
-----------------------------------------------------------------------------------------
This module defines the `DL` class, which constitutes the core neural network model for
predicting the behavior of materials, specifically designed to be thermodynamically
consistent. This model architecture is based on the concepts presented in the research
paper "A thermodynamically consistent physics-informed deep learning material model for
short fiber/polymer nanocomposites" (Comput. Methods Appl. Mech. Engrg., 2024).

The model integrates several neural network components:
-   **Long Short-Term Memory (LSTM) Networks:** Two stacked LSTM layers are used to
    capture the history-dependent nature of material behavior by predicting the
    evolution of internal state variables.
-   **Feed-Forward Neural Networks (FFNNs):**
    -   One set of TimeDistributed FFNNs processes the LSTM outputs to yield the
        internal variables ($z_i$).
    -   Another set of TimeDistributed FFNNs takes these internal variables and a
        measure of strain (e.g., related to the Right Cauchy-Green tensor $C$) as input
        to approximate the Helmholtz free energy function ($\Psi$) of the material.
-   **Thermodynamic Consistency (Physics-Informed Nature):**
    -   The stress tensor ($\sigma$) is derived analytically from the learned free energy
        function ($\Psi$) using automatic differentiation (e.g., $\sigma = 2 \partial\Psi / \partial C$).
    -   The rate of internal dissipation ($D$) is calculated based on the evolution of
        internal variables and their thermodynamically conjugate forces ($\tau_i = -\partial\Psi / \partial z_i$).
    -   Physics-based loss terms are added to the training objective to enforce:
        1.  Non-negative dissipation ($D \ge 0$), satisfying the second law of thermodynamics.
        2.  Non-negative free energy ($\Psi \ge 0$), a common physical requirement.
        3.  The free energy at the reference state (e.g., undeformed state) is constrained to be zero.

Key Architectural Components and Operations:
-----------------------------------------------------------------------------------------
-   **Custom Activation Functions:** Several custom activation functions are defined to
    potentially aid in representing physical quantities or ensuring constraints like
    non-negativity.
-   **Input Processing:** The model expects normalized sequential input data. Specific
    features (strain, timestep, environmental factors, previous stress) are sliced
    from this input for different parts of the network.
-   **Internal Variable Prediction:** LSTMs followed by dense layers predict a set of
    internal variables ($z_i$) that evolve over the sequence.
-   **Free Energy Prediction:** A separate network predicts the free energy ($\Psi$) based on
    the current internal variables and strain-like input.
-   **Automatic Differentiation:** TensorFlow's `GradientTape` is used to compute
    derivatives of the free energy with respect to strain (for stress) and internal
    variables (for thermodynamic forces).
-   **Loss Function:** The total loss minimized during training includes a data-fitting
    term (e.g., Mean Absolute Error between predicted and experimental stress) and
    the aforementioned physics-based penalty terms.
-   **Normalization Handling:** The model stores scaling parameters (mean/offset and
    scale/range) for inputs and outputs, allowing it to work with normalized data
    internally and convert outputs back to physical scales if needed.

Notes for Adaptation:
-----------------------------------------------------------------------------------------
For researchers aiming to adapt this model for different material systems (e.g., alloys,
other types of composites, biological tissues) or different physical phenomena:
-   **Input Features:** The slicing and interpretation of input features in the `call()`
    and `obtain_output()` methods are tied to the data structure prepared by `Main_ML.py`
    (derived from `misc.py`). If input features change, these sections need careful revision.
-   **Network Architecture:** The number of LSTM layers, units per layer, number of internal
    variables, and the structure of the FFNNs for internal variables and free energy
    are hyperparameters that can be modified. For instance, for problems with different
    complexity or different numbers of state variables, these would need adjustment.
-   **Free Energy Formulation:** The inputs to the free energy network (currently internal
    variables and a strain-like measure) and the network structure itself might need
    to be adapted based on the thermodynamic theory of the new material system.
-   **Physics-Based Constraints:** The specific form of the dissipation calculation and
    the stress derivation (e.g., the factor of 2 in $2 \partial\Psi / \partial C$) depends on
    the chosen continuum mechanics framework. These, along with the physics-based loss terms,
    must be reformulated to reflect the relevant physical laws for the new application.
-   **Activation Functions:** While several custom activations are provided, standard
    activations (ReLU, tanh, swish, etc.) can be used. Constraints like non-negativity
    on weights or specific activations (e.g., softplus for $\Psi \ge 0$) are important for
    enforcing physical plausibility.
"""

import numpy as np
import tensorflow as tf
# It's standard practice to set floatx to 'float32' or 'float64'.
# 'float33' is highly unusual and likely a typo. Assuming 'float32'.
tf.keras.backend.set_floatx('float32')
import matplotlib.pyplot as plt # Imported but not used in this class.
from tensorflow import keras # For Keras layers, constraints, etc.

# --- Custom Activation Functions ---
# These functions provide non-linear transformations within neural network layers.
# Their specific mathematical forms can influence learning dynamics and output ranges.

def act_elu_squared(x):
    """
    Custom activation: ELU (Exponential Linear Unit) applied to the square of the input.
    ELU(x^2, alpha=1.0). Tends to produce non-negative activations.
    """
    return tf.keras.backend.elu(tf.keras.backend.pow(x, 2), alpha=1.0)
# Registering for use as a string identifier in Keras layers.
# The name 'act' is used in the original code, mapping to this function.
tf.keras.utils.get_custom_objects().update({'act': tf.keras.layers.Activation(act_elu_squared)})

def my_softplus(x):
    """
    Custom shifted Softplus activation: log(0.5 + 0.5 * exp(x)).
    This function is always positive and smooth. It's a variation of the standard softplus.
    As x -> -inf, output -> log(0.5) = -0.693. As x -> inf, output -> x - log(2).
    To ensure positivity, a standard softplus tf.nn.softplus (log(1+exp(x))) is more common.
    """
    return tf.math.log(0.5 + 0.5 * tf.exp(x))
tf.keras.utils.get_custom_objects().update({'my_softplus': tf.keras.layers.Activation(my_softplus)})

def activation_Exp(x):
    """
    Custom shifted Exponential activation: 1.0 * (exp(x) - 1.0).
    Similar to ELU for x > 0. Output is > -1.0.
    """
    return 1.0 * (tf.math.exp(x) - 1.0)
# Note: This activation was defined but not explicitly registered with get_custom_objects
# in the provided script. If used by string name, it would need registration.

def soft_pp(x):
    """
    SoftplusPlus activation function: log(1 + exp(k*x)) + x/c - log(2).
    This is a variation of softplus with an added linear term.
    Here, k=1 and c=1 are hardcoded.
    """
    k_factor = 1.0 # Scaling factor for the exponential term's input
    c_factor = 1.0 # Divisor for the linear term
    return tf.math.log(1.0 + tf.exp(k_factor * x)) + (x / c_factor) - tf.math.log(2.0)
tf.keras.utils.get_custom_objects().update({'softplusplus': tf.keras.layers.Activation(soft_pp)})

class DL(tf.keras.Model):
    """
    Deep Learning (DL) model class for Thermodynamics-informed Artificial Neural Network (ThermoANN).
    This class defines the network architecture, forward pass, and training setup.
    """
    def __init__(self, s_all_inputs, m_all_inputs, s_out_targets, m_out_targets,
                 layer_size, internal_variables, layer_size_fenergy, max_psi_heuristic,
                 training_silent=True):
        """
        Initializes the layers and parameters of the ThermoANN model.

        Args:
            s_all_inputs (np.ndarray): Scaling factors (range = max - min) for each input feature.
                                       Order must match features in `train_x_tf` from `Main_ML.py`.
            m_all_inputs (np.ndarray): Offset values (min value) for each input feature.
            s_out_targets (np.ndarray): Scaling factors for each output target variable (e.g., stress).
            m_out_targets (np.ndarray): Offset values for each output target variable.
            layer_size (int): Number of units in the general LSTM and Dense hidden layers.
            internal_variables (int): Number of internal state variables ($z_i$) to be predicted.
            layer_size_fenergy (int): Number of units in hidden layers of the FFNN predicting free energy.
            max_psi_heuristic (float): A heuristic maximum value for free energy, potentially used
                                     for scaling the dissipation loss term or other physics constraints.
            training_silent (bool, optional): If True, suppresses some verbose output during
                                            the `setTraining` method. Defaults to True.
        """
        super(DL, self).__init__() # Call the parent tf.keras.Model constructor
        self.training_silent = training_silent

        # --- Define Neural Network Layers ---
        # Stacked LSTM layers for processing sequences and capturing history.
        # These predict features that then determine the internal variables.
        self.lstm_f0 = tf.keras.layers.LSTM(
            units=layer_size, name="history_lstm_1",
            return_sequences=True,  # Returns the full sequence of outputs for each timestep
            return_state=False,     # Does not return the final hidden and cell states separately
            use_bias=True
        )
        self.lstm_f02 = tf.keras.layers.LSTM(
            units=layer_size, name="history_lstm_2",
            return_sequences=True,
            return_state=False,
            use_bias=True
        )

        # TimeDistributed Dense layers for predicting internal variables ($z_i$).
        # These layers apply a Dense transformation independently to each timestep of the LSTM output.
        self.dense_f1_for_iv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size, activation='swish', name="td_dense_1_for_iv"))
        self.dense_f12_for_iv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size, activation='swish', name="td_dense_2_for_iv"))
        # Final layer outputting the internal variables.
        self.dense_f2_internal_vars_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(internal_variables, use_bias=True, name="td_dense_out_internal_vars"))

        # TimeDistributed Dense layers for the free energy ($\Psi$) sub-network.
        # This network takes internal variables and strain-like features as input.
        # 'softplus' activation helps in ensuring non-negativity if weights are also constrained.
        self.dense_f3_psi_hidden1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size_fenergy, activation='softplus', name="td_dense_1_for_psi"))
        # `kernel_constraint=keras.constraints.NonNeg()` ensures weights are non-negative,
        # which, with softplus, contributes to a non-negative free energy prediction.
        self.dense_f32_psi_hidden2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size_fenergy, activation='softplus', name="td_dense_2_for_psi",
                                kernel_constraint=keras.constraints.NonNeg()))
        # Final layer outputting the scalar free energy value per timestep.
        self.dense_f4_psi_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, use_bias=False, name='td_dense_out_free_energy', # Output is Psi (scalar)
                                kernel_constraint=keras.constraints.NonNeg()))

        # Store input scaling factors. The order of unpacking must match the order of
        # features in `all_x_for_model` (and thus `s_all_inputs`) from `Main_ML.py`.
        # Expected order: [C_strain-like, deltaT, NV, Zita, Temp, Sigma_previous_step]
        try:
            self.s_input_C, self.s_input_dt, self.s_input_nv, self.s_input_zita, \
            self.s_input_temp, self.s_input_sig_prev = s_all_inputs[:6]

            self.m_input_C, self.m_input_dt, self.m_input_nv, self.m_input_zita, \
            self.m_input_temp, self.m_input_sig_prev = m_all_inputs[:6]
        except (TypeError, ValueError, IndexError) as e:
            print(f"Error unpacking input scaling factors (s_all_inputs, m_all_inputs): {e}", file=sys.stderr)
            print("Expected at least 6 elements for s_all_inputs and m_all_inputs.", file=sys.stderr)
            raise

        # Store output target scaling factors (for the primary output, e.g., stress).
        # Assuming `s_out_targets` and `m_out_targets` are arrays (even if for a single response).
        try:
            self.s_target_stress = s_out_targets[0] # Scale for the first (and likely only) target response
            self.m_target_stress = m_out_targets[0] # Offset for the first target response
        except (TypeError, IndexError) as e:
            print(f"Error unpacking output scaling factors (s_out_targets, m_out_targets): {e}", file=sys.stderr)
            print("Expected s_out_targets and m_out_targets to be array-like with at least one element.", file=sys.stderr)
            raise

        self.max_psi_heuristic = max_psi_heuristic # Store for potential use in scaling physics losses.

    def unnormalize_output(self, normalized_tensor, scale, offset):
        """
        Converts a tensor normalized to the [-1, 1] range back to its physical scale.
        Formula: physical_value = (normalized_value + 1)/2 * range + min_value
        where range = `scale` and min_value = `offset`.
        """
        # Shift from [-1, 1] to [0, 1]
        tensor_0_to_1 = (normalized_tensor + 1.0) / 2.0
        # Scale to [0, range] and add offset
        return (tensor_0_to_1 * scale) + offset

    def normalize_input(self, physical_tensor, scale, offset): # Not actively used in this script's flow
        """
        Converts a tensor from its physical scale to the normalized [-1, 1] range.
        Formula: normalized_value = (physical_value - min_value) / range * 2 - 1
        """
        # Handle potential division by zero if scale is 0 (constant feature)
        safe_scale = tf.where(tf.equal(scale, 0.0), tf.keras.backend.epsilon(), scale)
        return ((physical_tensor - offset) / safe_scale) * 2.0 - 1.0

    def call(self, normalized_inputs_seq):
        """
        Defines the forward pass of the model during training.
        It takes a batch of normalized input sequences, predicts internal variables,
        free energy, stress, and computes physics-based loss terms.

        Args:
            normalized_inputs_seq (tf.Tensor): Batch of normalized input sequences.
                Shape: (batch_size, num_timesteps, num_features_in_train_x).
                This corresponds to `train_x_tf` from `Main_ML.py`.
                `num_timesteps` is the original sequence length before target shifting.

        Returns:
            tf.Tensor: Predicted normalized stress (or primary target).
                       Shape: (batch_size, num_timesteps-1, 1) to match the shifted target `train_y_tf`.
        """
        # --- Slice Input Features ---
        # The input `normalized_inputs_seq` contains features for timesteps t=0 to N-1.
        # Features are ordered as in `train_x_sequences` from `Main_ML.py`:
        # 0: Strain-like (C-related), 1: Timestep size (deltaT), 2: NV (nanoparticle/fiber related),
        # 3: Zita (moisture/condition related), 4: Temperature, 5: Previous step's Stress.

        # Full sequences (length N, for t=0 to N-1)
        norm_C_full_seq = tf.slice(normalized_inputs_seq, [0, 0, 0], [-1, -1, 1]) # Strain-like feature
        # norm_dt_full_seq = tf.slice(normalized_inputs_seq, [0, 0, 1], [-1, -1, 1]) # Not used with this name directly below
        # norm_nv_full_seq = tf.slice(normalized_inputs_seq, [0, 0, 2], [-1, -1, 1])
        # norm_zita_full_seq = tf.slice(normalized_inputs_seq, [0, 0, 3], [-1, -1, 1])
        # norm_temp_full_seq = tf.slice(normalized_inputs_seq, [0,0,4],[-1,-1,1]) # Not directly in LSTM input combination
        # norm_sigma_prev_full_seq = tf.slice(normalized_inputs_seq, [0,0,5],[-1,-1,1]) # Not directly in LSTM input

        # Sequences for LSTM input (length N-1, for t=1 to N-1 of original inputs, driving state from t=0)
        # These represent inputs at the "current" time step $t$ (from $t=1..N-1$ of original)
        # to predict changes or states relevant for the "next" step prediction.
        # Slicing from index 1 effectively means these are inputs for predicting the sequence that matches `train_y_tf`.
        norm_C_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 0], [-1, -1, 1]) # Strain-like, from t=1 of original input
        norm_dt_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 1], [-1, -1, 1]) # deltaT, from t=1
        norm_nv_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 2], [-1, -1, 1]) # NV, from t=1
        norm_zita_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 3], [-1, -1, 1]) # Zita, from t=1
        # (Temperature is also a feature but not explicitly concatenated into `combined_input_for_lstm` here)

        # Un-normalize deltaT (timestep size) for physical calculations (e.g., rates $\dot{z}_i$).
        # This `unnorm_dt_for_rates` corresponds to the timesteps over which $\Delta z_i$ occurs.
        unnorm_dt_for_rates = self.unnormalize_output(norm_dt_lstm_input_seq, self.s_input_dt, self.m_input_dt)

        with tf.GradientTape(persistent=True) as tape:
            # Watch the full normalized strain-like sequence for gradient calculation (stress = 2 dPsi/dC).
            tape.watch(norm_C_full_seq) # Watched for $\partial\Psi / \partial C$

            # --- Internal Variable Evolution Path (LSTMs + Dense) ---
            # Prepare combined input for the LSTM layers. This selection of features is model-specific.
            # These are features at $t+dt$ (i.e., from timestep 1 to N-1 of original input).
            combined_input_for_lstm = tf.concat([
                norm_C_lstm_input_seq,
                norm_dt_lstm_input_seq,
                norm_nv_lstm_input_seq,
                norm_zita_lstm_input_seq
            ], axis=-1)

            # Pass through stacked LSTM layers. Output shape: (batch, timesteps-1, lstm_units)
            lstm_hidden_seq1 = self.lstm_f0(combined_input_for_lstm)
            lstm_hidden_seq2 = self.lstm_f02(lstm_hidden_seq1)

            # The LSTM output `lstm_hidden_seq2` represents history from t=1 to N-1.
            # To predict internal variables $z_i$ starting from an initial state $z_i(t=0)$,
            # we need to prepend the LSTM's effective initial hidden state (usually zeros)
            # to this sequence.
            initial_lstm_h_state = self.lstm_f0.get_initial_state(combined_input_for_lstm)[0] # Get h_0 for first LSTM
            initial_lstm_h_state_expanded = tf.expand_dims(initial_lstm_h_state, axis=1) # Shape: (batch, 1, lstm_units)

            # Concatenate initial state with LSTM outputs. Sequence length becomes `timesteps`.
            # This represents a sequence of hidden states [h(0), h(1), ..., h(N-1)]
            # where h(0) is initial state, and h(1)...h(N-1) are outputs from LSTM processing inputs from t=1...N-1.
            full_history_sequence = tf.concat([initial_lstm_h_state_expanded, lstm_hidden_seq2], axis=1)

            # Predict internal variables $z_i$ using FFNNs applied to the full history sequence.
            # $z_i$ sequence will have length `timesteps`, corresponding to $z_i(0), ..., z_i(N-1)$.
            z_ffnn_hidden1 = self.dense_f1_for_iv(full_history_sequence)
            z_ffnn_hidden2 = self.dense_f12_for_iv(z_ffnn_hidden1)
            # `z_i_sequence_pred` shape: (batch, timesteps, num_internal_variables)
            z_i_sequence_pred = self.dense_f2_internal_vars_output(z_ffnn_hidden2)


            # --- Free Energy ($\Psi$) Prediction Path ---
            # $\Psi$ is a function of strain $C$ and internal variables $z_i$.
            # To ensure $\Psi(C=I, z_i=z_{i0}, t=0) = 0$ (or $\Psi_{ref}=0$), we calculate $\Psi$ at the initial
            # state (t=0) and subtract this initial $\Psi$ from the entire $\Psi$ sequence.

            # Get $z_i(t=0)$ and $C(t=0)$
            z_i_initial_t0 = z_i_sequence_pred[:, 0:1, :]      # Shape: (batch, 1, num_internal_vars)
            norm_C_initial_t0 = norm_C_full_seq[:, 0:1, :]    # Shape: (batch, 1, 1)
            # Combined input for $\Psi(t=0)$ prediction
            psi_input_initial_t0 = tf.concat([z_i_initial_t0, norm_C_initial_t0], axis=-1)

            # Predict $\Psi(t=0)$
            psi_ffnn_h1_init = self.dense_f3_psi_hidden1(psi_input_initial_t0)
            psi_ffnn_h2_init = self.dense_f32_psi_hidden2(psi_ffnn_h1_init)
            psi_at_t0_val = self.dense_f4_psi_output(psi_ffnn_h2_init) # Shape: (batch, 1, 1)

            # Predict $\Psi$ for the entire sequence (t=0 to N-1)
            # Input to Psi network: $z_i(0..N-1)$ and $C(0..N-1)$
            psi_input_full_seq = tf.concat([z_i_sequence_pred, norm_C_full_seq], axis=-1)
            psi_ffnn_h1_full = self.dense_f3_psi_hidden1(psi_input_full_seq)
            psi_ffnn_h2_full = self.dense_f32_psi_hidden2(psi_ffnn_h1_full)
            psi_raw_full_seq = self.dense_f4_psi_output(psi_ffnn_h2_full) # Shape: (batch, timesteps, 1)

            # Shifted free energy: $\Psi_{final}(t) = \Psi_{raw}(t) - \Psi(t=0)$
            # This ensures $\Psi_{final}(t=0) = 0$.
            psi_final_full_sequence = psi_raw_full_seq - psi_at_t0_val # Broadcasting subtraction

            # --- Thermodynamic Calculations (Gradients for Stress and Dissipation) ---
            # Thermodynamic forces conjugate to internal variables: $\tau_i = - \partial\Psi / \partial z_i$.
            # The paper uses $\partial\Psi / \partial z_i$ for $\tau$. GradientTape gives $\partial(\text{sum of losses}) / \partial z_i$.
            # Here, we need $\partial\Psi_{final} / \partial z_i$.
            # `tau_forces_seq` shape: (batch, timesteps, num_internal_variables)
            tau_forces_seq = tape.gradient(psi_final_full_sequence, z_i_sequence_pred)
            if tau_forces_seq is None:
                raise ValueError("Gradient of Psi w.r.t. z_i (tau_forces_seq) is None. Check model graph.")

            # Rate of change of internal variables: $\dot{z}_i = \Delta z_i / \Delta t$.
            # $\Delta z_i = z_i(t+1) - z_i(t)$. This sequence has length N-1 (from t=0 to N-2).
            delta_z_i = tf.experimental.numpy.diff(z_i_sequence_pred, n=1, axis=1) # Shape: (batch, timesteps-1, num_iv)
            # `unnorm_dt_for_rates` has length N-1. Ensure it's not zero.
            safe_unnorm_dt_for_rates = tf.where(tf.equal(unnorm_dt_for_rates, 0.0),
                                                tf.keras.backend.epsilon(), # Small non-zero value
                                                unnorm_dt_for_rates)
            z_i_dot_seq = delta_z_i / safe_unnorm_dt_for_rates # Shape: (batch, timesteps-1, num_iv)

            # Dissipation Rate $D = - \sum_i (\tau_i \cdot \dot{z}_i)$.
            # $\tau_i$ needs to align with $\dot{z}_i$. $\dot{z}_i$ is for intervals ending at t=1 to N-1.
            # So, use $\tau_i$ from t=1 to N-1 (representing $\tau_i(t+dt)$ or an average over interval).
            # `tau_forces_for_diss_seq` shape: (batch, timesteps-1, num_internal_variables)
            tau_forces_for_diss_seq = tf.slice(tau_forces_seq, [0, 1, 0], [-1, -1, -1]) # Slices $\tau(t=1)$ to $\tau(t=N-1)$
            
            # Element-wise product for each internal variable's contribution
            dissipation_terms_prod = tau_forces_for_diss_seq * z_i_dot_seq
            # Sum over all internal variables and apply scaling and sign.
            # The paper defines $D = -\sum (\partial\Psi/\partial z_{\alpha}) \dot{z}_{\alpha}$.
            # Result shape: (batch, timesteps-1, 1)
            dissipation_rate_final = tf.reduce_sum(dissipation_terms_prod, axis=-1, keepdims=True) \
                                     * (-1.0) * self.max_psi_heuristic # Scale by heuristic max_psi


            # Stress calculation: e.g., $\sigma = 2 \partial\Psi / \partial C$ (PK2 stress if C is Right Cauchy-Green)
            # `norm_stress_from_psi_seq` shape: (batch, timesteps, 1)
            norm_stress_from_psi_seq = tape.gradient(psi_final_full_sequence, norm_C_full_seq)
            if norm_stress_from_psi_seq is None:
                raise ValueError("Gradient of Psi w.r.t. C (stress) is None. Check model graph.")

            # The predicted stress for loss calculation needs to match the target `train_y_tf`
            # which has length `timesteps-1`. So, slice the stress sequence.
            # This `norm_pred_stress_for_loss` represents stress at $t=1, ..., N-1$.
            # The factor of 2 is from constitutive theory (e.g., $S = 2 \partial\Psi/\partial C$).
            norm_pred_stress_for_loss = norm_stress_from_psi_seq[:, 1:, :] * 2.0

        del tape # Release GradientTape resources

        # --- Physics-Based Loss Terms (added to the model's total loss) ---
        # 1. Non-negative Dissipation: $D \ge 0$. Penalize if $D < 0$.
        #    Use ReLU(-D) which is > 0 only if D < 0.
        #    `dissipation_rate_final` has length `timesteps-1`.
        dissipation_penalty = tf.reduce_mean(tf.nn.relu(-dissipation_rate_final))
        self.add_loss(dissipation_penalty * 10.0) # Apply a weighting factor (e.g., 10.0)

        # 2. Non-negative (or appropriately bounded) Free Energy: $\Psi \ge 0$.
        #    `psi_final_full_sequence` is already constructed such that $\Psi(t=0)=0$.
        #    The NonNeg constraint on the final layer of Psi network also helps.
        #    Penalize if $\Psi < 0$.
        free_energy_penalty = tf.reduce_mean(tf.nn.relu(-psi_final_full_sequence))
        self.add_loss(free_energy_penalty * 10.0) # Apply a weighting factor

        # The primary return of `call()` is the prediction that will be compared against true labels
        # by the loss function specified in `compile()` (e.g., MAE for stress).
        return norm_pred_stress_for_loss

    def obtain_output(self, normalized_inputs_seq, normalized_true_outputs=None):
        """
        Performs a forward pass (similar to `call`) to obtain detailed model outputs
        for postprocessing, evaluation, or inference. Returns key physical quantities,
        including unnormalized stress.

        Args:
            normalized_inputs_seq (tf.Tensor): Batch of normalized input sequences.
                                               Shape: (batch_size, timesteps, num_features).
            normalized_true_outputs (tf.Tensor, optional): True outputs, not used in this
                                                           method but included for API consistency.

        Returns:
            tuple: A tuple containing:
                - unnorm_pred_stress (tf.Tensor): Predicted stress in physical (unnormalized) scale.
                                                  Shape: (batch_size, timesteps-1, 1).
                - psi_final_full_sequence (tf.Tensor): Predicted free energy sequence (shifted so $\Psi(t=0)=0$).
                                                     Shape: (batch_size, timesteps, 1).
                - dissipation_rate_final (tf.Tensor): Predicted dissipation rate.
                                                      Shape: (batch_size, timesteps-1, 1).
                - z_i_sequence_pred (tf.Tensor): Predicted internal variable sequence.
                                                 Shape: (batch_size, timesteps, num_internal_variables).
        """
        # This method largely mirrors `call()` to ensure consistent computation paths,
        # but additionally unnormalizes the final stress prediction.

        # --- Slice Input Features (same as in `call`) ---
        norm_C_full_seq = tf.slice(normalized_inputs_seq, [0, 0, 0], [-1, -1, 1])
        norm_dt_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 1], [-1, -1, 1])
        norm_nv_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 2], [-1, -1, 1])
        norm_zita_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 3], [-1, -1, 1])
        norm_C_lstm_input_seq = tf.slice(normalized_inputs_seq, [0, 1, 0], [-1, -1, 1])

        unnorm_dt_for_rates = self.unnormalize_output(norm_dt_lstm_input_seq, self.s_input_dt, self.m_input_dt)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(norm_C_full_seq)

            combined_input_for_lstm = tf.concat([
                norm_C_lstm_input_seq, norm_dt_lstm_input_seq,
                norm_nv_lstm_input_seq, norm_zita_lstm_input_seq
            ], axis=-1)

            lstm_hidden_seq1 = self.lstm_f0(combined_input_for_lstm)
            lstm_hidden_seq2 = self.lstm_f02(lstm_hidden_seq1)

            initial_lstm_h_state = self.lstm_f0.get_initial_state(combined_input_for_lstm)[0]
            initial_lstm_h_state_expanded = tf.expand_dims(initial_lstm_h_state, axis=1)
            full_history_sequence = tf.concat([initial_lstm_h_state_expanded, lstm_hidden_seq2], axis=1)

            z_ffnn_hidden1 = self.dense_f1_for_iv(full_history_sequence)
            z_ffnn_hidden2 = self.dense_f12_for_iv(z_ffnn_hidden1)
            z_i_sequence_pred = self.dense_f2_internal_vars_output(z_ffnn_hidden2)

            z_i_initial_t0 = z_i_sequence_pred[:, 0:1, :]
            norm_C_initial_t0 = norm_C_full_seq[:, 0:1, :]
            psi_input_initial_t0 = tf.concat([z_i_initial_t0, norm_C_initial_t0], axis=-1)
            psi_ffnn_h1_init = self.dense_f3_psi_hidden1(psi_input_initial_t0)
            psi_ffnn_h2_init = self.dense_f32_psi_hidden2(psi_ffnn_h1_init)
            psi_at_t0_val = self.dense_f4_psi_output(psi_ffnn_h2_init)

            psi_input_full_seq = tf.concat([z_i_sequence_pred, norm_C_full_seq], axis=-1)
            psi_ffnn_h1_full = self.dense_f3_psi_hidden1(psi_input_full_seq)
            psi_ffnn_h2_full = self.dense_f32_psi_hidden2(psi_ffnn_h1_full)
            psi_raw_full_seq = self.dense_f4_psi_output(psi_ffnn_h2_full)
            psi_final_full_sequence = psi_raw_full_seq - psi_at_t0_val

            tau_forces_seq = tape.gradient(psi_final_full_sequence, z_i_sequence_pred)
            if tau_forces_seq is None: raise ValueError("Gradient tau is None in obtain_output.")

            delta_z_i = tf.experimental.numpy.diff(z_i_sequence_pred, n=1, axis=1)
            safe_unnorm_dt_for_rates = tf.where(tf.equal(unnorm_dt_for_rates, 0.0),
                                                tf.keras.backend.epsilon(),
                                                unnorm_dt_for_rates)
            z_i_dot_seq = delta_z_i / safe_unnorm_dt_for_rates

            tau_forces_for_diss_seq = tf.slice(tau_forces_seq, [0, 1, 0], [-1, -1, -1])
            dissipation_terms_prod = tau_forces_for_diss_seq * z_i_dot_seq
            dissipation_rate_final = tf.reduce_sum(dissipation_terms_prod, axis=-1, keepdims=True) \
                                     * (-1.0) * self.max_psi_heuristic

            norm_stress_from_psi_seq = tape.gradient(psi_final_full_sequence, norm_C_full_seq)
            if norm_stress_from_psi_seq is None: raise ValueError("Gradient sigma is None in obtain_output.")
            
            norm_pred_stress_output = norm_stress_from_psi_seq[:, 1:, :] * 2.0
        
        del tape # Release GradientTape resources

        # Un-normalize the predicted stress to its physical scale.
        unnorm_pred_stress = self.unnormalize_output(
            norm_pred_stress_output,
            self.s_target_stress, # Use scaling parameters for the primary target (stress)
            self.m_target_stress
        )

        return unnorm_pred_stress, psi_final_full_sequence, dissipation_rate_final, z_i_sequence_pred


    def setTraining(self, normalized_train_input, normalized_train_output,
                    learning_rate_schedule_val, num_epochs, batch_size_training,
                    normalized_val_input, normalized_val_output, L2_reg=None): # L2_reg arg was in main script call
        """
        Configures the model for training and initiates the Keras `fit` process.

        Args:
            normalized_train_input (tf.Tensor): Normalized training input data sequences.
            normalized_train_output (tf.Tensor): Normalized training target data sequences (shifted).
            learning_rate_schedule_val (float or tf.keras.optimizers.schedules.LearningRateSchedule):
                                           Initial learning rate value or a Keras learning rate schedule object.
            num_epochs (int): Maximum number of epochs for training.
            batch_size_training (int): Batch size for training.
            normalized_val_input (tf.Tensor): Normalized validation input data sequences.
            normalized_val_output (tf.Tensor): Normalized validation target data sequences (shifted).
            L2_reg (float, optional): L2 regularization factor. If provided, it might be intended
                                      for application to layer kernels, though this method doesn't
                                      explicitly apply it here; layers should have `kernel_regularizer`
                                      set at initialization if L2 is desired per-layer.

        Returns:
            tf.keras.callbacks.History: Keras History object containing training/validation loss
                                        and metrics per epoch.
        """
        if not self.training_silent:
            keras_verbose_level = 1 # 0=silent, 1=progress bar, 2=one line per epoch
        else:
            keras_verbose_level = 0

        # Setup learning rate: if a float is passed, create an ExponentialDecay schedule.
        if isinstance(learning_rate_schedule_val, float):
            lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate_schedule_val,
                decay_steps=2000,  # Number of steps before decay is applied.
                decay_rate=0.90    # Factor by which LR is multiplied.
            )
        else: # Assume it's already a Keras LearningRateSchedule object
            lr_scheduler = learning_rate_schedule_val

        # Optimizer: Nadam is an Adam variant with Nesterov momentum.
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr_scheduler)

        # Compile the model.
        # The 'loss' specified here ('mae') applies to the direct output of the `call` method
        # (i.e., the predicted normalized stress) compared against `normalized_train_output`.
        # Physics-based losses (dissipation, free energy penalties) are added internally
        # via `self.add_loss()` within the `call` method and are part of the total loss
        # minimized by the optimizer.
        # `loss_weights` applies to the specified 'loss' components. Here, only one primary loss.
        primary_loss_weight = 1.0
        self.compile(
            optimizer=optimizer,
            loss=['mae'], # Mean Absolute Error for the stress prediction.
            loss_weights=[primary_loss_weight], # Weight for the MAE loss.
            run_eagerly=False # False for performance, True for easier debugging of `call` method.
        )

        # Callbacks to enhance training:
        # EarlyStopping: Stops training if validation loss doesn't improve for a set number of epochs.
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',       # Monitor validation loss for improvement.
            min_delta=1.e-6,        # Minimum change in monitored quantity to qualify as improvement.
            patience=1000,          # Number of epochs with no improvement to wait before stopping.
            verbose=1 if not self.training_silent else 0,
            mode='min',             # Stop when the monitored quantity has stopped decreasing.
            restore_best_weights=True # Restore model weights from the epoch with the best `val_loss`.
        )

        # ModelCheckpoint: Saves model weights periodically, especially the best ones found so far.
        # The filepath used here matches the `compat_output_dirs['checkpoints_training_orig']`
        # from `Main_ML.py` which was './checkpoints/checkpoint'.
        # Ensure this directory exists.
        checkpoint_base_dir = './checkpoints' # Default from original script
        if not os.path.exists(checkpoint_base_dir):
            os.makedirs(checkpoint_base_dir, exist_ok=True)
        checkpoint_filepath = os.path.join(checkpoint_base_dir, 'checkpoint') # Keras appends epoch/loss info or saves to this name
        
        print(f"Model checkpoints (best weights during training) will be saved to: {checkpoint_filepath}", file=sys.stderr)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,     # Only save the model's weights.
            monitor='val_loss',         # Monitor validation loss.
            mode='min',                 # Save when `val_loss` decreases.
            save_best_only=True,        # Only save if `val_loss` is the best seen so far.
            # save_freq='epoch',        # Can be 'epoch' or an integer number of batches.
                                        # Original `save_freq = 300` (batches) - use 'epoch' for simplicity unless batch-wise needed.
            verbose=1 if not self.training_silent else 0
        )

        # Start the training process using Keras `fit` method.
        history = self.fit(
            normalized_train_input,
            normalized_train_output,
            epochs=num_epochs,
            batch_size=batch_size_training,
            verbose=keras_verbose_level,
            validation_data=(normalized_val_input, normalized_val_output),
            callbacks=[earlystop_callback, model_checkpoint_callback]
        )

        if not self.training_silent:
            epochs_ran = len(history.history['loss'])
            print(f"\nTraining completed in {epochs_ran} epochs (max_epochs: {num_epochs}).", file=sys.stderr)
        
        return history