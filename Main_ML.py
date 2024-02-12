# -*- coding: utf-8 -*-
"""
A Pinn for calibration using experimental data 

1. Get experimental data for training and validation
2. Settings for hyperparameters
3. Calculate invariants, C etc. from inputs
4. Normalize inputs and outputs
5. Create folders for postprocessing
6. Train the model
7. Postprocessing (Plots)
"""
import sys
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
from misc import *
import shutil
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    import nvidia.cudnn
    cudnn_path = Path(nvidia.cudnn.__file__).parent
    cudnn_lib_path = cudnn_path / "lib"
    os.environ["LD_LIBRARY_PATH"] = str(cudnn_lib_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "") 
except:
    pass
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')),file=sys.stderr)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


numFeatures = 6
numResponses = 1
# Root for training data
root = os.path.abspath(os.curdir) + "/data_experiments_train"
nr_files = len([entry for entry in os.listdir(root) if os.path.isfile(os.path.join(root, entry))])
pattern = 'epoxy_*_*_*.mat'
print("Take data from subdirectories",file=sys.stderr)
input_data_exp,output_data_exp,k = read_dictionaries_exp(root,pattern,nr_files,numFeatures,numResponses)                
numObservations_exp = k

# Root for validation data
root = os.path.abspath(os.curdir) + "/data_experiments_validation"
nr_files = len([entry for entry in os.listdir(root) if os.path.isfile(os.path.join(root, entry))])
pattern = 'epoxy_*_*_*.mat'
print("Take data from subdirectories",file=sys.stderr)
input_data_exp_val,output_data_exp_val,k = read_dictionaries_exp(root,pattern,nr_files,numFeatures,numResponses)                
numObservations_exp_val = k

print("Number of generated inputs: ", numObservations_exp,file=sys.stderr)  
# Hyperparameters and options
layer_size = 30
layer_size_fenergy = 30
internal_variables = 8
print('Layer Size: ',layer_size,file=sys.stderr)
print('Layer Size for Free Energy: ',layer_size_fenergy,file=sys.stderr)
print('Internal variables: ',internal_variables,file=sys.stderr)
# For adaptive constant
initiate = 999999
update_adapt = 100
beta = 1.0
decay_rate_beta = 0.9
decay_steps_beta = 1000
initial_adapt_const = 10
# Learning settings
learning_rate = 0.0001
num_epochs = 5000
L2 =1e-6 # regularization strength 
batch_size = 16
print_out = 100
timesteps = 1000
timesteps_val = timesteps 
# Get final data
all_y_exp = output_data_exp[:,:timesteps_val,:]
all_x_exp = np.concatenate((input_data_exp[:,0:timesteps,[0,2,1,4,3]],all_y_exp),axis=-1)
all_y_exp_val = output_data_exp_val[:,:timesteps_val,:]
all_x_exp_val = np.concatenate((input_data_exp_val[:,0:timesteps,[0,2,1,4,3]],all_y_exp_val),axis=-1)
all_x = all_x_exp
all_y = all_y_exp

# Training and validation data seperation
train_y = all_y[:,0:timesteps,:]
train_x = all_x[:,0:timesteps,:]
E_grla_train = np.expand_dims(all_x[:,0:timesteps,0],axis=-1)
validation_x = all_x_exp_val[:,0:timesteps,:]
validation_y = all_y_exp_val[:,0:timesteps,:]
E_grla_val = np.expand_dims(all_x_exp_val[:,0:timesteps,0],axis=-1)
# Use only t+dt data
train_y = train_y[:,1:,:]
validation_y = validation_y[:,1:,:]
#Plot training and validation data
for i in range(train_x.shape[0]):
    plt.plot(E_grla_train[i,1:,0],train_y[i,:,0])
    print('Zita: ',train_x[i,0,3])
    print('NV: ', train_x[i,0,2])
    print('Temp: ', train_x[i,0,4])
    plt.title('Zita = '+str(train_x[i,0,3])+', NV = '+str(train_x[i,0,2])+', temp = '+str(train_x[i,0,4]))
    plt.pause(0.1)

for i in range(validation_x.shape[0]):
    plt.plot(E_grla_val[i,1:,0],validation_y[i,:,0])
    print('Zita: ',validation_x[i,0,3])
    print('NV: ', validation_x[i,0,2])
    print('Temp: ', validation_x[i,0,4])
    plt.title('Zita = '+str(validation_x[i,0,3])+', NV = '+str(validation_x[i,0,2])+', temp = '+str(validation_x[i,0,4]))
    plt.pause(0.1)
initial_input_size = 7 # C, dt or I1,I2,I3,I4,I5,I6,I7,dt
print("Normalization",file=sys.stderr) 
# Normalization of input
max_values = np.max(np.max(all_x,axis=1),axis=0)
min_values = np.min(np.min(all_x,axis=1),axis=0)
s_all = (max_values - min_values) 
m_all = min_values
normalized_train = (train_x - m_all) / s_all * 2 -1 
normalized_validation = (validation_x - m_all) / s_all * 2 - 1
max_c = max_values[0]
s_c11,s_dt,s_vp,s_zita,s_temp,s_sig11 = s_all
m_c11,m_dt,m_vp,m_zita,m_temp,m_sig11= m_all
# Normalization of output (for stress)
max_values = np.max(np.max(all_y,axis=1),axis=0)
min_values = np.min(np.min(all_y,axis=1),axis=0)
s_out = (max_values - min_values) 
m_out = min_values
normalized_train_output = (train_y - m_out) / s_out * 2 -1 
normalized_validation_output = (validation_y - m_out) / s_out * 2 -1 
s_sig11 = s_out
m_sig11= m_out
max_sig11 = max_values[0]
max_psi = max_sig11 * max_c # To convert PSI to real units
# Convert to tensorflow
train_x_tf = tf.convert_to_tensor(normalized_train,dtype=tf.float32)
train_y_tf = tf.convert_to_tensor(normalized_train_output,dtype=tf.float32)
train_x_un = tf.convert_to_tensor(train_x,dtype=tf.float32)


val_x_tf = tf.convert_to_tensor(normalized_validation,dtype=tf.float32)
val_y_tf = tf.convert_to_tensor(normalized_validation_output,dtype=tf.float32)
val_x_un = tf.convert_to_tensor(validation_x,dtype=tf.float32)

# Create folders for postprocessing
dir = './stress_exact'
dir2= './final_predictions'
dir3 = './checkpoints'
dir4 = './final_validation'
dir5 = './stress_exact_validation'
dir6 = './input'
dir7 = './input_validation'
dir8 = './strain'
dir9 = './strain_validation'
dir10 = './checkpoints_v2'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
if os.path.exists(dir2):
    shutil.rmtree(dir2)
os.makedirs(dir2)
if os.path.exists(dir3):
    shutil.rmtree(dir3)
os.makedirs(dir3)
if os.path.exists(dir4):
    shutil.rmtree(dir4)
os.makedirs(dir4)
if os.path.exists(dir5):
    shutil.rmtree(dir5)
os.makedirs(dir5)
if os.path.exists(dir6):
    shutil.rmtree(dir6)
os.makedirs(dir6)
if os.path.exists(dir7):
    shutil.rmtree(dir7)
os.makedirs(dir7)
if os.path.exists(dir8):
    shutil.rmtree(dir8)
os.makedirs(dir8)
if os.path.exists(dir9):
    shutil.rmtree(dir9)
os.makedirs(dir9)
if os.path.exists(dir10):
    shutil.rmtree(dir10)
os.makedirs(dir10)

from DL_model import DL
silent=False; silent_summary=True #silent=True: avoid info messages
# Training, evaluation against training, validation, and test data-sets, and weights export
ThermoANN=DL(s_all,m_all,s_out,m_out,layer_size,internal_variables,layer_size_fenergy,max_psi,silent)
inputs=(None,train_x.shape[1],train_x.shape[2]); 
# ThermoANN.build(inputs)
if silent==False: print("\n... Training")
# Loads the weights
# ThermoANN.load_weights('./weights/ThermoTANN_weights')
# ThermoANN.load_weights('./weights/checkpoint')
# ThermoANN.compile(loss=['mae'],run_eagerly=False)
# Train the model
historyTraining=ThermoANN.setTraining(ThermoANN,normalized_train,normalized_train_output,
                                      learning_rate,num_epochs,16,
                                      normalized_validation,normalized_validation_output)
if silent==False: print("\n... Saving weights")
ThermoANN.save_weights('./weights/ThermoTANN_weights', save_format='tf')
print("\n... Completed!")
# Evaluate the model
ThermoANN.evaluate(normalized_train,normalized_train_output); 
stress_predict,psi,dissipation_rate,z = ThermoANN.obtain_output(tf.convert_to_tensor(normalized_train,dtype=tf.float32),
                                                                tf.convert_to_tensor(normalized_train_output,dtype=tf.float32))
# Predict outputs using training input
stress_predict,psi, dissipation_rate, z = ThermoANN.obtain_output(train_x_tf,train_y_tf)
stress_predict = stress_predict.numpy()
psi = psi.numpy()
dissipation_rate = dissipation_rate.numpy()
z = z.numpy()
# Save the training values
for i in range(train_x.shape[0]):
    filename1 = f"./final_predictions/fnergy_{i}.txt"  # Naming each file as array_0.txt, array_1.txt, ...
    filename2 = f"./final_predictions/zi_{i}.txt"
    filename3 = f"./final_predictions/diss_{i}.txt" 
    filename4 = f"./final_predictions/stress_pred_{i}.txt" 
    filename5 = f"./stress_exact/stress_{i}.txt" 
    filename6 = f"./input/input_{i}.txt" 
    filename7 = f"./strain/strain_{i}.txt" 
    
    np.savetxt(filename1, psi[i].reshape(-1), delimiter=',', fmt='%f')    
    # Reshape the 2D slice to (500, 9) and then flatten it to (500*9) before saving
    reshaped_data = z[i].reshape(z.shape[1], -1)
    np.savetxt(filename2, reshaped_data, delimiter=',', fmt='%f')
    reshaped_data2 = dissipation_rate[i].reshape(dissipation_rate.shape[1], -1)
    np.savetxt(filename3, reshaped_data2, delimiter=',', fmt='%f')
    reshaped_data3 = stress_predict[i].reshape(stress_predict.shape[1], -1)
    np.savetxt(filename4, reshaped_data3, delimiter=',', fmt='%f')
    reshaped_data4 = train_y[i].reshape(train_y.shape[1], -1)
    np.savetxt(filename5, reshaped_data4, delimiter=',', fmt='%f')
    reshaped_data6 = train_x[i].reshape(train_x.shape[1], -1)
    np.savetxt(filename6, reshaped_data6, delimiter=',', fmt='%f')
    reshaped_data7 = E_grla_train[i].reshape(E_grla_train.shape[1], -1)
    np.savetxt(filename7, reshaped_data7, delimiter=',', fmt='%f')
print("\n... Output for training data printed out!")

# Predict outputs using validation input
stress_predict,psi, dissipation_rate, z = ThermoANN.obtain_output(val_x_tf,val_y_tf)
stress_predict = stress_predict.numpy()
psi = psi.numpy()
dissipation_rate = dissipation_rate.numpy()
z = z.numpy()
# Save the validation values
for i in range(validation_x.shape[0]):
    filename1 = f"./final_validation/fnergy_{i}.txt"  # Naming each file as array_0.txt, array_1.txt, ...
    filename2 = f"./final_validation/zi_{i}.txt"
    filename3 = f"./final_validation/diss_{i}.txt" 
    filename4 = f"./final_validation/stress_pred_{i}.txt" 
    filename5 = f"./stress_exact_validation/stress_{i}.txt" 
    filename6 = f"./input_validation/input_{i}.txt" 
    filename7 = f"./strain_validation/strain_{i}.txt" 
    
    np.savetxt(filename1, psi[i].reshape(-1), delimiter=',', fmt='%f')    
    # Reshape the 2D slice to (500, 9) and then flatten it to (500*9) before saving
    reshaped_data = z[i].reshape(z.shape[1], -1)
    np.savetxt(filename2, reshaped_data, delimiter=',', fmt='%f')
    reshaped_data2 = dissipation_rate[i].reshape(dissipation_rate.shape[1], -1)
    np.savetxt(filename3, reshaped_data2, delimiter=',', fmt='%f')
    reshaped_data3 = stress_predict[i].reshape(stress_predict.shape[1], -1)
    np.savetxt(filename4, reshaped_data3, delimiter=',', fmt='%f')
    reshaped_data4 = validation_y[i].reshape(validation_y.shape[1], -1)
    np.savetxt(filename5, reshaped_data4, delimiter=',', fmt='%f')
    reshaped_data6 = validation_x[i].reshape(validation_x.shape[1], -1)
    np.savetxt(filename6, reshaped_data6, delimiter=',', fmt='%f')
    reshaped_data7 = E_grla_val[i].reshape(E_grla_val.shape[1], -1)
    np.savetxt(filename7, reshaped_data7, delimiter=',', fmt='%f')
print("\n... Output for validation data printed out!")