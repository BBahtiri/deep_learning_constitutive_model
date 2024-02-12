# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:45:05 2023

@author: bahtiri
"""
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import scipy.io
from fnmatch import fnmatch
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation


def resize_array(x):
    i = 1000 # Sequence length
    z = i / len(x) 
    x_res = np.expand_dims(interpolation.zoom(x[:,0],z),axis=-1)    
    return x_res

# Import data from mat file
def getData_exp(inputF):
    C = scipy.io.loadmat(inputF)
    stress = C['expStress'].astype('float64')
    timeVec = C['timeVec'].astype('float64')
    trueStrain = (C['trueStrain'].astype('float64')+1)
    # Resize the arrays by using interpolation to reduce sequence length 
    stress_res = resize_array(stress)
    timeVec_res = resize_array(timeVec)
    trueStrain_res = resize_array(trueStrain)
    
    
    deltaT = np.mean(np.diff(timeVec,axis=0))
    deltaT_res = np.mean(np.diff(timeVec_res,axis=0))
    if deltaT_res < 0:
        stress_res = stress_res[:-10]
        timeVec_res = timeVec_res[:-10]
        trueStrain_res = trueStrain_res[:-10]
        deltaT_res = np.mean(np.diff(timeVec_res,axis=0))
        
    return stress_res, trueStrain_res, deltaT_res

# Import data from mat file
def getData(inputF):
    C = scipy.io.loadmat(inputF)
    s = C['t'].astype('float64')
    x = C['x'].astype('float64')
    
    return s, x

# Take data and put in nested dictionaries
def read_dictionaries(paths,pattern,nr_files,features,responses):
    k = 0
    input_array= np.zeros((nr_files,10000,features))
    output_array = np.zeros((nr_files,10000,responses))
    input_dict = {}
    output_dict = {}
    for path, subdirs, files in os.walk(paths):
        for name in files:
            if fnmatch(name, pattern):
                file = os.path.join(path, name)
                outputs,inputs = getData(file)            
                input_array[k,:np.shape(inputs)[1],:] = inputs.T
                output_array[k,:np.shape(outputs)[1],:] = outputs.T
                k = k + 1
                if k == nr_files:
                    break
    return input_array,output_array,k



# Take data and put in nested dictionaries from experiments
def read_dictionaries_exp(paths,pattern,nr_files,features,responses):
    k = 0
    input_array= np.zeros((nr_files,13700,features))
    output_array = np.zeros((nr_files,13700,responses))
    input_dict = {}
    output_dict = {}
    for path, subdirs, files in os.walk(paths):
        for name in files:
            if fnmatch(name, pattern):
                file = os.path.join(path, name)
                outputs,inputs,deltaT = getData_exp(file)
                
                if int(name[6]) == 1:
                    nv = 0
                elif int(name[6]) == 2:
                    nv = 5
                else:
                    nv = 10
                    
                if int(name[8]) == 1:
                    zita = 0
                else:
                    zita = 1
                    
                if int(name[10]) == 1:
                    temp = -20
                elif int(name[10]) == 2:
                    temp = 23
                elif int(name[10]) == 3:
                    temp = 50
                else:
                    temp = 60
                        
                    
                input_array[k,:np.shape(inputs)[0],0] = inputs.T
                input_array[k,:np.shape(inputs)[0],1] = nv
                input_array[k,:np.shape(inputs)[0],2] = deltaT
                input_array[k,:np.shape(inputs)[0],3] = temp
                input_array[k,:np.shape(inputs)[0],4] = zita
                input_array[k,:np.shape(inputs)[0],5] = outputs.T
                output_array[k,:np.shape(outputs)[0],0] = outputs.T
                k = k + 1
                if k == nr_files:
                    break
    return input_array,output_array,k

def preprocess_input(X_batch,samples,timesteps,numFeatures,postprocess_features):
    X_new = np.zeros((samples,timesteps,numFeatures))
    X_post = np.zeros((samples,timesteps,postprocess_features))
    X_E = np.zeros((samples,timesteps,6))
    I = np.eye(3)
    for k in range(0,samples):
        for i in range(0,timesteps):
            current_input = X_batch[k,i,:]
            #B = current_input[0:6]
            F = np.reshape(current_input[4:13],(3,3))
            all_zeros = not np.any(F)
            if all_zeros == True:
                continue
            a0_t = np.reshape(current_input[13:22],(3,3))
            vf = np.reshape(current_input[2],(1,))
            vp = np.reshape(current_input[1]*100,(1,))
            zita = np.reshape(current_input[3],(1,))
            J = np.reshape(np.linalg.det(F),(1,))
            #Fbar = J**(-1/3)*F
            Fbar = F
            Cbar = np.matmul(np.transpose(Fbar),Fbar)
            E_grla = (1/2)*(Cbar - I)
            cofCbar = np.linalg.det(Cbar)*np.linalg.inv(Cbar).T
            Ibar1 = np.expand_dims(np.trace(Cbar),axis=0)
            #Ibar2 = np.expand_dims((1/2)*(np.trace(Cbar)**2 - np.trace(np.matmul(Cbar,Cbar))),
            #                       axis=0)
            Ibar2 = np.expand_dims(np.trace(cofCbar),axis=0)
            Ibar3 =  np.reshape(np.linalg.det(Cbar),(1,))
            delta_t = np.reshape(current_input[0],(1,))*2
            for j in range(0,3):
                if j == 0:
                    a1 = np.reshape(np.array([1,0,0]),(3,1))
                    #Ibar4 = np.matmul(np.transpose(a1),np.matmul(Cbar,a1))[0]
                    #Ibar5 = np.matmul(np.transpose(a1),np.matmul(np.matmul(Cbar,Cbar)
                    #                                             ,a1))[0]
                    
                    A1 = np.outer(a1,a1)
                    Ibar4 = np.expand_dims(np.trace(Cbar*A1),axis=0)
                    Ibar5 = np.expand_dims(np.trace(cofCbar*A1),axis=0)
                    dCdIbar5 = Ibar5 * np.linalg.inv(Cbar) - cofCbar * A1 * np.linalg.inv(Cbar)
                    Inv_first = np.concatenate((Ibar4,Ibar5),axis=0)
                elif j == 1:
                    a2 = np.reshape(np.array([0,1,0]),(3,1))
                    #Ibar4 = np.matmul(np.transpose(a1),np.matmul(Cbar,a1))[0]
                    #Ibar5 = np.matmul(np.transpose(a2),np.matmul(np.matmul(Cbar,Cbar)
                    #                                             ,a1))[0]
                    
                    A2 = np.outer(a2,a2)
                    Ibar4 = np.expand_dims(np.trace(Cbar*A2),axis=0)
                    Ibar5 = np.expand_dims(np.trace(cofCbar*A2),axis=0)
                    dCdIbar7 = Ibar5 * np.linalg.inv(Cbar) - cofCbar * A2 * np.linalg.inv(Cbar)
                    Inv_second = np.concatenate((Ibar4,Ibar5),axis=0)
                elif j == 2:
                    a3 = np.reshape(np.array([0,0,1]),(3,1))
                    Ibar4 = np.matmul(np.transpose(a3),np.matmul(Cbar,a3))[0]
                    Ibar5 = np.matmul(np.transpose(a3),np.matmul(np.matmul(Cbar,Cbar)
                                                                 ,a3))[0]
                    Inv_third = np.concatenate((Ibar4,Ibar5),axis=0)
                    A3 = np.outer(a3,a3)
            dCdIbar1 = I
            dCdIbar2 = Ibar1*I-Cbar
            dCdIbar3 = Ibar3*np.linalg.inv(Cbar)
            dCdIbar4 = A1
            dCdIbar6 = A2
            Cbar_vec = Cbar[np.triu_indices(3)]
            E_vec = E_grla[np.triu_indices(3)]
            dCdIbar1_vec = dCdIbar1[np.triu_indices(3)]
            dCdIbar2_vec = dCdIbar2[np.triu_indices(3)]
            dCdIbar3_vec = dCdIbar3[np.triu_indices(3)]
            dCdIbar4_vec = dCdIbar4[np.triu_indices(3)]
            dCdIbar5_vec = dCdIbar5[np.triu_indices(3)]
            dCdIbar6_vec = dCdIbar6[np.triu_indices(3)]
            dCdIbar7_vec = dCdIbar7[np.triu_indices(3)]
            temp = 0
            temp = np.reshape(temp,(1,))
            new_input = np.concatenate((Cbar_vec,delta_t,vp,zita,temp,Ibar1,Ibar2,Ibar3,Inv_first,Inv_second),axis=0)
            new_input_post = np.concatenate((Cbar_vec,dCdIbar1_vec,dCdIbar2_vec,dCdIbar3_vec,
                                             dCdIbar4_vec,dCdIbar5_vec,dCdIbar6_vec,
                                             dCdIbar7_vec),axis=0)
            X_post[k,i,:] = new_input_post
            X_new[k,i,:] = new_input
            X_E[k,i,:] = E_vec
    return X_new, X_post, X_E

