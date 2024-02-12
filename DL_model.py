'''
PINN layers.
'''

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import matplotlib.pyplot as plt
from tensorflow import keras

def act(x):
    return tf.keras.backend.elu(tf.keras.backend.pow(x, 2), alpha=1.0) 
tf.keras.utils.get_custom_objects().update({'act': tf.keras.layers.Activation(act)})

#Shifted Softplus Activation Function
def my_softplus(x):
    return tf.math.log(0.5 + 0.5 * tf.exp(x))
tf.keras.utils.get_custom_objects().update({'my_softplus': tf.keras.layers.Activation(my_softplus)})

# exponential activation function
def activation_Exp(x):
    return 1.0 * (tf.math.exp(x) - 1.0)

#Soft++
def soft_pp(x):
    k = 1
    c = 1
    return tf.math.log(1 + tf.exp(tf.multiply(k,x))) + tf.divide(x, c) - tf.math.log(2)
tf.keras.utils.get_custom_objects().update({'softplusplus': tf.keras.layers.Activation(soft_pp)})

class DL(tf.keras.Model):
    def __init__(self,s_all,m_all,s_out,m_out,layer_size,internal_variables,layer_size_fenergy,max_psi,
                 training_silent=True):
        self.training_silent=training_silent
        super(DL, self).__init__()
        self.f0=tf.keras.layers.LSTM(units=layer_size, name="history_lstm",
                                                return_sequences=True,
                                                return_state=False,
                                                use_bias=True)
        self.f02=tf.keras.layers.LSTM(units=layer_size, name="history_lstm",
                                                return_sequences=True,
                                                return_state=False,
                                                use_bias=True)
        
        self.f1=tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size, activation='swish'))
        self.f12 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size, activation='swish'))
        self.f2=tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(internal_variables, use_bias=True))
        
        self.f3=tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size_fenergy, activation='softplus'))
        self.f32=tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(layer_size_fenergy, activation='softplus',
                                  kernel_constraint=keras.constraints.NonNeg()))
        self.f4=tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1,use_bias=False,kernel_constraint=keras.constraints.NonNeg(),
            name='free_energy'))
        
        self.s_c11,self.s_dt,self.s_np,self.s_zita,self.s_temp,self.s_sig11 = s_all
        self.m_c11,self.m_dt,self.m_np,self.m_zita,self.m_temp,self.m_sig11 = m_all

        self.max_psi = max_psi
        # self.s_sig11= s_out
        # self.m_sig11 = m_out
    
    def tf_u_stnd_nrml(self,output,a,b):
        ''' Un-standardize/un-normalize '''
        output_new = (output + 1) / 2.0
        return tf.add(tf.multiply(output_new,a),b)
    
    def tf_out_stnd_nrml(self,u,a,b):
        ''' Standardize/normalize '''
        return tf.divide(tf.add(u,-b),a)*2-1
    

    def call(self,un):   
        # Slice from inputs
        un_c_t_f = tf.slice(un,[0,0,0],[-1,-1,1])
        un_c_tdt_f = tf.slice(un,[0,1,0],[-1,-1,1])
        un_nv_t_f = tf.slice(un,[0,0,2],[-1,-1,1])
        un_nv_tdt_f = tf.slice(un,[0,1,2],[-1,-1,1])
        un_dt_t_f = tf.slice(un,[0,0,1],[-1,-1,1])
        un_dt_tdt_f = tf.slice(un,[0,1,1],[-1,-1,1])
        un_temp_t_f = tf.slice(un,[0,0,4],[-1,-1,1])
        un_temp_tdt_f = tf.slice(un,[0,1,4],[-1,-1,1])
        un_zita_t_f = tf.slice(un,[0,0,3],[-1,-1,1])
        un_zita_tdt_f = tf.slice(un,[0,1,3],[-1,-1,1])
        un_sigma_t_f = tf.slice(un,[0,0,5],[-1,-1,1])
        un_sigma_tdt_f = tf.slice(un,[0,1,5],[-1,-1,1])
        # Un-normalized dt
        u_dt_t = self.tf_u_stnd_nrml(un_dt_tdt_f, self.s_dt, self.m_dt)  
        u_dt_t_f = self.tf_u_stnd_nrml(un_dt_t_f, self.s_dt, self.m_dt)  
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(un_c_t_f)
            # Combine input for t+dt
            combined_input = tf.concat([un_c_tdt_f, un_dt_tdt_f,un_nv_tdt_f,un_zita_tdt_f],
                                           axis=-1)
            # Predict history information of LSTMs
            nf0_sub = self.f0(combined_input)
            nf0_sub_final = self.f02(nf0_sub)
            # Obtain the initial history output h(t=0) = 0.0
            init_state = tf.expand_dims(self.f0.get_initial_state(combined_input)[0],axis=1)
            # Combine t=0 with t+dt
            combined_states = tf.concat([init_state, nf0_sub_final],axis=1)
            # Use the FFN to predict the internal variables
            nf1 = self.f1(combined_states)
            nf12 = self.f12(nf1)
            z_i_1_final = self.f2(nf12)
            # Calculate psi for C = 1
            combined_input_int = tf.expand_dims(tf.concat([z_i_1_final[:,0,:], un_c_t_f[:,0,:]],
                                        axis=-1),1)
            # Predict free-energy
            psi = self.f3(combined_input_int) # energy @ t and C = 1
            psi_2 = self.f32(psi)
            psi_t_init = self.f4(psi_2)
            #Calculate psi for whole sequence
            combined_input = tf.concat([z_i_1_final, un_c_t_f],
                                           axis=-1)
            # Predict free-energy
            psi = self.f3(combined_input) # energy @ t
            psi_2 = self.f32(psi)
            psi_t = self.f4(psi_2)
            # Set free-energy (@ t = 0) = 0.0
            psi_final = psi_t - psi_t_init
        #Calculate dissipation
        s_psi = self.s_sig11 * self.s_c11
        s_tau = self.s_sig11
        z_delta = tf.experimental.numpy.diff(z_i_1_final,n=1,axis=1)
        z_dot = z_delta/u_dt_t
        tau = tape.gradient(psi_final,z_i_1_final)
        f_diss_tdt = tf.slice(tau,[0,1,0],[-1,-1,-1]) * z_dot 
        diss_rate = tf.reduce_sum(f_diss_tdt,axis=-1)*(-1) * self.max_psi
        # Get stress
        f_σ_tdt = tape.gradient(psi_final,un_c_t_f)[:,1:,:] # σ @ t+dt (normalized)
        # Add loss of free-energy (not needed but in case) and dissipation
        # Here we use a static weight factor instead of using the adaptive one
        self.add_loss(tf.reduce_mean(tf.nn.relu(-diss_rate))*10)
        self.add_loss(tf.reduce_mean(tf.nn.relu(-psi_final))*10)
        return 2*f_σ_tdt
    
    # Use this function for postprocessing (with dissipation and free-energy as outputs)
    def obtain_output(self,un,output):
        # Slice from inputs
        un_c_t_f = tf.slice(un,[0,0,0],[-1,-1,1])
        un_c_tdt_f = tf.slice(un,[0,1,0],[-1,-1,1])
        un_nv_t_f = tf.slice(un,[0,0,2],[-1,-1,1])
        un_nv_tdt_f = tf.slice(un,[0,1,2],[-1,-1,1])
        un_dt_t_f = tf.slice(un,[0,0,1],[-1,-1,1])
        un_dt_tdt_f = tf.slice(un,[0,1,1],[-1,-1,1])
        un_temp_t_f = tf.slice(un,[0,0,4],[-1,-1,1])
        un_temp_tdt_f = tf.slice(un,[0,1,4],[-1,-1,1])
        un_zita_t_f = tf.slice(un,[0,0,3],[-1,-1,1])
        un_zita_tdt_f = tf.slice(un,[0,1,3],[-1,-1,1])
        un_sigma_t_f = tf.slice(un,[0,0,5],[-1,-1,1])
        un_sigma_tdt_f = tf.slice(un,[0,1,5],[-1,-1,1])
        # Un-normalized dt
        u_dt_t = self.tf_u_stnd_nrml(un_dt_tdt_f, self.s_dt, self.m_dt)  
        u_dt_t_f = self.tf_u_stnd_nrml(un_dt_t_f, self.s_dt, self.m_dt) 
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(un_c_t_f)
            # Combine input for t+dt
            combined_input = tf.concat([un_c_tdt_f, un_dt_tdt_f,un_nv_tdt_f,un_zita_tdt_f],
                                           axis=-1)
            # Predict history information of LSTMs
            nf0_sub = self.f0(combined_input)
            nf0_sub_final = self.f02(nf0_sub)
            # Obtain the initial history output h(t=0) = 0.0
            init_state = tf.expand_dims(self.f0.get_initial_state(combined_input)[0],axis=1)
            # Combine t=0 with t+dt
            combined_states = tf.concat([init_state, nf0_sub_final],axis=1)
            # Use the FFN to predict the internal variables
            nf1 = self.f1(combined_states)
            nf12 = self.f12(nf1)
            z_i_1_final = self.f2(nf12)
            # Calculate psi for C = 1
            combined_input_int = tf.expand_dims(tf.concat([z_i_1_final[:,0,:], un_c_t_f[:,0,:]],
                                        axis=-1),1)
            # Predict free-energy
            psi = self.f3(combined_input_int) # energy @ t and C = 1
            psi_2 = self.f32(psi)
            psi_t_init = self.f4(psi_2)
            #Calculate psi for whole sequence
            combined_input = tf.concat([z_i_1_final, un_c_t_f],
                                           axis=-1)
            # Predict free-energy
            psi = self.f3(combined_input) # energy @ t
            psi_2 = self.f32(psi)
            psi_t = self.f4(psi_2)
            # Set free-energy (@ t = 0) = 0.0
            psi_final = psi_t - psi_t_init
        #Calculate dissipation
        s_psi = self.s_sig11 * self.s_c11
        s_tau = self.s_sig11
        z_delta = tf.experimental.numpy.diff(z_i_1_final,n=1,axis=1)
        z_dot = z_delta/u_dt_t
        tau = tape.gradient(psi_final,z_i_1_final)
        f_diss_tdt = tf.slice(tau,[0,1,0],[-1,-1,-1]) * z_dot
        diss_rate = tf.reduce_sum(f_diss_tdt,axis=-1)*(-1) * self.max_psi
        # Get stress
        f_σ_tdt = tape.gradient(psi_final,un_c_t_f)[:,1:,:] # σ @ t+dt (normalized)
        f_σ_tdt_u = self.tf_u_stnd_nrml(2*f_σ_tdt, self.s_sig11, self.m_sig11) 
        return f_σ_tdt_u,psi_final,diss_rate,z_i_1_final
    
    
    def setTraining(self,TANN,input_training,output_training,learningRate,nEpochs,bSize,
                    val_input,val_output):
        if self.training_silent==False: silent_verbose=2
        else: silent_verbose=0
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=learningRate,
                        decay_steps=2000,
                        decay_rate=0.9)
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr_schedule)
        wα=1; 
        TANN.compile(optimizer=optimizer,loss=['mae'],loss_weights=[wα],run_eagerly=False)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                              min_delta=1.e-6,
                                                              patience=1000,verbose=0,
                                                              mode='auto',baseline=None,
                                                              restore_best_weights=True)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                            filepath='./weights/checkpoint',
                                                            save_weights_only=True,
                                                            monitor='loss',
                                                            mode='auto',
                                                            save_best_only=True,
                                                            save_freq = 300,
                                                            verbose=1)
        history = TANN.fit(input_training,output_training,
                           epochs=nEpochs,batch_size=bSize,
                           verbose=silent_verbose,
                           validation_data=(val_input,val_output),
                           callbacks=[earlystop_callback,model_checkpoint_callback])
        if self.training_silent==False: print("\n Training completed in", nEpochs, " epochs")
        return history