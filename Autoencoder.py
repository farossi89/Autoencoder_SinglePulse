# File Regression.py
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import random

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn import metrics

from Function import *

# Variable and global
#######################################
keras.utils.set_random_seed(89)     # Set seeds for deterministic result
random.seed(89) 

TRAIN = 1                           # If 1 model is trained, if 0 model is loaded and only applied to test dataset 
PRUNE = 1

file    = 'best_model.h5'           # File for best float model
p_file  = 'p_best_model.h5'         # File for best pruned model
q_file  = 'q_best_model.h5'         # File for best quantized model

Input_Size = 48                     # Number of samples of each signal
latent_dim = 12                     # dimension of the compressed signal

# Data manipulation
# ############################################################################################################

# load data from file
#######################################
print('Load data from file...')
Data = LoadData('pulse_generated.csv', num_rows=25000)      #load the same number of signals from database

num_signal = 3
# Show some signals from Train dataset
plt.figure('Raw signals from database')
for i,signal in enumerate(Data[:num_signal]):
    plt.subplot(num_signal,1,i+1)
    plt.plot(signal)

# Extract only sample from each signal
#######################################
Samples = ExtractSamples(Data, 'Single')                    # Extract only signal's semples for Train and Test

# Show some signals from Train dataset
plt.figure('Signals from database')
for i,signal in enumerate(Samples[:num_signal]):
    plt.subplot(num_signal,1,i+1)
    plt.plot(signal)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

# Compute the integral of each signal
#######################################
Class = ComputeIntegral(Data, 'Single')                     # Compute integral of each signal as reference

#Show some signal during integration
plt.figure('Some integral')
ComputeIntegral(Data[:num_signal], 'Single', plot='True')


# Split database in test and train
#######################################
Train_Sample, Test_Sample, Train_Class, Test_Class = train_test_split(Samples, Class, test_size=0.25, random_state=42)

# Show Training dataset
print()
print('Training:')
print('Train sample')
print(Train_Sample)
print('Train class')
print(Train_Class)
print(f'Samples{Train_Sample.shape}, Class {Train_Class.shape}')
print()

# Show Test dataset
print('Test:') 
print('Test sample')
print(Test_Sample)
print('Test class')
print(Test_Class)
print(f'Samples{Test_Sample.shape}, Class {Test_Class.shape}')


# ###########################################################################################################
# Baseline model                                                                                            #        
# ###########################################################################################################
print()
print('Baseline model')
print('####################################################################################')
if TRAIN == 1:  
    # Model definition
    autoencoder_inputs = keras.Input(shape=(Input_Size,))
    
    x = layers.Dense(96, activation= 'relu')(autoencoder_inputs)               # Input layer
    x = layers.Dense(48, activation="relu")(x)                                  #
    #x = layers.Dense(24, activation="relu")(x)                                  #
    encoder_outputs = layers.Dense(latent_dim, activation="relu")(x)            # Output layer of encoder
    
    x = layers.Dense(latent_dim, activation='relu')(encoder_outputs)
    #x = layers.Dense(24, activation="relu")(x)                                  # Input layer of decoder
    x = layers.Dense(48, activation="relu")(x)                                  #
    x = layers.Dense(96, activation="relu")(x)                                 #
    decoder_outputs = layers.Dense(Input_Size)(x)                               # Output layer of decoder
    
    model = tf.keras.Model(inputs=autoencoder_inputs, outputs=decoder_outputs)  # Autoencoder model
         
    
    # Model compilation
    LR_ST=1e-3                                                                  # standard learning rate for Adam
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)                   # use of Adam optimizer
                
    model.compile(                                                              # Compile the model
                    optimizer=OPTIMIZER,                                        #
                    loss='mse',                                                 # optimization on mean square error
                    metrics=['mae'],                                            # considering also the mean absolute error during train
                    )
     
    # Model Training
    ####################################
    # def lr_decay(epoch)
    #
    # Function decrease learning rate exponentially after epoch number
    #
    # epoch: start of exponential decay of learning rate
    #
    # return learning rate value
    ####################################
    def lr_decay(epoch):
        if epoch < 50:
            return LR_ST
        else:
            return LR_ST * tf.math.exp(0.2 * (50 - epoch))
    
    # callbacks for decrease learning rate
    lr_callbacks = keras.callbacks.LearningRateScheduler(lr_decay)

    # callbacks for stop training before max epoch if no more improvement on loss function
    patience = 10                                                               # number of ephocs to wait if loss does't improve
    earlystop = EarlyStopping(                              
                                monitor = 'val_loss',                           # control of validation loss improvement
                                patience = patience                             # number of step after stop training with no monitor imnprovements
                             )
    
    # callbacks for save only the best model
    checkpoint = ModelCheckpoint(                                                              
                                'float_best.h5',                        # filename
                                monitor='val_loss',                     # check for validation loss
                                verbose=1,                              # print when file is saved
                                save_weights_only = True,               # save only the weights of model
                                save_best_only=True,                    # save only the best model   
                                mode='min'                              # min value of validation loss   
                                )
    
    print(model.summary())                                              # Show model parameters
    
    #input("Press Enter to continue...")

    # Model training
    num_epochs = 100                                                        # max number of epochs
    history = model.fit(
                        Train_Sample,                                       # Use Train dataset
                        Train_Sample,                                       # Use of the train dataset as Target
                        batch_size = 200,                                   # 100 signals each batch
                        epochs = num_epochs,                                # max number of epochs
                        validation_split = 0.2,                             # use 20% of train as validation dataset
                        callbacks = [lr_callbacks, checkpoint, earlystop]   # Insert the callbacks for save best model, stop if no improvements and modify learning rate
                        )
    
    # plot loss during training
    plt.figure ('Loss during Training')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')   
    # Plot 'X' for best model loss
    x_epoch = earlystop.stopped_epoch - (earlystop.patience)                    # retrieve stop epoch number
    testo = f'{history.history["loss"][x_epoch]:.3e}'                           # retieve val_loss value at stop epochs 
    if x_epoch < 0:                                                             # If training reach final epoch earlystop is set to 0
        x_epoch = num_epochs-1                                                  # put 'X' at the end of chart
    plt.plot(x_epoch, history.history['val_loss'][x_epoch], "x", color='black') # Plot 'X' at stop epoch
    plt.text(x_epoch-1, history.history["loss"][x_epoch], testo)                # Plot loss value near 'X'    
    
    #plt.show()
    
    # save best model
    model.load_weights('float_best.h5')                                         # load best weights in model
    tf.keras.models.save_model(model, file, include_optimizer=False)            # save model to use without training

    
# Evaluation of the trained model
#############################################################################################################
print("\n\nEvaluation")

model = load_model(file)                                    # Load best model

LR_ST=1e-3                                                  # standard learning rate for Adam
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)   # use of Adam optimizer

model.compile(                                              # Compile the model
                optimizer = OPTIMIZER,                      # Adam optimizer
                loss = 'mse',                               # optimization on mean square error
                metrics = ['mae'],                          # considering also the mean absolute error during train
            )


LossResult, StatResult, Error = EvalAuto(model, Test_Sample, Test_Class, 'Baseline', 'Single')        # Evaluation of model        

    
print('Original model statistics')
print(f'Mean: {StatResult["Mean_1"]:.2f} Std: {StatResult["Std_1"]:.2f}')        # print test loss and stat of reconstructed integral


num_signal = 3
# Apply the autoencoder to signals
recon = model(Test_Sample[:num_signal])

# Compute the integral before and after autoencoder
recon_inte = ComputeIntegral(recon, 'Single')

# Show some signal before and after the autoencoder for final comparison
plt.figure('Some reconstucted signals')
plt.tight_layout() 
for i,sgn in enumerate(Test_Sample[:num_signal]):
    plt.subplot(num_signal,1,i+1)
    plt.plot(sgn, label='Database')                         # Plot original signal from database
    plt.plot(recon[i], label='Baseline')                 # Plot reconstructed signal
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

# Print some percent error
print('Original Recon Error%')
for o,r in zip(Test_Class[:num_signal], recon_inte):
        error = 100.0*(o[0]-r[0])/o[0]
        print(f'{o[0]:.3f} {r[0]:.3f} {error:.3f}')


# ###########################################################################################################
# Pruned model                                                                                              #        
# ###########################################################################################################
print()
print('Pruned model')
print('####################################################################################')
# Load baseline model and compile
if PRUNE == 1:
    baseline_model = keras.models.load_model(file, compile=False)              # Load original model

    LR_ST=1e-3                                                  # standard learning rate for Adam
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)   # use of Adam optimizer
    baseline_model.compile(                                     # Compile the model
                        optimizer = OPTIMIZER,
                        loss='mse',                             # optimization on mean square error
                        metrics=['mae'],                        # considering also the mean absolute error during train
                        )

    # Model pruning and fine tuning
    epochs = 15                                                                         # during pruning train for a few epochs
    batch_size = 100                                                                    # number of signal at each batch
    end_step = np.ceil(Train_Sample.shape[0] / batch_size).astype(np.int32) * epochs    # compute the number of step during pruning

    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                                                                initial_sparsity = 0.50,      # start pruning 50% of the weigth
                                                                final_sparsity = 0.80,        # ends at 80% of weight
                                                                begin_step = 0,               # pruning from step 0
                                                                end_step = end_step)          # end after end_step step, slow descent
                    }

    ####################################
    # def apply_pruning_to_dense(layer)
    #
    # Function apply pruning only to Dense layer.
    # Useful for different type of models
    #
    # layer: layer of model
    #
    # return pruned layer
    ####################################
    def apply_pruning_to_dense(layer):                      
        if isinstance(layer, tf.keras.layers.Dense):                                            # isinstance verify Dense layers
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)            # remove only weight with lowest value
        return layer

    model_for_pruning = tf.keras.models.clone_model(                                            # build new model with pruned layer
                                                    baseline_model,                             # start from original model
                                                    clone_function = apply_pruning_to_dense,    # prune during clone
                                                    )

    # Pruned model compilation
    LR_ST=1e-3
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)               # same optimization of original model

    model_for_pruning.compile(
                                optimizer = OPTIMIZER,                      # compile model with the same parameter of original model
                                loss = 'mse',
                                metrics = ['mae']
                                )

    model_for_pruning.summary()

    # Definition of callbacks
    callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),                       # update pruning at each steps
                #tfmot.sparsity.keras.PruningSummaries(log_dir='test'),          # show pruning process
                keras.callbacks.ModelCheckpoint(                                # mae and save weights
                                                filepath = 'pruned_best.h5',            # save weights in a file
                                                monitor ='val_mae',                     # use mae on the validation for best model
                                                save_weights_only = True,               # save only the weights not the entire model
                                                save_best_only = True,                  # save only the best model
                                                save_freq = 'epoch'                     # check at each epochs
                                                )
                ]

    # Training of pruned model
    history = model_for_pruning.fit(
                                    Train_Sample,                       # Train dataset
                                    Train_Sample,                       # Use of the train dataset as class
                                    batch_size = batch_size,            # 100 signals each batch
                                    epochs = epochs,                    # max number of epochs
                                    validation_split = 0.2,             # use 20% of train as validation dataset
                                    callbacks = callbacks               # Callbacks for Checkpoint and pruning
                                    )
                   

    # plot loss during fine tuning
    plt.figure ('Loss of pruned model')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Fine tuning of pruned model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')   
    plt.legend()
    
    # save pruned model
    model_for_pruning.load_weights('pruned_best.h5')                    # Load the best parameters in the pruned model    
    p_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)     # remove extra-parameters used for pruning
    tf.keras.models.save_model(p_model, p_file)                         # Save pruned model to file


# Evaluation of the pruned model
################################################################################
pruned_model = keras.models.load_model(p_file, compile=False)       # compile=False to avoid warning

LR_ST=1e-3
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)           # same optimization of original model
pruned_model.compile(
                        optimizer = OPTIMIZER,                      # compile model with the same parameter of original model
                        loss = 'mse',
                        metrics = ['mae']
                        )

LossResult, StatResult_pruned, p_Error = EvalAuto(pruned_model, Test_Sample, Test_Class, 'Pruned', 'Single')   # Evaluation of model 

print('Pruned model')
print(f'Mean: {StatResult_pruned["Mean_1"]:.2f} Std: {StatResult_pruned["Std_1"]:.2f}')        # print test loss and stat of reconstructed integral

# Show some signal before and after the autoencoder
plt.figure('Some reconstucted signals')

# Apply the autoencoder to signals
p_recon = pruned_model(Test_Sample[:num_signal])

# Compute the integral after pruned autoencoder
p_recon_inte = ComputeIntegral(p_recon, 'Single')

for i,sgn in enumerate(Test_Sample[:num_signal]):
    plt.subplot(num_signal,1,i+1)
    plt.plot(p_recon[i], label='Pruned')
    plt.legend()

# Print some percent error
print('Original Recon Error%')
for o,r in zip(Test_Class[:num_signal], p_recon_inte):
        error = 100.0*(o[0]-r[0])/o[0]
        print(f'{o[0]:.3f} {r[0]:.3f} {error:.3f}')

print()
print('Original model')
print(model.get_weights()[0])               # Print some weights of the first layer
print_model_weights_sparsity(model)         # Print sparsity of model
print()
print('Pruned model')
print(pruned_model.get_weights()[0])        # Print some weights of the first layer
print_model_weights_sparsity(pruned_model)  # Print sparsity of model



# ###########################################################################################################
# Quantization of model                                                                                            #        
# ###########################################################################################################
print()
print('Quantized model')
print('####################################################################################')
# load pruned model and quantization
pruned_model = load_model(p_file, compile=False)                                           # load pruned model

q_model = tfmot.quantization.keras.quantize_annotate_model(pruned_model)             # Add parameters for quantization
q_model = tfmot.quantization.keras.quantize_apply(                                  # apply quantization
              q_model,
              tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme()   # preserve sparsity
              )


# Quantized Model compilation
#################################################################
LR_ST=1e-3                                                          # Standard learning rate
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)           # Adam optimizer

q_model.compile(
                optimizer = OPTIMIZER,
                loss = 'mse',
                metrics = ['mae']
                )

q_model.summary()


# Quantized Model fine tuning
#################################################################
# callbacks for save only the best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
                                                    filepath = 'q_best.h5',
                                                    monitor = 'val_mae',
                                                    save_weights_only = True, 
                                                    save_best_only = True,
                                                    save_freq = 'epoch'
                                                    )

# Train quantized model
history = q_model.fit(
                        Train_Sample,                   # Train dataset
                        Train_Sample,                   # Use of the train dataset as class
                        epochs = 10,                    # few ephocs for fine tuning
                        validation_split = 0.2,         # 20% for validation
                        verbose = 1,                    #
                        callbacks = [model_checkpoint]  #
                    )

# Save quantized model
q_model.load_weights('q_best.h5')                       # load best weights


# Show loss during fine tuning
plt.figure ('Loss of quantized model')
testo = f'{history.history["loss"][-1]:.3e}'
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Fine tuning of quantized model')
plt.xlabel('Epoch')
plt.ylabel('Loss')


# Evaluation of the quantized model
############################################################
print("\n\nEvaluation")

LossResult, StatResult_quantized, q_Error = EvalAuto(q_model, Test_Sample, Test_Class, 'Quantized', 'Single')   # Evaluation of model   

print('Quantized model statistics')
print(f'Mean: {StatResult_quantized["Mean_1"]:.2f} Std: {StatResult_quantized["Std_1"]:.2f}')        # print test loss and stat of reconstructed integral

plt.figure('Evaluation of quantized model') 
plt.xlabel('Error %')
plt.ylabel('Counts')
plt.hist(Error, bins=range(-10, 10), histtype='step', label='Original')     # Original model
plt.hist(p_Error, bins=range(-10, 10), histtype='step', label='Pruned')     # Pruned model
plt.hist(q_Error, bins=range(-10, 10), histtype='step', label='Quantized')  # Quantized model
plt.legend()   


# Show some signal before and after the autoencoder
plt.figure('Some reconstucted signals')

# Apply the autoencoder to signals
recon = q_model(Test_Sample[:num_signal])

# Compute the integral before and after autoencoder
q_recon_inte = ComputeIntegral(recon, 'Single')



# Plot some signals
for i,sgn in enumerate(Test_Sample[:num_signal]):
    plt.subplot(num_signal,1,i+1)
    plt.plot(recon[i], label='Quantized')
    plt.legend()


# Print some percent error
print('Original Recon error%')
for o,r in zip(Test_Class[:num_signal], q_recon_inte):
        error = 100.0*((o[0]-r[0])/o[0])
        print(f'{o[0]:.3f} {r[0]:.3f} {error:.3f}')


# Comparison between different models
# ########################################################################################
baseline_model = keras.models.load_model(file, compile=False)                   # Load original model
pruned_model = keras.models.load_model(p_file, compile=False)
# baseline model
float_converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
float_tflite_model = float_converter.convert()
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
    f.write(float_tflite_model)

# pruned model
pruned_converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
pruned_tflite_model = pruned_converter.convert()
_, pruned_file = tempfile.mkstemp('.tflite')
with open(pruned_file, 'wb') as f:
    f.write(pruned_tflite_model)

# quantized model
quantized_converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
quantized_converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = quantized_converter.convert()
_, quantized_file = tempfile.mkstemp('.tflite')
with open(quantized_file, 'wb') as f:
    f.write(quantized_tflite_model)


print()
print('Comparison between models')
print('####################################################################################')
LR_ST=1e-3                                                 
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST) 

# Original model
print('Baseline model:')

baseline_model.compile(optimizer = OPTIMIZER, loss='mse', metrics=['mae'])      # Compile with same option

start = time.time()
basescore = baseline_model.evaluate(Test_Sample, Test_Sample, verbose=2)        # evaluation of model
stop = time.time()
print(f'Mean: {StatResult["Mean_1"]:.2f} Std: {StatResult["Std_1"]:.2f}')       # print stat of reconstructed integral
print(f'Elapsed time = {(stop-start)/Test_Sample.shape[0]}')                    # Compute evaluation time
print(f'Float model in kb: {os.path.getsize(float_file) / float(2**10)}')
print()

# Pruned model
print('Pruned model:')

pruned_model.compile(optimizer = OPTIMIZER, loss = 'mse', metrics = ['mae'])

start = time.time()
score_p = pruned_model.evaluate(Test_Sample, Test_Sample, verbose=2)
stop = time.time()
print(f'Mean: {StatResult_pruned["Mean_1"]:.2f} Std: {StatResult_pruned["Std_1"]:.2f}')        # print stat of reconstructed integral
print(f'Elapsed time = {(stop-start)/Test_Sample.shape[0]}')                            # Compute evaluation time
print(f'Pruned model in kb: {os.path.getsize(pruned_file) / float(2**10)}')
print()

# Quantized model
print('Pruned+Quantized model:')

start = time.time()
score_q = q_model.evaluate(Test_Sample, Test_Sample, verbose=2)
stop = time.time()
print(f'Mean: {StatResult_quantized["Mean_1"]:.2f} Std: {StatResult_quantized["Std_1"]:.2f}')        # print stat of reconstructed integral
print(f'Elapsed time = {(stop-start)/Test_Sample.shape[0]}')                                # Compute evaluation time
print(f'Quantized model in kb: {os.path.getsize(quantized_file) / float(2**10)}')
print()


plt.show()
exit()
