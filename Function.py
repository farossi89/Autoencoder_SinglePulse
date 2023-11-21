# file Function.py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import integrate
from scipy.signal import find_peaks, peak_prominences

import tensorflow as tf
import keras
from tensorflow.keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras_visualizer import visualizer
from sklearn.model_selection import train_test_split

import tensorflow_model_optimization as tfmot

import time

# Global Variable
PERIOD = 1


####################################
# def LoadData(nomefile, num_rows=10)
#
# Function load data from file
#
# nomefile: File name with extension
# num_rows: Number of rows to load from file
#
# return samples as a list of list
####################################
def LoadData(nomefile, num_rows=10):
    samples = []                                                                    # Initialize empty sample list
    with open (nomefile) as f:                                                      # Open file passed as argument
        for i, line in enumerate(f.readlines()):                                    # read every lines in file
                if i < num_rows:                                                    # Read n row passed as argument     
                        samples.append(([float(x) for x in line[:-2].split(',')]))  # Convert line to list and append to dataset
                else:
                    break
    return samples


####################################
# def ExtractSamples(Data, key)
#
# Function return the Data in the form of np array
#
# Data: List of signals
# key:  Type of signals, one of the dictionary key (Single, Double or PileUp)
#
# return np.array of list of signals
####################################
def ExtractSamples(Data, key):
    samples = []
    init = 0                                # Initial number of skipped samples
    if key == 'Single':
        init = 1                            # 1 for Single pulse signal, first number is time
    if key == 'Double':
        init = 2                            # 2 for Double pulse signal, firsts two number are time
    if key == 'PileUp': 
        init = 4                            # 4 for PileUp pulse signal, firsts 4 number are time and integral of the two pulse
    for signal in Data:                     
        samples.append(signal[init:])       # create new array with only amplidute samples
    return np.array(samples)                # return as numpy array
 
#&
####################################
# def ComputeIntegral(Data, key, plot= False)
#
# Function use the correct integral function on each signal type
#
# Data: List of signals
# key:    type of database. Single, Double or PileUp
#
# return list of integral value as numpy array
####################################
def ComputeIntegral(Data, key, plot = False):

    if key == 'Single':
        result = ComputeIntegral_1(Data, plot)
    if key == 'Double':
        result = ComputeIntegral_2(Data, plot)
    if key == 'PileUp':
        result = ReadPileupIntegral(Data)
    return result



####################################
# def ComputeIntegral_1(Data, plot=False)
#
# Function compute the integral of single signals
#
# Data: List of signals
# plot: show plot of elaborated signal. True or false
#
# return list of integral value as numpy array
####################################
def ComputeIntegral_1(Data, plot=False): 
    ret_integ = []                                                      # List od integral of each signal
    
    y = []
    for i in range(0, 48,1):                                            # create correct timebase
        y.append(i * PERIOD)

    for num_sgn,x in enumerate(Data):                                   # for all signal in dataset
        # Find peaks
        peaks = []                                                      # list of peaks position
        my_integral = []                                                # list of integral per each signal
        #peaks, _ = find_peaks(x, distance=20, width=3, prominence=0.5) # find peacks with empirical parameters. Do not works on scaled signals...
        peaks.append(np.argmax(x[1:], axis=0) + 1 )                     # find only maximun of signal, ok for single signals    

        init = peaks[0] - 3                                             # init of peak empirical. Peak max - 3 samples
        end = peaks[0] + 8                                              # end of peak empirical. Peak max + 8 samples    
                
        # Baseline evaluation
        base = 0.0                                                      # Value of the baseline
        cnt = 0                                                         # counter for sample in the peak area
        for i,p in enumerate(x[1:len(x)]):                              # for each element in signal
            if ((i < init) | (i > end)):                                # compute baseline only outside peack area
                base = base + p
                cnt = cnt + 1                                           # count number of element for mean        
        base = base / cnt                                               # compute mean value
        
        # Baseline restoration
        xb = [item - base for item in x]                                # remove the mean value from all values   
        xb[0] = x[0]                                                    # keep time information
        
        # simple integral. Only sum of all peak samples
        ######################################################################################################
        cnt = 0
        integral = 0.0
        for i,p in enumerate(xb):                           # for each element in the signal
            if ((i >= init) & (i <= end)):                  # compute integral in the peack area
                integral = integral + p
                cnt = cnt + 1                               # count number of element
        my_integral.append(integral)
        ######################################################################################################
        
        # Trapezoidal integral. Use Trapezoidal algorithm
        ######################################################################################################
        # my_integral.append(integrate.trapezoid(xb[init:end], y[init+1:end+1]))
        ######################################################################################################
        
        # Simpson integral. Use Simpson algorithm
        ######################################################################################################
        # my_integral.append(integrate.simpson(xb[init:end], y[init+1:end+1]))
        ######################################################################################################
                
        my_integral.append(0)                           # Only one peak, the second is 0.
        
        if plot:
            print(f'Integral:{my_integral} Baseline:{base}')      
            
            plt.subplot(3, 1, num_sgn+1)  
            for p in peaks:
                plt.plot(y[p-1], xb[p], "x", color='black', label='peak')   # Plot black X in the graph   
            plt.plot(y[0:init-1], xb[1:init], 'r', label='baseline')        # plot first part of baseline
            plt.plot(y[init-1:end-1], xb[init:end], label='integral')       # plot peak
            plt.fill_between(y[init-1:end-1], xb[init:end], alpha=0.2)      # plot peak integral
            plt.plot(y[end-1:len(x)-2], xb[end:len(x)-1],'r')               # plot second part of baseline
            plt.legend()
            
        ret_integ.append(my_integral)                                       # Append the integral to the total list    

    return np.array(ret_integ)                                              # Result as numpy array


####################################
# def EvalAuto(model, Test_Sample, Test_Class, ml, test_type)
#
# Function reads the two integral value from raw data
#
# Data: List of signals
#
# return integral value as numpy array
####################################
def EvalAuto(model, Test_Sample, Test_Class, ml, test_type):

    # evaluation of the model
    loss, accuracy = model.evaluate(Test_Sample, Test_Sample)       # Evaluation of the model on test dataset
                                     
    recon = model.predict(Test_Sample)                              # prediction on test dataset    
    result = ComputeIntegral(recon, test_type)                      # Compute integral of reconstructed signals
        
    Percent_1 = []
    for r,o in zip(result, Test_Class):
        Percent_1.append((((o[0]-r[0])/o[0])*100.0))                # Compute error of each signal

    stat = {}
    stat['Mean_1'] = sum(Percent_1) / len(Percent_1)                # Mean value
    stat['Std_1'] = np.std(Percent_1)                               # Standard deviation
        
    return loss, stat, Percent_1

####################################
# def print_model_weights_sparsity(model)
#
# Function print the sparsity of model. How many weights are equal to 0
#
# model: model as imput 
#
####################################
def print_model_weights_sparsity(model):
    for layer in model.layers:                              # for each layer
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights               # considering only the trainable weights if wrapper are used
        else:
            weights = layer.weights                         # otherwise all layer
        for weight in weights:                              
            if "quantize_layer" in weight.name:             # ignore auxiliary quantization weights
                continue
            weight_size = weight.numpy().size               # count how many weights    
            zero_num = np.count_nonzero(weight == 0)        # count how many weights are equal to 0
            print(
                f"{weight.name}: {zero_num/weight_size:.2%} sparsity ", # Sparsity
                f"({zero_num}/{weight_size})",                          # Number of zeros / number of weight
            )