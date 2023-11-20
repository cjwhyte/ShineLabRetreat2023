"""
Function generates delay match to sample task environemnt for RNN:
    
arguments: randomseed for stimulus noise
    
return: inputs = time x input size, labels = time

Christopher Whyte 19/11/2023

"""
import numpy as np

def WMEnv(condition):
    
    trial_length = 50
    
    mu, sigma = 0,.1;
    
    noise = np.random.normal(mu,sigma,[1,trial_length])

    # inputs
    inputs = np.zeros([1,trial_length])

    if condition == 0:
        # cue
        inputs[0,0:10] += 1
        # sample
        inputs[0,40:] += 1
    elif condition == 1:
        # cue
        inputs[0,0:10] += 1
        # sample
        inputs[0,40:] += -1   
    elif condition == 2:
        # cue
        inputs[0,0:10] += 1
        # sample
        inputs[0,40:] += 1  
    elif condition == 3:
        # cue
        inputs[0,0:10] += 1
        # sample
        inputs[0,40:] += -1  
        
    inputs += noise;
        
    # trial info
    labels = np.zeros([1,trial_length])
    if condition == 0:
        labels[:,40:] = 1; 
    elif condition == 1:
        labels[:,40:] = 2; 
    elif condition == 2:
        labels[:,40:] = 1; 
    elif condition == 3:
        labels[:,40:] = 2; 

    return inputs, labels, condition

