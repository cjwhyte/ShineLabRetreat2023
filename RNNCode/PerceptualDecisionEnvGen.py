"""
Function generates perceptual decision making task environemnt for RNN:
    
arguments: randomseed for stimulus noise
    
return: inputs = time x input size, labels = time

Christopher Whyte 19/11/2023

"""
import numpy as np

def PerceptualDecisionEnv(direction):
    
    trial_length = 50
    
    mu, sigma = 0,.1;
    
    noise = np.random.normal(mu,sigma,[1,trial_length])

    # inputs
    inputs = np.zeros([1,trial_length])

    if direction == 0:
        inputs[0,0:trial_length] += np.linspace(0,1,trial_length).T
    else:
        inputs[0,0:trial_length] += np.linspace(0,-1,trial_length).T
        
    inputs += noise;
        
    # trial info
    labels = np.zeros([1,trial_length])
    if direction == 0:
        labels[:,40:] = 1; 
    else:
        labels[:,40:] = 2; 

    return inputs, labels, direction

