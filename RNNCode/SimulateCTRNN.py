"""
@author: Christopher Whyte
"""

import torch
import numpy as np

task = 1; # Perceptual decision making = 0, WM = 1

# %% Import supervised dataset 

num_trials = 1000

if task == 0:
    from PerceptualDecisionEnvGen import PerceptualDecisionEnv
    trl_type = np.tile([0,1],int(num_trials/2))
    np.random.shuffle(trl_type)
    ob, gt, direction = PerceptualDecisionEnv(0)
elif task == 1:
    from WMEnvGen import WMEnv
    trl_type = np.tile([0,1,2,3],int(num_trials/4))
    np.random.shuffle(trl_type)
    ob, gt, direction = WMEnv(0)

trial_length = np.size(ob,1)
      
# %% import and initialise network 

# network
from CTRNN import RNNNet

# Instantialise the network
hidden_size = 50; input_size = 1; output_size = 3
net = RNNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=20)

# %% run sims


# import pretrained networks
varname = 'CRTNN_PerceptualDecision.pth'
net.load_state_dict(torch.load(varname))

# initialise arrays 
activity = np.zeros([trial_length,hidden_size,num_trials])
labels =  np.zeros([trial_length,num_trials])
RNN_choice = np.zeros([trial_length,num_trials])
trial_condition = np.zeros([1,num_trials])

print('Simulation: start')
for trl in range(num_trials):
    
    if task == 0:
        # Generate input and target, convert to pytorch tensor
        ob, gt, trial_type = PerceptualDecisionEnv(trl_type[trl])
    elif task == 1:
        # Generate input and target, convert to pytorch tensor
        ob, gt, trial_type = WMEnv(trl_type[trl])
    
    inputs = ob.T
    
    # inputs = torch.from_numpy(inputs[:, np.newaxis, :]).type(torch.float)
    inputs = torch.from_numpy(inputs).type(torch.float)
    output, rnn_activity = net(inputs)
    
    z = output.detach().numpy()
    
    rnn_activity = np.squeeze(rnn_activity.detach().numpy())
    
    
    # output = output.view(-1, output_size) # Reshape to (SeqLen x Batch, OutputSize)
    output = torch.squeeze(output)
    choices = torch.argmax(output, dim=1)
    choices = choices.numpy()
    
    activity[:,:,trl] = rnn_activity
    labels[:,trl] = gt
    RNN_choice[:,trl] = choices
    trial_condition[:,trl] = trial_type
    
    # Compute the running loss every 100 steps
    if trl % 100 == 99:
        print('trial {}'.format(trl))
        

activity_array = np.reshape(activity,[trial_length,hidden_size*num_trials], order = 'F') # reshape activity for exporting (needs to be a matrix)

# save activity array for each random seed
if task == 0:
    np.savetxt("activity_PD1.csv", activity_array, delimiter=",")
    np.savetxt("labels_PD1.csv", labels, delimiter=",")
    np.savetxt("choices_PD1.csv", choices, delimiter=",")
    np.savetxt("trial_condition_PD1.csv", trial_condition, delimiter=",")
if task == 1:
    np.savetxt("activity_WM1.csv", activity_array, delimiter=",")
    np.savetxt("labels_WM1.csv", labels, delimiter=",")
    np.savetxt("choices_WM1.csv", choices, delimiter=",")
    np.savetxt("trial_condition_WM1.csv", trial_condition, delimiter=",")

print('Simulation: finished')

