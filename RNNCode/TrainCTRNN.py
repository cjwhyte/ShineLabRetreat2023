# %% train network on WM or perceptual decision task

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


task = 1; # Perceptual decision making = 0, WM = 1

# %% Import supervised dataset 

training_iterations = 5000

if task == 0:
    from PerceptualDecisionEnvGen import PerceptualDecisionEnv
    trl_type = np.tile([0,1],int(training_iterations/2))
    np.random.shuffle(trl_type)
elif task == 1:
    from WMEnvGen import WMEnv
    trl_type = np.tile([0,1,2,3],int(training_iterations/4))
    np.random.shuffle(trl_type)

# %% import network

from CTRNN import RNNNet

# Instantiate the network and print information

hidden_size = 50; input_size = 1; output_size = 3
net = RNNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=20)

# Use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

losses = []; accuracies = []
running_loss = 0; running_acc = 0

print('Training network:')
for i in range(training_iterations):
    
    
    if task == 0:
        # Generate input and target, convert to pytorch tensor
        ob, gt, trial_type = PerceptualDecisionEnv(trl_type[i])
    elif task == 1:
        # Generate input and target, convert to pytorch tensor
        ob, gt, trial_type = WMEnv(trl_type[i])
    
    inputs = ob.T; labels = gt.T

    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels.flatten()).type(torch.long)

    # boiler plate pytorch training:
    optimizer.zero_grad() # zero the gradient buffers
    output, _ = net(inputs)
    output = torch.squeeze(output)
    # output = output.view(-1, output_size) # Reshape to (SeqLen x Batch, OutputSize)

    # cross entropy loss function
    loss = criterion(output, labels)

    loss.backward()
    optimizer.step() # Does the update
    
    # grab choice and compare to ground truth
    choices = torch.argmax(output, dim=1) 
    acc = torch.mean(torch.eq(choices, labels).type(torch.float32)) #torch.eq computs elementwise equality

    # compute running accuracy and loss
    running_loss += loss.item()
    running_acc += acc.item()
    
    accuracies.append(acc.item())
    losses.append(loss.item())
    
    # Compute the running loss every 100 steps
    if i % 100 == 99:
        running_loss /= 100
        running_acc /= 100
        print('Step {}, Loss {:0.4f}, Acc {:0.4f}'.format(i+1, running_loss, running_acc))
        running_acc = 0
        running_loss = 0
   
# save network weights    
if task == 0:
    varname = 'CRTNN_PerceptualDecision.pth' 
elif task == 1:
    varname = 'CRTNN_WM.pth' 
    
torch.save(net.state_dict(), varname)

# %% Plot the network performance and loss over training iterations

# import matplotlib.pyplot as plt

# fig1, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 4))
# ax1.plot(losses, color= "orange")
# ax2.plot(accuracies, color= "blue")

# ax1.set_xlabel("Training steps")
# ax2.set_xlabel("Training steps")

# ax1.set_title("Losses")
# ax2.set_title("Accuracies")
