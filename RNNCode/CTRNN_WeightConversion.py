"""
Conversion of network weights into numpy array that can be analysed in matlab

@author: christopherwhyte
"""

import torch
import numpy as np

from CTRNN import RNNNet

task = 1

# Instantialise the network
hidden_size = 50; input_size = 1; output_size = 3
net = RNNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=20)

# import pretrained networks
if task == 0:
    varname = 'CRTNN_PerceptualDecision.pth' 
elif task == 1:
    varname = 'CRTNN_WM.pth' 
net.load_state_dict(torch.load(varname))


# weights
W_in = np.squeeze(net.rnn.input2h.weight.detach().numpy())
W_out = net.fc.weight.detach().numpy() 
W_res = net.rnn.h2h.weight.detach().numpy()

in_bias = net.rnn.input2h.bias.detach().numpy()
res_bias = net.rnn.h2h.bias.detach().numpy()
out_bias = net.fc.weight.detach().numpy() 

# save weights 
if task == 0:
    np.savetxt("W_in_PD1", W_in, delimiter=",")
    np.savetxt("W_out_PD1", W_out, delimiter=",")
    np.savetxt("W_res_PD1", W_res, delimiter=",")
    np.savetxt("in_bias_PD1", in_bias, delimiter=",")
    np.savetxt("out_bias_PD1", out_bias, delimiter=",")
    np.savetxt("res_bias_PD1", res_bias, delimiter=",")
elif task == 1:
    np.savetxt("W_in_WM1", W_in, delimiter=",")
    np.savetxt("W_out_WM1", W_out, delimiter=",")
    np.savetxt("W_resWM1", W_res, delimiter=",")
    np.savetxt("in_bias_WM1", in_bias, delimiter=",")
    np.savetxt("out_bias_WM1", out_bias, delimiter=",")
    np.savetxt("res_bias_WM1", res_bias, delimiter=",")
    