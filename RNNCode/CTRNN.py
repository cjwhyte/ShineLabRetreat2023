
""""
Continuous time RNN modified from:
    
Yang, G. R., & Wang, X. J. (2020). Artificial neural networks for 
neuroscientists: A primer. Neuron, 107(6), 1048-1070.

Christopher Whyte 18/10/2022

"""

import torch
import torch.nn as nn


class CTRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, sigma_gain=1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau # alpha = dt/tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self.input2h = nn.Linear(input_size, hidden_size,bias='False')
        self.h2h = nn.Linear(hidden_size, hidden_size,bias='False')
        self.gain = sigma_gain

    def init_hidden(self, input_shape):
        batch_size = 1 # batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.h2h(hidden) + self.input2h(input) 
        h_new = hidden * self.oneminusalpha + self.alpha*torch.sigmoid(self.gain*(pre_activation))
        # h_new = hidden * self.oneminusalpha + self.alpha*torch.tanh(self.gain*(pre_activation))
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class RNNNet(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity