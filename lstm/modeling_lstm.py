import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_size, hps, birnn=False):
        super(LSTM, self).__init__()
        # input_dim     : dimension for each input
        # num_steps     : number of token in a sequence
        # num_layers    : number of layers of RNN
        # cell_type     : 'LSTM', 'GRU'

        self.hps = hps

        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_layer

        self.birnn = birnn
        self.num_numerator = 2 if self.birnn else 1

        self.rnn = nn.LSTM(input_dim,
                            self.hidden_dim // self.num_numerator,
                            num_layers=self.num_rnn_layers,
                            bidirectional=self.birnn,
                            batch_first=True
                            )  # [batch_size, input_dim] --> [batch_size, hidden_dim]

        self.linear = nn.Linear(hidden_dim, output_size)

    def init_hidden(self, device, batch_size):
        if self.birnn:
            _num_rnn_layers = self.num_rnn_layers * 2
        else:
            _num_rnn_layers = self.num_rnn_layers

        B = batch_size
        h0 = Variable(torch.zeros(_num_rnn_layers, B, self.hidden_dim // self.num_numerator)).to(device)
        c0 = Variable(torch.zeros(_num_rnn_layers, B, self.hidden_dim // self.num_numerator)).to(device)

        return (h0, c0)

    def reset_initial_state(self, device, batch_size):
        self.initial_state = self.init_hidden(device, batch_size)

    def forward(self, input):
        # input = [batch_size, num_steps, input_dim]
        #
        # return rnn_output and last output

        batch_size = input.shape[0]
        device = input.device

        self.reset_initial_state(device, batch_size)

        rnn_out, self.initial_state = self.rnn(input, self.initial_state)
        # rnn_out = [batch_size, num_steps, output_dim]

        last_rnn_out = rnn_out[:, -1, :]

        predictions = self.linear(last_rnn_out.view(len(input), -1))
        # output
        #   - rnn_out : [batch_size, num_steps, output_dim]
        #   - last_rnn_out : [batch_size, output_dim]
        return predictions[-1]


# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#
#         self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
#                             torch.zeros(1,1,self.hidden_layer_size))
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]


'''
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
        '''