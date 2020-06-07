import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layer, hps):
        super(LSTM, self).__init__()
        self.hps = hps
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.rnn = nn.LSTM(input_dim, hidden_size, num_layer)
        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden = None

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).requires_grad_()

        return (h0, c0)

    def forward(self, input):
        device = input.device

        h0, c0 = self.hidden

        h0 = h0.to(device)
        c0 = c0.to(device)

        out, (hn, cn) = self.rnn(input, (h0.detach(), c0.detach()))

        self.hidden = (hn, cn)

        out = self.linear(out[:, -1, :])

        return out

