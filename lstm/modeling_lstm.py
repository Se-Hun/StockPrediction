import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layer, hps):
        super(LSTM, self).__init__()
        self.hps = hps # 모델의 하이퍼파라미터 정보 저장
        self.hidden_size = hidden_size # hidden unit의 수를 저장
        self.num_layer = num_layer # LSTM Layer의 수를 저장

        self.rnn = nn.LSTM(input_dim, hidden_size, num_layer)
        self.linear = nn.Linear(hidden_size, output_size) # 회귀 문제로 바꾸기 위한 선형 Layer

        self.dropout = nn.Dropout(p=0.1) # 과적합 방지를 위한 DropOut

        self.hidden = None # 이전 예측 결과 사용을 위해 hidden unit 예측 결과 저장

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

        out = self.dropout(out)

        out = self.linear(out[:, -1, :])

        return out


