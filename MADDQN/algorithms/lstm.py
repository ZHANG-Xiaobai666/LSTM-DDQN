import torch
import torch.nn as nn

class LstmLayer(nn.Module):
    def __init__(self, obs_dim, hid_shape):
        super(LstmLayer, self).__init__()
        self.hid_shape = hid_shape
        self.lstm = nn.LSTM(obs_dim, self.hid_shape[-1], batch_first = True)
        self.hidden_state = (torch.zeros(1, 1, self.hid_shape[-1]), torch.zeros(1, 1, self.hid_shape[-1]))

    def forward(self, obs):
        lstm_out, self.hidden_state = self.lstm(obs, self.hidden_state)
        return lstm_out

    def reset(self):
        self.hidden_state = (torch.zeros(1, 1, self.hid_shape[-1]), torch.zeros(1, 1, self.hid_shape[-1]))