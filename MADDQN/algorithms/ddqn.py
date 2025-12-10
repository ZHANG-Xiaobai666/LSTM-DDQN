import torch
import torch.nn as nn

from algorithms.lstm import LstmLayer

class DDQNLayer(nn.Module):
    def __init__(self, obs_dim, action_dim, hid_shape):
        super(DDQNLayer, self).__init__()
        self.lstm = LstmLayer(obs_dim, hid_shape)
        self.V = nn.Linear(hid_shape[-1], 1)
        self.A = nn.Linear(hid_shape[-1], action_dim)

    def forward(self, s):
        s = self.lstm(s)
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))
        return Q

    def reset_lstm_state(self):
        self.lstm.reset()