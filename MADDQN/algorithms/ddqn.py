import torch
import torch.nn as nn
import copy
from algorithms.lstm import LstmLayer

class DDQNLayer(nn.Module):
    def __init__(self, obs_dim, action_dim, hid_shape):
        super(DDQNLayer, self).__init__()
        self.lstm = nn.LSTM(obs_dim, hid_shape[-1], batch_first=True)
        self.V = nn.Linear(hid_shape[-1], 1)
        self.A = nn.Linear(hid_shape[-1], action_dim)
        self.hidden_state = None

    def forward(self, s, is_update=True):
        if is_update:
            s, self.hidden_state = self.lstm(s, self.hidden_state)
        else:
            s, _ = self.lstm(s, self.hidden_state)
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))
        return Q

    def get_eval_q(self, obs, hidden_state):
        s, _ = self.lstm(obs, hidden_state)
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))
        return Q
    def reset_lstm_state(self):
        self.hidden_state = None
    def get_hidden_state(self):
        return (self.hidden_state[0].detach(), self.hidden_state[1].detach())