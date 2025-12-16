
import torch.nn.functional as F
import torch
import torch.nn as nn
from algorithms.ddqn import DDQNLayer
import numpy as np
import copy


class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim

        self.net_width = args.net_width

        self.alpha_max = args.alpha
        self.alpha = self.alpha_max # exploration probability
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.beta = self.beta_min  # temperature
        self.gamma = args.gamma # discount factor

        self.lr = args.lr # learning rate
        self.device = args.dvc

        self.ddqn = DDQNLayer(self.obs_dim, self.action_dim, (self.net_width, self.net_width))
        self.ddqn_optimizer = torch.optim.Adam(self.ddqn.parameters(), self.lr)

        self.ddqn_target = copy.deepcopy(self.ddqn)
        #self.tpdv = dict(dtype=torch.float32, device=device)


    def select_action(self, obs, deterministic=False):
        #obs = check(obs).to(**self.tpdv)
        #rnn_states = check(rnn_states).to(**self.tpdv)

        obs = torch.tensor(obs, dtype=torch.float32).view(1, 1, self.obs_dim)

        Q = self.ddqn(obs)
        _ = self.ddqn_target(obs) # update the hidden state of eval network

        # action_eval = Q.argmax(dim=2).view(1, 1, 1)
        # with torch.no_grad():
        #     Q_eval = self.ddqn_target(obs)

        if deterministic:
            action_taken = Q.argmax(dim=2).view(1, 1, 1)
            return Q.gather(dim=2, index=action_taken), action_taken.item()
        else:
            if np.random.rand() < self.alpha:
                action_taken = torch.randint(0, self.action_dim, (1, 1, 1))
                return Q.gather(dim=2, index=action_taken),  action_taken.item()
            else:
                pros = torch.softmax(self.beta * Q, dim=-1)
                action_taken = torch.distributions.Categorical(pros.squeeze()).sample().view(1, 1, 1)
                return Q.gather(dim=2, index=action_taken), action_taken.item()
    def eval_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).view(1, 1, self.obs_dim)

        with torch.no_grad():
            Q = self.ddqn(obs, is_update=False)
            action = Q.argmax(dim=2).view(1, 1, 1)
            Q_eval = self.ddqn_target.get_eval_q(obs, self.ddqn.get_hidden_state())

            return Q_eval.gather(dim=2, index=action)

    def train_mini_batch(self, batchQ):
        actual_Qs = torch.stack([a for (a, t) in batchQ])
        traget_Qs = torch.stack([t for (a, t) in batchQ])
        q_loss = F.mse_loss(actual_Qs, traget_Qs).mean()
        #print(q_loss)

        self.ddqn_optimizer.zero_grad()
        q_loss.backward()
        self.ddqn_optimizer.step()
        self.ddqn.reset_lstm_state()
        self.ddqn_target.reset_lstm_state()

    # def get_targetQ(self, action, feedback):
    #     _, action_new = self.select_action(action, feedback, deterministic=True)
    #     Q_eval = self.eval_action(action, feedback)
    #     action_new = torch.tensor(action_new).view(1, 1, 1)
    #     return self.gamma * Q_eval.gather(dim=2, index=action_new)

    def deep_copy_ddqn(self):
        self.ddqn_target.load_state_dict(self.ddqn.state_dict())

    def save(self, save_dir):
        torch.save(self.ddqn.state_dict(), save_dir)

    def load(self, load_dir):
        self.ddqn.load_state_dict(torch.load(load_dir, weights_only=True,  map_location=self.device))
        self.ddqn.eval()

    def lr_decay(self, episode, episodes):
        lr = self.lr - (self.lr * (episode / float(episodes)))
        for param_group in self.ddqn_optimizer.param_groups:
            param_group['lr'] = lr

    def update_alpha(self, episode, episodes):
        self.alpha = self.alpha_max - (self.alpha_max * (episode / float(episodes)))

    def update_beta(self, episode, episodes):
        self.beta = self.beta_min + ((self.beta_max - self.beta_min) * (episode / float(episodes)))

    def lstm_ret(self):
        self.ddqn.reset_lstm_state()
        self.ddqn_target.reset_lstm_state()