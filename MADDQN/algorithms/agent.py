
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

    def select_action(self, pre_action, pre_feedback, deterministic=False):
        #obs = check(obs).to(**self.tpdv)
        #rnn_states = check(rnn_states).to(**self.tpdv)

        pre_action_vector = torch.zeros((1, self.action_dim))
        pre_action_vector[0][pre_action] = 1 if pre_action > 0 else 0

        channel_capacity = torch.ones((1, int(self.action_dim - 1)))    # [1,...1] for equal channel
        pre_feedback = torch.ones((1, 1)) * pre_feedback
        obs = torch.cat((pre_action_vector, channel_capacity, pre_feedback), dim = 1).view(1, 1, self.obs_dim)
        # obs = pre_action + [1 for _ in range(int((self.obs_dim-2) / 2))] + pre_feedback
        Q = self.ddqn(obs)
        action_eval = Q.argmax(dim=2).view(1, 1, 1)
        with torch.no_grad():
            Q_eval = self.ddqn_target(obs)

        if deterministic:
            #action_taken = Q.argmax(dim=2).view(1, 1, 1)
            action_taken = action_eval
            return Q.gather(dim=2, index=action_taken), Q_eval.gather(dim=2, index=action_eval), action_taken.item()
        else:
            if np.random.rand() < self.alpha:
                action_taken = torch.randint(0, self.action_dim, (1, 1, 1))
                return Q.gather(dim=2, index=action_taken), Q_eval.gather(dim=2, index=action_eval), action_taken.item()
            else:
                pros = torch.softmax(self.beta * Q, dim=-1)
                action_taken = torch.distributions.Categorical(pros.squeeze()).sample().view(1, 1, 1)
                return Q.gather(dim=2, index=action_taken), Q_eval.gather(dim=2, index=action_eval), action_taken.item()

    # def eval_action(self, pre_action, pre_feedback):
    #     pre_action_vector = torch.zeros((1, self.action_dim))
    #     pre_action_vector[0][pre_action] = 1 if pre_action > 0 else 0
    #
    #     channel_capacity = torch.ones((1, int(self.action_dim - 1)))    # [1,...1] for equal channel
    #     pre_feedback = torch.ones((1, 1)) * pre_feedback
    #     obs = torch.cat((pre_action_vector, channel_capacity, pre_feedback), dim = 1).view(1, 1, self.obs_dim)
    #     with torch.no_grad():
    #         Q = self.ddqn_target(obs)
    #         return Q

    def train_mini_batch(self, batchQ):
        actual_Qs = torch.stack([a for (a, t) in batchQ])
        traget_Qs = torch.stack([t for (a, t) in batchQ])
        q_loss = F.mse_loss(actual_Qs, traget_Qs)
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
        self.ddqn_target = copy.deepcopy(self.ddqn)

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