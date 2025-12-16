
import time
import numpy as np
import torch
from algorithms.agent import Agent
import os
import copy

class EnvRunner():
    def __init__(self, args, env):
        self.max_train_times = args.max_train_times
        self.deep_copy_per_episode = args.deep_copy_per_episode
        self.episodes = args.episode_num
        self.episode_length = args.episode_length
        self.reward_type = args.reward_type
        self.env = env

        self.num_agent = args.num_agent
        self.num_channel = args.num_channel

        self.save_dir = args.save_dir

        self.agents = []
        for _ in range(self.num_agent):
            agent = Agent(args)
            self.agents.append(agent)

        self.buffers = [[] for _ in range(self.num_agent)]

    def run(self):

        start = time.time()
        for train_time in range(self.max_train_times):

            if train_time % self.deep_copy_per_episode == 0:
                self.update_eval_ddqn()
                self.save_model()


            for episode in range(self.episodes):

                self.env.reset()
                self.lstm_state_reset()
                obs_next = self.env.get_obs()
                obs = copy.deepcopy(obs_next)

                for step in range(0, self.episode_length):
                    Q_values, actions_taken = self.collect(obs)

                    obs_next, rewards = self.env.step(actions_taken, step)
                    self.push(Q_values, obs_next, rewards, step)   # push actual Q  and target Q into buffer
                    obs = copy.deepcopy(obs_next)

            self.train()
            self.buffers = [[] for _ in range(self.num_agent)]
            self.update_par(train_time + 1, self.max_train_times)


            print(f"Iteration: {train_time+1} / {self.max_train_times}")
            print(f"Throughput {self.env.get_throughput()}")




        end = time.time()
    def collect(self, obs):
        actions_taken = []
        Q_values = []
        for agent in range(int(self.num_agent)):
            Q, action_taken = self.agents[agent].select_action(obs[agent])
            actions_taken.append(action_taken)
            Q_values.append(Q)
        return Q_values, actions_taken

    def push(self, Q_values, obs_next, rewards, step):
        if step == self.episode_length-1:
            for agent in range(self.num_agent):
                target_Q = torch.tensor(rewards[agent], dtype=torch.float32).view(1, 1, 1)
                self.buffers[agent].append((Q_values[agent], target_Q))
        else:
            for agent in range(self.num_agent):
                target_Q = rewards[agent] + self.agents[agent].eval_action(obs_next[agent])
                self.buffers[agent].append((Q_values[agent], target_Q))
    def train(self):
        for agent in range(self.num_agent):
            self.agents[agent].train_mini_batch(self.buffers[agent])


    def update_eval_ddqn(self):
        for agent in range(self.num_agent):
            self.agents[agent].deep_copy_ddqn()

    def save_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].save(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))

    def load_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].load(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))

    def update_par(self, episode, episodes):
        for agent in range(self.num_agent):
            self.agents[agent].lr_decay(episode, episodes)
            self.agents[agent].update_alpha(episode, episodes)
            self.agents[agent].update_beta(episode, episodes)

    def lstm_state_reset(self):
        for agent in range(self.num_agent):
            self.agents[agent].lstm_ret()