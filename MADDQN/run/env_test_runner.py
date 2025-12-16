
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

        self.load_model()
        start = time.time()
        for train_time in range(self.max_train_times):

            # if train_time % self.deep_copy_per_episode == 0:
            #     self.update_eval_ddqn()
            #     self.save_model()

            for episode in range(self.episodes):

                self.env.reset()
                self.lstm_state_reset()
                obs = self.env.get_obs()



                for step in range(1, self.episode_length):

                    actions_taken = self.collect(obs)

                    obs, _ = self.env.step(actions_taken, step)

                    #elf.push(pre_Q_values, Q_values_eval, pre_nodes_feedbacks) # push actual Q
                        # and target Q
                    # into buffer

                # pre_Q_values = Q_values
                # pre_nodes_feedbacks = nodes_feedbacks
                # _, Q_values_eval, _ = self.collect(pre_actions, pre_nodes_feedbacks)
                # self.push(pre_Q_values, Q_values_eval, pre_nodes_feedbacks)
            #self.train()
            #self.buffers = [[] for _ in range(self.num_agent)]
            #self.update_par(train_time + 1, self.max_train_times)


            print(f"Iteration: {train_time+1} / {self.max_train_times}")
            print(f"Throughput {self.env.get_throughput()}")




        end = time.time()

    def collect(self, obs, deterministic=False):
        actions_taken = []
        if deterministic:
            for agent in range(int(self.num_agent)):
                _, action_taken = self.agents[agent].select_action(obs[agent], deterministic=True)
                actions_taken.append(action_taken)
        else:
            for agent in range(int(self.num_agent)):
                _, action_taken = self.agents[agent].select_action(obs[agent])
                actions_taken.append(action_taken)
        return actions_taken


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
