
import time
import numpy as np
import torch
from algorithms.agent import Agent
import os

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
                actions_taken = [0 for _ in range(self.num_agent)]                        # initial action
                nodes_feedbacks = [0 for _ in range(self.num_agent)]                     # initial feedback

                Q_values, _, actions_taken = self.collect(actions_taken, nodes_feedbacks)
                nodes_feedbacks = self.env.step(actions_taken, 0)

                for step in range(1, self.episode_length):
                    pre_actions = actions_taken
                    pre_nodes_feedbacks = nodes_feedbacks
                    pre_Q_values = Q_values

                    Q_values, Q_values_eval, actions_taken = self.collect(pre_actions, pre_nodes_feedbacks)

                    nodes_feedbacks = self.env.step(actions_taken, step)

                    self.push(pre_Q_values, Q_values_eval, pre_nodes_feedbacks) # push actual Q
                        # and target Q
                    # into buffer

                pre_Q_values = Q_values
                pre_nodes_feedbacks = nodes_feedbacks
                _, Q_values_eval, _ = self.collect(pre_actions, pre_nodes_feedbacks)
                self.push(pre_Q_values, Q_values_eval, pre_nodes_feedbacks)
            self.train()
            self.buffers = [[] for _ in range(self.num_agent)]
            self.update_par(train_time + 1, self.max_train_times)


            print(f"Iteration: {train_time+1} / {self.max_train_times}")
            print(f"Throughput {self.env.get_throughput()}")




        end = time.time()
    def collect(self, pre_actions, pre_nodes_feedback):
        actions_taken = []
        Q_values = []
        Q_values_eval = []
        for agent in range(int(self.num_agent)):
            Q, Q_eval, action_taken = self.agents[agent].select_action(pre_actions[agent], pre_nodes_feedback[agent])

            actions_taken.append(action_taken)
            Q_values.append(Q)
            Q_values_eval.append(Q_eval)
        return Q_values, Q_values_eval, actions_taken

    def push(self, act_Q_values, eval_Q_values, nodes_feedbacks):
        
        if self.reward_type == "sum_rate":
            # if step == self.episode_length - 1:
            #     ri = sum(self.env.get_sum_success())
            # else:
            #     ri = 0
            ri = sum(self.env.get_sum_success())
            r = [ri for _ in range(self.num_agent)]
        elif self.reward_type == "proportional":
            ri = np.dot(np.array(nodes_feedbacks), 1/(np.array(self.env.get_sum_success()) + 1e-8))
            r = [ri for _ in range(self.num_agent)]
        else:                     # self.reward_type == "competitive":
            r = nodes_feedbacks
        #ri = sum(np.array(nodes_feedbacks))
        #r = [ri for _ in range(self.num_agent)]
        for agent in range(self.num_agent):
            target_Q = r[agent] + eval_Q_values[agent]
            self.buffers[agent].append((act_Q_values[agent], target_Q))
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