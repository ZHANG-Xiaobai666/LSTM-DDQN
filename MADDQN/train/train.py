import numpy as np
import time
import argparse
import os

from env.env_core import EnvCore
from run.env_train_runner import EnvRunner
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--max_train_times', type=int, default=int(1e4), help='maximum training times')
parser.add_argument('--episode_num', type=int, default=int(16), help='num of episodes per train')
parser.add_argument('--episode_length', type=int, default=int(50), help='number of time slots per episode')
parser.add_argument('--deep_copy_per_episode', type=int, default=int(5), help='num of train times to deepcopy eva DQN')


parser.add_argument('--alpha', type=float, default=0.05, help='exploration probability')
parser.add_argument('--beta_min', type=float, default=1, help='initial temperature')
parser.add_argument('--beta_max', type=float, default=20, help='final temperature')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')

parser.add_argument('--action_dim', type=int, default=int(2), help='action space {0,..., num_channel}')
parser.add_argument('--obs_dim', type=int, default=int(4), help='obs space 2*num_channel+2')
parser.add_argument('--net_width', type=int, default=int(100), help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')


parser.add_argument('--reward_type', type=str, default="sum_rate", help='sum_rate / competitive / proportional')



args = parser.parse_args()

def make_env(args, num_agent, num_channel, arr_pro):
    return EnvCore(args, num_agent, num_channel, arr_pro)

def main():

    num_agent = int(3)
    num_channel = int(2)
    arr_pro = 1

    args.num_agent = num_agent
    args.num_channel = num_channel
    args.action_dim = int(num_channel+1)
    args.obs_dim = int(2*num_channel+2)

    root_path = os.path.dirname(os.getcwd())
    args.save_dir = os.path.join(root_path, "results")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.reward_type)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, "N" + str(args.num_agent) + "K" + str(args.num_channel))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    env = make_env(args, num_agent, num_channel, arr_pro)
    runner = EnvRunner(args, env)
    runner.run()




if __name__ == "__main__":
    main()
