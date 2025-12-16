import random
import numpy as np

class EnvCore:

    def __init__(self, args, num_agent, num_channel, arr_pro):
        self.episode_length = args.episode_length # Number of time slot per train/execute
        self.num_agent = num_agent       # Number of nodes
        self.arr_pro = arr_pro           # aggregate arrival probability
        self.num_channel = num_channel   # Number of channels

        actions_taken = np.zeros(self.num_channel+1)  # initial action
        node_feedback = np.zeros(1)  # initial feedback
        self.channel_capacity = np.ones(self.num_channel)   #
        obs_ = np.concatenate((actions_taken, self.channel_capacity, node_feedback))
        self.obs = [obs_ for _ in range(self.num_agent)]

        self.queue_length = [0 for _ in range(self.num_agent)]   # backlogged packets
        #self.generate_time = [[] for _ in range(self.num_agent)] # For derivation of delay

        self.num_successful_packets = [0 for i in range(self.num_agent)] # successful transmitted packets
        #self.mean_delay = [0 for _ in range(self.num_agent)]             # mean queueing delay of data packets
        self.throughput = 0                                             # network throughput sum(
                                                                        # num_successful_packets)/episode_length)

        self.reward_type = args.reward_type

    def reset(self):
        actions_taken = np.zeros(self.num_channel+1)  # initial action
        node_feedback = np.zeros(1)  # initial feedback
        channel_capacity = np.ones(self.num_channel)   #
        obs_ = np.concatenate((actions_taken, channel_capacity, node_feedback))
        self.obs = [obs_ for _ in range(self.num_agent)]

        self.queue_length = [0 for _ in range(self.num_agent)]    #  backlogged packets
        #self.generate_time = [[] for _ in range(self.num_agent)]  #  queue buffer

        self.num_successful_packets = [0 for _ in range(self.num_agent)]
        #self.mean_delay = [0 for _ in range(self.num_agent)]
        self.throughput = 0



    def step(self, actions, time):

        actions = np.array(actions)
        nodes_feedback = np.zeros(self.num_agent)

        """packet generation"""
        random_numbers = np.array([random.random() for _ in range(self.num_agent)])
        index = np.where(random_numbers < self.arr_pro)[0]
        for idx in index:
            #self.generate_time[idx].append(time)
            self.queue_length[idx] += 1

        index = np.where(actions == 0)[0]       # feedback must be 0 if it does not transmit
        for idx in index:
            nodes_feedback[idx] = 0

        for idx in range(self.num_channel):      # get feedback for each channel
            x = np.where(actions == idx + 1)[0]           # actions N X 1
            if len(x) == 1:                      # successful if only one node transmits
                nodes_feedback[x[0]] = 1
                self.num_successful_packets[x[0]] += 1
                #self.mean_delay[x[0]] += (self.generate_time[x[0]][0] - time + 1)/self.num_successful_packets[x[0]]
                #del self.generate_time[x[0]][0]
                self.queue_length[x[0]] -= 1
            elif len(x) > 1:                   # collision
                for node in x:
                    nodes_feedback[node] = 0
            #else:                            # idle

        """calculate rewards"""
        if self.reward_type == "sum_rate":
            ri = sum(nodes_feedback)
            r = [ri for _ in range(self.num_agent)]
        elif self.reward_type == "proportional":
            ri = sum(nodes_feedback * 1/(np.array(self.num_successful_packets) + 1e-20))
            r = [ri for _ in range(self.num_agent)]
        else:                     # self.reward_type == "competitive":
            r = nodes_feedback


        """update obs"""
        for agent in range(self.num_agent):
            action_vector = np.zeros(self.num_channel+1)
            action_vector[actions[agent]] = 1
            self.obs[agent] = np.concatenate((action_vector, self.channel_capacity, [nodes_feedback[agent]]))

        if time == self.episode_length-1:
            self.throughput = sum(self.num_successful_packets)/self.episode_length

        return self.obs, r

    def get_throughput(self):
        return self.throughput

    def get_sum_success(self):
        return self.num_successful_packets

    def get_short_term_throughput(self, time):
        return sum(self.num_successful_packets)/(time + 1)

    def get_obs(self):
        return self.obs