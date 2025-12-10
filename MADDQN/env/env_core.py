import random
import numpy as np

class EnvCore:

    def __init__(self, args, num_agent, num_channel, arr_pro):
        self.episode_length = args.episode_length # Number of time slot per train/execute
        self.num_agent = num_agent       # Number of nodes
        self.arr_pro = arr_pro           # aggregate arrival probability
        self.num_channel = num_channel   # Number of channels

        self.channel_state = [0 for _ in range(self.num_channel)]  # 0:idle, 1:successful 2:collision;
        self.nodes_feedback = [0 for _ in range(self.num_agent)]  # 1 if ACK 0 Otherwise

        self.queue_length = [0 for _ in range(self.num_agent)]   # backlogged packets
        self.generate_time = [[] for _ in range(self.num_agent)] # For derivation of delay

        self.num_successful_packets = [0 for i in range(self.num_agent)] # successful transmitted packets
        self.mean_delay = [0 for _ in range(self.num_agent)]             # mean queueing delay of data packets
        self.throughput = 0                                             # network throughput sum(
                                                                        # num_successful_packets)/episode_length)

    def reset(self):
        self.channel_state = [0 for _ in range(self.num_channel)]  # 0:idle, 1:successful 2:collision;

        self.queue_length = [0 for _ in range(self.num_agent)]    #  backlogged packets
        self.generate_time = [[] for _ in range(self.num_agent)]  #  queue buffer

        self.num_successful_packets = [0 for _ in range(self.num_agent)]
        self.mean_delay = [0 for _ in range(self.num_agent)]
        self.throughput = 0

    def step(self, actions, time):
        self.channel_state = [0 for _ in range(self.num_channel)]  # reset channel state
        actions = np.array(actions)

        random_numbers = np.array([random.random() for _ in range(self.num_agent)])
        index = np.where(random_numbers < self.arr_pro)[0]  # packet generation
        for idx in index:
            self.generate_time[idx].append(time)
            self.queue_length[idx] += 1

        index = np.where(actions == 0)[0]       # feedback must be 0 if it does not transmit
        for idx in index:
            self.nodes_feedback[idx] = 0

        for idx in range(self.num_channel):      # get feedback for each channel
            x = np.where(actions == idx + 1)[0]           # actions N X 1
            if len(x) == 1:                      # successful if only one node transmits
                self.channel_state[idx] = 1
                self.nodes_feedback[x[0]] = 1
                self.num_successful_packets[x[0]] += 1
                self.mean_delay[x[0]] += (self.generate_time[x[0]][0] - time + 1)/self.num_successful_packets[x[0]]
                del self.generate_time[x[0]][0]
                self.queue_length[x[0]] -= 1
            elif len(x) > 1:                   # collision
                self.channel_state[idx] = 2
                for node in x:
                    self.nodes_feedback[node] = 0
            else:                            # idle
                self.channel_state[idx] = 0

        if time == self.episode_length-1:
            self.throughput = sum(self.num_successful_packets)/self.episode_length

        return self.nodes_feedback

    def get_throughput(self):
        return self.throughput

    def get_sum_success(self):
        return self.num_successful_packets

    def get_short_term_throughput(self, time):
        return sum(self.num_successful_packets)/(time + 1)

