import gym
from gym import spaces
import numpy as np

class TrafficGenerator(gym.Env):
    
    #doesnt really workj
    def getObs(self):
        # Loop over the packet types
        for i in range(len(self.mean_delay)):

            # Get the current queue for the packet type
            current_queue = self.queues[i]

            # Calculate the length and average waiting time of the current queue
            current_length = len(current_queue)
            current_waiting_time = np.average(current_queue) if current_length > 0 else 0.0
            #print(np.average(current_queue), current_length, current_waiting_time )
            
            
           
            observation.append([current_length, current_waiting_time])

       
        observation = np.array(observation)
        return observation
    
    def __init__(self):
        
        self.action_space = spaces.Discrete(3)
        
        # Define the observation space (number of packets in each queue and their waiting time)
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,2), dtype=int)
        
        # pancket info (DataType, arrival_rate, mean_delay )
        #self.packetInfo = [[0, 0.3, 6],[1, 0.25, 4],[2, 0.4, float('inf')]]
        
        self.dataType = [0, 1, 2]
        self.arrival_rate = [0.6, 0.5, 0.4]
        self.mean_delay_req = [6, 4, float('inf')]
        self.packetInfo = [[elem1, elem2, elem3] for elem1, elem2, elem3 in zip(self.dataType, self.arrival_rate, self.mean_delay_req)]
        self.curr_mean_delay_best_effort = 0
        self.packet = 1
        self.timeslot = 1
        self.totaltime = self.timeslot
        
        # Initialize the queues
        self.queues = [[], [], []]
      
    
    def step(self, action):  
        self.totaltime += self.timeslot
        print(self.totaltime)
        #packet generator
        for i in range(len(self.packetInfo)):
            for sublist in self.packetInfo:
                if sublist[0] == i and np.random.uniform() < sublist[1]:
                    self.queues[i].append(self.packet)
                    #print("Appending to queue", i)
        
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.queues[i][j] -= self.timeslot
                
        
        # services the queues
        #for i in range(len(self.queues)):
            #print("queue ", i, ": has a length of", len(self.queues[i]))
        #print("ACTION queue ", action, ": has a length of", len(self.queues[action]))
        if action < 3 and len(self.queues[action]) > 0:
            self.queues[action].pop(0)
            for i in range(len(self.queues[action])):
                self.queues[action][i] -= 1
        

        
        #TODO rewardoptions?
        reward = 1
        
        observation = []
        for i in range(len(self.mean_delay_req)):

            # Get the current queue for the packet type
            current_queue = self.queues[i]

            # Calculate the length and average waiting time of the current queue
            current_length = len(current_queue)
            current_waiting_time = np.average(current_queue) if current_length > 0 else 0.0
            #print(np.average(current_queue), current_length, current_waiting_time )
            
    
            observation.append([current_length, current_waiting_time])
            if i == 2:
                #reward option
                if self.curr_mean_delay_best_effort == 0:
                    curr_mean_delay_best_effort = current_waiting_time
                if self.curr_mean_delay_best_effort < current_waiting_time:
                    self.curr_mean_delay_best_effort = current_waiting_time
                    reward = 1
                elif self.curr_mean_delay_best_effort > current_waiting_time:
                    self.curr_mean_delay_best_effort = current_waiting_time
                    reward = -1
                else:
                    reward = 1
        
                #print(self.curr_mean_delay_best_effort)
                #print(current_waiting_time)
        print(f"observation{observation}")
        if observation[0][1] < -4:
            reward = -10
        elif observation[1][1] < -6:
            reward = -10
        observation = np.array(observation)

         # Loop over the packet types
        """
        observation = []
        for i in range(len(self.mean_delay)):

            # Get the current queue for the packet type
            current_queue = self.queues[i]

            # Calculate the length and average waiting time of the current queue
            current_length = len(current_queue)
            current_waiting_time = np.average(current_queue) if current_length > 0 else 0.0
            #print(np.average(current_queue), current_length, current_waiting_time )
            
            
    
            observation.append([current_length, current_waiting_time])

        observation = np.array(observation)
        """
        info = {"mean_delay_0": self.mean_delay_req[0],
        "mean_delay_1": self.mean_delay_req[1],
        "mean_delay_2": self.mean_delay_req[2],
        "arrival_rate_0": self.arrival_rate[0],
        "arrival_rate_1": self.arrival_rate[1],
        "arrival_rate_2": self.arrival_rate[2]}
        done = len(self.queues[0]) + len(self.queues[1]) + len(self.queues[2]) == 0 or (self.totaltime >= 100)
        return observation, reward, done, info
        
    def reset(self):
        # Reset the queues
        self.queues = [[], [], []]
        return np.array([[len(self.queues[i]), 0.0] for i in range(len(self.mean_delay_req))])
        
    def render(self):
        pass

env = TrafficGenerator()
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

print('State space Low: ', env.observation_space.low)
print('State space High: ', env.observation_space.high)


def discretize_observations(observations, bins):
    """
    Discretize the second column of each observation in a given array.
    
    Args:
        observations (np.ndarray): Array of observations.
        bins (int): The number of bins to discretize into.
    
    Returns:
        np.ndarray: The discretized observations.
    """
    discretized_observations = np.copy(observations)
    for obs in discretized_observations:
        obs[:, 1] = np.digitize(obs[:, 1], np.linspace(obs[:, 1].min(), obs[:, 1].max(), bins))
    
    return discretized_observations

# Sample observations
observations = np.array([env.observation_space.sample() for i in range(10)])


# Discretize observations
discretized_observations = discretize_observations(observations, bins=100)

print(discretized_observations)

obs = env.reset()
done = False
x = 0
while x < 100:
    print("~~~~~~~~~")
    print(obs[:, 0])
    
    if obs[:, 0][0] > 0 or obs[:, 0][1] > 0:
        action = np.argmax(obs[:2, 0])
    else:
        action = 2 
    
    print("chosen", action)
    obs,reward, done, info = env.step(action)
    print(obs)
    # print(np.array(obs))
    # print(discretize_observations(np.array(obs), 100))
    print("reward", reward)
    
    x = x +1