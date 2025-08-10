import numpy as np


class PPOMemory:
    def __init__(self, agents, critic_dims, actor_dims, n_actions, args):
        self.args = args
        buffer_capacity = args.buffer_capacity
        self.states = np.zeros((buffer_capacity, critic_dims), dtype=np.float32)
        self.new_states = np.zeros((buffer_capacity, critic_dims), dtype=np.float32)
        self.dones = np.zeros(buffer_capacity, dtype=np.float32)

        self.actor_states = {a: np.zeros((buffer_capacity, actor_dims[a])) for a in agents}
        self.actor_new_states = {a: np.zeros((buffer_capacity, actor_dims[a])) for a in agents}
        self.actions = {a: np.zeros((buffer_capacity, n_actions[a])) for a in agents}
        self.probs = {a: np.zeros((buffer_capacity, n_actions[a])) for a in agents}
        self.rewards = {a: np.zeros(buffer_capacity) for a in agents}

        self.mem_cntr = 0
        self.buffer_capacity = buffer_capacity
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.agents = agents
        self.n_agents = len(agents)
        self.batch_size = args.batch_size

    def recall(self):
        return self.actor_states, \
            self.states, \
            self.actions, \
            self.probs, \
            self.rewards, \
            self.actor_new_states, \
            self.new_states, \
            self.dones

    def generate_batches(self):

        n_batches = int(self.buffer_capacity // self.batch_size)
        indices = np.arange(self.buffer_capacity, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, raw_obs:dict, state:list, action:dict, probs:dict, reward:dict,
                     raw_obs_:dict, state_:list, done:float):
        index = self.mem_cntr % self.buffer_capacity

        self.states[index] = state
        self.new_states[index] = state_
        self.dones[index] = done

        for agent in self.agents:
            self.actions[agent][index] = action[agent]
            self.actor_states[agent][index] = raw_obs[agent]
            self.actor_new_states[agent][index] = raw_obs_[agent]
            self.probs[agent][index] = probs[agent]
            self.rewards[agent][index] = reward[agent]
        self.mem_cntr += 1

    def clear_memory(self):
        self.states = np.zeros((self.buffer_capacity, self.critic_dims),
                               dtype=np.float32)
        self.dones = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.new_states = np.zeros((self.buffer_capacity,
                                    self.critic_dims), dtype=np.float32)

        self.actor_states = {a: np.zeros(
            (self.buffer_capacity, self.actor_dims[a]))
                             for a in self.agents}
        self.actor_new_states = {a: np.zeros(
            (self.buffer_capacity, self.actor_dims[a]))
                                 for a in self.agents}
        self.actions = {a: np.zeros(
            (self.buffer_capacity, self.n_actions[a]))
                        for a in self.agents}
        self.probs = {a: np.zeros(
            (self.buffer_capacity, self.n_actions[a]))
                      for a in self.agents}
        self.rewards = {a: np.zeros(self.buffer_capacity) for a in self.agents}

