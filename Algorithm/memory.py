import numpy as np
import copy

class PPOMemory:
    def __init__(self, batch_size, buffer_size, agents,
                 critic_dims, actor_dims, n_actions):

        self.states = np.zeros((buffer_size, critic_dims), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.new_states = np.zeros((buffer_size, critic_dims), dtype=np.float32)

        self.actor_states = {agent: np.zeros((buffer_size, actor_dims[agent]), dtype=np.float32) for agent in agents}
        self.actor_new_states = {agent: np.zeros((buffer_size, actor_dims[agent]), dtype=np.float32) for agent in agents}
        self.actions = {agent: np.zeros((buffer_size, n_actions[agent]), dtype=np.float32) for agent in agents}
        self.probs = {agent: np.zeros((buffer_size, n_actions[agent]), dtype=np.float32) for agent in agents}
        self.rewards = {agent: np.zeros(buffer_size, dtype=np.float32) for agent in agents}
        self.dead_or_win = {agent: np.zeros(buffer_size,dtype=np.bool_) for agent in agents}
        self.done = {agent: np.zeros(buffer_size,dtype=np.bool_) for agent in agents}

        self.mem_cntr = 0
        self.n_states = buffer_size
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = len(agents)
        self.agents = agents
        self.batch_size = batch_size

    def recall(self):
        return copy.deepcopy(self.actor_states), \
            copy.deepcopy(self.states), \
            copy.deepcopy(self.actions), \
            copy.deepcopy(self.probs), \
            copy.deepcopy(self.rewards), \
            copy.deepcopy(self.actor_new_states), \
            copy.deepcopy(self.new_states), \
            copy.deepcopy(self.dead_or_win),\
            copy.deepcopy(self.done)

#   将一整个batch分为几块，以块的形式返回
    def generate_batches(self):

        n_batches = int(self.n_states // self.batch_size) # 将整个buffer分成几块
        indices = np.arange(self.n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, raw_obs, raw_obs_, state, state_,
                     action, probs, reward, done, dw):
        index = self.mem_cntr % self.n_states
        self.states[index] = copy.deepcopy(state)
        self.new_states[index] = copy.deepcopy(state_)

        for agent in self.agents:
            self.actions[agent][index] = copy.deepcopy(action[agent])
            self.actor_states[agent][index] = copy.deepcopy(raw_obs[agent])
            self.actor_new_states[agent][index] = copy.deepcopy(raw_obs_[agent])
            self.probs[agent][index] = copy.deepcopy(probs[agent])
            self.rewards[agent][index] = copy.deepcopy(reward[agent])
            self.done[agent][index] = copy.deepcopy(done[agent]) # 任务终止
            self.dead_or_win[agent][index] = copy.deepcopy(dw[agent]) # 任务成功或失败
        self.mem_cntr += 1

    def clear_memory(self):
        self.states = np.zeros((self.n_states, self.critic_dims), dtype=np.float32)
        self.dones = np.zeros((self.n_states), dtype=np.float32)
        self.new_states = np.zeros((self.n_states, self.critic_dims), dtype=np.float32)

        self.actor_states = {agent: np.zeros([self.n_states, self.actor_dims[agent]], dtype=np.float32) for agent in self.agents}
        self.actor_new_states = {agent: np.zeros([self.n_states, self.actor_dims[agent]], dtype=np.float32) for agent in self.agents}
        self.actions = {agent: np.zeros([self.n_states, self.n_actions[agent]], dtype=np.float32) for agent in self.agents}
        self.probs = {agent: np.zeros([self.n_states, self.n_actions[agent]], dtype=np.float32) for agent in self.agents}
        self.rewards = {agent: np.zeros(self.n_states, dtype=np.float32) for agent in self.agents}
        self.dead_or_win = {agent: np.zeros(self.n_states,dtype=np.bool_) for agent in self.agents}
        self.done = {agent: np.zeros(self.n_states,dtype=np.bool_) for agent in self.agents}
