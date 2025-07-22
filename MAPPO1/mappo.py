from agent import Agent
from memory import PPOMemory
import numpy as np


class MAPPO:
    def __init__(self, dim_info, args, agents_name):

        self.agent_name = agents_name
        actor_dims = {agent_id:dim_info[agent_id][0] for agent_id in agents_name}
        n_actions = {agent_id:dim_info[agent_id][1] for agent_id in agents_name}
        critic_dims = list(dim_info.values())[0][2]

        self.memory = PPOMemory(agents=agents_name,
                                critic_dims=critic_dims, actor_dims=actor_dims,
                                n_actions=n_actions, args=args)
        self.agents = []
        for agent_idx, agent in enumerate(agents_name):
            self.agents.append(Agent(actor_dims=actor_dims[agent], critic_dims=critic_dims,
                 n_actions=n_actions[agent], agent_idx=agent_idx, agent_name=agent, args=args))

    def store_memory(self,observation, state, action, prob, reward, observation_, state_, mask):
        self.memory.store_memory(observation, state, action, prob, reward, observation_, state_, mask)


    def choose_action(self, raw_obs, evalute=False):
        debug = False
        actions = {}
        probs = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            if not debug:
                action, prob = agent.choose_action(raw_obs[agent_id],evalute=evalute)
                # action, prob = self.action_mask(action, prob, DW[agent_id])
            else:
                action = np.zeros(3)+0.5
                prob = np.zeros(3)
            actions[agent_id] = action
            probs[agent_id] = prob
        return actions, probs

    def action_mask(self,action, action_prob, is_dw):
        if is_dw!=0: # 若卫星死亡，则动作为0
            return np.zeros_like(action,dtype=float)+0.5, np.zeros_like(action_prob,dtype=float)-1e10
        return action,action_prob

    def learn(self):
        for agent in self.agents:
            agent.learn(self.memory)

    def clear_memory(self):
        self.memory.clear_memory()
