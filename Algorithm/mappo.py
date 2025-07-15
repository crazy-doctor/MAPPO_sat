import numpy as np

from Algorithm.agent import Agent

class MAPPO:
    def __init__(self,agents_name, dim_info,file=None,args=None):

        self.agents_names = agents_name
        # critic的输入维度
        global_obs_dim = list(dim_info.values())[0][2]#?
        # actor的观测输入维度
        obs_dim = {agent_name: dim_info[agent_name][0] for agent_name in self.agents_names}
        # actor动作维度
        act_dim = {agent_name: dim_info[agent_name][1] for agent_name in self.agents_names}
        self.agents = {}

        for agent_idx, agent_name in enumerate(self.agents_names):
            self.agents[agent_name] = Agent(obs_dim[agent_name], global_obs_dim,act_dim[agent_name], agent_idx=agent_idx,
                                            agent_name=agent_name,file=file,args=args)

    def choose_action(self, raw_obs, DW):
        actions = {}
        probs = {}
        for agent_id in self.agents_names:

            action, prob = self.agents[agent_id].choose_action(raw_obs[agent_id])
            action, prob = self.action_mask(action, prob, DW[agent_id])
            actions[agent_id] = action
            probs[agent_id] = prob
        return actions, probs

    def action_mask(self,action, action_prob, is_dw):
        if is_dw!=0: # 若卫星死亡，则动作为0，log_prob为取不到
            return np.zeros_like(action,dtype=float),np.zeros_like(action_prob,dtype=float)-1e10
        return action,action_prob

    def store_memory(self, observation, observation_,
                     action, prob, reward, dw, done,global_obs,global_obs_,idx):

        for agent_name, agent in self.agents.items():
            agent.put_data(observation[agent_name],
                           action[agent_name],
                           reward[agent_name],
                           observation_[agent_name],
                           prob[agent_name],
                           done[agent_name],
                           dw[agent_name],global_obs,global_obs_, idx)

    def learn(self):
        for agent in self.agents.values():
            agent.learn()
