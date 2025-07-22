from agent import Agent
from memory import PPOMemory
import numpy as np
import os
import torch


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

    def save(self, episode, dir):
        result_num_dir = os.path.join(dir, f'{episode}th_episode')
        if not os.path.exists(result_num_dir):
            os.makedirs(result_num_dir)

        torch.save(
            {agent.agent_name: agent.actor.state_dict() for agent in self.agents},  # actor parameter
            os.path.join(result_num_dir, f'ep_{episode}_actor.pt')
        )

        torch.save(
            {agent.agent_name: agent.critic.state_dict() for agent in self.agents},  # actor parameter
            os.path.join(result_num_dir, f'ep_{episode}_critic.pt')
        )

    @classmethod
    # dir : 文件所在的文件夹的上一级，存储着不同episode的信息
    # def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, args):
    def load(cls, dim_info, args, agents_name, load_dir, episode_num):
        instance = cls(dim_info, args, agents_name)
        load_dir = os.path.join(load_dir, f'{episode_num}th_episode')
        file_actor = f'ep_{episode_num}_actor.pt'
        file_critic = f'ep_{episode_num}_critic.pt'
        # 加载actor
        data_actor = torch.load(load_dir+ "\\"+ file_actor)
        for agent in instance.agents:
            agent.actor.load_state_dict(data_actor[agent.agent_name])
        # 加载critic
        data_critic = torch.load(load_dir+ "\\"+ file_critic)
        for agent in instance.agents:
            agent.critic.load_state_dict(data_critic[agent.agent_name])

        return instance