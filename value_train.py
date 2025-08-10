# 系统包
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import torch
import copy
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm

# 自建包
# from maddpg.Agent import Actor_Network
from Env.scenario import scenario
import Env.parameter as parameter
from Tool.init import create_net, get_reset_env


class State_Net(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=128, args=None):
        super(State_Net, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.01)


    def forward(self, x_in):
        x2 = torch.relu(self.fc1(x_in))
        x3 = torch.relu(self.fc2(x2))
        x4 = torch.relu(self.fc3(x3))
        x_out = self.fc4(x4)

        return x_out



class Buffer:
    def __init__(self, buffer_size, state_dim, args):
        self.args = args
        self._index = 0
        self._size = buffer_size
        self.len = 0

        self.s = np.empty([self._size, state_dim])
        self.s_ = np.empty([self._size, state_dim])
        self.r = np.empty(self._size)
        self.done = np.empty(self._size)

    def add(self, state, state_next, reward, done):
        if self.len>=self._size:
            self.len = self._size
        else:
            self.len+=1

        self.s[self._index] = state
        self.s_[self._index] = state_next
        self.r[self._index] = reward
        self.done[self._index] = done
        self._index += 1
        self._index = self._index % self._size

    def sample(self,batch_size):
        if batch_size > self.len:
            index = np.arange(0, self.len)
        else:
            index = np.random.randint(0, self.len, batch_size)

        sample_s = torch.from_numpy(self.s[index]).float().to(self.args.device)
        sample_s_ = torch.from_numpy(self.s_[index]).float().to(self.args.device)
        sample_r = torch.from_numpy(self.r[index]).float().to(self.args.device)
        sample_done = torch.from_numpy(self.done[index]).float().to(self.args.device)

        return sample_s, sample_s_, sample_r, sample_done


# 收集数据->训练
class state_value_train:
    def __init__(self, args, obs_dim):

        self.args = args
        self.state_net = State_Net(in_dim=obs_dim, out_dim=1, args=args).to(args.device)
        self.target_states = deepcopy(self.state_net).to(args.device)
        self.buffer = Buffer(buffer_size=args.value_batch_size, state_dim=obs_dim, args=args)
        self.state_optimizer = Adam(self.state_net.parameters(), lr=self.args.value_lr)



    def train(self):
        s, s_, r, done = self.buffer.sample(self.args.value_batch_size)
        Vs = self.state_net(s).squeeze(1)

        next_target_state_value = self.target_states(s_).squeeze(1)
        target_value = r + self.args.value_gamma * next_target_state_value * (1 - done)

        loss = F.mse_loss(Vs, target_value.detach(), reduction='mean')
        # 记录loss值
        with open(r"E:\code_list\red_battle_blue\value_net\loss\loss.txt", "a") as f:
            f.write(str(loss.item()) + "\n")
        self.update_state_value(loss)
        self.update_target_state_value()


    # 优化器反向传播
    def update_state_value(self, loss):
        self.state_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.state_net.parameters(), 0.5)
        self.state_optimizer.step()

    # 更新target_value
    def update_target_state_value(self):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(self.args.value_tau * from_p.data + (1.0 - self.args.tau) * to_p.data)
        soft_update(self.state_net, self.target_states)

    # path为保存路径，不包含文件名称
    def save(self, path, episode):
        # 保存actor
        torch.save(self.state_net.state_dict(), os.path.join(path, f'{episode}value.pt'))
        # 保存target_actor
        torch.save(self.target_states.state_dict(),  # actor parameter
            os.path.join(path, f'{episode}target_value.pt')
        )

    @classmethod
    def load(cls, args, obs_dim):
        new_class = cls(args, obs_dim)
        new_class.state_net.load_state_dict(torch.load(args.value_net_path))
        return new_class

    def state_value_eva(self, obs:torch.tensor):
        obs = torch.from_numpy(obs).float().to(self.args.device)
        return self.state_net(obs).squeeze(0).detach().cpu().numpy()




if __name__ == '__main__':
    args = parameter.get_args()
    env, dim_info_red, dim_info_blue = get_reset_env(red_num=1, blue_num=1, evaluate=False, args=args)

    EP_NUM = 2000
    STEP_NUM = 60
    BATCH_SIZE = 128
    LEARN_INTERVAL = 20
    SAVE_INTERVAL = 500
    red_actor_path = r".\\selected_actor\\" + "red_actor.pt"
    blue_actor_path = r".\\selected_actor\\" + "blue_actor.pt"

    val_net_train = state_value_train(args, dim_info_red["r0"][0])
    select_act = create_net(args=args, red_path=args.red_actor_path, dim_inf_red=dim_info_red,
                            blue_path=blue_actor_path, dim_inf_blue=dim_info_blue)

    step = 0

    for episode in range(EP_NUM):
        obs_red, obs_blue = env.reset()
        for i in tqdm(range(STEP_NUM),desc=f"episode{episode+1}"):
            step += 1
            obs = copy.deepcopy({**obs_red, **obs_blue})
            # actor做出决策
            act = select_act.act_select(obs)
            assign_red = {"r0":"b0"}
            assign_blue = {"b0":"r0"}

            obs_red_next, obs_blue_next, reward_red, reward_blue, done_red, blue_done = (
                env.step(action=act))
            # 环境得出s, s_,r,done，将其存入state_value的buffer中
            val_net_train.buffer.add(state=obs_red["r0"], state_next=obs_red_next["r0"], reward=reward_red["r0"], done=done_red["r0"])

            obs_red, obs_blue = obs_red_next, obs_blue_next
            if step % LEARN_INTERVAL == 0:
                val_net_train.train()
            if (episode+1) % SAVE_INTERVAL == 0:
                val_net_train.save(path=r".\value_net\model",episode=episode+1)

