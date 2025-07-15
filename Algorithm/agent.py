import numpy
import numpy as np
import torch as T
from Algorithm.networks import ContinuousActorNetwork, ContinuousCriticNetwork
import copy
from torch.utils.tensorboard import SummaryWriter
import os
import math

class Agent:
    def __init__(self, actor_dims, critic_dims,
                 n_actions, agent_idx, agent_name, file,args=None):
        self.args = args
        self.gamma = args.gamma
        self.policy_clip = 0.2
        self.n_epochs = args.n_epochs
        self.gae_lambda = 0.95
        self.agent_idx = agent_idx
        self.agent_name = agent_name

        self.actor = ContinuousActorNetwork(n_actions, actor_dims, self.args.actor_lr, args=args)
        self.critic = ContinuousCriticNetwork(critic_dims, self.args.critic_lr, args=args)

        self.n_actions = n_actions
        self.device = args.device

        # if self.agent_name[0]=='r':
        #     self.writer = SummaryWriter(os.path.join(str(file.tensor_draw_path_red), self.agent_name))
        # elif self.agent_name[0]=='b':
        #     self.writer = SummaryWriter(os.path.join(str(file.tensor_draw_path_blue), self.agent_name))
        self.update_times = 0

        self.actor_obs_hoder = np.zeros((args.buffer_capacity, actor_dims), dtype=np.float32)
        self.state_hoder = np.zeros((args.buffer_capacity, critic_dims), dtype=np.float32)
        self.a_hoder = np.zeros((args.buffer_capacity, n_actions), dtype=np.float32)
        self.r_hoder = np.zeros((args.buffer_capacity, 1), dtype=np.float32)
        self.actor_obs_next_hoder = np.zeros((args.buffer_capacity, actor_dims), dtype=np.float32)
        self.state_next_hoder = np.zeros((args.buffer_capacity, critic_dims), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((args.buffer_capacity, n_actions), dtype=np.float32)
        self.done_hoder = np.zeros((args.buffer_capacity, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((args.buffer_capacity, 1), dtype=np.bool_)

    def choose_action(self, observation: numpy.ndarray):
        with T.no_grad():
            state = T.FloatTensor(observation.reshape(1, -1)).to(self.device)
            dist = self.actor.get_dist(state)
            a = dist.sample()
            a = T.clamp(a, 0, 1)
            logprob_a = dist.log_prob(a).cpu().numpy().flatten()
            return a.cpu().numpy()[0], logprob_a  # both are in shape (adim, 0)

    def put_data(self, s, a, r, s_next, logprob_a, done, dw,state, state_, idx):
        self.actor_obs_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.actor_obs_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.state_hoder[idx] = state
        self.state_next_hoder[idx] = state_
        self.dw_hoder[idx] = dw

    def learn(self):
        self.args.entropy_coef *= self.args.entropy_coef_decay
        device = self.critic.device

        '''Prepare PyTorch data from Numpy data'''
        actor_obs = T.from_numpy(self.actor_obs_hoder).to(device)
        a = T.from_numpy(self.a_hoder).to(device)
        r = T.from_numpy(self.r_hoder).to(device)
        actor_obs_next = T.from_numpy(self.actor_obs_next_hoder).to(device)
        logprob_a = T.from_numpy(self.logprob_a_hoder).to(device)
        state = T.from_numpy(self.state_hoder).to(device)
        state_next = T.from_numpy(self.state_next_hoder).to(device)
        done = T.from_numpy(self.done_hoder).to(device)  # 任务成功或者失败以及到达一局中允许的最大时间
        dw = T.from_numpy(self.dw_hoder).to(device)  # 任务成功或者中途死亡
        # dw: dead&win; tr: truncated
        # done = (dw or tr)
        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with T.no_grad():
            vs = self.critic(state)
            vs_ = self.critic(state_next)

            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()  # 优势
            adv = [0]

            '''done for GAE'''
            # for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
            #     advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (~mask)
            #     adv.append(advantage)
            for dlt, dw_mask, done_mask in zip(deltas[::-1], dw.cpu().flatten().numpy()[::-1],
                                               done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (~done_mask) * (~dw_mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = T.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(actor_obs.shape[0] / self.args.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(actor_obs.shape[0] / self.args.c_optim_batch_size))
        for i in range(self.n_epochs):

            perm = np.arange(actor_obs.shape[0])
            np.random.shuffle(perm)
            perm = T.LongTensor(perm).to(device)
            actor_obs, a, td_target, adv, logprob_a = \
                actor_obs[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

            '''update the actor'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.args.a_optim_batch_size, min((i + 1) * self.args.a_optim_batch_size, actor_obs.shape[0]))
                distribution = self.actor.get_dist(actor_obs[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)

                logprob_a_now = distribution.log_prob(a[index])

                ratio = T.exp(logprob_a_now.sum(1, keepdim=True) - logprob_a[index].sum(1, keepdim=True))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv[index]
                a_loss = -T.min(surr1, surr2) - self.args.entropy_coef * dist_entropy

                self.actor.optimizer.zero_grad()
                a_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

            '''update the critic'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.args.c_optim_batch_size, min((i + 1) * self.args.c_optim_batch_size, actor_obs.shape[0]))
                c_loss = (self.critic(state[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.args.l2_reg

                self.critic.optimizer.zero_grad()
                c_loss.backward()
                self.critic.optimizer.step()

            # self.writer.add_scalars('loss', {'weighted_probs': -surr1.mean().item(),
            #                                  'weighted_clipped_probs': -weighted_clipped_probs.mean().item(),
            #                                  'entropy': -entropy.mean().item(),
            #                                  'actor_loss': actor_loss.mean().item(),
            #                                  'critic_loss': critic_loss.item(),
            #                                  'advantage': adv.mean().item()},
            #                         self.update_times
            #                         )
