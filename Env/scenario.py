# 系统包
import numpy as np
import copy
from abc import ABC, abstractclassmethod
# 自建包
from Tool.astro_tool import Tool
## 观测、奖励、结束方案
from Env.done_judge import done_judge # 环境中卫星的任务完成、死亡情况
from Env import reward

# 有don_judge
class info_generate(ABC):
    @abstractclassmethod
    def red_obs(self, red_name, blue_name, inf):
        pass

    @abstractclassmethod
    def red_reward(self, red_name, blue_name, inf):
        pass

    @abstractclassmethod
    def red_done(self, red_name, blue_name, inf):
        pass

    @abstractclassmethod
    def blue_obs(self, red_name, blue_name, inf):
        pass

    @abstractclassmethod
    def blue_reward(self, red_name, blue_name, inf):
        pass

    @abstractclassmethod
    def blue_done(self, red_name, blue_name, inf):
        pass

class scenario(Tool):
    def __init__(self, args = None,evalute=False):
        self.args = args
        self.data = None  # 仿真环境返回的json数据

        self.red_sat = ["r"+str(i) for i in range(args.red_num)]
        self.blue_sat = ["b"+str(i) for i in range(args.blue_num)]
        self.ref_sat = ["main_sat"]
        self.satellite = self.red_sat + self.blue_sat + self.ref_sat
        self.mode = "evalute" if evalute==True else "train"

        self.done_judge = done_judge(red_sat=self.red_sat,
                                     blue_sat=self.blue_sat, args=args)

        ## 引擎接口
        if args.fast_calculate:
            from Env.interface_sim import Self_Sim
            self.sim = Self_Sim(args=args,red_sat=self.red_sat,blue_sat=self.blue_sat)
        else:
            from Env.interface_sim import Mixed_Sim
            self.sim = Mixed_Sim(args=args,red_sat=self.red_sat,blue_sat=self.blue_sat)


        self.assign_res = {}

        self.random_ep = 3000

        # 指标量
        self.done_distance = 70  # km 星收敛距离，小于该距离为成功完成任务

        # 记录训练状态
        self.episode_num = 0
        self.step_num = 0
        self.rod = reward.reward_obs_done(args=self.args,red_sat=self.red_sat,blue_sat=self.blue_sat)
        ## 分配方案
        # from task_assign import TaskAssign as task_assign
        # self.task_assign = task_assign(args=args, red_sat=self.red_sat, blue_sat=self.blue_sat,
        #                                red_obs_dim=self.observation_space()["red"])


    # 交互函数1
    def reset(self):
        self.episode_num += 1
        self.step_num = 0

        self.sim.Reset_Env()
        self.rod.reset()

        # 分配
        if self.mode == "evalute":
            self.assign_res = self.task_assign.assign(self.sim.inf,blue_die=self.rod.blue_done)
        else:
            self.assign_res = copy.deepcopy({**{red_id: "b" + red_id[1] for red_id in self.red_sat},
                               **{blue_id: "r" + blue_id[1] for blue_id in self.blue_sat}})

        # 根据分配结果制作观测
        obs_red, obs_global_red, obs_blue, obs_global_blue = self.rod.obs_generate(self.assign_res, self.sim.inf)

        return obs_red, obs_global_red, obs_blue, obs_global_blue


    def step(self, action,noise=False):
        self.step_num += 1
        # 1.先将动作规划输入到引擎
        action_copy = copy.deepcopy(action) #在action作为实际参数传进来后，函数内部对action进行改变，那么外部的action也会改变
        # 训练的情况下需要噪声的加入
        if noise and self.mode=="train":
            action_copy = self.action_add_noise(action_copy) # 这里的动作单位是m/s
        action_main_star = {name:np.zeros(3) for name in self.ref_sat} #不给main_sat卫星动作
        action_copy = {**action_copy, **action_main_star}

        delta_v_dict = self.convert_v(action_copy)

        red_done, blue_done = self.rod.done_judge(
            inf=self.sim.inf,
            assign_res=self.assign_res)


        red_reward, blue_reward = self.rod.reward_genarate(assign_res=self.assign_res,
                                                           inf=self.sim.inf,
                                                           done_judge=self.done_judge,
                                                           action=delta_v_dict)

        self.sim.Step_Env(delta_v_dict=delta_v_dict) #抽象接口的输入单位为km/s

        # 任务分配
        if(self.mode=="train"):
            self.assign_res = copy.deepcopy({**{red_id: "b" + red_id[1] for red_id in self.red_sat},
                               **{blue_id: "r" + blue_id[1] for blue_id in self.blue_sat}}) ##训练状态下为，单智能体训练，只有在测试阶段需要进行多对多分配
        else:
            if self.step_num%10==0 and sum(list(self.rod.blue_done.values()))<len(list(self.rod.blue_done.values())):
                self.assign_res = self.task_assign.assign(self.sim.inf,blue_die=self.rod.blue_done)

        # 根据分配结果制作观测
        red_obs, global_obs_red, blue_obs, global_obs_blue = \
            self.rod.obs_generate(assign_res=self.assign_res, inf=self.sim.inf)

        return red_obs, blue_obs,\
               red_reward, blue_reward,\
               red_done, blue_done,\
               global_obs_red, global_obs_blue

    def action_add_noise(self, action):
        k = float(self.random_ep)/(float(self.episode_num)+float(self.random_ep)) #随机数衰减系数

        for agent_id in self.red_sat:
            action_noise = np.random.uniform(-1, 1, action[agent_id].shape[0]) * k
            action[agent_id] += action_noise

        for agent_id in self.blue_sat:
            action_noise = np.random.uniform(-1, 1, action[agent_id].shape[0]) * k
            action[agent_id] += action_noise
        return action

    def convert_v(self, action):
        act_dict = copy.deepcopy(action)
        for sat_id, act in act_dict.items():
            if sat_id[0]=="r":
                action[sat_id] = (act-0.5) * 2 * self.args.red_delta_v_limit/1000 # 网络输出的是-1到1，将其转换为红方卫星的限幅，并将其转换为km
            elif sat_id[0]=="b":
                action[sat_id] = (act-0.5) * 2 * self.args.blue_delta_v_limit/1000 # 网络输出的是-1到1，将其转换为蓝方卫星的限幅，并将其转换为km
            elif sat_id=="main_sat":
                action[sat_id] = act / 1000 # 一定会是【0,0,0】
        return action

    def observation_space(self):
        return self.rod.observation_space()

    def action_space(self):
        return {"red":3, "blue": 3}

