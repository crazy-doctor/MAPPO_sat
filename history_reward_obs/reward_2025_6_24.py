# 系统包
import numpy as np
# 自建包
from Tool.astro_tool import Tool


class reward_obs_done(Tool):
    def __init__(self, args, red_sat, blue_sat):
        self.args = args
        self.red_sat = red_sat
        self.blue_sat = blue_sat

    def GlobalObs_Red(self,inf, done_judge, every_obs:dict):
        global_obs = np.zeros(0)
        for red_id in self.red_sat:
            global_obs = np.concatenate((global_obs,every_obs[red_id]),axis=0)
        return global_obs
    def single_red_obs(self, red_name, blue_name, inf, done_judge):
        time = np.array([inf.time/self.args.episode_length],dtype=float)
        ref_info = np.concatenate((inf.pos["main_sat"] / 42157, inf.vel["main_sat"] / 7),axis=0)

        # 在每个卫星位置速度前加上卫星的dw次数，如果是第一次
        zero_p_v = np.zeros(6)

        if done_judge.RedIsDw[red_name] == 2:
            my_info = np.concatenate((np.array([float(2)]),zero_p_v), axis=0)
        else:
            my_info = np.concatenate((
                np.array([done_judge.RedIsDw[red_name]],dtype=float),
                inf.pos_cw[red_name] / 1000, inf.vel_cw[red_name] * 10),axis=0)

        target_blue_info = np.concatenate((inf.pos_cw[blue_name] / 1000, inf.vel_cw[blue_name] * 10),axis=0)

        other_info = np.zeros(0)
        for red_id in self.red_sat:
            if red_id!=red_name:
                if done_judge.RedIsDw[red_id] == 2:
                    tmp_info = np.concatenate((np.array([float(2)]), zero_p_v), axis=0)
                else:
                    tmp_info = np.concatenate((
                    np.array([done_judge.RedIsDw[red_id]],dtype=float),
                    inf.pos_cw[red_id] / 1000, inf.vel_cw[red_id] * 10),axis=0)

                other_info = np.concatenate((other_info, tmp_info), axis=0)

        return np.concatenate((time, ref_info, my_info, target_blue_info, other_info),axis=0)

    def single_red_reward(self, red_name, blue_name, act, inf, done_judge):
        # 终端奖励
        # 1.追击成功并且此时任务没结束（第一次追击成功）
        if done_judge.RedSuccess[red_name] and done_judge.RedIsDw[red_name]==1: return 50
        # 2.追击失败并且此时卫星还没死（第一次卫星死亡）
        if done_judge.RedIsDie[red_name] and done_judge.RedIsDw[red_name]==1: return -10
        # 3.如果卫星在之前就已经死亡，则不给予奖励
        if done_judge.RedIsDw[red_name]>1: return 0

        # if done_judge.BlueIsDw[blue_name]!=0 :return 0
        # 引导奖励
        dis_feature = self.dis_ocursion(red_name=red_name,action=act ,blue_name=blue_name, inf=inf, step=2)
        dis_current = np.linalg.norm(inf.pos[red_name] - inf.pos[blue_name])

        reward = 0

        reward += (dis_current - dis_feature) / 20

        return reward

    def single_red_done(self, red_name, blue_name, inf, step_num):
        if np.linalg.norm(inf.pos[red_name]-inf.pos[blue_name])<self.args.done_distance:
            return True
        if(step_num==self.args.episode_length-1):
            return True
        return False

    def GlobalObs_Blue(self,inf, done_judge, every_obs:dict):
        global_obs = np.zeros(0)
        for blue_id in self.blue_sat:
            global_obs = np.concatenate((global_obs,every_obs[blue_id]),axis=0)
        return global_obs

    def single_blue_obs(self, red_name, blue_name, inf, done_judge):

        time = np.array([inf.time / self.args.episode_length], dtype=float)

        ref_info = np.concatenate((inf.pos["main_sat"] / 42157, inf.vel["main_sat"] / 7),axis=0)

        zero_p_v = np.zeros(6)
        if done_judge.BlueIsDw[blue_name] == 2:
            my_info = np.concatenate((np.array([float(2)]), zero_p_v), axis=0)
        else:
            my_info = np.concatenate((
                np.array([done_judge.BlueIsDw[blue_name]],dtype=float), # 活着
                inf.pos_cw[blue_name] / 1000, inf.vel_cw[blue_name] * 10),axis=0)

        other_info = np.zeros(0)
        for blue_id in self.blue_sat:
            if blue_id!=blue_name:
                if done_judge.BlueIsDw[blue_id] == 2:
                    tmp_info = np.concatenate((np.array([float(2)]), zero_p_v), axis=0)
                else:
                    tmp_info = np.concatenate((
                    np.array([done_judge.BlueIsDw[blue_id]], dtype=float),
                    inf.pos_cw[blue_id] / 1000, inf.vel_cw[blue_id] * 10),axis=0)

                other_info = np.concatenate((other_info, tmp_info), axis=0)
        return np.concatenate((time, ref_info, my_info, other_info),axis=0)


    def single_blue_reward(self, red_name, blue_name, inf, done_judge,action):
        # 终端奖励
        if done_judge.BlueIsDw[blue_name] == 0: return 0.1
        # 1.第一次卫星死亡
        if done_judge.BlueIsDie[blue_name] and done_judge.BlueIsDw[blue_name] == 1: return -10
        # 2.如果卫星在之前就已经死亡，则不给予奖励,如果动作此时不为0给予惩罚
        return 0

    def single_blue_done(self, red_name, blue_name, inf, step_num):
        # 如果被追上，判定死亡
        if np.linalg.norm(inf.pos[red_name]-inf.pos[blue_name])<self.args.done_distance:
            return True
        if(step_num==self.args.episode_length-1):
            return True
        return False

    def dis_ocursion(self, red_name, blue_name,action, inf, step):  # 以当前的状态，向前推理step步后的距离
        pos_red_cw, vel_red_cw = Tool.CW(self, r_sat=inf.pos[red_name], v_sat=inf.vel[red_name],
                                         r_ref=inf.pos["main_sat"], v_ref=inf.vel["main_sat"]+action,
                                         t=step * self.args.step_time)

        pos_blue_cw, vel_blue_cw = Tool.CW(self, r_sat=inf.pos[blue_name],
                                           v_sat=inf.vel[blue_name],
                                           r_ref=inf.pos["main_sat"], v_ref=inf.vel["main_sat"],
                                           t=step * self.args.step_time)
        return np.linalg.norm(pos_red_cw - pos_blue_cw)

    def red_obs(self, assign_res, inf, done_judge):
        obs_red = {red_id: self.single_red_obs(red_name=red_id, blue_name=assign_res[red_id],
                                               inf=inf, done_judge=done_judge) for red_id in self.red_sat}
        obs_global = self.GlobalObs_Red(inf, done_judge, obs_red)
        return obs_red, obs_global

    def blue_obs(self, assign_res, inf, done_judge):
        obs_blue = {blue_id: self.single_blue_obs(red_name=assign_res[blue_id], blue_name=blue_id, inf=inf,
                                                  done_judge=done_judge) for blue_id in self.blue_sat}
        obs_global = self.GlobalObs_Blue(inf, done_judge, obs_blue)
        return obs_blue, obs_global

    def red_done(self, assign_res, inf, step_num):
        done_red = {red_id: self.single_red_done(red_id, assign_res[red_id], inf, step_num) for red_id in
                    self.red_sat}
        return done_red

    def blue_done(self, assign_res, inf, step_num):
        done_blue = {blue_id: self.single_blue_done(assign_res[blue_id], blue_id, inf, step_num)
                     for blue_id in self.blue_sat}
        return done_blue

    def red_reward(self, assign_res, inf, done_judge, action):
        reward_red = {red_id: self.single_red_reward(red_id, assign_res[red_id],action[red_id], inf,
                                                     done_judge=done_judge) for red_id in self.red_sat}

        return reward_red

    def blue_reward(self, assign_res, inf, done_judge,action):
        reward_blue = {blue_id: self.single_blue_reward(assign_res[blue_id],
                       blue_id, inf,done_judge=done_judge, action=action[blue_id]) for blue_id in self.blue_sat}
        return reward_blue

    def observation_space(self):
        return {"red":34, "global_red":34*3,
                "blue": 28, "global_blue":28*3}
