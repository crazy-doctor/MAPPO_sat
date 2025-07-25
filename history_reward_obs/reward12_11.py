# 系统包
import numpy as np
# 自建包
from Tool.astro_tool import Tool

class reward_obs_done(Tool):
    def __init__(self, args,red_sat, blue_sat):
        self.args = args
        self.red_sat = red_sat
        self.blue_sat = blue_sat

    def single_red_obs(self, red_name, blue_name, inf):
        obs = np.concatenate(
            (inf.pos["main_sat"] / 42157, inf.pos_cw[red_name] / 1000, inf.pos_cw[blue_name] / 1000,
                              inf.vel["main_sat"] / 7, inf.vel_cw[red_name] * 10, inf.vel_cw[blue_name] * 10)
            , axis=0)
        return obs

    def single_red_reward(self, red_name, blue_name, inf):
        dis_feature = self.dis_ocursion(red_name=red_name,blue_name=blue_name,inf=inf, step=10)
        dis_current = np.linalg.norm( inf.pos[red_name]- inf.pos[blue_name])

        reward = 0

        reward += (dis_current - dis_feature) / 50
        if dis_current<100:
            reward+=0.7
        return reward

    def single_red_done(self, red_name, blue_name, inf, done_distance):
        
        distance = np.linalg.norm(inf.pos[red_name] - inf.pos[blue_name])
        
        if(distance<done_distance):
            return True
        else:
            return False
       

    def single_blue_obs(self, red_name, blue_name, inf):
        # r0和b0使用cw坐标系进行训练
        obs = np.concatenate(( inf.pos[red_name] / 42157,  inf.pos[blue_name] / 42157,
                               inf.vel[red_name] / 5,  inf.vel[blue_name] / 5),axis=0)

        return obs

    def single_blue_reward(self, red_name, blue_name, inf):
        reward = 0
        punishment = 0

        dis1 = np.linalg.norm( inf.pos_cw[red_name] -  inf.pos_cw[blue_name])  # 与敌方卫星距离
        dis2 = np.linalg.norm( inf.pos_cw[blue_name])  # 距离初始位置距离
        if dis2 > 1000:
            punishment = 3
        reward += (dis1 / 500 - punishment)
        return reward

    def single_blue_done(self, red_name, blue_name, inf):
        return False
        
    def dis_ocursion(self, red_name, blue_name,inf, step):  # 以当前的状态，向前推理step步后的距离
        pos_red_cw, vel_red_cw = Tool.CW(self, r_sat= inf.pos[red_name], v_sat= inf.vel[red_name],
                                         r_ref= inf.pos["main_sat"], v_ref= inf.vel["main_sat"],
                                         t=step * self.args.step_time)

        pos_blue_cw, vel_blue_cw = Tool.CW(self, r_sat= inf.pos[blue_name],
                                           v_sat= inf.vel[blue_name],
                                           r_ref= inf.pos["main_sat"], v_ref= inf.vel["main_sat"],
                                           t=step * self.args.step_time)
        return np.linalg.norm(pos_red_cw - pos_blue_cw)

    def red_obs(self, assign_res, inf):
        obs_red = {red_id: self.single_red_obs(red_name=red_id, blue_name=assign_res[red_id], inf=inf) for red_id in self.red_sat}
        return obs_red

    def blue_obs(self, assign_res, inf):
        obs_blue = {blue_id: self.single_blue_obs(red_name=assign_res[blue_id], blue_name=blue_id, inf=inf) for blue_id in self.blue_sat}
        return obs_blue

    def red_done(self, assign_res, inf, done_distance):
        done_red = {red_id: self.single_red_done(red_id, assign_res[red_id], inf, done_distance) for red_id in self.red_sat}
        return done_red

    def blue_done(self, assign_res, inf):
        done_blue = {blue_id: self.single_blue_done(assign_res[blue_id], blue_id, inf) for blue_id in self.blue_sat}
        return done_blue

    def red_reward(self, assign_res, inf):
        reward_red = {red_id: self.single_red_reward(red_id, assign_res[red_id], inf) for red_id in self.red_sat}
        return reward_red

    def blue_reward(self, assign_res, inf):
        reward_blue = {blue_id: self.single_blue_reward(assign_res[blue_id], blue_id, inf) for blue_id in self.blue_sat}
        return reward_blue
        