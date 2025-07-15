# 系统包
import numpy as np
# 自建包
from Tool.astro_tool import Tool


class reward_obs_done(Tool):
    def __init__(self, args, red_sat, blue_sat):
        self.args = args
        self.red_sat = red_sat
        self.blue_sat = blue_sat
    def single_red_obs(self, red_name, blue_name, inf):
        # [主卫星位置速度，其他三颗红方卫星的状态，目标蓝方卫星状态]
        # 红方卫星状态：红方卫星速度位置
        # 蓝方卫星状态：蓝方卫星是否死亡，蓝方卫星速度位置

        if sat_dead or sat_success:
            ref_state = np.zeros(6)
            is_done = np.array([1])
            my_state = np.zeros(6, axis=0)
            # 蓝方目标卫星状态
            target_state = np.zeros(6, axis=0)
            other_red_obs = np.zeros(len(self.red_sat)-1)
            obs = np.concatenate((ref_state,is_done, my_state, target_state, other_red_obs), axis=0)
        else:
            ref_state = np.concatenate((inf.pos["main_sat"] / 42157, inf.vel["main_sat"] / 1000), axis=0)
            is_done = np.array([0])
            my_state = np.concatenate((inf.pos_cw[red_name] / 1000, inf.vel_cw[red_name] * 10), axis=0)
            # 蓝方目标卫星状态
            target_state = np.concatenate((inf.pos_cw[blue_name] / 1000, inf.vel_cw[blue_name] * 10), axis=0)
            other_red_obs = np.zeros(0)
            for sat_id in inf.red_sat:
                if sat_id != red_name:
                    state_tmp = np.concatenate((inf.pos_cw[sat_id] / 1000, inf.vel_cw[sat_id] * 10), axis=0)
                    red_state = np.concatenate((other_red_obs, state_tmp), axis=0)
            obs = np.concatenate((ref_state,is_done, my_state, target_state, other_red_obs), axis=0)

        return obs

    def single_red_reward(self, red_name, blue_name, inf):

        # # 检测碰撞或者超出通信距离
        # collision_or_overcomm = False
        # min_distance = 42157*2  # 设置较大值作为初始值
        # max_distance = 0
        # for red_id in inf.red_sat:
        #     if(red_id!=red_name):
        #         min_distance = min(min_distance, inf.dis_sat(red_name, red_id))
        #         max_distance = max(max_distance, inf.dis_sat(red_name, red_id))
        # if min_distance<5 or max_distance>80:
        #     collision_or_overcomm = True

        # with open(r"D:\shen\code_list\MADDPG\min_distance.txt", "a") as f:
        #     f.write(str(min_distance.item()) + "\n")
        #
        # with open(r"D:\shen\code_list\MADDPG\max_distance.txt", "a") as f:
        #     f.write(str(max_distance.item()) + "\n")

        dis_feature = self.dis_ocursion(red_name=red_name, blue_name=blue_name, inf=inf, step=2)
        dis_current = np.linalg.norm(inf.pos[red_name] - inf.pos[blue_name])

        reward = 0

        reward += (dis_current - dis_feature) / 50

        if dis_current < self.args.done_distance:
            reward += 50

        # if collision_or_overcomm:
        #     reward -= 1
        return reward

    def single_red_done(self, red_name, blue_name, inf, step_num):
        # if np.linalg.norm(inf.pos[red_name]-inf.pos[blue_name])<self.args.done_distance:
        #     return True
        if(step_num==self.args.episode_length-1):
            return True
        return False

    def single_blue_obs(self, red_name, blue_name, inf):
        # r0和b0使用cw坐标系进行训练
        obs = np.concatenate((inf.pos_cw[red_name] / 1000, inf.vel_cw[red_name] * 10,
                              inf.pos_cw[blue_name] / 1000, inf.vel_cw[blue_name] * 10), axis=0)

        return obs

    def single_blue_reward(self, red_name, blue_name, inf):
        reward = 0
        punishment = 0

        dis1 = np.linalg.norm(inf.pos_cw[red_name] - inf.pos_cw[blue_name])  # 与敌方卫星距离
        dis2 = np.linalg.norm(inf.pos_cw[blue_name])  # 距离初始位置距离
        if dis2 > 1000:
            punishment = 3
        reward += (dis1 / 500 - punishment)
        return reward

    def single_blue_done(self, red_name, blue_name, inf):
        # # 如果被追上，判定死亡
        # if np.linalg.norm(inf.pos[red_name]-inf.pos[blue_name])<self.args.done_distance:
        #     return True
        return False

    def dis_ocursion(self, red_name, blue_name, inf, step):  # 以当前的状态，向前推理step步后的距离
        pos_red_cw, vel_red_cw = Tool.CW(self, r_sat=inf.pos[red_name], v_sat=inf.vel[red_name],
                                         r_ref=inf.pos["main_sat"], v_ref=inf.vel["main_sat"],
                                         t=step * self.args.step_time)

        pos_blue_cw, vel_blue_cw = Tool.CW(self, r_sat=inf.pos[blue_name],
                                           v_sat=inf.vel[blue_name],
                                           r_ref=inf.pos["main_sat"], v_ref=inf.vel["main_sat"],
                                           t=step * self.args.step_time)
        return np.linalg.norm(pos_red_cw - pos_blue_cw)

    def red_obs(self, assign_res, inf):
        obs_red = {red_id: self.single_red_obs(red_name=red_id, blue_name=assign_res[red_id], inf=inf) for red_id in
                   self.red_sat}
        return obs_red

    def blue_obs(self, assign_res, inf):
        obs_blue = {blue_id: self.single_blue_obs(red_name=assign_res[blue_id], blue_name=blue_id, inf=inf) for blue_id
                    in self.blue_sat}
        return obs_blue

    def red_done(self, assign_res, inf, step_num):
        done_red = {red_id: self.single_red_done(red_id, assign_res[red_id], inf, step_num) for red_id in
                    self.red_sat}
        return done_red

    def blue_done(self, assign_res, inf):
        done_blue = {blue_id: self.single_blue_done(red_name=assign_res[blue_id], blue_name=blue_id, inf=inf) for blue_id in self.blue_sat}
        return done_blue

    def red_reward(self, assign_res, inf):
        reward_red = {red_id: self.single_red_reward(red_id, assign_res[red_id], inf) for red_id in self.red_sat}
        return reward_red

    def blue_reward(self, assign_res, inf):
        reward_blue = {blue_id: self.single_blue_reward(assign_res[blue_id], blue_id, inf) for blue_id in self.blue_sat}
        return reward_blue

    def observation_space(self):
        return {"red":18, "blue": 12}
