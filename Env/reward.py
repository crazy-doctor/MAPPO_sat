# 系统包
import numpy as np
# 自建包
from Tool.astro_tool import Tool
from Env import information

class reward_obs_done(Tool):
    def __init__(self, args, red_sat, blue_sat):
        self.args = args
        self.red_sat = red_sat
        self.blue_sat = blue_sat
        self.red_done = {sat_id: False for sat_id in self.red_sat}
        self.blue_done = {sat_id: False for sat_id in self.blue_sat}

    def reset(self):
        self.red_done = {sat_id: False for sat_id in self.red_sat}
        self.blue_done = {sat_id: False for sat_id in self.blue_sat}

##############################  判断卫星是否死亡或成功（结束）  #################################################
    def done_judge(self, inf, assign_res):
        done_tmp_red = self.red_done
        for red_id in self.red_sat:
            target_blue = assign_res[red_id]
            done_tmp_red[red_id] = self.single_red_done(red_id, target_blue, inf)

        done_tmp_blue = self.blue_done
        for blue_id in self.blue_sat:
            target_red = assign_res[blue_id]
            done_tmp_blue[blue_id] = self.single_blue_done(target_red, blue_id, inf)

        self.red_done = done_tmp_red
        self.blue_done = done_tmp_blue
        return done_tmp_red, done_tmp_blue

    def single_red_done(self, red_name, blue_name, inf):
        # 如果已经死亡，则不用再判断
        if self.red_done[red_name]: return True
        # 追到目标
        if inf.dis_sat(red_name, blue_name) < self.args.done_distance:
            return True
        # # 编队
        # for red_id in self.red_sat:
        #     if red_id != red_name and (not self.red_done[red_id]):
        #         dis = inf.dis_sat(name1=red_id, name2=red_name)
        #         if dis < self.args.safe_dis or dis > self.args.comm_dis:
        #             return True
        if (inf.time + 1 == self.args.episode_length):
            return True
        return False

    def single_blue_done(self, red_name, blue_name, inf):
        # 如果已经死亡，则不用再判断
        # if self.blue_done[blue_name]: return True
        # # 追到目标
        # if inf.dis_sat(red_name, blue_name) < self.args.done_distance:
        #     return True
        # # 编队
        # for blue_id in self.blue_sat:
        #     if blue_id != blue_name and (not self.blue_done[blue_id]):
        #         dis = inf.dis_sat(name1=blue_id, name2=blue_name)
        #         if dis < self.args.safe_dis or dis > self.args.comm_dis:
        #             return True
        if (inf.time + 1 == self.args.episode_length):
            return True
        return False
#############################################   观测    ################################################

    def obs_generate(self, assign_res, inf):
        # 红方观测
        obs_red = {red_id: self.single_red_obs(red_name=red_id, blue_name=assign_res[red_id],
                                               inf=inf) for red_id in self.red_sat}
        obs_global_red = self.GlobalObs_Red(inf, obs_red)
        # 蓝方观测
        obs_blue = {blue_id: self.single_blue_obs(red_name=assign_res[blue_id], blue_name=blue_id,
                                                  inf=inf) for blue_id in self.blue_sat}
        obs_global_blue = self.GlobalObs_Blue(inf, obs_blue)

        return obs_red, obs_global_red, obs_blue, obs_global_blue

    def single_red_obs(self, red_name, blue_name, inf):
        done_mask = 0 if self.red_done[red_name] else 1
        time = np.array([inf.time/self.args.episode_length],dtype=float)
        ref_info = np.concatenate((inf.pos["main_sat"] / 42157, inf.vel["main_sat"] / 7),axis=0)*done_mask

        my_info = np.concatenate((np.array([done_mask]),
                                  inf.pos_cw[red_name] / 1000,
                                  inf.vel_cw[red_name] * 10),axis=0)*done_mask

        target_blue_info = np.concatenate((inf.pos_cw[blue_name] / 1000,
                                  inf.vel_cw[blue_name] * 10),axis=0)*done_mask

        other_info = np.zeros(0)
        for red_id in self.red_sat:
            if red_id!=red_name:
                this_sat_done = 1.0-float(self.red_done[red_id])
                tmp_info = np.concatenate((np.array([this_sat_done],dtype=float),
                inf.pos_cw[red_id] / 1000, inf.vel_cw[red_id] * 10),axis=0)
                tmp_info *= this_sat_done
                other_info = np.concatenate((other_info, tmp_info), axis=0)
        other_info *= done_mask
        return np.concatenate((time, ref_info, my_info, target_blue_info, other_info),axis=0)

    def GlobalObs_Red(self,inf, every_obs:dict):
        global_obs = np.zeros(0)
        for red_id in self.red_sat:
            global_obs = np.concatenate((global_obs,every_obs[red_id]),axis=0)
        return global_obs

    def single_blue_obs(self, red_name, blue_name, inf):

        done_mask = 0 if self.blue_done[blue_name] else 1
        time = np.array([inf.time / self.args.episode_length], dtype=float)

        ref_info = np.concatenate((inf.pos["main_sat"] / 42157, inf.vel["main_sat"] / 7),axis=0)*done_mask

        my_info = np.concatenate((np.array([done_mask],dtype=float),
                                  inf.pos_cw[blue_name] / 1000,
                                  inf.vel_cw[blue_name] * 10),axis=0)*done_mask
        other_info = np.zeros(0)
        for blue_id in self.blue_sat:
            if blue_id!=blue_name:
                this_sat_is_die = 1-int(self.blue_done[blue_name])
                tmp_info = np.concatenate((
                np.array([this_sat_is_die], dtype=float),
                inf.pos_cw[blue_id] / 1000, inf.vel_cw[blue_id] * 10),axis=0)
                tmp_info*=this_sat_is_die
                other_info = np.concatenate((other_info, tmp_info), axis=0)
        other_info *= done_mask
        return np.concatenate((time, ref_info, my_info, other_info),axis=0)

    def GlobalObs_Blue(self, inf, every_obs: dict):
        global_obs = np.zeros(0)
        for blue_id in self.blue_sat:
            global_obs = np.concatenate((global_obs, every_obs[blue_id]), axis=0)
        return global_obs
#############################################   奖励    #######################################################
    def reward_genarate(self, assign_res, inf, done_judge, action):
        reward_red = {red_id: self.single_red_reward(red_id, assign_res[red_id], action[red_id], inf,
                                                     done_judge=done_judge) for red_id in self.red_sat}

        reward_blue = {blue_id: self.single_blue_reward(assign_res[blue_id],
                       blue_id, inf,done_judge=done_judge, action=action[blue_id]) for blue_id in self.blue_sat}

        return reward_red, reward_blue


    def single_red_reward(self, red_name, blue_name, act, inf, done_judge):
        # # 任务成功奖励
        # if inf.dis_sat(red_name, blue_name) < self.args.done_distance:
        #     return 50
        # # 卫星死亡惩罚
        # for red_id in self.red_sat:
        #     if red_id != red_name and self.red_done[red_id]:
        #         dis = inf.dis_sat(name1=red_id, name2=red_name)
        #         if dis < self.args.safe_dis or dis > self.args.comm_dis:
        #             return -10
        # 否则就以当前时刻距离惩罚，鼓励智能体减少距离
        dis_current = inf.dis_sat(red_name, blue_name)
        return -dis_current/20


    def single_blue_reward(self, red_name, blue_name, inf, done_judge,action):

        for blue_id in self.blue_sat:
            if blue_id != blue_name and self.blue_done[blue_id]:
                dis = inf.dis_sat(name1=blue_id, name2=blue_name)
                if dis < self.args.safe_dis or dis > self.args.comm_dis:
                    return -10
        return 0.1


    def dis_ocursion(self, red_name, blue_name,action, inf, step):  # 以当前的状态，向前推理step步后的距离
        pos_red_cw, vel_red_cw = Tool.CW(self, r_sat=inf.pos[red_name], v_sat=inf.vel[red_name],
                                         r_ref=inf.pos["main_sat"], v_ref=inf.vel["main_sat"]+action,
                                         t=step * self.args.step_time)

        pos_blue_cw, vel_blue_cw = Tool.CW(self, r_sat=inf.pos[blue_name],
                                           v_sat=inf.vel[blue_name],
                                           r_ref=inf.pos["main_sat"], v_ref=inf.vel["main_sat"],
                                           t=step * self.args.step_time)
        return np.linalg.norm(pos_red_cw - pos_blue_cw)


    def observation_space(self):
        assign_res = {**{red_id: "b" + red_id[1] for red_id in self.red_sat},
                           **{blue_id: "r" + blue_id[1] for blue_id in self.blue_sat}}
        inf = information.sc_info(self.red_sat, self.blue_sat, ["main_sat"], self.args)
        inf.init_state()

        obs_red, obs_global_red, obs_blue, obs_global_blue = \
            self.obs_generate(assign_res, inf)
        return {"red":list(obs_red.values())[0].shape[0], "global_red":obs_global_red.shape[0],
                "blue": list(obs_blue.values())[0].shape[0], "global_blue":obs_global_blue.shape[0]}
