import copy

from Env.information import sc_info as inf

import numpy as np

class done_judge:
    def __init__(self, red_sat, blue_sat, args):
        self.red_sat = red_sat
        self.blue_sat = blue_sat
        self.args = args
        # 多智能体强化学习中，由于单颗卫星任务结束（死亡或者成功）但是总体任务还没
        # 结束的情况下，环境不再给予该颗卫星的奖励即奖励为0
        self.RedIsDie = {red_id: False for red_id in self.red_sat} #超出通信距离或者小于安全距离
        self.RedSuccess = {red_id: False for red_id in self.red_sat} #追到目标卫星
        # 记录卫星第几次退出任务，如果是第一次，给奖励或惩罚，如果不是，不给予奖励
        self.RedIsDw = {red_id: 0 for red_id in self.red_sat}
        self.RedIsDone = {red_id: False for red_id in self.red_sat}

        self.BlueIsDie = {blue_id: False for blue_id in self.blue_sat} #被红方卫星追到
        self.BlueIsDw = {blue_id: 0 for blue_id in self.blue_sat}
        self.BlueIsDone = {blue_id: False for blue_id in self.blue_sat}


    def reset_dict(self):
        # 红方
        self.RedIsDie = {red_id: False for red_id in self.red_sat} #超出通信距离或者小于安全距离
        self.RedSuccess = {red_id: False for red_id in self.red_sat} #追到目标卫星
        self.RedIsDw = {red_id: 0 for red_id in self.red_sat} # 死亡或者成功
        self.RedIsDone = {red_id: False for red_id in self.red_sat} # 全都DW或者时间到
        # 蓝方
        self.BlueIsDie = {blue_id: False for blue_id in self.blue_sat} #被红方卫星追到
        self.BlueIsDw = {blue_id: 0 for blue_id in self.blue_sat}
        self.BlueIsDone = {blue_id: False for blue_id in self.blue_sat}

    def update_dict(self, inf, assign_res):
        # martix = np.zeros(3)
        # martix[0] = inf.dis_sat("b0","b1")
        # martix[1] = inf.dis_sat("b0", "b2")
        # martix[2] = inf.dis_sat("b1", "b2")
        # min_dis = np.min(martix)
        # max_dis = np.max(martix)
        # if min_dis<self.args.safe_dis or max_dis>self.args.comm_dis:
        #     a = 0
        self.RedIsDie = {red_id: False for red_id in self.red_sat}  # 一步量，只反应当前时刻状态
        self.RedSuccess = {red_id: False for red_id in self.red_sat}  # 一步量，只反应当前时刻状态
        self.BlueIsDie = {blue_id: False for blue_id in self.blue_sat}  # 一步量，只反应当前时刻状态
        # 判断红方卫星是否死亡
        self.RedIsDie = self.__red_die(inf)
        # 判断红方卫星是否追击成功
        self.RedSuccess = self.__red_success(inf, assign_res)
        # 红方卫星DW,记录的是第几次判断到该卫星DW，只关心0和1，其他时刻不关心，不给智能体奖励或者惩罚
        # 为什么在判断条件中加入self.RedIsDw[red_id]，是因为如果上一时刻已经dw而这一时刻没有dw，
        # 智能体观测会停留在dw=1的状态,无法理解为什么上一时刻(dw=1)奖励大，而这一时刻奖励少
        for red_id in self.red_sat:
            if self.RedIsDie[red_id] or self.RedSuccess[red_id] or \
                    self.RedIsDw[red_id]!=0 or inf.time+1 == self.args.episode_length:
                # 对dw进行限幅，因为dw需要加入到观测当中，如果太大会导致观测分布不均匀
                self.RedIsDw[red_id] = min(2,self.RedIsDw[red_id]+1)
        # 判断蓝方卫星是否死亡
        self.BlueIsDie = self.__blue_die(inf, assign_res)
        for blue_id in self.blue_sat:
            if self.BlueIsDie[blue_id] or self.BlueIsDw[blue_id]!=0 \
                    or inf.time+1 == self.args.episode_length:
                self.BlueIsDw[blue_id] = min(2,self.BlueIsDw[blue_id]+1)
        # 判断任务是否结束
        self.__TaskIsDone()

    def __red_die(self, inf):
        tmp = copy.deepcopy(self.RedIsDie) # 拷贝一份，只在判断结束之后修改成员变量值，保证判断的原子性
        for red_id in self.red_sat:
            tmp[red_id] = self.__SingleRedSatIsDie(inf, red_id) # 改变RedIsDie
        return tmp

    def __SingleRedSatIsDie(self, inf, red_name):
        exit_redsat = []
        for red_id, dw_times in self.RedIsDw.items():
            if dw_times==0: exit_redsat.append(red_id)
        # 例如r1卫星死亡后，卫星r0就不再考虑和他的位置关系
        for sat in exit_redsat:
            if sat!=red_name:
                dis = inf.dis_sat(sat, red_name)
                if dis<self.args.safe_dis or dis>self.args.comm_dis:
                    return True
        return False

    def __red_success(self, inf, assign_res):
        tmp = copy.deepcopy(self.RedSuccess)
        for red_id in self.red_sat:
            tmp[red_id] = self.__SingleRedSuccess(red_id, inf, assign_res) # 改变RedIsDie
        return tmp

    def __SingleRedSuccess(self, red_name, inf, assign_res):
        if self.RedIsDw[red_name] != 0: return False
        if inf.dis_sat(assign_res[red_name], red_name) < self.args.done_distance:
            return True
        return False

    def __blue_die(self, inf, assign_res):
        tmp = copy.deepcopy(self.BlueIsDie)
        for blue_id in self.blue_sat:
            tmp[blue_id] = self.__SingleBlueSatIsDie(inf, blue_id, assign_res)
        return tmp

    def __SingleBlueSatIsDie(self, inf, blue_name, assign_res):
        exit_sat = []
        for sat_id, dw_times in self.BlueIsDw.items():
            if dw_times == 0: exit_sat.append(sat_id)
        for sat in exit_sat:
            if sat != blue_name:
                dis = inf.dis_sat(name1=sat, name2=blue_name)
                if dis < self.args.safe_dis or dis > self.args.comm_dis:
                    return True

        # 暂时不考虑红方的追击
        # if (inf.dis_sat(assign_res[blue_name], blue_name) < self.args.done_distance and
        #         self.RedIsDw[assign_res[blue_name]]==0):
        #     return True
        return False

    # 判断任务是否结束
    def __TaskIsDone(self):
        # 蓝方死完（红方任务成功）
        task_done_cnt = 0
        for cnt in list(self.BlueIsDw.values()):
            if cnt!=0: task_done_cnt+=1
        if task_done_cnt==len(self.blue_sat):
            for red_id in self.red_sat:
                self.RedIsDone[red_id] = True
            for blue_id in self.blue_sat:
                self.BlueIsDone[blue_id] = True

        # 红方死完
        task_done_cnt = 0
        for cnt in list(self.RedIsDw.values()):
            if cnt!=0: task_done_cnt+=1
        if task_done_cnt==len(self.red_sat):
            for red_id in self.red_sat:
                self.RedIsDone[red_id] = True
            for blue_id in self.blue_sat:
                self.BlueIsDone[blue_id] = True
