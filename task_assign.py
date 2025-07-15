# 系统包
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
# 自建包
from value_train import state_value_train

class TaskAssign:
    def __init__(self, args, red_sat, blue_sat, red_obs_dim):
        from history_reward_obs.reward12_11 import reward_obs_done as rod
        self.args = args
        self.red_sat = red_sat
        self.blue_sat = blue_sat
        self.state_value = state_value_train(args=args,obs_dim=red_obs_dim)
        # self.state_value = state_value_train.load(args=args,obs_dim=red_obs_dim)
        self.rod = rod(args=self.args, red_sat=self.red_sat, blue_sat=self.blue_sat)

    # 计算红方分配代价矩阵
    def cal_red_cost_matrix(self, inf):
        cost_matrix = np.zeros([len(self.red_sat), len(self.blue_sat)])
        # 更新代价矩阵,行是红色卫星，列是蓝方卫星
        # 注意！！蓝方卫星被消灭之后不再算入代价矩阵当中
        for agent_red in self.red_sat:
            for agent_blue in self.blue_sat:
                obs = self.rod.single_red_obs(red_name=agent_red, blue_name=agent_blue, inf=inf)
                cost_matrix[int(agent_red[1]),int(agent_blue[1])] = self.state_value.state_value_eva(obs)
        return cost_matrix

    def assign_policy_red(self, inf, blue_die):
        if sum(list(blue_die.values()))==len(inf.blue_sat): return {}
        # 得到代价矩阵
        matrix_cost = self.cal_red_cost_matrix(inf)
        gain = copy.deepcopy(matrix_cost)
        cost = -gain

        assign_res = {} # 分配结果

        # 如果蓝方卫星死亡，则将其对应列删除
        # 建立映射关系
            # 列代表蓝方卫星（蓝方卫星死，对应那行也要删除）
        cost_column_index = np.array(self.blue_sat,dtype=object) # 建立列索引（如果在cost中要删除列，则对应该索引也要删除）
            # 行代表红方卫星（红方卫星完成分配，则要删除那行）
        cost_row_index = np.array(self.red_sat,dtype=object) # 建立行索引（如果在cost中要删除行，则对应该索引也要删除）

        delete_column = []
        for i, blue_id in enumerate(cost_column_index):
            # 如果蓝卫星死，则删除该列
            if(blue_die[blue_id]):
                delete_column.append(i)

        try:
            cost = np.delete(cost, delete_column, axis=1)
            cost_column_index = np.delete(cost_column_index, delete_column)
        except Exception as e:
            print(e)

        circle_times = 0
        # 友>敌:只需要考虑每颗红卫星都能分配到敌人，若敌人多与红，则只需要保证红卫星有目标即可
        while (len(assign_res.keys())<len(inf.red_sat)):
            circle_times += 1
            if circle_times > int(len(inf.red_sat)/len(cost_column_index))+1: raise Exception("红方分配进入死循环")
            red_index_, blue_index_ = linear_sum_assignment(cost)
            for red_name,blue_name in zip(red_index_,blue_index_):
                assign_res[cost_row_index[red_name]] = cost_column_index[blue_name]

            cost = np.delete(cost,red_index_,axis=0)
            cost_row_index = np.delete(cost_row_index, red_index_)
        return assign_res

    def assign_policy_blue(self, inf, blue_die):
        assign_res = {}
        for blue_id in inf.blue_sat:
            for red_id in inf.red_sat:
                dis_array = []
                dis = np.linalg.norm(inf.pos[blue_id]-inf.pos[red_id])
                dis_array.append(dis)
            dis_array = np.array(dis_array)
            res = "r"+str(np.argmax(dis_array).item())
            assign_res[blue_id] = res
        return assign_res

    def assign(self, inf, blue_die):
        # red_assign_res = self.assign_policy_red(inf, blue_die)

        blue_assign_res = self.assign_policy_blue(inf, blue_die)
        red_assign_res = {red_id: "b"+red_id[1] for red_id in inf.red_sat}
        # blue_assign_res = {blue_id: "b" + blue_id[1] for blue_id in inf.blue_sat}
        res = copy.deepcopy({**red_assign_res, **blue_assign_res})

        # print(res)

        return res