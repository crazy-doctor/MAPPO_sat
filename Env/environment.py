import json
import math
import numpy as np
import copy


# todo 改写轨道六根数次序
class Environment:
    def __init__(self, args=None):
        # 启动仿真软件
        if args.fast_calculate:
            from interface_sim import Self_Sim
            self.sim = Self_Sim(args=args)
        else:
            from interface_sim import Mixed_Sim
            self.sim = Mixed_Sim(args=args)
            
        self.args = args
        self.red_num = 3
        self.blue_num = 3
        self.red_sat = ["r"+str(i) for i in range(self.red_num)]
        self.blue_sat = ["b" + str(i) for i in range(self.blue_num)]
        self.main_sat = ["main_sat"]
        self.satellite = self.red_sat+self.blue_sat

        # 指标量
        self.done_distance = 100  # km 星收敛距离，小于该距离为成功完成任务
        self.orbit_alt = 42157 # 轨道高度


        self.done = {agent_id: False for agent_id in self.red_sat}
        self.obs = {}
        self.reward = {}

        # 以下状态不在update_state(self.data)函数中更新
        # 在step后中action更新，在reset中复位

        # 判断一局是否结束标志
        self.done_step = {name: 0 for name in self.red_sat}
        # 如果连续完成指标三步，判定为结束，在reset中初始化，在step中进行修改和判断
        # 如果距离小于30km，则认为完成一帧。有两种情况，在进判断时，如果不满足迫近条件，则判断self.done是否为0
        # 如果不等于0，需要进行置零操作，这是因为上一帧满足了迫近条件，而这一帧没有满足
        # 如果满足迫近条件，需要对该标志位加一
        self.episode_num = 0
        self.step_num = 0

        # 为了和强化学习接口接上，多余加的一些属性
        self.agents = self.red_sat
        self.num_agents = self.red_num

    def reset(self):

        obs = {sat_id: self.single_obs(sat_id) for sat_id in self.red_sat}
        return obs


    def action_explain(self,action):
        action_main_star = {name:np.zeros(3) for name in self.ref_sat}
        action_e = {**action, **action_main_star}
        return action

    def maneuver_step(self, delta_v_dict=None):
        if self.man_style == "implusive":
            while self.af.GetNumOfStepToAdvance() != 0:
                pass
            for name in self.satellite:
                delata_v = delta_v_dict[name]
                self.af.SetSatDeltaV(name, delata_v[0]*1000, delata_v[1]*1000, delata_v[2]*1000)
            while self.af.GetNumOfStepToAdvance() != 0:
                pass
            # 推进！推进推进！冲TNND
            self.af.SetNumOfStepToAdvance(self.step_time)

            while self.af.GetNumOfStepToAdvance() != 0:
                pass
            self.Current_Time = self.af.GetSimData().simTime

            while abs(self.Current_Time - self.lastTimeStamp) < 0.001:
                self.Current_Time = self.af.GetSimData().simTime
            # 动作完成后立即更新，从而进入下一步
            self.lastTimeStamp = self.Current_Time
            self.MainSat_Ocursion.SetNumOfStepToAdvance(self.step_time)

        elif self.man_style == "lasting":
            for i in range(self.maneuver_time):
                while self.af.GetNumOfStepToAdvance() != 0:
                    pass
                for name in self.satellite:
                    delata_v = delta_v_dict[name]
                    self.af.SetSatDeltaV(name, delata_v[0]*1000, delata_v[1]*1000, delata_v[2]*1000)
                self.af.SetNumOfStepToAdvance(1)

                # 仿真环境中有bug，需要将第一步和其他步数区别开
                while self.af.GetNumOfStepToAdvance() != 0:
                    pass
                self.Current_Time = self.af.GetSimData().simTime

                while abs(self.Current_Time - self.lastTimeStamp) < 0.001:
                    self.Current_Time = self.af.GetSimData().simTime
                # 动作完成后立即更新，从而进入下一步
                self.lastTimeStamp = self.Current_Time
            while self.af.GetNumOfStepToAdvance() != 0:
                pass
            self.af.SetNumOfStepToAdvance(self.step_time - self.maneuver_time)

            while self.af.GetNumOfStepToAdvance() != 0:
                pass
            self.Current_Time = self.af.GetSimData().simTime

            while abs(self.Current_Time - self.lastTimeStamp) < 0.001:
                self.Current_Time = self.af.GetSimData().simTime
            # 动作完成后立即更新，从而进入下一步
            self.lastTimeStamp = self.Current_Time




    def step(self, action, greedy_pro):
        # 上一步要么为reset要么为推进步骤,到这里，此刻时间和上一刻时间应该是相等的，因为成功
        # 推进之后执行self.lastTimeStamp = self.Current_Time
        # 利用先验知识，对action做出处理，避免明显不正确的动作
        action = self.action_mask(action, greedy_pro)
        delta_v_dict = self.action_explain(action)
        # 持续推力变轨
        self.maneuver_step(delta_v_dict=delta_v_dict)
        # 给main_sat更新位置
        data = self.MainSat_Ocursion.get_data()
        self.pos["main_sat"] = data["satellite"]["main_sat"]["pos"]
        self.vel["main_sat"] = data["satellite"]["main_sat"]["vel"]
        # 给其他卫星更新位置
        self.data = self.get_data()
        self.update_state(self.data)
        for name in self.satellite:
            if np.linalg.norm(delta_v_dict[name]) != 0:
                self.maneuver_cnt[name] += 1
        self.obs = {sat_id: self.single_obs(sat_id) for sat_id in self.red_sat}
        self.done = {sat_id: self.done_judge(sat_id) for sat_id in self.red_sat}
        self.reward = {sat_id: self.single_reward(sat_id) for sat_id in self.red_sat}
        return action, self.obs, self.reward, self.done

    # [轨道倾角， 升交点赤经， 半长轴， 偏心率， 真近点角， 近地点幅角]
    def orbit_init(self):
        # 初始化CW坐标系
        cw_sat_orbit_ele = np.array([np.random.uniform(0,3),np.random.uniform(0,300),self.orbit_alt,0,
                                     np.random.uniform(0,300),np.random.uniform(0,300)])
        pos, vel = self.orbital_elements_to_pv(cw_sat_orbit_ele)

        self.pos["main_sat"] = pos
        self.vel["main_sat"] = vel

        # 蓝方卫星初始化
        # 1.蓝方卫星的参考点是cw坐标系的原点
        reference_point_blue = self.pos["main_sat"]
        # 2.初始化蓝方卫星轨道
        for name in self.blue_sat:
            self.orbital_ele[name] = cw_sat_orbit_ele + np.array([0, 0, int(name[1])*2, 0, 0, 0])
            self.pos[name], self.vel[name] = self.orbital_elements_to_pv(orbital_elements=self.orbital_ele[name])

        # 红方卫星轨道初始化
        # 红方卫星的参考点在球面上，该球以cw坐标系原点为球心，半径在150~200km中
        r = np.random.uniform(150,200) # cw坐标系下的坐标
                                                                               # [轨道倾角， 升交点赤经， 半长轴， 偏心率， 真近点角， 近地点幅角]
        reference_point_red = self.generate_random_point_on_sphere(radius=r)
        pos_red,vel_red = self.CW_to_Inertial(pos_main=self.pos["main_sat"], vel_main=self.vel["main_sat"],
                                              pos_cw=reference_point_red, vel_cw=np.zeros(3))
        orbit_ele_red = self.pv_to_orbital_elements(pos=pos_red, vel=vel_red)
        for name in self.red_sat:
            self.orbital_ele[name] = orbit_ele_red + np.array([0, 0, int(name[1])*2, 0, 0, 0])
            self.pos[name], self.vel[name] = self.orbital_elements_to_pv(orbital_elements=self.orbital_ele[name])

    # [轨道倾角， 升交点赤经， 半长轴， 偏心率， 真近点角， 近地点幅角]
    # 智能体n： [相对距离（9维），六颗卫星的六根数（36维），六颗卫星的位置（18维），太阳位置（3）维，发动机工作时间（1维）]
    def single_reward(self, name):
        red_name = name
        blue_name = "b"+name[1]

        ref_last_pos = self.last_pos["main_sat"]
        vel_last_pos = self.last_vel["main_sat"]

        ref_pos_t_plus = self.pos["main_sat"]
        ref_vel_t_plus = self.vel["main_sat"]

        blue_pos_cw_last, blue_vel_cw_last = self.Inertial_to_CW(pos_main=ref_last_pos, vel_main=vel_last_pos,
                                                  pos_sat=self.last_pos[blue_name], vel_sat=self.last_vel[blue_name])
        red_pos_cw_last, red_vel_cw_last = self.Inertial_to_CW(pos_main=ref_last_pos, vel_main=vel_last_pos,
                                                   pos_sat=self.last_pos[red_name], vel_sat=self.last_vel[red_name])
        dis_dif_last = np.linalg.norm(blue_pos_cw_last - red_pos_cw_last)
        vel_dif_last = np.linalg.norm(blue_vel_cw_last - red_vel_cw_last)

        # 下一刻，红蓝卫星的cw位置
        blue_pos_cw, blue_vel_cw = self.Inertial_to_CW(pos_main=ref_pos_t_plus, vel_main=ref_vel_t_plus,
                                                  pos_sat=self.pos[blue_name], vel_sat=self.vel[blue_name])
        red_pos_cw, red_vel_cw = self.Inertial_to_CW(pos_main=ref_pos_t_plus, vel_main=ref_vel_t_plus,
                                                   pos_sat=self.pos[red_name], vel_sat=self.vel[red_name])
        dis_dif = np.linalg.norm(blue_pos_cw - red_pos_cw)
        vel_dif = np.linalg.norm(blue_vel_cw - red_vel_cw)
        self.dis_diference_between_cw_ine[red_name] = dis_dif
        self.vel_diference_between_cw_ine[red_name] = vel_dif
        self.p_dis[red_name] = dis_dif_last-dis_dif
        reward = -dis_dif/150 - 35*vel_dif + (dis_dif_last-dis_dif)/7
        if dis_dif<70:
            reward += 8
        return reward

    def done_judge(self, name):
        sucess_distance = self.done_distance
        sucess_time = 5
        distance = np.linalg.norm(self.pos[name] - self.pos["b" + name[1]])
        if distance < sucess_distance:
            self.done_step[name] += 1
        elif distance >= sucess_distance and self.done_step[name] < sucess_time:
            self.done_step[name] = 0

        if self.done_step[name] >= sucess_time:
            #训练过程中暂时屏蔽 ，之后改为True
            return True
        else:
            return False

    # 智能体n： [相对距离（9维），六颗卫星的六根数（36维），六颗卫星的位置（18维），太阳位置（3）维，发动机工作时间（1维）]
    # 调用： step
    # [轨道倾角， 升交点赤经， 半长轴， 偏心率， 真近点角， 近地点幅角]
    def single_obs(self, name):
        red_name = name
        blue_name = "b"+name[1]

        orbit_element_red = copy.deepcopy(self.orbital_ele[red_name])
        # 归一化
        orbit_element_red = orbit_element_red.astype(float)
        # 轨道倾角（Inclination）： 0 <= i <= 180°
        orbit_element_red[0] = orbit_element_red[0] / 90
        # 升交点赤经（Longitude of the ascending node）：0 <= Ω < 360°
        orbit_element_red[1] = orbit_element_red[1] / 360
        # 半长轴（Semi - majoraxis）：半长轴必须大于零。
        orbit_element_red[2] = orbit_element_red[2] / self.orbit_alt
        # 偏心率（Eccentricity）： 0 <= e < 1
        orbit_element_red[3] = orbit_element_red[3]
        # 近地点参数（Argument of periapsis）： 0 <= ω < 360°
        orbit_element_red[4] = orbit_element_red[4] / 360
        # 平均近点角（Mean anomaly at epoch）： -180° <= M <= 180°
        orbit_element_red[5] = orbit_element_red[5] / 360

        pos_red = self.pos[red_name]/self.orbit_alt
        vel_red = self.vel[red_name]


        orbit_element_blue = copy.deepcopy(self.orbital_ele[blue_name])
        # 归一化
        orbit_element_blue = orbit_element_blue.astype(float)
        # 轨道倾角（Inclination）： 0 <= i <= 180°
        orbit_element_blue[0] = orbit_element_blue[0] / 90
        # 升交点赤经（Longitude of the ascending node）：0 <= Ω < 360°
        orbit_element_blue[1] = orbit_element_blue[1] / 360
        # 半长轴（Semi - majoraxis）：半长轴必须大于零。
        orbit_element_blue[2] = orbit_element_blue[2] / self.orbit_alt
        # 偏心率（Eccentricity）： 0 <= e < 1
        orbit_element_blue[3] = orbit_element_blue[3]
        # 近地点参数（Argument of periapsis）： 0 <= ω < 360°
        orbit_element_blue[4] = orbit_element_blue[4] / 360
        # 平均近点角（Mean anomaly at epoch）： -180° <= M <= 180°
        orbit_element_blue[5] = orbit_element_blue[5] / 360
        pos_blue = self.pos[blue_name]/self.orbit_alt
        vel_blue = self.vel[blue_name]

        obs = np.concatenate((orbit_element_red, pos_red, vel_red, orbit_element_blue, pos_blue, vel_blue))

        return obs

    # 调用：update_state
    def extract_info(self, data):
        pos = {}
        vel = {}
        sun_pos = self.extract_xyz(data["sun"])
        # 提取所有卫星的位置信息
        for satellite, value in data["satellites"].items():
            pos[satellite] = self.extract_xyz(value["pos"]) / 1000
            vel[satellite] = self.extract_xyz(value["vel"]) / 1000  # 转化为km单位
        return sun_pos, pos, vel

    # 更新卫星状态(每一步更新一次)
    # 调用：step函数、reset函数
    # 参数说明：data是指从仿真软件读取出来的数据
    def update_state(self, data):

        if data is None:
            print("输入数据有错误")
            return 0
        self.last_pos = copy.deepcopy(self.pos)
        self.sun_pos, self.pos, self.vel = self.extract_info(data)

        for name in self.satellite:
            self.orbital_ele[name] = self.pv_to_orbital_elements(self.pos[name], self.vel[name])

    # 输入的是numpy向量
    def pv_to_orbital_elements(self, pos, vel):

        I = np.array([1, 0, 0])
        J = np.array([0, 1, 0])
        K = np.array([0, 0, 1])
        mu = 398600
        rvec = pos
        vvec = vel
        r = np.linalg.norm(rvec)
        v = np.linalg.norm(vvec)
        hvec = np.cross(rvec, vvec)
        h = np.linalg.norm(hvec)
        energy = .5 * v ** 2 - mu / r
        a = -mu / (2 * energy)
        p = h ** 2 / mu
        hhat = hvec / h
        i = math.acos(np.dot(hhat, K))
        rhat = rvec / r
        evec = np.cross(vvec, hvec) / mu - rhat
        e = np.linalg.norm(evec)
        ehat = evec / e
        if np.dot(rvec, vvec) >= 0:
            if abs(np.dot(ehat, rhat)-1) < 0.001:
                f = math.acos(1)
            elif abs(np.dot(ehat, rhat)+1) < 0.001:
                f = math.acos(-1)
            else:
                f = math.acos(np.dot(ehat, rhat))

        elif np.dot(rvec, vvec) < 0:
            if abs(np.dot(ehat, rhat) - 1) < 0.00001:
                f = 2*math.pi - math.acos(1)
            elif abs(np.dot(ehat, rhat) + 1) < 0.00001:
                f = 2*math.pi - math.acos(-1)
            else:
                f = 2 * math.pi - math.acos(np.dot(ehat, rhat))

        ph = np.cross(K, hvec)
        phn = ((ph[0] ** 2) + (ph[1] ** 2) + (ph[2] ** 2)) ** (1 / 2)
        nhat = np.cross(K, hvec) / phn

        if np.dot(ehat, K) >= 0:
            if abs(np.dot(ehat, K) - 1) < 0.00001:
                w = math.acos(1)
            elif abs(np.dot(ehat, rhat) + 1) < 0.00001:
                w = math.acos(-1)
            else:
                w = math.acos(np.dot(nhat, ehat))

        elif np.dot(ehat, K) < 0:
            if abs(np.dot(ehat, K) - 1) < 0.00001:
                w = 2 * math.pi - math.acos(1)
            elif abs(np.dot(ehat, rhat) + 1) < 0.00001:
                w = 2 * math.pi - math.acos(-1)
            else:
                w = 2 * math.pi - math.acos(np.dot(nhat, ehat))
        omega = math.atan2(nhat[1], nhat[0])
        if omega < 0:
            omega += 2 * np.pi
        # print('半长轴:', a)
        # print('真进点角:', np.degrees(f))
        # print('偏心率:', e)
        # print('轨道倾角:', np.degrees(i))
        # print('近地点幅角:', np.degrees(w))
        # print('升交点赤经:', np.degrees(omega))
        return np.array([np.degrees(i), np.degrees(omega), a, e, np.degrees(f), np.degrees(w)])

    # [轨道倾角， 升交点赤经， 半长轴， 偏心率， 真近点角， 近地点幅角]

    def orbital_elements_to_pv(self, orbital_elements):
        # 解包各个值
        inclination = orbital_elements[0]
        inclination = np.radians(inclination)

        right_ascension_of_the_ascending_node = orbital_elements[1]
        right_ascension_of_the_ascending_node = np.radians(right_ascension_of_the_ascending_node)

        semimajor_axis = orbital_elements[2]
        eccentricity = orbital_elements[3]

        true_anomaly = orbital_elements[4]
        true_anomaly = np.radians(true_anomaly)

        argument_of_periapsis = orbital_elements[5]
        argument_of_periapsis = np.radians(argument_of_periapsis)
        # 轨道地心坐标系的单位向量在惯性坐标系下的坐标
        e1 = np.array([np.cos(right_ascension_of_the_ascending_node) * np.cos(argument_of_periapsis) - np.sin(
            right_ascension_of_the_ascending_node) * np.sin(argument_of_periapsis) * np.cos(inclination),
                       np.sin(right_ascension_of_the_ascending_node) * np.cos(argument_of_periapsis) + np.cos(
                           right_ascension_of_the_ascending_node) * np.sin(argument_of_periapsis) * np.cos(inclination),
                       np.sin(argument_of_periapsis) * np.sin(inclination)])

        e2 = np.array([-np.cos(right_ascension_of_the_ascending_node) * np.sin(argument_of_periapsis) - np.sin(
            right_ascension_of_the_ascending_node) * np.cos(argument_of_periapsis) * np.cos(inclination),
                       -np.sin(right_ascension_of_the_ascending_node) * np.sin(argument_of_periapsis) + np.cos(
                           right_ascension_of_the_ascending_node) * np.cos(argument_of_periapsis) * np.cos(inclination),
                       np.cos(argument_of_periapsis) * np.sin(inclination)])

        # 计算卫星的位置矢量
        u = 398600.44
        p = semimajor_axis * (1 - eccentricity ** 2)
        pos_mag = p / (1 + eccentricity * np.cos(true_anomaly))
        pos = pos_mag * np.cos(true_anomaly) * e1 + pos_mag * np.sin(true_anomaly) * e2

        # 计算速度矢量
        vel = -(np.sin(true_anomaly) * e1 - (np.cos(true_anomaly) + eccentricity) * e2) * (u / p) ** 0.5
        return pos, vel

    def extract_xyz(self, dict_in):
        return np.array([value for name, value in dict_in.items()])

    def get_data(self):
        # 如果获取信息成功，就跳出循环
        get_success_flag = False
        error_info = []
        data = self.af.GetSimData().message
        # 转化为字典类型
        while not get_success_flag:
            if len(error_info) == 5:
                print("反复出错，跳出循环！！！")
                return None
            try:
                data = json.loads(data)
                t = data["satellites"]["r0"]["pos"]
                # 出错，此句不执行
                get_success_flag = True
            except:
                error_info.append(data)
                get_success_flag = False
                data = self.af.GetSimData().message
                print(f"第{len(error_info)}次信息获取失败")
        return data

    # axis:旋转轴,numpy
    # theta:逆时针旋转角度,输入的单位是角度制
    # vector: 所要旋转的向量，numy数组
    def rotate_around_axis(self, axis, theta, vector):
        a_x, a_y, a_z = axis[0], axis[1], axis[2]
        theta = np.radians(theta)
        rotate_matrix = np.array([
            [np.cos(theta) + (1 - np.cos(theta)) * (a_x ** 2), a_x * a_y * (1 - np.cos(theta)) - a_z * np.sin(theta),
             a_x * a_z * (1 - np.cos(theta)) + a_y * np.sin(theta)],
            [a_y * a_x * (1 - np.cos(theta)) + a_z * np.sin(theta), np.cos(theta) + (1 - np.cos(theta)) * (a_y ** 2),
             a_y * a_z * (1 - np.cos(theta)) - a_x * np.sin(theta)],
            [a_z * a_x * (1 - np.cos(theta)) - a_y * np.sin(theta),
             a_z * a_y * (1 - np.cos(theta)) + a_x * np.sin(theta), np.cos(theta) + (1 - np.cos(theta)) * (a_z ** 2)]
        ])
        vector_trans = np.dot(rotate_matrix, vector)
        return vector_trans

    # 为了和强化学习接口对接，额外添加的一些方法
    def observation_space(self, agent_name):
        return (24,)

    def action_space(self, agent_name):
        return (3,)

    def action_random(self, name):
        action1 = np.random.uniform(-1, 1)*self.delta_v_implusive
        action2 = np.random.uniform(-1, 1)*self.delta_v_implusive
        action3 = np.random.uniform(-1, 1)*self.delta_v_implusive
        return np.array([action1, action2, action3])

    def z_score_f(self, data):
        # 计算均值
        mean = np.mean(data)
        # 计算标准差
        std_dev = np.std(data)
        # Z-score 标准化
        z_scores = (data - mean) / std_dev
        return z_scores

    def generate_random_point_on_sphere(self, radius):

        # 使用球坐标生成随机点，球坐标系的三个参数为r,φ，θ
        phi = np.random.uniform(0, 2 * np.pi)  # Phi角在0到2π之间
        theta = np.arccos(np.random.uniform(-1, 1))  # Theta角在0到π之间

        # 将球坐标转换为笛卡尔坐标
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        return np.array([x, y, z])

    def CW_to_Inertial(self, pos_main, vel_main, pos_cw, vel_cw):
        # 坐标转换
        R = np.array([
            np.cross(np.cross(pos_main, vel_main), pos_main) / np.linalg.norm(np.cross(np.cross(pos_main, vel_main), pos_main)),
            np.cross(vel_main, pos_main) / np.linalg.norm(np.cross(vel_main, pos_main)),
            -pos_main / np.linalg.norm(pos_main)
        ])

        h = np.linalg.norm(np.cross(pos_main, vel_main))
        w = np.array([0, 0, h / (np.linalg.norm(pos_main) ** 2)])
        pos_inertial = np.dot(pos_cw,R) + pos_main
        vel_inertial = np.dot((vel_cw + np.cross(w,pos_cw)),R) + vel_main
        return pos_inertial, vel_inertial