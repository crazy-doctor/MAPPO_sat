import numpy as np
import copy
import multiprocessing
# 单个卫星的状态
class satellite:
    def __init__(self,name):
        self.name = name
        self.pos = None
        self.vel = None
    # 1.设置卫星的位置和速度
    def SetSatSate(self,pos,vel):
        self.pos = pos
        self.vel = vel
    # 2.轨道递推,
    # 描写递推函数
    def StateEq(self,t, RV):  ##状态方程
        mu = 398600
        Re = 6378.137  # 地球半径
        J2 = 0.00108263  # J2项
        J2 = 0  # J2项
        x = RV[0]
        y = RV[1]
        z = RV[2]
        vx = RV[3]
        vy = RV[4]
        vz = RV[5]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        gx = -mu * x / r ** 3
        gy = -mu * y / r ** 3
        gz = -mu * z / r ** 3
        dgx = -3 / 2 * J2 * Re ** 2 * mu * x / r ** 5 * (1 - 5 * (z / r) ** 2)
        dgy = -3 / 2 * J2 * Re ** 2 * mu * y / r ** 5 * (1 - 5 * (z / r) ** 2)
        dgz = -3 / 2 * J2 * Re ** 2 * mu * z / r ** 5 * (3 - 5 * (z / r) ** 2)
        f = np.array([vx, vy, vz, gx + dgx, gy + dgy, gz + dgz])
        return f

    def RungeKutta(self,t0, r0, h):  ## 龙格库塔算法
        K1 = self.StateEq(t0, r0)
        K2 = self.StateEq(t0 + 2 / h, r0 + h / 2 * K1)
        K3 = self.StateEq(t0 + h, r0 + h / 2 * K2)
        K4 = self.StateEq(t0 + h, r0 + h * K3)
        r1 = r0 + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return r1

        # t:递推多长时间 pos0初始位置 vel0初始速度
    def orbit_recursion(self,t):
        h = 1  # 步长
        N = t  # 外推时间
        # 初始位置和速度
        RV0 = np.concatenate((self.pos, self.vel))
        data = np.array(np.zeros((np.int32(N / h + 1), 6)))
        data[0] = RV0
        for i in range(np.int32(N / h)):
            data[i + 1] = self.RungeKutta(i * h, RV0, h)
            RV0 = data[i + 1]
        R_tf = copy.deepcopy(data[-1][0:3])
        V_tf = copy.deepcopy(data[-1][3:])
        self.pos = R_tf
        self.vel = V_tf
        return self

    def SetDeltaV(self,delta_v):
        self.vel = self.vel + delta_v

    # 3.变轨状态
    def get_sat_data(self):
        info_dict = {"pos": self.pos,"vel": self.vel}
        return info_dict


# 场景创建
class Afsim:
    def __init__(self,red_sat,blue_sat,main_sat=[]):

        self.ref_main = main_sat
        self.sat = red_sat + blue_sat + self.ref_main
        self.satellite = {sat_id: satellite(sat_id) for sat_id in self.sat}
        self.pos = None
        self.vel = None


        self.pool = multiprocessing.Pool(processes=min(len(self.sat), multiprocessing.cpu_count() - 5))
    # 1.卫星位置速度初始化
    def SetLocationAndvcelocitECI(self, name, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z):
        pos = np.zeros(3)
        vel = np.zeros(3)
        pos[0] = pos_x #km
        pos[1] = pos_y
        pos[2] = pos_z
        vel[0] = vel_x
        vel[1] = vel_y
        vel[2] = vel_z
        self.satellite[name].SetSatSate(pos=pos,vel=vel)
    # 2.action
    def SetSatDeltaV(self, name, delata_vx, delata_vy, delata_vz):
        delta_vel = np.zeros(3)
        delta_vel[0] = delata_vx
        delta_vel[1] = delata_vy
        delta_vel[2] = delata_vz
        self.satellite[name].SetDeltaV(delta_vel)

    def SetNumOfStepToAdvance(self, t):
        async_results = [self.pool.apply_async(self.satellite[sat_id].orbit_recursion, args=(t,)) for sat_id in self.sat]
        for res in async_results:
            sat = res.get()
            self.satellite[sat.name] = sat
        # for sat in self.sat:
        #     self.satellite[sat].orbit_recursion(t)

    # 3.状态更新
    def get_data(self):
        info_every_sat = {}
        for sat_id in self.sat:
            info_every_sat[sat_id] = self.satellite[sat_id].get_sat_data()
        sun_pos = np.zeros(3)
        return {"satellites":info_every_sat, "sun":sun_pos}
