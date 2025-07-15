import numpy as np
import copy
import math
from Tool.astro_tool import Tool
import random


class sc_info(Tool):
    def __init__(self,red_name, blue_name, main_sat, args):
        # 列表类型，装载各卫星名称
        self.red_sat = red_name
        self.blue_sat = blue_name
        self.ref_sat= main_sat
        self.args = args
        # 字典类型

        # 卫星在惯性坐标系下的位置和速度
        self.pos = {}
        self.vel = {}
        # 卫星在轨道坐标系下的位置和速度
        self.pos_cw = {}
        self.vel_cw = {}
        # 卫星的轨道六根数
        self.orbital_ele = {}
        # 太阳位置
        self.sun_pos = {}

        # 上一时刻，惯性坐标位置和速度
        self.pos_last = {}
        self.vel_last = {}
        # 上一时刻，轨道坐标系位置和速度
        self.pos_cw_last = {}
        self.vel_cw_last = {}
        # 上一时刻，轨道六根数
        self.orbital_ele_last = {}
        # 上一时刻，太阳位置
        self.sun_pos_last = {}


    def init_state(self):
        self.time = 0
        self.random_init(self.args.orbit_alt) # todo:初始化卫星初始状态,初始化时，确保卫星不会小于安全距离死亡
        # self.debug_init()

    def random_init(self, orbit_alt):
        ref_sat_ele = np.array([np.random.uniform(0, 3), np.random.uniform(0, 300), orbit_alt, 0,
                                     np.random.uniform(0, 300), np.random.uniform(0, 300)])  # 初始化参考卫星的轨道六根数
        r = np.random.uniform(200, 220)

        theta = 2 * math.asin(r / (42157 * 2)) * 180 / math.pi
        # single = random.choice([-1, 1])
        single = 1
        red_ref_ele = ref_sat_ele + single*np.array([0,0,0,0,theta,0])
        blue_ref_pos, blue_ref_vel = Tool.orbital_elements_to_pv(self, ref_sat_ele)  # 给蓝方设置一个参考点，三颗卫星以这个参考点为基准
        # red_ref_pos, red_ref_vel = Tool.orbital_elements_to_pv(self, red_ref_ele)

        # 设置各个卫星的轨道六根数
        for sat_id in self.red_sat + self.blue_sat + self.ref_sat:
            if sat_id == "main_sat":
                self.orbital_ele[sat_id] = ref_sat_ele
                self.pos[sat_id], self.vel[sat_id] = Tool.orbital_elements_to_pv(self, self.orbital_ele[sat_id])
                self.pos_cw[sat_id], self.vel_cw[sat_id] = Tool.Inertial_to_CW(self, blue_ref_pos, blue_ref_vel,
                                                                       self.pos[sat_id], self.vel[sat_id])
            elif sat_id[0] == "b":
                dis = 5*int(sat_id[1])
                theta = 2 * math.asin(dis / (42157 * 2)) * 180 / math.pi
                ele_tmp = ref_sat_ele + single * np.array([0, 0, 0, 0, theta, 0])
                self.orbital_ele[sat_id] = ele_tmp
                self.pos[sat_id], self.vel[sat_id] = Tool.orbital_elements_to_pv(self, self.orbital_ele[sat_id])
                self.pos_cw[sat_id], self.vel_cw[sat_id] = Tool.Inertial_to_CW(self, blue_ref_pos, blue_ref_vel,
                                                                       self.pos[sat_id], self.vel[sat_id])

            elif sat_id[0] == "r":  # 红方卫星
                dis = 5*int(sat_id[1])
                theta = 2 * math.asin(dis / (42157 * 2)) * 180 / math.pi
                ele_tmp = red_ref_ele + single * np.array([0, 0, 0, 0, theta, 0])
                self.orbital_ele[sat_id] = ele_tmp
                self.pos[sat_id], self.vel[sat_id] = Tool.orbital_elements_to_pv(self, self.orbital_ele[sat_id])
                self.pos_cw[sat_id], self.vel_cw[sat_id] = Tool.Inertial_to_CW(self, blue_ref_pos, blue_ref_vel,
                                                                       self.pos[sat_id], self.vel[sat_id])

            else:
                raise Exception("装载着卫星名字的列表中出现了意料之外的名字")

        # 设置卫星的惯性坐标系
        for sat_id in self.red_sat + self.blue_sat + self.ref_sat:
            self.pos[sat_id], self.vel[sat_id] = Tool.CW_to_Inertial(self, pos_main=blue_ref_pos,
                                                                     vel_main=blue_ref_vel,
                                                                     pos_cw=self.pos_cw[sat_id],
                                                                     vel_cw=self.vel_cw[sat_id])
            self.orbital_ele[sat_id] = Tool.pv_to_orbital_elements(self, pos=self.pos[sat_id], vel=self.vel[sat_id])
        self.pos_last = copy.deepcopy(self.pos)

    #
    def debug_init(self):
        orbital_elements = np.array([3, 10, 42157, 0, 10, 10])
        blue_ref_pos, blue_ref_vel = Tool.orbital_elements_to_pv(self, orbital_elements)  # 给蓝方设置一个参考点，三颗卫星以这个参考点为基准
        self.pos_cw["main_sat"], self.vel_cw["main_sat"] = np.zeros(3), np.zeros(3)
        for sat_id in self.blue_sat:
            self.pos_cw[sat_id] = np.array([3,-2,1])
            self.vel_cw[sat_id] = np.zeros(3)
        for sat_id in self.red_sat:
            self.pos_cw[sat_id] = np.array([130,130,130])
            self.vel_cw[sat_id] = np.zeros(3)

        for sat_id in self.red_sat + self.blue_sat + self.ref_sat:
            self.pos[sat_id], self.vel[sat_id] = Tool.CW_to_Inertial(self, pos_main=blue_ref_pos,
                                                                     vel_main=blue_ref_vel,
                                                                     pos_cw=self.pos_cw[sat_id],
                                                                     vel_cw=self.vel_cw[sat_id])
            self.orbital_ele[sat_id] = Tool.pv_to_orbital_elements(self, pos=self.pos[sat_id], vel=self.vel[sat_id])
        self.pos_last = copy.deepcopy(self.pos)



    def update_data(self, sun_pos, pos, vel):

        self.time += 1

        self.pos_last = copy.deepcopy(self.pos)
        self.vel_last = copy.deepcopy(self.vel)

        # 上一时刻，轨道坐标系位置和速度
        self.pos_cw_last = copy.deepcopy(self.pos_cw)
        self.vel_cw_last = copy.deepcopy(self.vel_cw)
        # 上一时刻，轨道六根数
        self.orbital_ele_last = copy.deepcopy(self.orbital_ele)
        # 上一时刻，太阳位置
        self.sun_pos_last = copy.deepcopy(self.sun_pos)

        # 1.更新位置和速度
        self.sun_pos, self.pos, self.vel = sun_pos, pos, vel

        # 2.计算在cw坐标系下的坐标
        pos_main = self.pos["main_sat"]
        vel_main = self.vel["main_sat"]
        for sat_id in self.red_sat+self.blue_sat+self.ref_sat:
            self.pos_cw[sat_id], self.vel_cw[sat_id] = self.Inertial_to_CW(pos_main=pos_main, vel_main=vel_main,
                                                           pos_sat=self.pos[sat_id], vel_sat=self.vel[sat_id])
        # 3.轨道六根数计算
        for name in self.red_sat+self.blue_sat+self.ref_sat:
            self.orbital_ele[name] = self.pv_to_orbital_elements(self.pos[name], self.vel[name])


    def dis_sat(self,name1, name2):
        return np.linalg.norm(self.pos[name1] - self.pos[name2])


