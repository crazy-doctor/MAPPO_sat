# 系统包
import json
import numpy as np
import copy
from abc import ABC, abstractclassmethod
import os
import time
# 自建包
from AfSimLib.AfSim import *
import Env.environment_ as environment_
from Env.information import sc_info

class Sim_Abstract(ABC): #参考性使用自己的引擎推演，其他卫星使用公司给的引擎
    # 外界无需给任何信息，一切交给随机，只需要给外界提供一个接口即可
    @abstractclassmethod
    def Reset_Env(self):
        pass

    # 我只需要告诉引擎，每个卫星需要执行什么动作就行，而引擎提供的只是一个调用状态的接口，以保证外界能调用这个接口
    @abstractclassmethod
    def Step_Env(self, delta_v):
        pass

# 实现1：采用我自己的推理引擎
class Self_Sim(Sim_Abstract):
    def __init__(self, args=None, red_sat=[], blue_sat=[]):
        self.args = args
        self.red_sat = red_sat
        self.blue_sat = blue_sat
        self.ref_sat = ["main_sat"]

        self.af = environment_.Afsim(red_sat=self.red_sat,blue_sat=self.blue_sat,
                                     main_sat=self.ref_sat)

        self.inf = sc_info(red_name=self.red_sat, blue_name=self.blue_sat, main_sat=self.ref_sat,args=args)

    def extract_xyz(self, dict_in):
        return np.array(dict_in)

    def extract_info(self, data):
        pos = {}
        vel = {}
        sun_pos = self.extract_xyz(data["sun"])

        for satellite, value in data["satellites"].items():
            pos[satellite] = self.extract_xyz(value["pos"])
            vel[satellite] = self.extract_xyz(value["vel"])  # 转化为km单位
        return sun_pos, pos, vel

    def Reset_Env(self):
        self.inf.init_state()
        # self.inf.debug_init()
        for name in self.red_sat+self.blue_sat+self.ref_sat:
            # 发送指令到引擎
            self.af.SetLocationAndvcelocitECI(name, self.inf.pos[name][0], self.inf.pos[name][1], self.inf.pos[name][2],
                                              self.inf.vel[name][0], self.inf.vel[name][1], self.inf.vel[name][2]) #km

    def Step_Env(self,delta_v_dict:dict):
        for name in self.red_sat+self.blue_sat+self.ref_sat:
            # 这个delta_v设为CW坐标系下的变速度
            delta_v = delta_v_dict[name]
            # 将惯性坐标系下的变速度输入
            self.af.SetSatDeltaV(name, delta_v[0], delta_v[1], delta_v[2])


        self.af.SetNumOfStepToAdvance(self.args.step_time)
        sun_pos, pos, vel = self.extract_info(self.af.get_data())
        self.inf.update_data(sun_pos,pos, vel)

# 实现2：采用混合式推理引擎
class Mixed_Sim(Sim_Abstract):
    def __init__(self,args=None,mission_server_path=r"",red_sat=[], blue_sat=[]):
        self.args = args
        self.red_sat = ["r0", "r1", "r2"]
        self.blue_sat = ["b0", "b1", "b2"]
        self.ref_sat = ["main_sat"]

        # 打开控制行,
        try:
            exit_code = os.system('taskkill /f /im %s' % 'mission_server.exe')
            time.sleep(0.1)
            os.startfile(args.mission_server_path)
            time.sleep(0.1)
        except:
            raise Exception("初始化环境时,mission_server命令行打开失败")

        self.MainSat_Ocursion = environment_.Afsim(red_sat=[],blue_sat=[],main_sat=["main_sat"])

        try:
            self.af = AfSimServer("127.0.0.1", 50051)
            if not args.visual_flag:
                self.af.OpenScenario("startup.txt", "demos/805-ai")
        except:
            raise Exception("在获得操作句柄时，发生异常")

        self.inf = sc_info(red_name=self.red_sat, blue_name=self.blue_sat, main_sat=self.ref_sat)

        self.lastTimeStamp = -1
        self.Current_Time = -1

    def get_data(self, data_str):
        # 如果获取信息成功，就跳出循环
        get_success_flag = False
        error_info = []
        data = None
        # 转化为字典类型
        while not get_success_flag:
            if len(error_info) == 5:
                print("反复出错，跳出循环！！！")
                return None
            try:
                data = json.loads(data_str)
                t = data["satellites"]["r0"]["pos"]
                # 出错，此句不执行
                get_success_flag = True
            except:
                error_info.append(data_str)
                get_success_flag = False
                data = self.af.GetSimData().message
                print(f"Afsim第{len(error_info)}次信息获取失败")
        return data

    def Reset_Env(self):
        self.inf.init_state(self.args.orbit_alt)
        self.lastTimeStamp = -1
        self.Current_Time = -1
        self.done_step = {name: 0 for name in self.red_sat}
        # 流程2.重置引擎
        while self.af.GetNumOfStepToAdvance() != 0:  # 不加这句调用self.af.RestartSimAndWait()好像会报错
            pass
        self.af.RestartSimAndWait()
        while self.af.GetNumOfStepToAdvance() != 0:  # 如果不加这句,状态无法更新，一般都是卡在这句，mission_server.exe关闭
            pass
        self.Current_Time = self.af.GetSimData().simTime
        while abs(self.Current_Time - self.lastTimeStamp) < 0.001:
            self.Current_Time = self.af.GetSimData().simTime
        self.lastTimeStamp = self.Current_Time

        for name in ["main_sat"]:
            self.MainSat_Ocursion.SetLocationAndvcelocitECI(name, self.inf.pos[name][0], self.inf.pos[name][1],
                                                            self.inf.pos[name][2],
                                                            self.inf.vel[name][0], self.inf.vel[name][1], self.inf.vel[name][2]) #km
        for name in self.red_sat+self.blue_sat:

            pos, vel = copy.deepcopy(self.inf.pos[name]), copy.deepcopy(self.inf.vel[name])
            pos *= 1000
            vel *= 1000  # 将单位从km转化为m
            # 4. 发送指令到引擎
            self.af.SetLocationAndvcelocitECI(name, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]) #m

        while self.af.GetNumOfStepToAdvance() != 0:
            pass
        # 这里如果不推进一步，状态就无法更新，默认返回想定中卫星的位置和速度
        self.af.SetNumOfStepToAdvance(1)
        while self.af.GetNumOfStepToAdvance() != 0:
            pass
        self.Current_Time = self.af.GetSimData().simTime
        while abs(self.lastTimeStamp - self.Current_Time) < 0.001:
            self.Current_Time = self.af.GetSimData().simTime
        self.lastTimeStamp = self.Current_Time

    def extract_xyz(self, dict_in):
        return np.array([value for name, value in dict_in.items()])

    def extract_info(self, data):
        pos = {}
        vel = {}
        sun_pos = self.extract_xyz(data["sun"])
        # 提取所有卫星的位置信息
        for satellite, value in data["satellites"].items():
            pos[satellite] = self.extract_xyz(value["pos"]) / 1000
            vel[satellite] = self.extract_xyz(value["vel"]) / 1000  # 转化为km单位
        return sun_pos, pos, vel

    def extract_mainsat_xyz(self, dict_in):
        return np.array(dict_in)

    def extract_mainsat_info(self, data):
        pos = {}
        vel = {}
        sun_pos = self.extract_mainsat_xyz(data["sun"])

        for satellite, value in data["satellites"].items():
            pos[satellite] = self.extract_mainsat_xyz(value["pos"])
            vel[satellite] = self.extract_mainsat_xyz(value["vel"])  # 转化为km单位
        return sun_pos, pos, vel

    def Step_Env(self,delta_v_dict:dict): #输入单位是km
        while self.af.GetNumOfStepToAdvance() != 0:
            pass
        for name in self.red_sat+self.blue_sat:
            delata_v = delta_v_dict[name]*1000
            self.af.SetSatDeltaV(name, delata_v[0], delata_v[1], delata_v[2])
        while self.af.GetNumOfStepToAdvance() != 0:
            pass
        # 推进！推进推进！冲TNND
        self.af.SetNumOfStepToAdvance(self.args.step_time)

        while self.af.GetNumOfStepToAdvance() != 0:
            pass
        self.Current_Time = self.af.GetSimData().simTime

        while abs(self.Current_Time - self.lastTimeStamp) < 0.001:
            self.Current_Time = self.af.GetSimData().simTime
        # 动作完成后立即更新，从而进入下一步
        self.lastTimeStamp = self.Current_Time
        self.MainSat_Ocursion.SetNumOfStepToAdvance(self.args.step_time)
        # 从可视化引擎中读取信息
        sun_pos, pos, vel = self.extract_info(self.get_data(self.af.GetSimData().message))
        # 在sun_pos, pos, vel中添加入main_sat的信息
        sun_mainsat_pos, pos_mainsat, vel_mainsat = \
            self.extract_mainsat_info(self.MainSat_Ocursion.get_data())

        pos_mix = copy.deepcopy({**pos, **pos_mainsat})
        vel_mix = copy.deepcopy({**vel, **vel_mainsat})

        self.inf.update_data(sun_pos, pos_mix, vel_mix)
