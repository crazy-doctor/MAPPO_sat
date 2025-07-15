import sys
import time
from concurrent.futures import thread

import grpc
from AfSimLib import AfSim_pb2, AfSim_pb2_grpc
from AfSimLib.AfSim_Enum import EngineState, SimType


class AfSimServer(object):
    def __init__(self, afsimIP="127.0.0.1", afsimPort=50051):
        self.timeStamp = 0
        self.channel = grpc.insecure_channel(afsimIP + ":" + str(afsimPort),options=[('grpc.max_receive_message_length', 10 * 1024 * 1024)])#设置消息大小限制为10M
        self.server = AfSim_pb2_grpc.AfSimGrpcServiceStub(self.channel)

    # 判断是否执行到指定时间
    def IsFinishedStepToAdvance(self, finishTime):
        while True:
            epsilon = 1e-3  # 设定一个小的容差值，例如1e-9
            while self.GetNumOfStepToAdvance() != 0:
                pass
            data = self.GetSimData()
            mistiming = data.simTime - finishTime
            if mistiming < -epsilon:
                continue
            elif mistiming > epsilon:
                print("finishTime错误，仿真时间已经超过此时间. finishTime：" + str(finishTime) + ", simTime:" + str(data.simTime))
                return False
            return True

    """Changes the simulation clock_rate. This only has effect in realtime mode."""
    def SetClockRate(self, clockRate=1):
        self.RunScriptCmd("WsfSimulation.SetClockRate({});".format(clockRate));

    """删除平台"""
    def DeletePlatform(self, platformName):
        self.RunScriptCmd("WsfSimulation.DeletePlatform(\"{}\");".format(platformName));

    """平台位置修改为指定坐标(纬度、经度、高度) 单位：度、米"""
    def SetLocation(self, platformName, lat, lon, alt):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.SetLocation({},{},{});".format(platformName, lat, lon, alt));

    """平台位置修改为指定坐标(纬度、经度、高度) 单位：度、米"""
    def SetLocationAndvcelocitECI(self, platformName, x, y, z, u, v, w):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          +"WsfSpaceMover mover = (WsfSpaceMover)plat.Mover();"
                          +"Vec3 locationECI = Vec3.Construct({},{},{});".format(x, y, z)
                          +"Vec3 vcelocityECI = Vec3.Construct({},{},{});".format(u, v, w)
                          +"mover.SetOrbit(locationECI, vcelocityECI);")

    """平台机动至指定坐标(纬度、经度、高度) 单位：度、米"""
    def GotoLocation(self, platformName, lat, lon, alt=sys.float_info.max):
        if alt == sys.float_info.max:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.GoToLocation({},{});".format(platformName, lat, lon));
        else:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.GoToLocation({},{},{});".format(platformName, lat, lon, alt));

    """平台机动至指定高度(米）；爬升/下降速率（米/秒）；是否保持当前路径"""
    def GoToAltitude(self, platformName, alt, altRateOfChange=sys.float_info.max, keepRoute=True):
        if altRateOfChange == sys.float_info.max:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.GoToAltitude({});".format(platformName, alt));
        else:
            if keepRoute == True:
                self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                                  "plat.GoToAltitude({},{},true);".format(platformName, alt, altRateOfChange));
            else:
                self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                                  "plat.GoToAltitude({},{},false);".format(platformName, alt, altRateOfChange));

    """平台速度提升/降低到指定数值(米/秒）；加速度（米/秒²）；是否保持当前路径"""
    def GoToSpeed(self, platformName, speed, linearAccel=sys.float_info.max, keepRoute=True):
        if linearAccel == sys.float_info.max:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.GoToSpeed({});".format(platformName, speed));
        else:
            if keepRoute == True:
                self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                                  "plat.GoToSpeed({},{},true);".format(platformName, speed, linearAccel));
            else:
                self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                                  "plat.GoToSpeed({},{},false);".format(platformName, speed, linearAccel));

    """平台转向指定绝对航向[0,360]；角加速度（度/秒²）"""
    def TurnToHeading(self, platformName, heading, radicalAccel=sys.float_info.max):
        if radicalAccel == sys.float_info.max:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.TurnToHeading({});".format(platformName, heading));
        else:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.TurnToHeading({},{});".format(platformName, heading, radicalAccel));

    """平台转向指定相对航向[-180,180]；角加速度（度/秒²）"""
    def TurnToRelativeHeading(self, platformName, heading, radicalAccel=sys.float_info.max):
        if radicalAccel == sys.float_info.max:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.TurnToRelativeHeading({});".format(platformName, heading));
        else:
            self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                              "plat.TurnToRelativeHeading({},{});".format(platformName, heading, radicalAccel));

    """设置卫星变轨，输入参数为速度增量，需确保速度增量为有效值，指定的卫星存在"""
    def SetSatDeltaV(self, satName, deltaX, deltaY, deltaZ):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(satName)
                          +"WsfSpaceMover mover = (WsfSpaceMover)plat.Mover();"
                          +"Vec3 deltaV = Vec3.Construct({},{},{});".format(deltaX, deltaY,deltaZ)
                          +"WsfDeltaV_Maneuver mvr = WsfDeltaV_Maneuver.Construct(WsfOrbitalEventCondition.NONE(),deltaV,WsfOrbitalReferenceFrame.INERTIAL());"
                          +"mover.ExecuteManeuver(mvr);");

    """设置传感器Yaw角度：角度制"""
    def SetSensorYaw(self, platformName, sensorName, val):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          + "plat.Sensor(\"{}\").SetYaw({});".format(sensorName,val));

    """设置传感器Pitch角度：角度制"""
    def SetSensorPitch(self, platformName, sensorName, val):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          + "plat.Sensor(\"{}\").SetPitch({});".format(sensorName, val));

    """设置传感器Roll角度：角度制"""
    def SetSensorRoll(self, platformName, sensorName, val):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          + "plat.Sensor(\"{}\").SetRoll({});".format(sensorName, val));

    """设置DXN武器Yaw角度：角度制"""
    def SetImplicitWeaponYaw(self, platformName, weaponName, val):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          + "plat.Weapon(\"{}\").SetYaw({});".format(weaponName, val));

    """设置DXN武器Pitch角度：角度制"""
    def SetImplicitWeaponPitch(self, platformName, weaponName, val):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          + "plat.Weapon(\"{}\").SetPitch({});".format(weaponName, val));

    """设置DXN武器Roll角度：角度制"""
    def SetImplicitWeaponRoll(self, platformName, weaponName, val):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(platformName)
                          + "plat.Weapon(\"{}\").SetRoll({});".format(weaponName, val));

    """设置卫星姿态：ECI Psi 角度制"""
    def SetSatOrientationECI(self, satName, PsiECI, ThetaECI, PhiECI):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");".format(satName)
                          + "WsfSpaceMover mover = (WsfSpaceMover)plat.Mover();"
                          + "mover.SetOrientationECI({},{},{});".format(PsiECI, ThetaECI, PhiECI));

    """通信设备开机"""
    def TurnCommOn(self, platformName, commName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnCommOn(\"{}\");".format(platformName, commName));

    """通信设备关机"""
    def TurnCommOff(self, platformName, commName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnCommOff(\"{}\");".format(platformName, commName));

    """传感器开机"""
    def TurnSensorOn(self, platformName, sensorName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnSensorOn(\"{}\");".format(platformName, sensorName));

    """传感器关机"""
    def TurnSensorOff(self, platformName, sensorName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnSensorOff(\"{}\");".format(platformName, sensorName));

    """处理器开机"""
    def TurnProcessorOn(self, platformName, processorName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnProcessorOn(\"{}\");".format(platformName, processorName));

    """处理器关机"""
    def TurnProcessorOff(self, platformName, processorName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnProcessorOff(\"{}\");".format(platformName, processorName));

    """激活路径"""
    def TurnRouterOn(self, platformName, routerName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnRouterOn(\"{}\");".format(platformName, routerName));

    """关闭路径"""
    def TurnRouterOff(self, platformName, routerName):
        self.RunScriptCmd("WsfPlatform plat = WsfSimulation.FindPlatform(\"{}\");"
                          "plat.TurnRouterOff(\"{}\");".format(platformName, routerName));

        """暂停仿真"""
    def PauseSim(self):
        self.RunScriptCmd("WsfSimulation.Pause();");

    """结束仿真"""
    def TerminateSim(self):
        self.RunScriptCmd("WsfSimulation.Terminate();");

    """仿真重新开始"""
    def RestartSim(self):
        self.RunScriptCmd("WsfSimulation.RequestReset();");

    # 重新开始仿真并等待想定重新加载完成的函数
    def RestartSimAndWait(self):
        self.RestartSim()
        restarted = False
        while (restarted == False):
            try:
                state = self.GetSimulationState()
                #print("GetSimulationState():  " + str(EngineState(state)));
                if (EngineState(state) == EngineState.cACTIVE):
                    restarted = True
                    #self.SetNumOfStepToAdvance(1)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    print("调用超时")
                else:
                    print("调用发生异常(可能是AfSim引擎在重启中):", e.details())
            time.sleep(1)

    """继续/开始仿真"""
    def StartResumeSim(self):
        self.RunScriptCmd("WsfSimulation.Resume();");

    def SetNumOfStepToAdvance(self, numOfStep):
        request = AfSim_pb2.NumOfStepToAdvance(simTime=self.timeStamp, numOfStep=numOfStep)
        self.server.SetNumOfStepToAdvance(request);

    def GetNumOfStepToAdvance(self):
        while (True):
            try:
                request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message="")
                response = self.server.GetNumOfStepToAdvance(request)
                return response.numOfStep
            except grpc.RpcError as e:
                print("调用GetNumOfStepToAdvance()发生异常:", e.details())
        return 0

    def GetSimulationState(self):
        request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message="")
        response = self.server.GetSimulationState(request);
        return response.val;

    def GetSimulationType(self):
        request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message="")
        response = self.server.GetSimulationType(request);
        return SimType(response.val);

    #暂停 返回True；没暂停，返回False
    def IsSimulationPaused(self):
        request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message="")
        response = self.server.IsSimulationPaused(request);
        return response.val;

    #fileName：不包含路径的文件名称
    #tgtDirName：相对路径，以bin目录的上一级目录为基准；如果目标目录不存在，会自动创建目标目录
    #fileContent：文件内容，字节数组
    def SendFile(self,fileName,relDirName,fileContent):
        request = AfSim_pb2.FileInfo(fileName=fileName, dirName=relDirName,content=fileContent)
        response = self.server.SendFile(request);
        if response.success == False:
            print("文件发送失败:" + response.message)
            return False
        else:
            return True
    
     #fileName：不包含路径的文件名称：文件名称为空时，删除目录
    #tgtDirName：相对路径，以bin目录的上一级目录为基准；如果文件名称不为空，则删除本目录下的指定文件；否则删除整个目录
    def DeleteFile(self,fileName,relDirName):
        request = AfSim_pb2.FileInfo(fileName=fileName, dirName=relDirName)
        response = self.server.DeleteFile(request);
        if response.success == False:
            print("删除文件/目录失败:" + response.message)
            return False
        else:
            return True

    # fileName：不包含路径的文件名称
    # tgtDirName：相对路径，以bin目录的上一级目录为基准；如果目标文件不存在，会返回false
    def OpenScenario(self,fileName,relDirName):
        request = AfSim_pb2.FileInfo(fileName=fileName, dirName=relDirName)
        response = self.server.OpenScenario(request);
        if response.success == False:
            print("打开想定文件失败:" + response.message)
            return False
        self.TerminateSim()
        restarted = False
        while (restarted == False):
            try:
                state = self.GetSimulationState()
                # print("GetSimulationState():  " + str(EngineState(state)));
                if (EngineState(state) == EngineState.cACTIVE):
                    restarted = True
                    # self.SetNumOfStepToAdvance(1)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    print("调用超时")
                else:
                    print("想定打开中,请稍后:", e.details())
            time.sleep(2)
        return True

    # 获取想定中收集的仿真数据：此函数只用于4y4b的项目
    def GetData(self):
        request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message="")
        return self.server.GetData(request);

    # 获取想定中自定义的收集的仿真数据
    def GetSimData(self):
        request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message="")
        return self.server.GetCustomData(request);

    # 执行脚本指令
    def RunScriptCmd(self, cmdContent, simTime=-1):
        if simTime == -1:
            simTime = self.timeStamp
        request = AfSim_pb2.DefaultPara(simTime=self.timeStamp, message=cmdContent)
        response = self.server.RunScript(request)
        if response.success == False:
            print("脚本执行失败:" + cmdContent)
            return False
        else:
            return True

    # 手动设置数据包的时间戳
    def SetTimeStamp(self, timestamp):
        self.timeStamp = timestamp
