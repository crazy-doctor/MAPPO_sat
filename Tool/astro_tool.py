import numpy as np
import math

class Tool:
    # 输入的是惯性坐标系下的坐标
    # 输出的是CW坐标系（pos方向为x轴、pos叉乘vel的方向为z轴）下的坐标
    def Inertial_to_CW(self, pos_main, vel_main, pos_sat, vel_sat):
        # 坐标转换
        R = np.array([
            pos_main / np.linalg.norm(pos_main),
            np.cross(np.cross(pos_main, vel_main), pos_main) / np.linalg.norm(
                np.cross(np.cross(pos_main, vel_main), pos_main)),
            np.cross(pos_main, vel_main) / np.linalg.norm(np.cross(pos_main, vel_main)),
        ])

        h = np.linalg.norm(np.cross(pos_main, vel_main))
        w = np.array([0, 0, h / (np.linalg.norm(pos_main) ** 2)])

        r_pos_cw = np.dot(R, (pos_sat - pos_main))
        r_vel_cw = np.dot(R, (vel_sat - vel_main)) - np.cross(w, r_pos_cw)
        # R_TO_VVLH = np.array([
        #     [0, 1, 0],
        #     [0, 0, -1],
        #     [-1, 0, 0]
        # ])
        # r_pos_cw = np.dot(R_TO_VVLH, r_pos_cw)
        # r_vel_cw = np.dot(R_TO_VVLH, r_vel_cw)
        return r_pos_cw, r_vel_cw

    ## 输入的是参考卫星在惯性坐标系下的坐标，以及主卫星在cw系（pos方向为x轴、pos叉乘vel的方向为z轴）下的坐标
    ## 返回的是卫星在ECI下的坐标
    def CW_to_Inertial(self, pos_main, vel_main, pos_cw, vel_cw):
        # 坐标转换
        # R = np.array([
        #     np.cross(np.cross(pos_main, vel_main), pos_main) / np.linalg.norm(
        #         np.cross(np.cross(pos_main, vel_main), pos_main)),
        #     np.cross(vel_main, pos_main) / np.linalg.norm(np.cross(vel_main, pos_main)),
        #     -pos_main / np.linalg.norm(pos_main)
        # ])
        R = np.array([
            pos_main / np.linalg.norm(pos_main),
            np.cross(np.cross(pos_main, vel_main), pos_main) / np.linalg.norm(
                np.cross(np.cross(pos_main, vel_main), pos_main)),
            np.cross(pos_main, vel_main) / np.linalg.norm(np.cross(pos_main, vel_main)),
        ])

        h = np.linalg.norm(np.cross(pos_main, vel_main))
        w = np.array([0, 0, h / (np.linalg.norm(pos_main) ** 2)])
        pos_inertial = np.dot(pos_cw, R) + pos_main
        vel_inertial = np.dot((vel_cw + np.cross(w, pos_cw)), R) + vel_main
        return pos_inertial, vel_inertial

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
            if abs(np.dot(ehat, rhat) - 1) < 0.001:
                f = math.acos(1)
            elif abs(np.dot(ehat, rhat) + 1) < 0.001:
                f = math.acos(-1)
            else:
                f = math.acos(np.dot(ehat, rhat))

        elif np.dot(rvec, vvec) < 0:
            if abs(np.dot(ehat, rhat) - 1) < 0.00001:
                f = 2 * math.pi - math.acos(1)
            elif abs(np.dot(ehat, rhat) + 1) < 0.00001:
                f = 2 * math.pi - math.acos(-1)
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


    def CW(self, r_sat, v_sat, r_ref, v_ref, t):
        mu = 398600
        H = np.cross(r_ref, v_ref)
        h = np.linalg.norm(H)
        p = h ** 2 / mu
        E = np.cross(v_ref, H) / mu - r_ref / np.linalg.norm(r_ref)
        e = np.linalg.norm(E)
        a = (p / (1 - e ** 2))
        n = np.sqrt(mu / (a ** 3))
        tau = n * t
        T_RR = np.array([
            [4 - 3 * np.cos(tau), 0, 0],
            [6 * (np.sin(tau) - tau), 1, 0],
            [0, 0, np.cos(tau)]
        ])
        T_RV = np.array([
            [np.sin(tau) / n, 2 * (1 - np.cos(tau)) / n, 0],
            [2 * (np.cos(tau) - 1) / n, (4 * np.sin(tau) - 3 * tau) / n, 0],
            [0, 0, np.sin(tau) / n]
        ])
        T_VR = np.array([
            [3 * n * np.sin(tau), 0, 0],
            [6 * n * (np.cos(tau) - 1), 0, 0],
            [0, 0, -n * np.sin(tau)]
        ])
        T_VV = np.array([
            [np.cos(tau), 2 * np.sin(tau), 0],
            [-2 * np.sin(tau), 4 * np.cos(tau) - 3, 0],
            [0, 0, np.cos(tau)]
        ])

        r_pos_ric, r_vel_ric = self.Inertial_to_CW(pos_main=r_ref, vel_main=v_ref, pos_sat=r_sat, vel_sat=v_sat)

        r_new = np.dot(T_RR, r_pos_ric) + np.dot(T_RV, r_vel_ric)
        v_new = np.dot(T_VR, r_pos_ric) + np.dot(T_VV, r_vel_ric)
        R_TO_VVLH = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ])
        r_pos_cw = np.dot(R_TO_VVLH, r_new)
        r_vel_cw = np.dot(R_TO_VVLH, v_new)
        return r_pos_cw, r_vel_cw

    def extract_xyz(self, dict_in):
        return np.array(dict_in)

    def generate_random_point_on_sphere(self, radius):

        # 使用球坐标生成随机点，球坐标系的三个参数为r,φ，θ
        phi = np.random.uniform(0, 2 * np.pi)  # Phi角在0到2π之间
        theta = np.arccos(np.random.uniform(-1, 1))  # Theta角在0到π之间

        # 将球坐标转换为笛卡尔坐标
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        return np.array([x, y, z])
