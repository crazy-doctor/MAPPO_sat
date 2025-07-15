import matplotlib.pyplot as plt
import os
import numpy as np
import copy

class draw_distance:
    def __init__(self, data_length, save_dir_red, save_dir_blue,args):
        self.sat_red = ["r"+str(i) for i in range(args.red_num)]
        self.sat_blue = ["b"+str(i) for i in range(args.blue_num)]
        self.data_length = data_length
        self.data_dict_red = {name: [] for name in self.sat_red}
        self.data_dict_blue = {name: [] for name in self.sat_blue}
        self.fig, self.ax = plt.subplots()
        self.save_dir_red = save_dir_red
        self.save_dir_blue = save_dir_blue

    def draw_save_fig(self, data_dict, dir,red_or_blue):
        if red_or_blue == "r":
            sat = self.sat_red
        elif red_or_blue == "b":
            sat = self.sat_blue

        x = list(range(len(data_dict[sat[0]])))

        # 在同一个轴上几根曲线
        for agent_id in sat:
            self.ax.plot(x, data_dict[agent_id], label=agent_id)
        # 添加图例
        self.ax.legend()
        # 添加标题和轴标签
        self.ax.set_title('distance')
        self.ax.set_xlabel('step')
        self.ax.set_ylabel('dis/km')
        # 显示网格
        self.ax.grid(True)
        # 保存图形
        plt.savefig(dir, dpi=300)
        plt.cla()

    def update_data(self, dict, episode,done_flag):

        for name in self.sat_red:
            self.data_dict_red[name].append(dict[name])

        for name in self.sat_blue:
            self.data_dict_blue[name].append(dict[name])

        if len(list(self.data_dict_red.values())[0]) == self.data_length or done_flag:
            pic_dir_red = self.save_dir_red + f"\\ep{episode}"
            pic_dir_blue = self.save_dir_blue + f"\\ep{episode}"

            self.draw_save_fig(data_dict=self.data_dict_red,dir=pic_dir_red, red_or_blue="r")
            self.draw_save_fig(data_dict=self.data_dict_blue, dir=pic_dir_blue, red_or_blue="b")

            self.data_dict_red = {name: [] for name in self.sat_red}
            self.data_dict_blue = {name: [] for name in self.sat_blue}

    def clear_data(self):
        self.data_dict_red = {name: [] for name in self.sat_red}
        self.data_dict_blue = {name: [] for name in self.sat_blue}


class draw_orbit_ele():
    def __init__(self, data_length):
        self.sat = ["r0", "r1", "r2"]
        self.data_length = data_length
        self.data_dict = {name: np.empty([0,6]) for name in self.sat}
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))

    def draw_save_fig(self, dir):
        # [轨道倾角， 升交点赤经， 半长轴， 偏心率， 真近点角， 近地点幅角]
        data_dict = {
        "a": {name: copy.deepcopy(self.data_dict[name][:, 2]) for name in self.sat},
        "e": {name: copy.deepcopy(self.data_dict[name][:, 3]) for name in self.sat},
        "inclination": {name: copy.deepcopy(self.data_dict[name][:, 0]) for name in self.sat},
        "RAAN": {name: copy.deepcopy(self.data_dict[name][:, 1]) for name in self.sat},
        "Argument": {name: copy.deepcopy(self.data_dict[name][:, 5]) for name in self.sat},
        "f": {name: copy.deepcopy(self.data_dict[name][:, 4]) for name in self.sat}
        }
        # data_dict[六根数属性][卫星n]

        recity_key = ["inclination", "RAAN", "Argument", "f"]
        for key in recity_key:
            for agent_name in self.sat:
                for i in range(len(data_dict[key][agent_name])):
                    if data_dict[key][agent_name][i]>180:
                        data_dict[key][agent_name][i] = 360 - data_dict[key][agent_name][i]

        time = np.arange(self.data_length)
        # 创建一个2x3的子图网格（2行3列）
        # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        # 曲线颜色
        colors = ['blue', 'green', 'red']
        # 遍历每个子图
        for i, (name_orbit_ele, data_dict) in enumerate(data_dict.items()):
            ax = self.axs[i // 3, i % 3]
            for j, (satellite, color) in enumerate(zip(self.sat, colors)):
                ax.plot(time, data_dict[satellite], label=f'r{j} ({color})', color=color)
            ax.set_title(name_orbit_ele)
            ax.set_xlabel('time')
            ax.set_ylabel('value')
            ax.legend()
        # 调整布局
        plt.tight_layout()
        # 显示图形
        # 保存图形
        plt.savefig(dir, dpi=300)
        plt.cla()

    def update_data(self, dict, dir, episode,done_flag):
        for name in self.sat:
            self.data_dict[name] = np.vstack((self.data_dict[name], dict[name]))
        if len(self.data_dict["r0"]) == self.data_length or done_flag:
            dir += f"\\ep{episode}"
            self.draw_save_fig(dir)
            self.data_dict = {name: np.empty([0,6]) for name in self.sat}


