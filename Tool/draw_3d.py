import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class trajectory_draw:
    def __init__(self,red_sat,blue_sat):
        self.red_sat = red_sat
        self.blue_sat = blue_sat
        self.sat = self.red_sat+self.blue_sat
        self.tra = {sat_id:np.array([]).reshape(0,3) for sat_id in self.sat}

    def update_data(self, pos):
        for sat_id in self.sat:
            self.tra[sat_id] = np.vstack((self.tra[sat_id], pos[sat_id].reshape(1, -1)))

    def draw_fig(self,episode,save_dir):

        # 创建3D图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for sat_id in self.red_sat:
            # # 绘制点
            ax.scatter(self.tra[sat_id][:,0], self.tra[sat_id][:,1], self.tra[sat_id][:,2], c='red', marker='o')
            # 绘制曲线
            ax.plot(self.tra[sat_id][:,0], self.tra[sat_id][:,1], self.tra[sat_id][:,2], label=sat_id)

        for sat_id in self.blue_sat:
            # # 绘制点
            ax.scatter(self.tra[sat_id][:,0], self.tra[sat_id][:,1], self.tra[sat_id][:,2], c='blue', marker='o')
            # 绘制曲线
            ax.plot(self.tra[sat_id][:,0], self.tra[sat_id][:,1], self.tra[sat_id][:,2], label=sat_id)

        # 设置图形标签和标题
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Curve Plot')

        fig.canvas.draw()  # 这将确保图形被渲染
        # 显示图例
        ax.legend()
        # 显示图形
        plt.savefig(fname=save_dir + str(episode) + "CW.png")
        # plt.savefig(fname=r"E:\code_list\red_battle_blue\results\CW_fig\\"+str(episode)+"CW.png")
        plt.close(fig)
        plt.cla()
        self.tra = {sat_id:np.array([]).reshape(0,3) for sat_id in self.sat}

    # def draw_fig_2d(self, episode, save_dir):





