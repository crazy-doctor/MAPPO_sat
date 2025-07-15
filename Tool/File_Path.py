import os


# class uniform_format:
#     def __init__(self):
#

# --->root_path---->results--->1th_run
#                          --->2th_run
#                          --->3th_run--->tensorboard_draw--->r0
#                                                         --->r1
#                                     --->1000th_episode--->picture
#                                     --->2000th_episode--->picture
# root_path:代码文件所在文件夹
# mission_server_path："mission_server.exe"所在路径
#
class file_operate:
    def __init__(self, root_path="./",mission_server_path=r"D:/shen/software/AfSim290/bin/mission_server.exe",red_num=3,blue_num=3):
        # 这个用来保存各种结果的根目录
        self.root_path = root_path
        # mission_server.exe执行文件的位置
        self.mission_server_path = mission_server_path
        # 保存结果的路径
        self.results_dir = os.path.join(self.root_path, "results/")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.run_path = self.create_run_dir(self.results_dir)  # 3th_run文件夹路径



        # 创建error文件夹，代表此次运行出现过哪些错误
        self.erroe_log_dir = os.path.join(self.run_path,"error")
        if not os.path.exists(self.erroe_log_dir):
            os.makedirs(self.erroe_log_dir)

        self.RED = os.path.join(self.run_path,"RED")
        if not os.path.exists(self.RED):
            os.makedirs(self.RED)

        self.BLUE = os.path.join(self.run_path,"BLUE")
        if not os.path.exists(self.BLUE):
            os.makedirs(self.BLUE)


        # 红方picture保存运行图像
        self.picture_path_red = os.path.join(self.RED, "picture/")
        if not os.path.exists(self.picture_path_red):
            os.makedirs(self.picture_path_red)
            #1.距离图片
        self.picture_distance_red = os.path.join(self.picture_path_red,"dis")
        if not os.path.exists(self.picture_distance_red):
            os.makedirs(self.picture_distance_red)
            # 2.六根数图片
        self.picture_orbit_red = os.path.join(self.picture_path_red,"orbit")
        if not os.path.exists(self.picture_orbit_red):
            os.makedirs(self.picture_orbit_red)

        # 蓝方picture保存运行图像
        self.picture_path_blue = os.path.join(self.BLUE, "picture/")
        if not os.path.exists(self.picture_path_blue):
            os.makedirs(self.picture_path_blue)
            #1.距离图片
        self.picture_distance_blue = os.path.join(self.picture_path_blue,"dis")
        if not os.path.exists(self.picture_distance_blue):
            os.makedirs(self.picture_distance_blue)
            # 2.六根数图片
        self.picture_orbit_blue = os.path.join(self.picture_path_blue,"orbit")
        if not os.path.exists(self.picture_orbit_blue):
            os.makedirs(self.picture_orbit_blue)

        # 红方tensorboard的文件，保存智能体奖励信息
        self.tensor_draw_path_red = os.path.join(self.RED,"tensor_draw")
        if not os.path.exists(self.tensor_draw_path_red):
            os.makedirs(self.tensor_draw_path_red)
            # 创建不同红方卫星的奖励文件
        if red_num!=0:
            for i in range(red_num):
                agent_num = "r"+str(i)
                t = os.path.join(self.tensor_draw_path_red, agent_num)
                if not os.path.exists(t):
                    os.makedirs(t)
            # 创建不同蓝方卫星的奖励文件
        # 蓝方tensorboard的文件，保存智能体奖励信息
        self.tensor_draw_path_blue = os.path.join(self.BLUE, "tensor_draw")
        if not os.path.exists(self.tensor_draw_path_blue):
            os.makedirs(self.tensor_draw_path_blue)
        if blue_num!=0:
            for i in range(blue_num):
                agent_num = "b"+str(i)
                t = os.path.join(self.tensor_draw_path_blue, agent_num)
                if not os.path.exists(t):
                    os.makedirs(t)


    def create_run_dir(self, results_dir):
        total_files = len([file for file in os.listdir(results_dir)])
        run_path = os.path.join(results_dir, f'{total_files + 1}th_run/')
        os.makedirs(run_path)
        return run_path

    # 在当前的th_run文件夹下，创建一个文件夹，用来保存模型,返回地址
    def create_now_episode_folder(self, episode_num):
        now_folder_path = os.path.join(self.run_path, f"{episode_num}th_episode")
        if not os.path.exists(now_folder_path):
            os.makedirs(now_folder_path)
        return now_folder_path

    # 在load模型的时候使用，返回的是模型文件路径
    def get_episode_path_load(self, run_th, side):
        side_path = os.path.join(self.results_dir, f'{run_th}th_run/', side)
        return side_path

    # 在evaluate模型的时候使用，在evaluate创建文件夹
    # 并且返回所测试模型的路径，以及图片保存的地址
    def get_episode_path_evaluate(self, run_th, episode_num):
        episode_path = os.path.join(self.results_dir, f'{run_th}th_run/', f'{episode_num+1}th_run/')
        if not os.path.exists(episode_path):
            raise FileNotFoundError("evaluate时文件未找到，请再次检查") from None

        evaluate_path = os.path.join(episode_path, "evaluate")
        if not os.path.exists(evaluate_path):
            os.makedirs(evaluate_path)
        # 创建picture保存运行图像
        picture_path = os.path.join(evaluate_path, "picture/")
        if not os.path.exists(picture_path):
            os.makedirs(picture_path)
            #1.距离图片
        picture_distance = os.path.join(picture_path,"dis")
        if not os.path.exists(picture_distance):
            os.makedirs(picture_distance)
            # 2.六根数图片
        picture_orbit = os.path.join(picture_path,"orbit")
        if not os.path.exists(picture_orbit):
            os.makedirs(picture_orbit)

        return episode_path, picture_distance, picture_orbit

    # 产生模型指针，指向第几次运行的第多少局，方便加载模型
    def gen_model_point(self, run_th, episode):
        path = os.path.join(self.root_path,"results",f"{run_th}th_tun",f"{episode}th_episode")
        return path

