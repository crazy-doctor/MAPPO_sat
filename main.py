# 系统包
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
# 自建包
from Tool.init import get_reset_env,create_net
import Tool.parameter as parameter
from Tool.draw_3d import trajectory_draw
from matplotlib.ticker import MaxNLocator

args = parameter.get_args()
# 导入actor
RED_NUM, BLUE_NUM = 3,3
FIG_DIR = r".\results\dis\\"
# 初始化环境（几打几）
env, dim_info_red, dim_info_blue = get_reset_env(args=args,red_num=RED_NUM,blue_num=BLUE_NUM,evaluate=True)

PICTURE_3D_DIR = r".\results\CW_fig\\"
PICTURE_ASSIGN = r".\results\assign_res\\"
# 导入actor
actor_net = create_net( args=args, red_path=args.red_actor_path,blue_path=args.blue_actor_path,
                        dim_inf_red=dim_info_red,  dim_inf_blue=dim_info_blue)
t_d = trajectory_draw(red_sat=env.red_sat,blue_sat=env.blue_sat)

fig, ax = plt.subplots()

success_time = 0
dis_term = {red_name: [] for red_name in env.red_sat}
fuel_consume = {red_name: [] for red_name in env.red_sat}
time_record = np.zeros(0)
for episode in tqdm(range(50)):
    env.episode_num = episode
    obs_red, obs_blue = env.reset()
    # 将分配策略做成观测输入到actor中，得出动作
    obs = copy.deepcopy({**obs_red, **obs_blue})
    dis = {red_name: [] for red_name in env.red_sat}
    fuel = {red_name: [] for red_name in env.red_sat}

    for red_name in env.red_sat:
        dis[red_name].append(env.sim.inf.dis_sat(red_name=red_name,blue_name=env.assign_res[red_name]))
    # for i in tqdm(range(args.episode_length),desc=f"episode{episode+1}"):
    for i in range(args.episode_length):
        env.step_num = i
        t_d.update_data(pos=env.sim.inf.pos_cw)
        act = actor_net.act_select(obs=obs)
        obs_red, obs_blue, _, _, _, _ = env.step(action=act)
        obs = copy.deepcopy({**obs_red, **obs_blue})
        if i % 10 == 0: ##任务分配完成
            assign=[[],[]]
            for red_id in env.red_sat:
                assign[0].append(int(red_id[1]))
                assign[1].append(int(env.assign_res[red_id][1]))
            # 创建散点图
            plt.scatter(assign[0], assign[1], s=100)

            # 设置图表标题和坐标轴标签
            plt.title('assign_result')
            plt.xlabel('red')
            plt.ylabel('blue')
            # 设置X轴和Y轴的刻度定位器，使其只显示整数
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlim(-0.5, RED_NUM-0.5)  # X轴显示范围从0到10
            plt.ylim(-0.5, BLUE_NUM-0.5)  # Y轴显示范围从0到10
            plt.savefig(fname=PICTURE_ASSIGN+"ep"+str(episode+1)+"_step"+str(i+1)+"assign.png")
            plt.cla()

        action_copy = copy.deepcopy(act)
        delta_v_dict = env.convert_v(action_copy)

        for red_name in env.red_sat:
            dis[red_name].append(env.sim.inf.dis_sat(red_name=red_name, blue_name=env.assign_res[red_name]))

        for red_name in env.red_sat:
            fuel[red_name].append(np.linalg.norm(delta_v_dict[red_name]))

        if(len(obs_red)==0):
            success_time += 1
            time_record = np.append(arr=time_record, values=i)
            break

    t_d.draw_fig(episode=episode,save_dir=PICTURE_3D_DIR)

    for red_name in env.red_sat: dis_term[red_name].append(
        env.sim.inf.dis_sat(red_name=red_name, blue_name=env.assign_res[red_name]))
    for red_name in env.red_sat: fuel_consume[red_name].append(env.fuel_consume[red_name])



    for red_name in env.red_sat: ax.plot(dis[red_name], label=red_name)
    ax.legend()
    ax.grid(True)
    plt.savefig(fname=FIG_DIR+str(episode)+"dis.png", dpi=300)
    plt.cla()

    for red_name in env.red_sat: ax.plot(fuel[red_name], label=red_name)
    ax.legend()
    ax.grid(True)
    plt.savefig(fname=r".\results\fuel\\"+str(episode)+"fuel.png", dpi=300)
    plt.cla()

np.save(file=r"E:\code_list\red_battle_blue\results\file_record\no_assign.npy", arr=time_record)

fig, ax = plt.subplots()
for red_name in env.red_sat: ax.plot(dis_term[red_name], label=red_name)
ax.legend()
ax.grid(True)
plt.savefig(fname=r".\results\dis_term"+"\dis_term.png", dpi=300)
plt.cla()

for red_name in env.red_sat: ax.plot(fuel_consume[red_name], label=red_name)
ax.legend()
ax.grid(True)
plt.savefig(fname=r".\results\fuel_term"+"\\fuel_consume.png", dpi=300)
plt.cla()

print(f"50次中成功侦察{success_time}次")