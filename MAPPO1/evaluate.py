import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from Tool import File_Path
def evalute(args, env, mappo_red, mappo_blue, writer_red, writer_blue,evalute_times):

    print("=============================================================")
    print(f"第{int(evalute_times)}次评估开始")
    Evalute_ep = 10 # 评估一次需要几轮
    episode = 1
    traj_length = 0
    total_steps = 0
    evalute_episode = 100
    file_result = args.result_save_path
    # 工程文件夹名称（文件夹代表不同的方法）
    current_folder = os.path.basename(os.getcwd())
    # 存放结果的文件夹
    result_path = file_result+current_folder+"\\"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ##################指标#########################
    min_distance_list = {sat_name:[] for sat_name in env.red_sat} #红方指标，用于计算平均最短距离
    die_times = {sat_name:0 for sat_name in env.red_sat+env.blue_sat} #红蓝双方卫星死亡次数
    ##############################################

    while episode <= Evalute_ep:
        red_obs, blue_obs, global_obs_red, global_obs_blue = env.reset()
        terminal = False
        min_dis_ep = {sat_name:env.sim.inf.dis_sat(sat_name,"b"+sat_name[1]) for sat_name in env.red_sat}
        is_die_ep = {sat_name:False for sat_name in env.red_sat+env.blue_sat}
        while not terminal:
            total_steps += 1
            action_red, prob_red = mappo_red.choose_action(red_obs, evalute=True)
            action_blue, prob_blue = mappo_blue.choose_action(blue_obs,evalute=True)
            act = {**action_red, **action_blue}

            red_obs_next, blue_obs_next, \
            red_reward, blue_reward, \
            red_dead_win, blue_dead_wim, \
            red_done, blue_done, \
            global_obs_red_next, global_obs_blue_next \
            = env.step(act)

            # 维护min_dis_ep
            for red_name in env.red_sat:
                dis_now = env.sim.inf.dis_sat(red_name,"b"+red_name[1])
                min_dis_ep[red_name] = min(dis_now, min_dis_ep[red_name])
                dis_array = []
                for red_id in env.red_sat:
                    if red_name != red_id:
                        dis_array.append(env.sim.inf.dis_sat(red_name, red_id))

                min_dis = min(dis_array)
                max_dis = max(dis_array)
                if min_dis < args.safe_dis or max_dis > args.comm_dis:
                    is_die_ep[red_name] = True
                    break

            for blue_name in env.blue_sat:
                dis_array = []
                for blue_id in env.blue_sat:
                    if blue_id != blue_name:
                        dis_array.append(env.sim.inf.dis_sat(blue_name, blue_id))
                min_dis = min(dis_array)
                max_dis = max(dis_array)
                if min_dis < args.safe_dis or max_dis > args.comm_dis:
                    is_die_ep[blue_name] = True
                    break

            traj_length += 1
            # 这里所有智能体的done，和trunc都是同时置True，在真实场景中
            # 达到terminal有两种情况，第一种是所有智能体死亡，或智能体胜利，第二种是达到
            # 任务时间，强行任务终止
            # 全局任务终止条件，使用dw判断和使用时间判断
            terminal = list(blue_done.values())[0]
            if terminal: break
            red_obs, blue_obs = red_obs_next, blue_obs_next

        # 处理一局运行的结果
        for red_id in env.red_sat:
            min_distance_list[red_id].append(min_dis_ep[red_id])
            if is_die_ep[red_id]: die_times[red_id] += 1

        for blue_id in env.blue_sat:
            if is_die_ep[blue_id]: die_times[blue_id] += 1

        episode += 1

    for red_id in env.red_sat:
        writer_red[red_id].add_scalars('result_eva',
                                       {'die_times': die_times[red_id],
                                        'avr_min_dis': sum(min_distance_list[red_id])/len(min_distance_list[red_id])
                                       },
                                        evalute_times)

    for blue_id in env.blue_sat:
        writer_blue[blue_id].add_scalars('result_eva',
                                       {'die_times': die_times[blue_id]},
                                        evalute_times)

    avr_dis = {red_id:sum(min_distance_list[red_id])/len(min_distance_list[red_id]) for red_id in env.red_sat}
    print("评估结果:")
    print(f"红方最小距离: {avr_dis}")
    print(f"死亡次数: {die_times}")
    print("=============================================================")