import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import copy

from Tool import File_Path
from Tool import draw_picture
def evalute(args, env, mappo_red, mappo_blue, writer_red, writer_blue,evalute_times,file_o):

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

    # draw_dis = draw_picture.draw_distance(data_length=args.episode_length,
    #                                       save_dir_red=file_o.picture_path_red,
    #                                       save_dir_blue=file_o.picture_path_blue,
    #                                       args=args)

    ##################指标#########################
    min_distance_list = {sat_name:[] for sat_name in env.red_sat} #红方指标，用于计算平均最短距离
    die_times = {sat_name:0 for sat_name in env.red_sat+env.blue_sat} #红蓝双方卫星死亡次数
    reward_avr = {sat_name:0 for sat_name in env.red_sat+env.blue_sat} #奖励
    ##############################################

    while episode <= Evalute_ep:
        red_obs, global_obs_red, blue_obs, global_obs_blue = env.reset()
        terminal = False
        # 红方卫星在一局中距离目标的最小值
        min_dis_ep = {sat_name:env.sim.inf.dis_sat(sat_name,"b"+sat_name[1]) for sat_name in env.red_sat}
        # 每个卫星在一局中的死亡情况
        is_die_ep = {sat_name:False for sat_name in env.red_sat+env.blue_sat}
        while not terminal:
            total_steps += 1
            action_red, prob_red = mappo_red.choose_action(red_obs, evalute=True)
            # action_blue, prob_blue = mappo_blue.choose_action(blue_obs,evalute=True)
            action_blue = {sat_id: np.ones(3) * 0.5 for sat_id in env.blue_sat}
            act = {**action_red, **action_blue}

            red_obs_next, blue_obs_next, \
            red_reward, blue_reward, \
            red_done, blue_done, \
            global_obs_red_next, global_obs_blue_next \
            = env.step(act)

            # 维护min_dis_ep
            die_tmp1 = copy.deepcopy(is_die_ep)
            for red_name in env.red_sat:
                # 奖励叠加
                reward_avr[red_name] += red_reward[red_name]
                dis_now = env.sim.inf.dis_sat(red_name,"b"+red_name[1])
                min_dis_ep[red_name] = min(dis_now, min_dis_ep[red_name])
                for red_id in env.red_sat:
                    if red_name != red_id and not is_die_ep[red_id]:
                        dis_tmp = env.sim.inf.dis_sat(red_name, red_id)
                        if is_die_ep[red_name] or dis_tmp < args.safe_dis or dis_tmp > args.comm_dis:
                            die_tmp1[red_name] = True
            is_die_ep = die_tmp1

            die_tmp2 = copy.deepcopy(is_die_ep)
            for blue_name in env.blue_sat:
                reward_avr[blue_name] += blue_reward[blue_name]
                for blue_id in env.blue_sat:
                    if blue_id != blue_name and not is_die_ep[blue_id]:
                        dis_tmp = env.sim.inf.dis_sat(blue_name, blue_id)
                        if is_die_ep[blue_name] or dis_tmp < args.safe_dis or dis_tmp > args.comm_dis:
                            die_tmp2[blue_name] = True
            is_die_ep = die_tmp2

            traj_length += 1
            # 判断结束
            terminal = (sum(list(red_done.values())) == len(env.red_sat))
            red_obs, blue_obs = red_obs_next, blue_obs_next

        # 处理一局运行的结果
        for red_id in env.red_sat:
            min_distance_list[red_id].append(min_dis_ep[red_id])
            if is_die_ep[red_id]: die_times[red_id] += 1

        for blue_id in env.blue_sat:
            if is_die_ep[blue_id]: die_times[blue_id] += 1

        episode += 1

    for red_id in env.red_sat:
        reward_avr[red_id]/=total_steps
        writer_red[red_id].add_scalars('result_eva',
                                       {'die_times': die_times[red_id],
                                        'avr_min_dis': sum(min_distance_list[red_id])/len(min_distance_list[red_id]),
                                        "avr_reward": reward_avr[red_id]
                                       },
                                        evalute_times)

    # for blue_id in env.blue_sat:
    #     reward_avr[blue_id] /= total_steps
    #     writer_blue[blue_id].add_scalars('result_eva',
    #                                    {'die_times': die_times[blue_id],
    #                                     "blue_reward": reward_avr[blue_id]
    #                                     },
    #                                     evalute_times)

    avr_dis = {red_id:sum(min_distance_list[red_id])/len(min_distance_list[red_id]) for red_id in env.red_sat}
    print("评估结果:")
    print(f"红方最小距离: {avr_dis}")
    print(f"死亡次数: {die_times}")
    print("=============================================================")