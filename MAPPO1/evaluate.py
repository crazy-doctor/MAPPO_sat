import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from Tool import File_Path
def evalute(args, env, mappo_blue, writer_blue,evalute_times):

    print("=============================================================")
    print(f"第{int(evalute_times)}次评估开始")
    Evalute_ep = 5 # 评估一次需要几轮
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

    while episode <= Evalute_ep:
        red_obs, blue_obs, global_obs_red, global_obs_blue = env.reset()
        terminal = False
        unregularTimes = 0

        while not terminal:
            total_steps += 1
            # action_red, prob_red = mappo_red.choose_action(red_obs, debug=False)

            action_red, prob_red = \
                {a_id: np.zeros(3)+0.5 for a_id in env.red_sat}, \
                    {a_id: np.zeros(3) for a_id in env.red_sat}
            action_blue, prob_blue = mappo_blue.choose_action(blue_obs,evalute=True)
            act = {**action_red, **action_blue}

            red_obs_next, blue_obs_next, \
            red_reward, blue_reward, \
            red_dead_win, blue_dead_wim, \
            red_done, blue_done, \
            global_obs_red_next, global_obs_blue_next \
            = env.step(act)
            for blue_name in env.blue_sat:
                dis_array = []
                for blue_id in env.blue_sat:
                    if blue_id != blue_name:
                        dis_array.append(env.sim.inf.dis_sat(blue_name, blue_id))
                min_dis = min(dis_array)
                max_dis = max(dis_array)
                if min_dis < args.safe_dis or max_dis > args.comm_dis:
                    unregularTimes+=1
                    break

            traj_length += 1
            # 这里所有智能体的done，和trunc都是同时置True，在真实场景中
            # 达到terminal有两种情况，第一种是所有智能体死亡，或智能体胜利，第二种是达到
            # 任务时间，强行任务终止
            # 全局任务终止条件，使用dw判断和使用时间判断
            terminal = list(blue_done.values())[0]
            red_obs, blue_obs = red_obs_next, blue_obs_next

        writer_blue.add_scalar("unregularTimes", unregularTimes, (evalute_times-1)*Evalute_ep + episode)
        print(f"第{episode}轮不合格步数为: {unregularTimes}")

        episode += 1
    print("评估结束")
    print("=============================================================")