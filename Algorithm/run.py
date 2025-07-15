# 系统包
import copy
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# 自建包
from Algorithm.mappo import MAPPO
from Tool.draw_picture import draw_distance
from Tool import File_Path
from Tool.init import get_reset_env
from Tool import parameter



def run():
    LOAD = True
    LOAD_RUN_TIME = 1
    LOAD_EP = 12000
    args = parameter.get_args()

    # 所有工程存储结果的地方
    file_result = args.result_save_path
    # 工程文件夹名称（文件夹代表不同的方法）
    current_folder = os.path.basename(os.getcwd())
    # 存放结果的文件夹
    result_path = file_result+current_folder+"\\"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_o = File_Path.file_operate(root_path=result_path)

    # dim_info_red = {卫星一：[观测维度，动作维度]}
    env, dim_info_red, dim_info_blue = get_reset_env(evaluate=False, args=args)

    # 将r0 r1 r2绘图数据分别保存到三个文件夹中
    writer_red = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_red), agent_id)) for agent_id in
                  env.red_sat}
    writer_blue = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_blue), agent_id)) for agent_id in
                  env.blue_sat}

    mappo_red = MAPPO(agents_name=env.red_sat, dim_info=dim_info_red,file=file_o,args=args)

    mappo_blue = MAPPO(agents_name=env.blue_sat, dim_info=dim_info_blue,file=file_o,args=args)

    total_steps = 0
    traj_length = 0

    for episode in range(args.episode_num):
        red_obs, blue_obs, global_obs_red, global_obs_blue = env.reset()
        this_ep_reward_sum_red = {agent_id: 0 for agent_id in env.red_sat}  # 每一个元素是代表一局当中获得奖励的总和
        this_ep_reward_sum_blue = {agent_id: 0 for agent_id in env.blue_sat}
        env.episode_num = episode

        # terminal: 任务成功/失败
        # trunction: 任务截止时间已到
        for step in tqdm(range(args.episode_length),desc=f"episode{episode+1}"):
            env.step_num = step
            action_r, prob_r =mappo_red.choose_action(red_obs, env.done_judge.RedIsDw)
            action_b, prob_b = mappo_blue.choose_action(blue_obs, env.done_judge.BlueIsDw)

            red_obs_, blue_obs_, \
            red_reward, blue_reward, \
            red_dw,blue_dw,\
            red_done, blue_done,\
            global_obs_red_, global_obs_blue_ = env.step({**action_r, **action_b})


            total_steps += 1
            traj_length += 1

            # dw, done,idx
            mappo_red.store_memory(red_obs, red_obs_,action_r, prob_r,
                     red_reward, red_dw ,red_done,global_obs_red,global_obs_red_, traj_length-1)

            mappo_blue.store_memory(blue_obs, blue_obs_,action_b, prob_b,
                     blue_reward, blue_dw, blue_done, global_obs_blue,global_obs_blue_, traj_length-1)

            for agent_id, r in red_reward.items():
                this_ep_reward_sum_red[agent_id] += r

            for agent_id, r in blue_reward.items():
                this_ep_reward_sum_blue[agent_id] += r

            if traj_length % args.buffer_capacity == 0:
                mappo_red.learn()
                mappo_blue.learn()
                traj_length = 0

            if red_done["r0"]:
                break

            global_obs_red = global_obs_red_
            global_obs_blue = global_obs_blue_

            red_obs = red_obs_
            blue_obs = blue_obs_
        # 更新奖励数组，用于保存到tensorboard的绘图文件中
        for agent_id, r in this_ep_reward_sum_red.items():  # record reward
            writer_red[agent_id].add_scalar("reward_agent", r, episode + 1)

        for agent_id, r in this_ep_reward_sum_blue.items():  # record reward
            writer_blue[agent_id].add_scalar("reward_agent", r, episode + 1)


if __name__ == '__main__':
    # profiler = LineProfiler()
    # profiler.add_function(run)  # 添加要分析的函数
    # profiler.run('run()')  # 执行分析
    # profiler.print_stats()  # 打印结果
    run()

