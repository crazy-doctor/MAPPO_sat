import numpy as np
from mappo import MAPPO
import os
from torch.utils.tensorboard import SummaryWriter
import time
import torch

from Tool.init import get_reset_env
from Tool import parameter
from Tool import File_Path

def run():

    args = parameter.get_args()
    env, dim_info_red, dim_info_blue = get_reset_env(evaluate=False, args=args)

    # mappo_red = MAPPO(dim_info=dim_info_red, args=args, agents_name=env.red_sat)
    mappo_blue = MAPPO(dim_info=dim_info_blue, args=args, agents_name=env.blue_sat)

    # 所有工程存储结果的地方
    file_result = args.result_save_path
    # 工程文件夹名称（文件夹代表不同的方法）
    current_folder = os.path.basename(os.getcwd())
    # 存放结果的文件夹
    result_path = file_result+current_folder+"\\"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_o = File_Path.file_operate(root_path=result_path)

    writer_blue = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_blue), agent_id)) for agent_id in
                  env.blue_sat}

    episode = 1
    traj_length = 0
    total_steps = 0

    while total_steps < args.max_step:
        red_obs, blue_obs, global_obs_red, global_obs_blue = env.reset()
        terminal = False
        this_ep_reward_sum_blue = {agent_id: 0 for agent_id in env.blue_sat}
        start_time = time.time()
        while not terminal:
            total_steps += 1
            # action_red, prob_red = mappo_red.choose_action(red_obs, debug=False)

            action_red, prob_red = \
                {a_id: np.zeros(3)+0.5 for a_id in env.red_sat}, \
                    {a_id: np.zeros(3) for a_id in env.red_sat}
            action_blue, prob_blue = mappo_blue.choose_action(blue_obs, env.done_judge.BlueIsDw)
            act = {**action_red, **action_blue}

            red_obs_next, blue_obs_next, \
            red_reward, blue_reward, \
            red_dead_win, blue_dead_wim, \
            red_done, blue_done, \
            global_obs_red_next, global_obs_blue_next \
            = env.step(act)

            traj_length += 1

            # 这里所有智能体的done，和trunc都是同时置True，在真实场景中
            # 达到terminal有两种情况，第一种是所有智能体死亡，或智能体胜利，第二种是达到
            # 任务时间，强行任务终止
            # 全局任务终止条件，使用dw判断和使用时间判断
            terminal = list(blue_done.values())[0]
            mask = 0.0 if terminal else 1.0
            # 存储数据
            # mappo_red.store_memory(red_obs, global_obs_red, action_red,
            #                     prob_red, red_reward,
            #                     red_obs_next, global_obs_red_next, mask)

            mappo_blue.store_memory(blue_obs, global_obs_blue, action_blue,
                                prob_blue, blue_reward,
                                blue_obs_next, global_obs_blue_next, mask)
            for agent_id, r in blue_reward.items():
                this_ep_reward_sum_blue[agent_id] += r

            if traj_length % args.buffer_capacity == 0:
                # mappo_red.learn()
                mappo_blue.learn()

                # mappo_red.clear_memory()
                mappo_blue.clear_memory()
                # torch.cuda.empty_cache()

                traj_length = 0
            red_obs, blue_obs = red_obs_next, blue_obs_next
            global_obs_red, global_obs_blue = global_obs_red_next, global_obs_blue_next

        for agent_id, r in this_ep_reward_sum_blue.items():  # record reward
            writer_blue[agent_id].add_scalar("reward_agent", r, episode + 1)

        print(f"episode:{episode} {this_ep_reward_sum_blue} time:{time.time()-start_time}")
        episode += 1
        # print(torch.cuda.memory_summary())


if __name__ == '__main__':
    run()
