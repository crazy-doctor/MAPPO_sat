import numpy as np
from mappo import MAPPO
import os
from torch.utils.tensorboard import SummaryWriter
import time

from MAPPO.env.init import get_reset_env
from MAPPO.env import parameter
from Tool import File_Path
from evaluate import evalute


def run():

    LOAD = True
    LOAD_RUN_TIME = 2
    LOAD_EP = 15000

    args = parameter.get_args()
    env, dim_info_red, dim_info_blue = get_reset_env(evaluate=False, args=args)
    # 所有工程存储结果的地方
    file_result = args.result_save_path
    # 工程文件夹名称（文件夹代表不同的方法）
    current_folder = os.path.basename(os.getcwd())
    # 存放结果的文件夹
    result_path = file_result+current_folder+"\\"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_o = File_Path.file_operate(root_path=result_path)

    if LOAD:
        mappo_red = MAPPO.load(dim_info=dim_info_red,args=args,agents_name=env.red_sat,
                                 load_dir=file_o.get_episode_path_load(run_th=LOAD_RUN_TIME, side="RED"),
                                 episode_num=LOAD_EP)
        # mappo_blue = MAPPO.load(dim_info=dim_info_blue,args=args,agents_name=env.blue_sat,
        #                          load_dir=file_o.get_episode_path_load(run_th=LOAD_RUN_TIME, side="BLUE"),
        #                          episode_num=LOAD_EP)
    else:
        mappo_red = MAPPO(dim_info=dim_info_red, args=args, agents_name=env.red_sat)
        # mappo_blue = MAPPO(dim_info=dim_info_blue, args=args, agents_name=env.blue_sat)

    writer_red = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_red), agent_id)) for agent_id in
                  env.red_sat}
    # writer_blue = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_blue), agent_id)) for agent_id in
    #               env.blue_sat}

    episode = 1
    traj_length = 0
    total_steps = 0
    evalute_episode = 100
    while total_steps < args.max_step:
        red_obs, global_obs_red, blue_obs, global_obs_blue = env.reset()
        terminal = False
        this_ep_reward_sum_blue = {agent_id: 0 for agent_id in env.blue_sat}
        start_time = time.time()

        while not terminal:
            total_steps += 1
            action_red, prob_red = mappo_red.choose_action(red_obs)
            # action_blue, prob_blue = mappo_blue.choose_action(blue_obs)
            action_blue = {sat_id: np.ones(3)*0.5 for sat_id in env.blue_sat}
            prob_blue = {sat_id: np.ones(3) * 5 for sat_id in env.blue_sat}
            act = {**action_red, **action_blue}

            red_obs_next, blue_obs_next, \
            red_reward, blue_reward,\
            red_done, blue_done, \
            global_obs_red_next, global_obs_blue_next \
            = env.step(act)

            traj_length += 1
            # 结束判断
            terminal = False
            if sum(list(blue_done.values()))== len(env.blue_sat) or \
                    sum(list(red_done.values()))== len(env.red_sat):
                terminal = True

            mask = 0.0 if terminal else 1.0
            # 存储数据
            mappo_red.store_memory(red_obs, global_obs_red, action_red,
                                prob_red, red_reward,
                                red_obs_next, global_obs_red_next, mask)

            # mappo_blue.store_memory(blue_obs, global_obs_blue, action_blue,
            #                     prob_blue, blue_reward,
            #                     blue_obs_next, global_obs_blue_next, mask)
            for agent_id, r in blue_reward.items():
                this_ep_reward_sum_blue[agent_id] += r

            if traj_length % args.buffer_capacity == 0:
                mappo_red.learn()
                # mappo_blue.learn()

                mappo_red.clear_memory()
                # mappo_blue.clear_memory()

                traj_length = 0


            red_obs, blue_obs = red_obs_next, blue_obs_next
            global_obs_red, global_obs_blue = global_obs_red_next, global_obs_blue_next

        # for agent_id, r in this_ep_reward_sum_blue.items():  # record reward
        #     writer_blue[agent_id].add_scalar("reward_agent", r, episode + 1)
        # 保存模型

        if episode % args.save_episode == 0:
            # 保存replay buffer以及actor,以及critic
            print("各参数已经保存")
            mappo_red.save(episode=episode, dir=file_o.RED)  # save model
            # mappo_blue.save(episode=episode, dir=file_o.BLUE)
        print(f"episode:{episode} time:{time.time()-start_time}")
        if episode % evalute_episode == 0:
            evalute(args, env, mappo_red, None, writer_red, None, episode/evalute_episode, file_o)
        episode += 1


if __name__ == '__main__':
    run()
