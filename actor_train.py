# 系统包
import copy
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
# 自建包
from maddpg.MADDPG import MADDPG
from Tool.draw_picture import draw_distance
from Tool import File_Path
from Tool.init import get_reset_env
from Tool import parameter

if __name__ == '__main__':
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
    draw_distance_red = draw_distance(data_length=args.episode_length, save_dir_red=file_o.picture_distance_red,
                                      save_dir_blue=file_o.picture_distance_blue, args=args)

    env, dim_info_red, dim_info_blue = get_reset_env(red_num=args.red_num, blue_num=args.blue_num, evaluate=False, args=args)

    # 将r0 r1 r2绘图数据分别保存到三个文件夹中
    writer_red = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_red), agent_id)) for agent_id in
                  env.red_sat}
    writer_blue = {agent_id: SummaryWriter(os.path.join(str(file_o.tensor_draw_path_blue), agent_id)) for agent_id in
                   env.blue_sat}


    if LOAD:
        maddpg_red = MADDPG.load(dim_info=dim_info_red,
                                 load_dir=file_o.get_episode_path_load(run_th=LOAD_RUN_TIME, side="RED"),
                                 episode_num=LOAD_EP, args=args, load_buffer=False)

        maddpg_blue = MADDPG.load(dim_info=dim_info_blue,
                                  load_dir=file_o.get_episode_path_load(run_th=LOAD_RUN_TIME, side="BLUE"),
                                  episode_num=LOAD_EP, args=args, load_buffer=False)
    else:
        # 红方策略
        maddpg_red = MADDPG(dim_info_red, args.buffer_capacity, args.batch_size,
                            args.actor_lr, args.critic_lr, args)

        maddpg_blue = MADDPG(dim_info_blue, args.buffer_capacity, args.batch_size,
                            args.actor_lr, args.critic_lr, args)

    step = 0  # global step counter

    episode_rewards_red = {agent_id: np.empty(0) for agent_id in env.red_sat}
    episode_rewards_blue = {agent_id: np.empty(0) for agent_id in env.blue_sat}
    for episode in range(args.episode_num):
        # try:
        DONE_FLAG = False
        obs_red, obs_blue = env.reset()
        env.episode_num = episode
        this_ep_reward_sum_red = {agent_id: 0 for agent_id in env.red_sat}  # 每一个元素是代表一局当中获得奖励的总和
        this_ep_reward_sum_blue = {agent_id: 0 for agent_id in env.blue_sat}

        for i1 in tqdm(range(args.episode_length),desc=f"episode{episode+1}"):

            env.step_num = i1
            step += 1
            # 蓝方动作
            action_blue = maddpg_blue.select_action(obs_blue)
            # 红方动作
            action_red = maddpg_red.select_action(obs_red)
            # 蓝方红方动作合并输入到环境
            action = {**action_red, **action_blue}

            # 返回的action是经过mask之后的action
            next_obs_red, next_obs_blue, reward_red, reward_blue, done_red, done_blue = env.step(action)

            # 将采集到的数据都存储到buffer中
            action_red = copy.deepcopy({agent_id: action[agent_id] for agent_id in env.red_sat})
            maddpg_red.add(obs_red, action_red, reward_red, next_obs_red, done_red)

            action_blue = copy.deepcopy({agent_id: action[agent_id] for agent_id in env.blue_sat})
            maddpg_blue.add(obs_blue, action_blue, reward_blue, next_obs_blue, done_blue)

            for agent_id, r in reward_red.items():
                this_ep_reward_sum_red[agent_id] += r

            for agent_id, r in reward_blue.items():
                this_ep_reward_sum_blue[agent_id] += r

            if step % args.learn_interval == 0:  # learn every few steps
                if len(maddpg_red.buffers["r0"]) < args.batch_size:
                    batch_size = len(maddpg_red.buffers["r0"])
                else:
                    batch_size = args.batch_size
                maddpg_red.learn(batch_size, args.gamma, args)
                maddpg_red.update_target(args.tau)

                maddpg_blue.learn(batch_size, args.gamma, args)
                maddpg_blue.update_target(args.tau)

            # 绘制图片
            dict_dis_red = {name: np.linalg.norm(env.sim.inf.pos[name] - env.sim.inf.pos["b" + name[1]]) for name in env.red_sat}
            dict_dis_blue = {name: np.linalg.norm(env.sim.inf.pos[name] - env.sim.inf.pos["r" + name[1]]) for name in env.blue_sat}
            dict_dis = {**dict_dis_red, **dict_dis_blue}


            # 蓝方卫星都死亡
            if sum(list(done_blue.values()))==len(list(done_blue.values())):
                print("已成功完成任务!")
                DONE_FLAG = True
                draw_distance_red.update_data(dict_dis, episode=episode, done_flag=DONE_FLAG)
                break  # 一开始编程错误，忽略了break前保存图片
            draw_distance_red.update_data(dict_dis, episode=episode, done_flag=DONE_FLAG)

            with open(os.path.join(file_o.run_path, 'reward_log.txt'), 'a') as f:
                title = f"ep{episode} step{i1}:\t"
                dis = "dis: "
                vel = "vel: "
                for name in env.red_sat:
                    red_name = name
                    blue_name = "b" + name[1]
                    dis += name + f":{format(np.linalg.norm(env.sim.inf.pos_cw[name]-env.sim.inf.pos_cw['b'+name[1]]), '.2f')}  "
                    vel += name + f":{format(np.linalg.norm(env.sim.inf.vel_cw[name]-env.sim.inf.vel_cw['b'+name[1]]), '.5f')}  "
                message = title + dis + "\t" + vel + "\n"
                if i1 == args.episode_length-1 or DONE_FLAG:
                    message+="\n"
                f.write(message)

            obs_red = next_obs_red
            obs_blue = next_obs_blue


        # episode finishes
        for agent_id, r in this_ep_reward_sum_red.items():  # record reward
            episode_rewards_red[agent_id] = np.append(episode_rewards_red[agent_id], r)

        for agent_id, r in this_ep_reward_sum_blue.items():  # record reward
            episode_rewards_blue[agent_id] = np.append(episode_rewards_blue[agent_id], r)
        # 更新奖励数组，用于保存到tensorboard的绘图文件中
        for agent_id, r in this_ep_reward_sum_red.items():  # record reward
            writer_red[agent_id].add_scalar("reward_agent", r, episode + 1)

        for agent_id, r in this_ep_reward_sum_blue.items():  # record reward
            writer_blue[agent_id].add_scalar("reward_agent", r, episode + 1)

        if ((episode+1)%args.save_episode==0):
            # 保存replay buffer以及actor,以及critic
            print("各参数已经保存")
            maddpg_red.save(reward=episode_rewards_red, episode=episode+1, dir=file_o.RED)  # save model
            maddpg_blue.save(reward=episode_rewards_blue, episode=episode + 1, dir=file_o.BLUE)
        # except Exception as e:
        #     # 获取堆栈跟踪信息
        #     error_info = traceback.format_exc()
        #     # 将堆栈跟踪信息写入到文件
        #     with open(os.path.join(file_o.erroe_log_dir, 'error_log.txt'), 'a') as f:
        #         f.write(error_info + "\n\n")
        #     maddpg_red.save(reward=episode_rewards_red, episode=episode + 1)  # save model
        #     env, obs_red, obs_blue, _dim_info_red, _dim_info_blue = get_reset_env(args=args)
        #     draw_distance.data_dict = {name: [] for name in draw_distance.sat}
