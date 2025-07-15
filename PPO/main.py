import os
from tqdm import tqdm
import copy

import Env.scenario as scenario
from torch.utils.tensorboard import SummaryWriter
from utils import str2bool, Action_adapter, Reward_adapter, evaluate_policy
from PPO import PPO_agent
import Tool.parameter as parameter
import Tool.File_Path as file_p

def main():
    args = parameter.get_args()
    # Build Env
    env = scenario.scenario(args = args)
    state_dim_red = env.observation_space()["red"]
    state_dim_blue = env.observation_space()["blue"]
    action_dim_red = env.action_space()["red"]
    action_dim_blue = env.action_space()["blue"]

    # 所有工程存储结果的地方
    file_result = args.result_save_path
    # 工程文件夹名称（文件夹代表不同的方法）
    current_folder = os.path.basename(os.getcwd())
    # 存放结果的文件夹
    result_path = file_result+current_folder+"\\"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_oper = file_p.file_operate(root_path=result_path)

    # create_Agent
    ppo_red = PPO_agent(args=args, state_dim=state_dim_red, action_dim=action_dim_red)
    ppo_blue = PPO_agent(args=args, state_dim=state_dim_blue, action_dim=action_dim_blue)
    writer_red = {agent_id: SummaryWriter(os.path.join(str(file_oper.tensor_draw_path_red), agent_id)) for agent_id in
                  env.red_sat}

    traj_lenth = 0
    for ep_num in range(args.episode_num):
        red_obs, blue_obs = env.reset() # Do not use opt.seed directly, or it can overfit to opt.seed
        ep_reward = 0
        '''Interact & trian'''
        for step in tqdm(range(0,args.episode_length),desc=f"episode{ep_num+1}"):
            '''Interact with Env'''
            a_red, logprob_a_red = ppo_red.select_action(red_obs["r0"], deterministic=False) # use stochastic when training

            a_blue, logprob_a_blue = ppo_blue.select_action(blue_obs["b0"], deterministic=False)  # use stochastic when training
            # act_red = Action_adapter(copy.deepcopy(a_red), max_action=1) #[0,1] to [-max,max]
            # act_blue = Action_adapter(copy.deepcopy(a_blue), max_action=1)  # [0,1] to [-max,max]

            red_obs_, blue_obs_, red_reward, blue_reward, red_done, blue_done = \
                env.step({"r0": a_red, "b0": a_blue}) # dw: dead&win; tr: truncated
            ep_reward += red_reward["r0"]

            '''Store the current transition'''
            ppo_red.put_data(red_obs["r0"], a_red, red_reward["r0"],
                             red_obs_["r0"], logprob_a_red, red_done["r0"],
                             (step != args.episode_length-1) and blue_done["b0"], idx = traj_lenth)
            ppo_blue.put_data(blue_obs["b0"], a_blue, blue_reward["b0"],
                              blue_obs_["b0"], logprob_a_blue, blue_done["b0"],
                              (step != args.episode_length-1) and blue_done["b0"], idx=traj_lenth)

            done = blue_done["b0"]
            red_obs = red_obs_
            blue_obs = blue_obs_

            traj_lenth += 1

            '''Update if its time'''
            if (traj_lenth+1) % args.buffer_capacity == 0:
                ppo_red.train()
                ppo_blue.train()
                traj_lenth = 0

            if(done): break

        writer_red["r0"].add_scalar("reward_agent", ep_reward, ep_num + 1)

    ppo_red.save(path=file_oper.run_path+"red"+'\\', timestep = args.episode_num)
    ppo_blue.save(path=file_oper.run_path+"blue"+'\\', timestep=args.episode_num)



if __name__ == '__main__':
    main()





