import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='orbit_sim', help='name of the env')

    parser.add_argument('--episode_num', type=int, default=10000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=120, help='steps per episode')
    parser.add_argument("--step_time", type=int, default=300)
    parser.add_argument('--learn_interval', type=int, default=10,
                        help='steps interval between learning time')
    parser.add_argument('--action_std', type=float, default=0.2)
    parser.add_argument('--action_std_decay_rate', type=float, default=0.011)
    parser.add_argument('--min_action_std', type=float, default=0.00001)
    parser.add_argument('--action_std_decay_freq', type=int, default=500*120)


    parser.add_argument('--actor_lr', type=float, default=0.0003, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='learning rate of critic')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--visual_flag', type=bool, default=False)
    parser.add_argument('--fast_calculate', type=bool, default=True)
    parser.add_argument('--save_episode', type=int, default=1000)
    parser.add_argument("--orbit_alt", type=int, default=42157)

    # 场景布置
    parser.add_argument('--red_num', type=int, default=1)
    parser.add_argument('--blue_num', type=int, default=1)
    parser.add_argument('--red_delta_v_limit', type=float, default=2)
    parser.add_argument('--blue_delta_v_limit', type=float, default=1)
    parser.add_argument('--init_distance', type=int, default=200)
    parser.add_argument('--done_distance', type=int, default=70)


    # 价值网络训练的参数
    parser.add_argument('--value_batch_size', type=int, default=128)  # 折扣因子
    parser.add_argument('--value_gamma', type=float, default=0.96) #折扣因子
    parser.add_argument('--value_tau', type=float, default=0.02) # 更新target时，原网络与target网络的比值
    parser.add_argument('--value_lr', type=float, default=0.0001) # 学习率

    #路径
    parser.add_argument("--mission_server_path", type=str,
                        default=r"D:/shen/software/AfSim290/bin/mission_server.exe")
    parser.add_argument("--value_net_path", type=str,
                        default=r"E:\code_list\red_battle_blue\selected_actor\value_net.pt")
    parser.add_argument("--red_actor_path", type=str,
                        default=r"E:\code_list\red_battle_blue\selected_actor\red_actor.pt")
    parser.add_argument("--blue_actor_path", type=str,
                        default=r"E:\code_list\red_battle_blue\selected_actor\blue_actor.pt")
    parser.add_argument("--result_path", type=str,
                        default=r"D:\shen\result\PPO2\\")

    #PPO
    parser.add_argument("--traj_len", type=int, default=2048)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument('--K_epochs', type=int, default=10)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--a_optim_batch_size', type=int, default=128, help='lenth of sliced trajectory of actor')
    parser.add_argument('--c_optim_batch_size', type=int, default=128, help='lenth of sliced trajectory of critic')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')



    args = parser.parse_args()

    return args