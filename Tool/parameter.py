import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='orbit_sim')
    # 场景布置
    parser.add_argument('--episode_length', type=int, default=120)
    parser.add_argument("--step_time", type=int, default=300)
    parser.add_argument('--red_num', type=int, default=3)
    parser.add_argument('--blue_num', type=int, default=3)
    parser.add_argument('--red_delta_v_limit', type=int, default=1)
    parser.add_argument('--blue_delta_v_limit', type=int, default=0.5)
    parser.add_argument('--init_distance', type=int, default=200)
    parser.add_argument('--done_distance', type=int, default=70)
    parser.add_argument('--safe_dis', type=int, default=2)
    parser.add_argument('--comm_dis', type=int, default=70)
    parser.add_argument('--visual_flag', type=bool, default=False)
    parser.add_argument('--fast_calculate', type=bool, default=True)
    parser.add_argument("--orbit_alt", type=int, default=42157)
    # 仿真参数
    parser.add_argument('--max_step', type=int, default=2400000)
    parser.add_argument('--learn_interval', type=int, default=5)
    parser.add_argument('--save_episode', type=int, default=4000)
    parser.add_argument('--buffer_capacity', type=int, default=1024)
    parser.add_argument('--a_optim_batch_size', type=int, default=128)
    parser.add_argument('--c_optim_batch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--actor_lr', type=float, default=0.0003)
    parser.add_argument('--critic_lr', type=float, default=0.0003)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')

    parser.add_argument("--result_save_path", type=str, default=r"D:\shen\result\\")

    args = parser.parse_args()

    return args