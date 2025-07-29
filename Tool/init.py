import torch
from Env.scenario import scenario
# from maddpg.Agent import Actor_Network

# 初始化环境
def get_reset_env(evaluate, args):
    env = scenario(args=args, evalute=evaluate)
    _dim_info_red = {}
    _dim_info_blue = {}
    for agent_id in env.red_sat:
        _dim_info_red[agent_id] = []  # [obs_dim, act_dim]
        # todo 需要改一下，能不能改成自适应动作空间和状态空间
        _dim_info_red[agent_id].append(env.observation_space()["red"])
        _dim_info_red[agent_id].append(env.action_space()["red"])
        _dim_info_red[agent_id].append(env.observation_space()["global_red"])
    for agent_id in env.blue_sat:
        _dim_info_blue[agent_id] = []  # [obs_dim, act_dim]
        # todo 需要改一下，能不能改成自适应动作空间和状态空间
        _dim_info_blue[agent_id].append(env.observation_space()["blue"])
        _dim_info_blue[agent_id].append(env.action_space()["blue"])
        _dim_info_blue[agent_id].append(env.observation_space()["global_blue"])
    return env, _dim_info_red, _dim_info_blue


# create_net(args=args, red_path=args.red_actor_path, dim_inf_red=dim_info_red, blue_path=blue_actor_path, dim_inf_blue=dim_info_blue)
# 创建一个类，导入actor。输入环境的信息包，输出的是决策
class create_net:
    def __init__(self, args, red_path, dim_inf_red, blue_path, dim_inf_blue):

        self.args = args

        self.actor_blue = Actor_Network(in_dim=dim_inf_blue["b0"][0], out_dim=dim_inf_blue["b0"][1]).to(args.device)
        self.actor_blue.load_state_dict(torch.load(blue_path)["b0"])

        self.actor_red = (Actor_Network(in_dim=dim_inf_red["r0"][0], out_dim=dim_inf_red["r0"][1]).to(args.device))
        self.actor_red.load_state_dict(torch.load(red_path)["r0"])

    def single_act(self, obs, side):

        obs = torch.from_numpy(obs).float().to(self.args.device)
        if side =="blue":
            mx, my, mz = self.actor_blue(obs)
            logits = torch.cat((mx, my, mz), dim=0).squeeze(0).cpu().detach().float().numpy()
        elif side == "red":
            mx, my, mz = self.actor_red(obs)
            logits = torch.cat((mx, my, mz), dim=0).squeeze(0).cpu().detach().float().numpy()
        else:
            raise Exception("side must be either 'blue' or 'red'") #抛出异常
        return logits

    def act_select(self, obs: dict):
        act = {}
        for agent_id, obs in obs.items():
            if agent_id[0] == 'b':
                act[agent_id] = self.single_act(obs=obs, side="blue")
            elif agent_id[0] == "r":
                act[agent_id] = self.single_act(obs=obs, side="red")
        return act