import os
import sys
import time
import torch
sys.path.insert(0, '/home/jaeho/jaeho_ws/f16_combat/src/NeuralPlane')
from envs.control_env import ControlEnv
from algorithms.ppo.ppo_actor import PPOActor


class Args:
    def __init__(self):
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = False


def _t2n(x):
    return x.detach().cpu().numpy()


def run_eval(name, policy, num_envs, steps, device, seed):
    env = ControlEnv(num_envs=num_envs, config='control', model='F16', random_seed=seed, device=device)
    obs = env.reset()
    rnn_states = torch.zeros((env.n, 1, 128), device=torch.device(device))
    masks = torch.ones((env.n, 1), device=torch.device(device))

    total_done = 0
    total_bad_done = 0
    total_reward = 0.0

    start = time.time()
    for t in range(steps):
        actions, _, rnn_states = policy(obs, rnn_states, masks, deterministic=True)
        obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(actions, render=False)
        total_done += int(_t2n(dones).sum())
        total_bad_done += int(_t2n(bad_dones).sum())
        total_reward += float(_t2n(rewards).sum())
        if (t + 1) % 500 == 0:
            print(f'[{name}] step {t+1}/{steps}  episodes so far={total_done + total_bad_done}')
    elapsed = time.time() - start

    n_episodes = total_done + total_bad_done
    success_rate = total_done / n_episodes if n_episodes > 0 else float('nan')
    avg_reward_per_episode = total_reward / n_episodes if n_episodes > 0 else float('nan')
    avg_reward_per_step = total_reward / (num_envs * steps)

    print(f'\n=== {name} ===')
    print(f'num_envs={num_envs}, steps={steps}, wall time={elapsed:.1f}s')
    print(f'total episodes finished: {n_episodes}  (success={total_done}, fail={total_bad_done})')
    print(f'success rate: {success_rate:.4f}')
    print(f'avg reward per episode: {avg_reward_per_episode:.2f}')
    print(f'avg reward per step: {avg_reward_per_step:.4f}')
    return dict(name=name, n_episodes=n_episodes, success_rate=success_rate,
                avg_reward_per_episode=avg_reward_per_episode)


if __name__ == '__main__':
    device = 'cuda:0'
    num_envs = 200
    steps = 3000
    seed = 42

    probe_env = ControlEnv(num_envs=num_envs, config='control', model='F16', random_seed=seed, device=device)
    args = Args()

    trained_policy = PPOActor(args, probe_env.observation_space, probe_env.action_space, device=torch.device(device))
    trained_policy.eval()
    ego_run_dir = '/home/jaeho/jaeho_ws/f16_combat/src/NeuralPlane/scripts/runs/2026-11-10_19-52-40_Control_control_F16_ppo_v1/episode_2499'
    trained_policy.load_state_dict(torch.load(ego_run_dir + '/actor_latest.pt'))

    random_policy = PPOActor(args, probe_env.observation_space, probe_env.action_space, device=torch.device(device))
    random_policy.eval()  # freshly initialized weights, never trained -> acts as the "random/untrained" baseline

    del probe_env

    results = []
    results.append(run_eval('TRAINED (episode_2499)', trained_policy, num_envs, steps, device, seed))
    results.append(run_eval('RANDOM/UNTRAINED baseline', random_policy, num_envs, steps, device, seed))

    print('\n\n========== SUMMARY ==========')
    for r in results:
        print(f"{r['name']:30s}  episodes={r['n_episodes']:5d}  success_rate={r['success_rate']*100:5.1f}%  avg_reward/ep={r['avg_reward_per_episode']:8.1f}")
