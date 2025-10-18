from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.rl.envs.supplychain_env import SupplyChainEnv

def train_rl(total_timesteps=10000, n_envs=2):
    env = make_vec_env(lambda: SupplyChainEnv(num_nodes=10), n_envs=n_envs)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save('experiments/ppo_supplychain')
    return 'experiments/ppo_supplychain.zip'

if __name__ == '__main__':
    train_rl()
