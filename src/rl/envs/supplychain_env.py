import gym
from gym import spaces
import numpy as np

class SupplyChainEnv(gym.Env):
    """Custom env that uses predictor outputs as part of the observation."""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_nodes=10, seed=42):
        super().__init__()
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
        # observation: status (utilization) + predicted risk per node in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_nodes*2,), dtype=np.float32)
        # actions: 0 none, 1 reroute, 2 buffer, 3 mode-switch
        self.action_space = spaces.Discrete(4)
        self.state = None
        self.step_count = 0
        self.budget = 1.0  # simple cost budget per episode

    def reset(self):
        self.state = self.rng.random(self.observation_space.shape).astype(np.float32)
        self.step_count = 0
        self.budget = 1.0
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        util = self.state[:self.num_nodes]
        risk = self.state[self.num_nodes:]

        # Costs and effects
        action_costs = {0:0.0, 1:0.05, 2:0.07, 3:0.09}
        cost = action_costs[int(action)]
        self.budget -= cost

        # Effect: actions reduce risk; none slightly increases expected disruption
        if action == 0:
            risk = np.clip(risk + 0.02*self.rng.random(risk.shape), 0.0, 1.0)
        elif action == 1:
            risk = np.clip(risk - 0.1*self.rng.random(risk.shape), 0.0, 1.0)
        elif action == 2:
            risk = np.clip(risk - 0.12*self.rng.random(risk.shape), 0.0, 1.0)
        elif action == 3:
            risk = np.clip(risk - 0.15*self.rng.random(risk.shape), 0.0, 1.0)

        # Utility drifts
        util = np.clip(util + 0.01*self.rng.standard_normal(util.shape), 0.0, 1.0)

        # Compose next state
        self.state = np.concatenate([util, risk]).astype(np.float32)

        # Reward = negative of expected disruption (avg risk) minus cost
        expected_disruption = float(np.mean(risk))
        reward = - expected_disruption - cost

        self.step_count += 1
        done = self.step_count >= 100 or self.budget <= 0.0
        info = {"budget": self.budget, "expected_disruption": expected_disruption, "cost": cost}
        return self.state, reward, done, info

    def render(self, mode='human'):
        pass
