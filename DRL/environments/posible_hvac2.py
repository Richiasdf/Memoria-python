

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class HVACenv(gym.Env):
        environment_name = "HVAC"
        def __init__(self, use, weather):
            self.min_action = np.array([0, -30 , 0])
            self.max_action = np.array([5, 0, 0.9])
            self.min_goal_position = (22)
            self.max_goal_position = (25)
            self.weather = weather
            self.reward_threshold = 0.0
            self.id = "HVAC"
            self.trials = 100
            self.n_step = 0
            self.use = use
            self.good_steps = 0
            self.seed()
            self.params = params = {
                            "cp": 1012,
                            "C1": 9163000,
                            "C2": 169400000,
                            "R": 0.0017,
                            "Roa": 0.057,
                            "kf": 65,
                            "eta": 4,
                        }

            self.action_space = spaces.Box(
                low=self.min_action, high=self.max_action, dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low = np.array([-10, -10 , -10, 0]), high=np.array([50, 50, 50 , 10000000]), dtype=np.float32
            )

            
        def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

        def step(self, action, Ts= 60*15):
            air = self.state[0]
            solid = self.state[1]
            ms = min(max(action[0], self.min_action[0]), self.max_action[0])
            Tc = min(max(action[1], self.min_action[1]), self.max_action[1])
            delta = min(max(action[2], self.min_action[2]), self.max_action[2])
            Toa = self.weather[self.n_step]
            Use = self.use[self.n_step]
            self.state[0] = air+(Ts*ms*self.params["cp"]*((delta*air+(1-delta)*Toa+Tc)-air)+ Ts*(solid-air)/self.params["R"]+Ts*(Toa-air)/self.params["Roa"]+Ts*Use)/self.params["C1"]
            self.state[1] = solid+Ts*(air-solid)/self.params["R"]/self.params["C2"]; 
            self.state[2] = Toa
            self.state[3] = Use

            if self.max_goal_position >= self.state[0] >= self.min_goal_position:
                self.good_steps += 1
            # Convert a possible numpy bool to a Python bool.
            done = bool((self.max_goal_position >= self.state[0] >= self.min_goal_position and self.good_steps >= self.trials/3) or self.n_step == self.trials)

            reward = 0
            if not(self.max_goal_position >= self.state[0] >= self.min_goal_position):
                reward -= min(abs(self.state[0]-self.max_goal_position),abs(self.state[0]-self.min_goal_position))
            reward -= 0.03*(((self.params['cp']/4)*ms*Tc + self.params['kf']*math.pow(ms,2))/1000)
            self.n_step += 1
            #print(self.state[0:2])
            #print([ms, Tc, delta])
            return self.state, reward, done, {}

        def reset(self):
            self.state = np.array([self.np_random.uniform(low = 24, high = 36), self.np_random.uniform(low = 24, high = 34), self.weather[0], self.use[0]])
            self.good_steps = 0
            self.n_step = 0
            return np.array(self.state, dtype=np.float32)

        def _height(self, xs):
            return np.sin(3 * xs) * 0.45 + 0.55

        def close(self):
            if self.viewer:
                self.viewer.close()
                self.viewer = None