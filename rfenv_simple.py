
import elements as elem
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

def shift_range(r0, r1, v):
    min0, max0 = r0
    min1, max1 = r1

    frac = v / (max0 - min0)

    return (frac * (max1 - min1)) + min1

class PhasedArrayEnv(gym.Env):
    def __init__(self, num_anten, freq, amp):
        self.num_anten = num_anten

        # Frequency of all waves
        self.freq = freq

        # Amplitude of all waves
        self.amp = amp

        self.max_power_all = 2 * self.num_anten * self.amp
        self.max_power_two = 2 * 2 * self.amp

        self.rng = np.random.RandomState()
        self.rng.seed()

        # Phase offsets of each antenna, randomly initialized.
        self.phase_init = self.rng.rand(self.num_anten)

        self.phases = np.zeros(self.num_anten)

        # Locations of each antenna, at x = 0, ascending on y axis
        # TODO: we currently don't calculate phase shift due to position
        # locs = zip([0] * self.num_anten), list(range(self.num_anten))
        # self.locations = np.array(locs)

        # Phases of each antenna
        self.action_space = spaces.Box(
            low = np.array([-1]*self.num_anten),
            high = np.array([1]*self.num_anten),
            dtype = np.float32
        )

        # Power of anten 0 + anten n; n = (1, n)
        self.observation_space = spaces.Box(
            low = np.array([-1]*(self.num_anten-1)),
            high = np.array([1]*(self.num_anten-1)),
            dtype = np.float32
        )

        self.reward_range = (0, 1)

        self.ts = np.linspace(-np.pi, np.pi, 100)

        # ???
        self.seed()
    
    def make_wave_series(self, ts, index):
        ys = self.amp * np.cos(
            2 * np.pi * self.freq * ts + (self.phase_init[index] + self.phases[index]))
        return ys
    
    # Returns the total power (all phases on), then
    # the powers of each phase with phase 0: [0 + 1, 0 + 2, ... 0 + n].
    def get_powers(self):
        agg = np.zeros_like(self.ts)
        for idx in range(self.num_anten):
            agg += self.make_wave_series(self.ts, idx)
        
        self.l_agg = agg
        
        ret = []
        ret.append(np.max(agg) - np.min(agg))

        wave_0 = self.make_wave_series(self.ts, 0)
        self.l_yss = [wave_0]
        for i in range(1, self.num_anten):
            wave_n = self.make_wave_series(self.ts, i)
            self.l_yss.append(wave_n)

            pair_wave = wave_0 + wave_n
            
            ret.append(np.max(pair_wave) - np.min(pair_wave))
        
        return ret
    

    def step(self, action):
        self.phases = action * np.pi

        powers = self.get_powers()

        done = False
        err = self.max_power_all - powers[0]
        print(err)
        if (err < 0.1):
            done = True
        
        # Normalize reward -1 to 1
        reward = shift_range((0,1), (-1, 1), powers[0] / self.max_power_all)

        obs = np.array(powers[1:])
        obs = shift_range((0, self.max_power_two), (-1, 1), obs)
        
        return obs, reward, done, {}

    def reset(self):
        self.phase_init = 2 * np.pi * self.rng.rand(self.num_anten)
        self.phases = np.zeros(self.num_anten)

        powers = self.get_powers()
        obs = np.array(powers[1:])
        obs = shift_range((0, self.max_power_two), (-1, 1), obs)
        # print("env reset, new offsets = ", self.phase_init)
        return obs

    def close(self):
        pass

    def render(self, mode = 'human'):
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([0, self.ts[-1]])
        axes.set_ylim([-self.max_power_all / 2, self.max_power_all / 2])

        for series in self.l_yss:
            plt.plot(self.ts, series)
        plt.plot(self.ts, self.l_agg)
        plt.pause(0.1) 
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]