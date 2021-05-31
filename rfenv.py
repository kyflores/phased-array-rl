import elements as elem
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

class PhasedArrayEnv(gym.Env):
    def __init__(self):
        self.elements = 7
        self.seed()
        self.rng = np.random.RandomState()
        self.rng.seed(1234)
        self.antennas = []
        # gain is unaffected by frequency
        self.phase_fn = lambda x: 1
        self.ts = np.linspace(0, 0.5*np.pi*1e-10, 50)

        self.waves = [elem.Wave(24e+9, 1, 0, np.array((5, 0)))]
        self.anten = []
        for i in range(self.elements):
            self.anten.append(elem.Antenna(np.array((0, i)), self.rng.rand(1) * 2 * np.pi, self.phase_fn))

        self.phased_array = elem.PhasedArray(self.waves, self.anten)

        # Phases of each antenna
        self.action_space = spaces.Box(
            low = np.array([0.0]*self.elements),
            high = np.array([2*np.pi]*self.elements),
            dtype = np.float32
        )

        # Power of waves 1-4 summed with wave 0.
        self.observation_space = spaces.Box(
            low = np.array([0]*(self.elements-1)),
            # Waves are "2" peak to peak, and these are the sum of 2 waves
            high = np.array([4]*(self.elements-1)),
            dtype = np.float32
        )
    
    def step(self, action):
        self.phased_array.set_phases(action)

        power, obs = self.collect_powers()
    
        return obs, power, False, {}
    
    def collect_powers(self):
        agg, yss = self.phased_array.get_time_series(self.ts)
        # Saving old values to render in matplot
        self.agg = agg
        self.yss = yss

        power = np.max(agg) - np.min(agg)
    
        # Calculate interference pattern of element 0 with
        # every other element
        pairwise_pwr = np.zeros(self.elements-1, dtype = np.float32)
        for idx in range(1, self.elements-1):
            res = yss[idx] + yss[0]
            pairwise_pwr[idx -1] = np.max(res) - np.min(res)
        
        return power, pairwise_pwr


    def reset(self):
        print("ENV reset!")
        anten = []
        phases = []
        for i in range(self.elements):
            rand_phase = self.rng.rand(1) * 2 * np.pi
            phases.append(rand_phase)
            anten.append(elem.Antenna(np.array((0, i)), rand_phase, self.phase_fn))
        self.anten = anten
        self.phased_array = elem.PhasedArray(self.waves, self.anten)
        print("Randomized initial phases are", phases)

        _, obs = self.collect_powers()

        return obs

    def render(self, mode='human'):
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([0, self.ts[-1]])
        axes.set_ylim([-10,10])

        for series in self.yss:
            plt.plot(self.ts, series)
        plt.plot(self.ts, self.agg)
        plt.pause(1) 

    def close(self):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]