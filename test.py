import elements as elem
import rfenv
import numpy as np
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def test_plot():
    wave = elem.Wave(24e+9, 1, 0, np.array((0, 0)))
    ant0 = elem.Antenna(np.array((0, 1)), np.pi/2, lambda x: 1)
    ant1 = elem.Antenna(np.array((0, -1)), 0, lambda x: 1)

    phased_array = elem.PhasedArray([wave], [ant0, ant1])

    ts = np.linspace(0, 0.2*np.pi*1e-10, 100)

    plt.plot(ts, wave.make_timeseries(ts))
    plt.plot(ts, phased_array.get_time_series(ts))
    phased_array.set_phases([-np.pi/2, 0])
    plt.plot(ts, phased_array.get_time_series(ts))
    phased_array.set_phases([np.pi/2, 0])
    plt.plot(ts, phased_array.get_time_series(ts))
    plt.show()

def sample():
    env = rfenv.PhasedArrayEnv()
    env.reset()
    for _ in range(20):
        (obs, re, _, _) = env.step(env.action_space.sample())
        print(obs,re)
        env.render()

def learn():
    # env = gym.make("CartPole-v1")
    env = rfenv.PhasedArrayEnv()
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, device = "cpu")

    model.learn(total_timesteps=10000)
    model.save("./parl-model.pkl")
    print("Learning done and model saved.")

def predict():
    env = rfenv.PhasedArrayEnv()
    model = PPO.load("./parl-model.pkl", env=env)

    for i in range(10):
        obs = env.reset()
        print("Scenario", i)
        input("Press to view")
        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, info = env.step(action)
            print(reward)
            env.render()
            if done:
              obs = env.reset()

if __name__ == '__main__':
    learn()