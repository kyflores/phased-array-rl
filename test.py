from torch._C import device
import rfenv_simple as rfenv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env

def sample():
    env = rfenv.PhasedArrayEnv(3, 1, 1)
    env.reset()
    (obs, re, _, _) = env.step(np.array([0, 0, 0]))
    env.render()
    input("obs: {}".format(obs))
    input("re: {}".format(re))
    (obs, re, _, _) = env.step(np.array([0, 1, 2]))
    env.render()
    input("obs: {}".format(obs))
    input("re: {}".format(re))
    # for _ in range(100):
    #     (obs, re, _, _) = env.step(env.action_space.sample())
    #     print(obs,re)
    #     env.render()

def learn():
    # env = gym.make("CartPole-v1")
    env = rfenv.PhasedArrayEnv(3, 1, 1)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)

    model.learn(total_timesteps=500_000)
    model.save("./parl-model.pkl")
    print("Learning done and model saved.")

def predict():
    env = rfenv.PhasedArrayEnv(3, 1, 1)
    model = PPO.load("./parl-model.pkl", env=env)

    for i in range(10):
        obs = env.reset()
        print("Scenario", i)
        input("Press to view")
        env.render()
        for i in range(20):
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, info = env.step(action)
            print(reward)
            env.render()
            if done:
                obs = env.reset()
                break

if __name__ == '__main__':
    # sample()
    learn()
    predict()