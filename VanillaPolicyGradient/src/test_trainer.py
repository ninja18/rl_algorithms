import time

import gym
import matplotlib.pyplot as plt
import torch
from torchviz import make_dot

from src.VanillaPolicyGradient.agent import MLPAgent
from src.VanillaPolicyGradient.train import Trainer


def test_discounted_rewards():
    trainer = Trainer(None, [1, 1, 1], MLPAgent(1, [1, 1, 1]), 1, 1, gamma=1.0)
    trainer.rewards = [1, 1, 1, 1, 1]
    expected_rewards = torch.Tensor([1.2649, 0.6325, 0, -0.6325, -1.2649])
    assert torch.allclose(trainer.discounted_rewards(), expected_rewards, rtol=1e-3)


def test_preprocess():
    trainer = Trainer(None, [80, 80, 1], MLPAgent(1, [1, 1, 1]), 1, 1)
    env = gym.make("Pong-v0")
    obs = env.reset()
    new_obs = trainer.pre_process(obs)
    plt.imshow(new_obs.view(80, 80).numpy())
    plt.show()


def test_single_pass():
    env = gym.make("Pong-v0")
    agent = MLPAgent(2, 6400, layer_sizes=[200])
    trainer = Trainer(env, [80, 80, 1], agent, 0.0005, 1,
                      custom_actions=torch.tensor([2, 3], dtype=torch.int16),
                      save_per_epoch=50,
                      gamma=0.99,
                      batch_size=10, render=False)
    start = time.time()
    trainer.train()
    print(f"One iteration takes : {time.time() - start}")


def test_agent():
    env = gym.make("Pong-v0")
    agent = MLPAgent(2, 6400, layer_sizes=[200])
    obs = env.reset()
    obs = obs[35:195]  # crop
    obs = obs[::2, ::2, 0]
    obst = torch.from_numpy(obs).type(torch.FloatTensor)
    out = agent(obst)
    viz = make_dot(out)
    viz.view()
