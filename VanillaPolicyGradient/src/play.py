import os

import gym
import torch
from torch.distributions import Categorical

from VanillaPolicyGradient.src.agent import MLPAgent


class Player:
    def __init__(self, env, agent, model_file=None, no_games=5, custom_actions=None):
        self.env = env
        self.agent = agent
        self.model_file = model_file
        self.no_games = no_games
        self.custom_actions = custom_actions

    def load_model(self):
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, map_location=torch.device('cpu'))
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            current_epoch = checkpoint['epoch']
            print(f"Loaded existing model continuing from epoch {current_epoch}")
            return
        else:
            print(f"model doesn't exists in the given path {self.model_file}")

    def play(self):
        self.load_model()
        self.agent.eval()
        for i in range(self.no_games):
            obs = self.env.reset()
            done = False
            rews = 0
            while not done:
                self.env.render()
                obs, rew, done, _ = self.act(obs)
                rews += rew
            print(f"Game ended with total reward {rews}")

    def preprocess(self, obs):
        # return self.obs_transform(obs).unsqueeze(0)
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2, 0]  # downsample by factor of 2
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        return torch.tensor(obs).type(torch.FloatTensor)

    def act(self, obs):
        probs = self.agent(self.preprocess(obs))
        dis = Categorical(probs)
        action = dis.sample()

        if self.custom_actions is not None:
            action = self.custom_actions[action.item()]
        return self.env.step(action.item())


if __name__ == '__main__': # Todo Remove train and test code duplications
    env = gym.make('Pong-v0')
    env = gym.wrappers.Monitor(env, "../recordings/recordingAfter5000episodes")
    agent = MLPAgent(2, 6400, layer_sizes=[200])
    player = Player(env, agent, model_file="../checkpoints/model5000ep.pt", custom_actions=torch.tensor([2, 3], dtype=torch.int16))
    player.play()
