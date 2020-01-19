import os

import gym
import torch
from torch import optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from VanillaPolicyGradient.src.agent import MLPAgent


class Trainer:

    def __init__(self, env, obs_dims,
                 agent, lr, epoch,
                 gamma=0.95, batch_size=1,
                 custom_actions=None,
                 model_file="../checkpoints/model.pt", save_per_epoch=1,
                 render=False):
        self.env = env
        self.dev = torch.device("cpu")
        self.obs_dims = obs_dims
        self.obs_transform = None  # Todo get transform
        # if len(self.obs_dims) > 1:
        #     self.obs_transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Grayscale(num_output_channels=self.obs_dims[2]),
        #         transforms.Resize(self.obs_dims[:2]),
        #         transforms.ToTensor(),
        #     ])
        self.agent = agent
        self.agent.to(self.dev)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.custom_actions = custom_actions
        self.epoch = epoch
        self.current_epoch = 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_file = model_file
        self.render = render
        self.save_per_epoch = save_per_epoch
        self.writer = SummaryWriter(log_dir="../runs/")
        self.load_model()
        self.log_probs = []
        self.rewards = []

    def load_model(self):
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            if self.current_epoch == self.epoch:
                self.epoch += self.epoch
            print(f"Loaded existing model continuing from epoch {self.current_epoch}")

    def act(self, obs):
        if len(obs.shape) == 1:
            probs = self.agent(torch.from_numpy(obs).type(torch.FloatTensor))
        else:
            probs = self.agent(self.pre_process(obs))

        dis = Categorical(probs)
        action = dis.sample()
        self.log_probs.append(dis.log_prob(action))
        if self.custom_actions is not None:
            action = self.custom_actions[action.item()]
        return self.env.step(action.item())

    def pre_process(self, obs):  # Todo preprocess in torch
        # return self.obs_transform(obs).unsqueeze(0)
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2, 0]  # downsample by factor of 2
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        return torch.tensor(obs).type(torch.FloatTensor).to(self.dev)

    def train(self):
        self.agent.train()
        for i in range(self.current_epoch, (self.epoch * self.batch_size) + 1):
            obs = self.env.reset()
            while True:
                obs, rew, done, _ = self.act(obs)
                self.rewards.append(rew)
                if self.render:
                    self.env.render()
                if done:
                    if i % self.batch_size == 0:
                        loss = self.update_policy()
                        total_reward = sum(self.rewards) / self.batch_size
                        epoch = i // self.batch_size
                        print(f"Epoch: {epoch}"
                              f"  Loss: {loss.item()}"
                              f"  Total Average Reward: {total_reward}")
                        self.writer.add_scalar("Loss", loss, epoch)
                        self.writer.add_scalar("reward_per_episode", total_reward, epoch)
                        self.writer.flush()
                        self.rewards.clear()
                        self.log_probs = []
                    break

            if i % self.save_per_epoch == 0:
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.model_file)

        self.env.close()

    def update_policy(self):
        norm_rewards = self.discounted_rewards()
        cumulative_reward = - torch.cat(self.log_probs).to(self.dev) * norm_rewards
        loss = torch.sum(cumulative_reward, -1)
        # viz = make_dot(loss)
        # viz.view()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def discounted_rewards(self):
        reward_accumulator = 0
        discounted_rewards = []
        for r in reversed(self.rewards):
            if r != 0:
                reward_accumulator = 0
            reward_accumulator = r + reward_accumulator * self.gamma
            discounted_rewards.insert(0, reward_accumulator)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.dev)
        norm_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-06)
        return norm_rewards


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    agent = MLPAgent(2, 6400, layer_sizes=[200])
    trainer = Trainer(env, [80, 80, 1], agent, 0.0005, 2000,
                      custom_actions=torch.tensor([2, 3], dtype=torch.int8),
                      save_per_epoch=100,
                      gamma=0.99,
                      batch_size=10, render=False)
    trainer.train()
