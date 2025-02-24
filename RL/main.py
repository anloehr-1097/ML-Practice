from typing import Tuple, List
import random
from dataclasses import dataclass
import numpy as np  # type: ignore

import gymnasium as gym  # type: ignore
import torch  # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore

import debugpy

# debugpy.listen(("localhost", 5678))
# debugpy.wait_for_client()


@dataclass
class ReplayBuffer:
    buffer: list[Tuple]
    max_size: int = int(10 * 1e9)
    min_size: int = 10000

    def add(self, data: Tuple):
        self.buffer.append(data)
        self.make_space()

    def __len__(self):
        return len(self.buffer)

    def make_space(self):
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.min_size:]


class OfflineRLData(Dataset):

    def __init__(self, rp_buf: ReplayBuffer, size=100):
        assert size <= len(rp_buf.buffer)
        self.indices = np.random.randint(0, len(rp_buf.buffer), size)

        self.data = [rp_buf.buffer[i] for i in self.indices]

    def __len__(self):
        return self.indices.size

    def __getitem__(self, idx):
        return self.data[idx]


class Policy(torch.nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def select_random_action(): return torch.randint(0, 4, (1,))[0]


def collect_data(policy: Policy, replay_buffer: ReplayBuffer, env: gym.core.Env):

    EXPLORE_PROB: float = 0.1
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        cur_obs = observation

        if random.random() < EXPLORE_PROB:
            action = select_random_action()
        else:
            action = torch.argmax(policy.forward(torch.from_numpy(cur_obs)))

        observation, reward, terminated, truncated, info = env.step(action.item())

        replay_buffer.add((cur_obs.astype(float), action, reward, observation.astype(float)))

        if terminated or truncated:
            observation, info = env.reset()

    return None


def meta_train_loop(env: gym.core.Env, policy: Policy, critic: Policy, replay_buffer: ReplayBuffer):

    UPDATE_CRITIC: int = 3  # C in paper
    NUM_EPOCHS: int = 5
    COLLECT_DATA: int = 3

    for i in range(100):
        print(f"Meta loop: {i}")
        # meta loop
        # collect data using policy
        if i % COLLECT_DATA == 0:
            collect_data(policy, replay_buffer, env)

        # create dataset and dataloader
        dataset = OfflineRLData(replay_buffer, size=900)
        dataloader = DataLoader(dataset, batch_size=16)

        # train 3 epochs using data
        for j in range(NUM_EPOCHS):
            print(f"Train Epoch: {j}")
            train_one_epoch(policy, critic, dataloader, gamma=0.99)

        # update critic
        if i % UPDATE_CRITIC == 0:
            critic.load_state_dict(policy.state_dict())

    env.close()
    return None


def q_learning_loss(
    immediate_return: torch.Tensor,
    q_values: torch.Tensor,
    target_q_values: torch.Tensor,
        discount_factor: float) -> torch.Tensor:
    target_state_action_return: torch.Tensor = immediate_return + discount_factor * torch.argmax(target_q_values, dim=1)
    # return torch.nn.MSELoss()(q_values, target_state_action_return).to(torch.float)
    return torch.nn.functional.mse_loss(q_values, target_state_action_return).to(torch.float)


def train_one_epoch(policy: Policy, critic: Policy, data: DataLoader, gamma: float):
    # sample_indices: np.ndarray = np.random.choice(len(replay_buffer.buffer), buf_excerpt_size)
    # sample: List = [replay_buffer.buffer[i] for i in sample_indices]
    critic.eval()
    policy.train()

    optim: torch.optim.Optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optim.zero_grad()

    total_loss = 0
    for sample in data:
        # single batch
        cur_obs, action, reward, next_obs = sample
        cur_obs = cur_obs.to(dtype=torch.float32)

        next_obs = next_obs.to(dtype=torch.float32)
        q_values = policy(cur_obs)[torch.arange(action.shape[0]), action].to(torch.float)
        target_q_values = critic(next_obs).to(torch.float)
        immediate_return = reward.to(torch.float)
        loss = q_learning_loss(immediate_return, q_values, target_q_values, gamma)
        loss.backward()
        optim.step()
        optim.zero_grad()
        total_loss += loss.item()

    print(f"Total Loss: {total_loss}")
    return None


def main():

    policy = Policy()
    critic = Policy()
    replay_buffer = ReplayBuffer(buffer=[])
    env = gym.make("LunarLander-v3", render_mode="human")

    meta_train_loop(env, policy, critic, replay_buffer)

    # observation, info = env.reset(seed=42)
    # for _ in range(10):
    #     cur_obs = observation
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #
    #     replay_buffer.add((cur_obs, action, reward, observation))
    #
    #     if terminated or truncated:
    #         observation, info = env.reset()
    # env.close()
    #
    # print(replay_buffer.buffer)
    return None


if __name__ == "__main__":
    main()
