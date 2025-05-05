import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# DQN model with convolutional layers to process game frames
class DQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = 3136  # For 84x84 input after conv layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0  # Normalize input to [0,1]
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# Simple replay memory to store game experiences
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Set up the game environment with preprocessing
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)  # (240,256,1)
    env = ResizeObservation(env, 84)  # (84,84,1)
    # Handle different gym versions for FrameStack
    try:
        env = FrameStack(env, 4, enable_lazy=True)
    except TypeError:
        env = FrameStack(env, 4)
    return env

# Convert environment observations to the right format
def obs_to_state(obs):
    state = np.array(obs)  # (84,84,4)
    state = np.transpose(state, (2, 0, 1))  # (4,84,84)
    return state.astype(np.uint8)

# Training function for local use to generate the model
def train():
    BATCH_SIZE = 32
    GAMMA = 0.99
    REPLAY_SIZE = 100_000
    LEARNING_RATE = 1e-4
    TARGET_SYNC = 1000
    START_EPS = 1.0
    END_EPS = 0.05
    EPS_DECAY_FR = 1_000_000
    MAX_EPISODES = 1000
    device = torch.device("cpu")

    env = make_env()
    n_actions = env.action_space.n
    policy_net = DQN(4, n_actions).to(device)
    target_net = DQN(4, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(REPLAY_SIZE)

    frame = 0
    for episode in range(MAX_EPISODES):
        state = env.reset()
        state = obs_to_state(state)
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 10000:
            frame += 1
            step += 1
            eps = max(END_EPS, START_EPS - (START_EPS - END_EPS) * frame / EPS_DECAY_FR)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(1).item()

            next_obs, reward, done, _ = env.step(action)
            next_state = obs_to_state(next_obs)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(memory) >= BATCH_SIZE * 100:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(BATCH_SIZE)

                state_batch = torch.tensor(state_batch, dtype=torch.uint8, device=device)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.uint8, device=device)
                action_batch = torch.tensor(action_batch, dtype=torch.long, device=device).unsqueeze(1)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                done_batch = torch.tensor(done_batch, dtype=torch.bool, device=device)

                q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

                with torch.no_grad():
                    next_actions = policy_net(next_state_batch).argmax(1, keepdim=True)
                    next_q = target_net(next_state_batch).gather(1, next_actions).squeeze(1)
                    target_q = reward_batch + GAMMA * next_q * (~done_batch)

                loss = nn.SmoothL1Loss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if frame % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}, Reward: {episode_reward}, Epsilon: {eps:.2f}")

        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), "mario_dqn.pth")

    torch.save(policy_net.state_dict(), "mario_dqn.pth")
    env.close()

# Agent class for the leaderboard
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cpu")
        self.frame_stack = 4
        self.frames = collections.deque(maxlen=self.frame_stack)

        self.policy_net = DQN(4, self.action_space.n).to(self.device)
        try:
            self.policy_net.load_state_dict(torch.load("mario_dqn.pth", map_location=self.device, weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError("Error: 'mario_dqn.pth' not found. Please train the model locally using `python student_agent.py` and upload 'mario_dqn.pth' to the repository.")
        self.policy_net.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])

        self.reset()

    def reset(self):
        self.frames.clear()
        dummy_frame = torch.zeros((84, 84))
        for _ in range(self.frame_stack):
            self.frames.append(dummy_frame)

    def act(self, observation):
        processed_frame = self.transform(observation).squeeze(0)  # (84,84)
        self.frames.append(processed_frame)

        stacked_frames = torch.stack(list(self.frames), dim=0).to(self.device)  # (4,84,84)
        stacked_frames = stacked_frames.unsqueeze(0)  # (1,4,84,84)

        with torch.no_grad():
            q_values = self.policy_net(stacked_frames)
            action = q_values.argmax(1).item()

        return action

# Test the agent locally
if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    agent = Agent()

    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  

    print(f"Episode finished with total reward: {total_reward}")
    print(f"Final position: {info.get('x_pos', 0)}")
    env.close()