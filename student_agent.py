# I will import all the libraries I need for the agent and training
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

# I will define the DQN model here, which is like a neural network for picking actions
class DQN(nn.Module):
    # I will set up the network with convolutional layers to process game frames
    def __init__(self, input_channels, n_actions):
        super(DQN, self).__init__()
        # I will create conv layers to extract features from 84x84 images
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # I calculated this size (3136) for 84x84 input after conv layers
        conv_out_size = 3136
        # I will add fully connected layers to output Q-values for actions
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # I will define how the network processes input frames
    def forward(self, x):
        # I will normalize the input to [0,1] for better training
        x = x.float() / 255.0
        # I will pass the input through conv layers and flatten it
        conv_out = self.conv(x).view(x.size()[0], -1)
        # I will get Q-values for each action
        return self.fc(conv_out)

# I will create a simple replay memory to store game experiences
class ReplayMemory:
    # I will initialize the memory with a fixed capacity
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    # I will add a new experience to the memory
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # I will sample a batch of experiences for training
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    # I will return the size of the memory
    def __len__(self):
        return len(self.buffer)

# I will set up the game environment with preprocessing
def make_env():
    # I will create the Super Mario Bros environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # I will use COMPLEX_MOVEMENT for 12 actions
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # I will convert frames to grayscale to simplify input
    env = GrayScaleObservation(env, keep_dim=True)  # (240,256,1)
    # I will resize frames to 84x84 to reduce computation
    env = ResizeObservation(env, 84)  # (84,84,1)
    # I will stack 4 frames to capture motion
    env = FrameStack(env, 4)  # (4,84,84)
    return env

# I will convert environment observations to the right format
def obs_to_state(obs):
    # I will convert LazyFrames to a numpy array
    state = np.array(obs)  # (84,84,4)
    # I will transpose to get (4,84,84) for the network
    state = np.transpose(state, (2, 0, 1))
    return state.astype(np.uint8)

# I will write the training function to train the DQN model
def train():
    # I will define hyperparameters for training
    BATCH_SIZE = 32
    GAMMA = 0.99  # I will use 0.99 for future reward discount
    REPLAY_SIZE = 100_000  # I will store 100,000 experiences
    LEARNING_RATE = 1e-4  # I will use a small learning rate
    TARGET_SYNC = 1000  # I will update target network every 1000 steps
    START_EPS = 1.0  # I will start with random actions
    END_EPS = 0.05  # I will reduce exploration to 5%
    EPS_DECAY_FR = 1_000_000  # I will decay epsilon over 1M frames
    MAX_EPISODES = 1000  # I will train for 1000 episodes
    device = torch.device("cpu")  # I will use CPU for leaderboard

    # I will set up the environment
    env = make_env()
    n_actions = env.action_space.n
    # I will create policy and target networks
    policy_net = DQN(4, n_actions).to(device)
    target_net = DQN(4, n_actions).to(device)
    # I will copy policy weights to target network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # I will set target network to evaluation mode

    # I will set up the optimizer and memory
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(REPLAY_SIZE)

    frame = 0
    # I will loop through episodes
    for episode in range(MAX_EPISODES):
        state = env.reset()
        state = obs_to_state(state)
        episode_reward = 0
        done = False
        step = 0

        # I will play the game until done or max steps
        while not done and step < 10000:
            frame += 1
            step += 1
            # I will calculate epsilon for exploration
            eps = max(END_EPS, START_EPS - (START_EPS - END_EPS) * frame / EPS_DECAY_FR)

            # I will choose an action using epsilon-greedy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.uint8, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(1).item()

            # I will take a step in the environment
            next_obs, reward, done, _ = env.step(action)
            next_state = obs_to_state(next_obs)
            # I will store the experience
            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # I will train when enough experiences are collected
            if len(memory) >= BATCH_SIZE * 100:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(BATCH_SIZE)

                # I will convert batches to tensors
                state_batch = torch.tensor(state_batch, dtype=torch.uint8, device=device)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.uint8, device=device)
                action_batch = torch.tensor(action_batch, dtype=torch.long, device=device).unsqueeze(1)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                done_batch = torch.tensor(done_batch, dtype=torch.bool, device=device)

                # I will compute Q-values for current state
                q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

                # I will use Double DQN for target Q-values
                with torch.no_grad():
                    next_actions = policy_net(next_state_batch).argmax(1, keepdim=True)
                    next_q = target_net(next_state_batch).gather(1, next_actions).squeeze(1)
                    target_q = reward_batch + GAMMA * next_q * (~done_batch)

                # I will compute loss and optimize
                loss = nn.SmoothL1Loss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # I will sync target network periodically
            if frame % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # I will print training progress
        print(f"Episode {episode+1}, Reward: {episode_reward}, Epsilon: {eps:.2f}")

        # I will save the model every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), "mario_dqn.pth")

    # I will save the final model
    torch.save(policy_net.state_dict(), "mario_dqn.pth")
    env.close()

# I will define the Agent class for the leaderboard
class Agent(object):
    """Agent that acts using a pre-trained DQN model."""
    # I will initialize the agent with the required structure
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        # I will use CPU for leaderboard submission
        self.device = torch.device("cpu")
        # I will stack 4 frames for input
        self.frame_stack = 4
        self.frames = collections.deque(maxlen=self.frame_stack)

        # I will create the DQN model
        self.policy_net = DQN(4, self.action_space.n).to(self.device)
        # I will try to load the trained model
        try:
            self.policy_net.load_state_dict(torch.load("mario_dqn.pth", map_location=self.device))
        except FileNotFoundError:
            # I will train if the model is missing
            print("Model file 'mario_dqn.pth' not found. Running training...")
            train()
            self.policy_net.load_state_dict(torch.load("mario_dqn.pth", map_location=self.device))
        # I will set the network to evaluation mode
        self.policy_net.eval()

        # I will set up image preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])

        # I will reset the frame stack
        self.reset()

    # I will create a helper to reset the frame stack
    def reset(self):
        # I will clear the frame stack and fill with zeros
        self.frames.clear()
        dummy_frame = torch.zeros((84, 84))
        for _ in range(self.frame_stack):
            self.frames.append(dummy_frame)

    # I will implement the act function to choose actions
    def act(self, observation):
        # I will preprocess the raw RGB frame
        processed_frame = self.transform(observation).squeeze(0)  # (84,84)
        # I will add the frame to the stack
        self.frames.append(processed_frame)

        # I will stack the frames for the network
        stacked_frames = torch.stack(list(self.frames), dim=0).to(self.device)  # (4,84,84)
        stacked_frames = stacked_frames.unsqueeze(0)  # (1,4,84,84)

        # I will get the action with the highest Q-value
        with torch.no_grad():
            q_values = self.policy_net(stacked_frames)
            action = q_values.argmax(1).item()

        return action

# I will add a main block to test the agent
if __name__ == "__main__":
    # I will set up the environment for testing
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # I will create the agent
    agent = Agent()

    # I will run one episode
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # env.render()  # I will comment this out for submission

    # I will print the results
    print(f"Episode finished with total reward: {total_reward}")
    print(f"Final position: {info.get('x_pos', 0)}")
    env.close()