import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return np.argmax(act_values.data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            state = torch.tensor(state, dtype=torch.float32)
            target_f = self.model(state).data.numpy()[0].tolist()  # Convert to list

            # Ensure target_f is a list before attempting to modify it
            if isinstance(target_f, list) and len(target_f) > action:
                target_f[action] = target  # Modify the list

                target_f = torch.tensor([target_f], dtype=torch.float32)  # Convert back to tensor

                # Compute loss and optimize the model
                loss = self.criterion(self.model(state), target_f)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

