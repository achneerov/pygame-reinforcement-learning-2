import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(400, 1024)  # 400 input nodes, 1024 hidden nodes
        self.fc2 = nn.Linear(1024, 4)    # 1024 hidden nodes, 4 output nodes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
        self.model = DQN()  # Corrected: Initialize DQN model without any arguments
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        #print(q_values)
        selected = np.argmax(q_values.data.numpy())
        move = [0, 0, 0, 0]
        move[selected] = 1
        return move



    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state).data.numpy().tolist()  # Convert to list

            # Find the index of the action that was taken (where the 1 is)
            action_index = (np.argmax(action))
            #print(q_values)

            # Update the Q-value for the action that was taken
            q_values[action_index] = target

            # Convert back to tensor for loss computation
            target_f = torch.tensor([q_values], dtype=torch.float32)

            # Compute loss and optimize the model
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

