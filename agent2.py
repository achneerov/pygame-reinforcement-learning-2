#agent2.py

import numpy as np
from model2 import DQNAgent  # Assuming DQNAgent is implemented using PyTorch
from game2 import SnakeGameAi  # Assuming you have a SnakeGameAi class


class Agent:
    def __init__(self):
        self.game = SnakeGameAi()  # Initialize the Snake game environment
        self.agent = DQNAgent(state_size=20 * 20, action_size=4)  # Initialize DQN agent

    def preprocess_state(self, state):
        if not isinstance(state, list) or not all(isinstance(row, list) for row in state):
            print(f"Unexpected state type: {type(state)}")
            return None

        # Mapping symbols to numbers including 's' for snake body
        mapping = {'B': 0, '.': 1, 'H': 2, 'F': 3, 's': 4}

        numeric_state = [[mapping.get(symbol, 5) for symbol in row] for row in
                         state]  # 5 can be a default value or you can handle it differently

        flattened_state = np.array(numeric_state).flatten()

        return flattened_state if flattened_state.shape[0] == 400 else None

    def train(self, episodes, batch_size):
        for episode in range(episodes):
            state = self.game.get_board()
            state = self.preprocess_state(state)

            done = False
            step_count = 0  # Add a step count for debugging

            while not done:
                action = self.agent.act(state)

                next_state, reward, done = self.game.play_step(move=action)
                next_state = self.preprocess_state(next_state)

                # Ensure next_state is not None before proceeding
                if next_state is not None:
                    self.agent.remember(state, action, reward, next_state, done)

                    state = next_state

                    if len(self.agent.memory) > batch_size:
                        self.agent.replay(batch_size)

                # Debugging statement to track the step count
                step_count += 1
                if step_count > 1000:  # Limit the step count to avoid infinite loops
                    print("Max steps reached. Terminating episode.")
                    break

            print(f"Episode {episode + 1}/{episodes} completed with score: {self.game.score}")



if __name__ == "__main__":
    agent = Agent()
    agent.train(episodes=1000, batch_size=32)
