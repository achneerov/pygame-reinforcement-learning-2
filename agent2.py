import numpy as np
from model2 import DQNAgent  # Assuming DQNAgent is implemented using PyTorch
from game2 import SnakeGameAi  # Assuming you have a SnakeGameAi class


class Agent:
    def __init__(self):
        self.game = SnakeGameAi()  # Initialize the Snake game environment
        self.agent = DQNAgent(state_size=20 * 20, action_size=4)  # Initialize DQN agent

    def preprocess_state(self, state):
        print(state)
        """
        Preprocess the state to make it suitable for the DQN model.
        Convert symbols to numbers and flatten the state.

        Parameters:
            state (list or int): 2D list representing the game board or an integer.

        Returns:
            np.array: Flattened and converted state.
        """
        # Check if the state is a 2D list
        if not isinstance(state, list) or not all(isinstance(row, list) for row in state):
            print(f"Unexpected state type: {type(state)}")
            return None  # Return None or handle the error accordingly

        # Mapping symbols to numbers
        mapping = {'B': 0, '.': 1, 'H': 2, 'F': 3}

        # Convert symbols to numbers
        numeric_state = [[mapping[symbol] for symbol in row] for row in state]

        # Flatten the state
        flattened_state = np.array(numeric_state).flatten()

        return flattened_state if flattened_state.shape[
                                      0] == 400 else None  # Return flattened state if it has the expected shape

    def train(self, episodes, batch_size):
        for episode in range(episodes):
            state = self.game.get_board()  # Get initial state from the environment
            state = self.preprocess_state(state)

            done = False
            while not done:
                # Decide action based on the current state
                action = self.agent.act(state)

                # Take action and get next_state, reward, done from the environment
                next_state, reward, done = self.game.play_step(
                    move=action)  # Assuming play_step method returns next_state, reward, done
                next_state = self.preprocess_state(next_state)
                print(next_state.shape)  # Debugging line to inspect the shape of next_state

                # Remember the experience and train the DQN agent
                self.agent.remember(state, action, reward, next_state, done)

                # Update the current state
                state = next_state

                # Perform replay if memory has enough samples
                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)

            print(f"Episode {episode + 1}/{episodes} completed with score: {self.game.score}")

        # Save the trained model if needed
        # Assuming the DQNAgent class has a save method implemented
        self.agent.save_model("snake_dqn_model.pth")


if __name__ == "__main__":
    agent = Agent()
    agent.train(episodes=1000, batch_size=32)  # Train the agent for 1000 episodes with a batch size of 32
