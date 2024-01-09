import random
import torch
import torch.nn as nn
import torch.optim as optim

mode = "2"


class Game:
    def __init__(self):
        # Generate random values for A, B, and C
        self.A = random.uniform(0, 1)
        self.B = random.uniform(0, 1 - self.A)
        self.C = 1 - self.A - self.B

        # Create a list of tuples for percentages
        self.percentages = [("A", self.A), ("B", self.B), ("C", self.C)]

        # Generate random number of levels
        self.num_levels = random.randint(20, 30)

        # Extract characters and weights from percentages
        self.chars, self.weights = zip(*self.percentages)

        # Generate seed based on characters and weights
        self.seed = random.choices(self.chars, weights=self.weights, k=self.num_levels)

        # Generate random weapon costs
        self.knife = random.uniform(0, 10)
        self.gun = random.uniform(3, 30)
        self.missile = random.uniform(10, 100)

        # Set weapon costs dictionary
        self.weapon_costs = {"knife": self.knife, "gun": self.gun, "missile": self.missile}

        # Create enemy types based on seed
        self.enemy_types = ''.join(self.seed)
        self.game_status = "won"
        self.total_cost = 0

    def reset(self):
        self.game_status = "won"
        self.total_cost = 0

    def get_cost(self, num_knives, num_guns, num_missiles):
        """
        Calculate the total cost based on the number of each weapon type used.
        """
        total_cost = 0
        total_cost += num_knives * self.weapon_costs["knife"]
        total_cost += num_guns * self.weapon_costs["gun"]
        total_cost += num_missiles * self.weapon_costs["missile"]
        return total_cost

    def play(self, num_knives, num_guns, num_missiles):

        reward = 0
        # Calculate the total cost based on the weapons
        self.total_cost = self.get_cost(num_knives, num_guns, num_missiles)

        # Display the total cost
        print(f"Total cost for {num_knives} knives, {num_guns} guns, and {num_missiles} missiles: {self.total_cost}")

        # Loop through the seed to encounter enemies and use weapons accordingly

        for enemy in self.seed:
            reward += 1
            if enemy == "A":
                if num_knives > 0:
                    print("Enemy type A encountered. Using a knife.")
                    num_knives -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                elif num_guns > 0:
                    print("No knives left. Using a gun.")
                    num_guns -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                elif num_missiles > 0:
                    print("No knives or guns left. Using a missile.")
                    num_missiles -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    print("Out of weapons.")
                    self.game_status = "lost"
                    break

            elif enemy == "B":
                if num_guns > 0:
                    print("Enemy type B encountered. Using a gun.")
                    num_guns -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                elif num_missiles > 0:
                    print("No guns left. Using a missile.")
                    num_missiles -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    print("Out of weapons.")
                    self.game_status = "lost"
                    break
            elif enemy == "C":
                if num_missiles > 0:
                    print("Enemy type C encountered. Using a missile.")
                    num_missiles -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    print("Out of missiles.")
                    self.game_status = "lost"
                    break

        print("reward:", reward)
        if self.game_status == "won":
            print("All enemies defeated. You win!")
        else:
            print("you lost.")
        return reward


class WeaponSelector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WeaponSelector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    if mode == "1":
        # Play the game
        game = Game()
        print(game.percentages)
        print(game.num_levels)
        print(game.weapon_costs)

        num_knives = int(input("Enter the number of knives: "))
        num_guns = int(input("Enter the number of guns: "))
        num_missiles = int(input("Enter the number of missiles: "))

        game.play(num_knives, num_guns, num_missiles)

    if mode == "2":
        # Define parameters for training
        num_episodes = 10000  # Number of games for training
        plays_per_game = 100  # Number of plays per game
        input_size = 7  # Number of input features: percentages of A, B, and C, num_levels, and prices of three weapons
        hidden_size = 512  # Hidden layer size
        output_size = 3  # Number of output values: num_knives, num_guns, num_missiles

        # Initialize neural network and optimizer
        model = WeaponSelector(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        # Training loop
        # Training loop
        for episode in range(num_episodes):
            total_reward = 0
            episode_rewards = []  # List to store rewards for the current episode

            for _ in range(plays_per_game):
                # Initialize game
                game = Game()

                # Prepare input features including weapon prices
                input_features = torch.tensor([
                    game.A, game.B, game.C, game.num_levels,
                    game.knife, game.gun, game.missile
                ], dtype=torch.float32)

                # Forward pass through the neural network
                output = model(input_features)

                # Convert output to integer values for number of weapons
                num_knives = int(output[0].item())
                num_guns = int(output[1].item())
                num_missiles = int(output[2].item())

                # Play the game and get reward
                reward = game.play(num_knives, num_guns, num_missiles)
                total_reward += reward  # Accumulate the reward
                episode_rewards.append(reward)  # Store reward for the current play

            baseline_reward = sum(episode_rewards) / plays_per_game
            policy_loss = sum([(reward - baseline_reward) * output.sum() for reward, output in zip(episode_rewards, model(input_features))])

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            # Print total reward for the episode
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

