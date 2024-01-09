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
        enemies_defeated = 0
        reward = 0

        if num_knives < 0 or num_guns < 0 or num_missiles < 0:
            print("no negative weapons")
            return reward
        # Calculate the total cost based on the weapons
        self.total_cost = self.get_cost(num_knives, num_guns, num_missiles)

        # Display the total cost
        print(f"Total cost for {num_knives} knives, {num_guns} guns, and {num_missiles} missiles: {self.total_cost}")

        # Loop through the seed to encounter enemies and use weapons accordingly

        if self.total_cost < self.num_levels * self.missile:
            reward += 1
        for enemy in self.seed:
            enemies_defeated += 1

            if enemies_defeated == int(0.25 * self.num_levels):
                print("25% of enemies defeated! Reward: +0.5")
                reward += 0.5
            elif enemies_defeated == int(0.5 * self.num_levels):
                print("50% of enemies defeated! Reward: +1")
                reward += 1
            elif enemies_defeated == int(0.75 * self.num_levels):
                print("75% of enemies defeated! Reward: +1.5")
                reward += 1.5
            elif enemies_defeated == self.num_levels - 1:
                print("all enemies defeated")
                reward += 10

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

        if self.game_status == "won":
            print("All enemies defeated. You win!")
            return reward
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
        num_episodes = 1000  # Number of games for training
        plays_per_game = 10  # Number of plays per game
        input_size = 4  # Number of input features: percentages of A, B, and C, and num_levels
        hidden_size = 10  # Hidden layer size
        output_size = 3  # Number of output values: num_knives, num_guns, num_missiles

        # Initialize neural network and optimizer
        model = WeaponSelector(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        for episode in range(num_episodes):
            total_reward = 0

            for _ in range(plays_per_game):
                # Initialize game
                game = Game()

                # Prepare input features
                input_features = torch.tensor([game.A, game.B, game.C, game.num_levels], dtype=torch.float32)

                # Forward pass through the neural network
                output = model(input_features)

                # Convert output to integer values for number of weapons
                num_knives = int(output[0].item())
                num_guns = int(output[1].item())
                num_missiles = int(output[2].item())

                # Play the game and get reward
                reward = game.play(num_knives, num_guns, num_missiles)
                total_reward += reward

                # Compute loss (reward is used as a proxy for loss in this simplified example)
                loss = criterion(output, torch.tensor([num_knives, num_guns, num_missiles], dtype=torch.float32))

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print average reward for the episode
            avg_reward = total_reward / plays_per_game
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward}")

        # Save the trained model
        torch.save(model.state_dict(), "weapon_selector_model.pth")
