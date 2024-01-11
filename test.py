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
        self.num_levels = 100

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
        self.current_level = 0
        self.reward = 0
        self.num_knives = 0
        self.num_guns = 0
        self.num_missiles = 0

    def reset(self):
        self.game_status = "won"
        self.total_cost = 0
        self.current_level = 0
        self.reward = 0
        self.num_knives = 0
        self.num_guns = 0
        self.num_missiles = 0

    def get_cost(self):
        """
        Calculate the total cost based on the number of each weapon type used.
        """
        total_cost = 0
        total_cost += self.num_knives * self.weapon_costs["knife"]
        total_cost += self.num_guns * self.weapon_costs["gun"]
        total_cost += self.num_missiles * self.weapon_costs["missile"]
        return total_cost

    def play(self, num_knives, num_guns, num_missiles):
        self.num_knives = num_knives
        self.num_guns = num_guns
        self.num_missiles = num_missiles

        # Calculate the total cost based on the weapons
        self.total_cost = self.get_cost()

        # Display the total cost
        print(f"Total cost for {num_knives} knives, {num_guns} guns, and {num_missiles} missiles: {self.total_cost}")

        # Loop through the seed to encounter enemies and use weapons accordingly

        for enemy in self.seed:
            self.current_level += 1
            self.reward += 10
            if enemy == "A":
                if num_knives > 0:
                    # print("Enemy type A encountered. Using a knife.")
                    num_knives -= 1
                    # print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                elif num_guns > 0:
                    # print("No knives left. Using a gun.")
                    num_guns -= 1
                    # print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                elif num_missiles > 0:
                    # print("No knives or guns left. Using a missile.")
                    num_missiles -= 1
                    # print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    # print("Out of weapons.")
                    self.game_status = "lost"
                    break
            elif enemy == "B":
                if num_guns > 0:
                    # print("Enemy type B encountered. Using a gun.")
                    num_guns -= 1
                    # print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                elif num_missiles > 0:
                    # print("No guns left. Using a missile.")
                    num_missiles -= 1
                    # print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    # print("Out of weapons.")
                    self.game_status = "lost"
                    break
            elif enemy == "C":
                if num_missiles > 0:
                    # print("Enemy type C encountered. Using a missile.")
                    num_missiles -= 1
                    # print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    # print("Out of missiles.")
                    self.game_status = "lost"
                    break

        return self.reward

    def print_stats(self, game_num=None, round_num=None):
        print()
        print("current game: ", game_num, "current round: ", round_num, "current weights of enemies: ",
              "Weights of enemies: ", self.percentages, "Price of weapons: ", self.weapon_costs, "number of rounds in a game: ", self.num_levels,
              "levels beaten: ", self.current_level, "total price: ", self.get_cost(), "reward: ", self.reward)


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
        num_games = 1000  # Number of games for training
        plays_per_game = 100  # Number of plays per game
        input_size = 7  # Number of input features: percentages of A, B, and C, num_levels, and prices of three weapons
        hidden_size = 512  # Hidden layer size
        output_size = 3  # Number of output values: num_knives, num_guns, num_missiles

        # Initialize neural network and optimizer
        model = WeaponSelector(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        for game_num in range(num_games):
            total_game_reward = 0
            game_instance = Game()

            for round_num in range(plays_per_game):
                input_features = torch.tensor([
                    game_instance.A, game_instance.B, game_instance.C,
                    game_instance.num_levels, game_instance.knife, game_instance.gun, game_instance.missile
                ], dtype=torch.float32)

                # Predict actions using the neural network
                output = model(input_features)
                num_knives_pred, num_guns_pred, num_missiles_pred = output

                # Play the game with predicted actions
                round_reward = game_instance.play(int(num_knives_pred.item()), int(num_guns_pred.item()),
                                                  int(num_missiles_pred.item()))

                # Compute loss based on the difference between expected and actual rewards
                expected_reward_tensor = torch.tensor(round_reward, dtype=torch.float32)
                loss = criterion(output, expected_reward_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_game_reward += round_reward

                game_instance.print_stats(game_num, round_num)
                game_instance.reset()

            # Print the total reward accumulated for this game
            print(f"Game {game_num + 1}/{num_games}, Total Reward: {total_game_reward}")
