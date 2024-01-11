import random
import torch
import torch.nn as nn
import torch.optim as optim

mode = 2


class Game:
    def __init__(self):
        self.rat = 0.4
        self.goblin = 0.7
        self.dragon = 0.9
        self.percentages = [("R", self.rat), ("G", self.goblin), ("D", self.dragon)]
        self.num_levels = 100
        self.chars, self.weights = zip(*self.percentages)
        self.seed = "DRRGRRRRRDRGDRGRRGRRDDDDGRRDRRGDGGRDDGGDGDRRDRRRRRRDGRDDRGRDRRGDDDGRDGGRRGGGDDDRGGGGRDRGGRRGRRRGDRRG"
        self.num_rats = 44
        self.num_goblins = 29
        self.num_dragons = 27

        self.knife_price = 1
        self.gun_price = 5
        self.missile_price = 15
        self.weapon_costs = {"knife": self.knife_price, "gun": self.gun_price, "missile": self.missile_price}
        self.enemy_types = ''.join(self.seed)

        self.game_status = "won"
        self.total_cost = 0
        self.current_level = 0
        self.reward = 0
        self.initial_num_knives = 0
        self.initial_num_guns = 0
        self.initial_num_missiles = 0
        self.num_knives = 0
        self.num_guns = 0
        self.num_missiles = 0

    def reset(self):
        self.game_status = "won"
        self.total_cost = 0
        self.current_level = 0
        self.reward = 0
        self.initial_num_knives = 0
        self.initial_num_guns = 0
        self.initial_num_missiles = 0
        self.num_knives = 0
        self.num_guns = 0
        self.num_missiles = 0
        self.num_rats = 0
        self.num_goblins = 0
        self.num_dragons = 0

    def get_state(self):
        return [self.num_levels, self.knife_price, self.gun_price, self.missile_price, self.rat, self.goblin,
                self.dragon]

    def get_cost(self):
        total_cost = 0
        total_cost += self.num_knives * self.weapon_costs["knife"]
        total_cost += self.num_guns * self.weapon_costs["gun"]
        total_cost += self.num_missiles * self.weapon_costs["missile"]
        return total_cost

    def play(self, num_knives, num_guns, num_missiles):
        self.initial_num_knives = num_knives
        self.initial_num_guns = num_guns
        self.initial_num_missiles = num_missiles
        self.num_knives = num_knives
        self.num_guns = num_guns
        self.num_missiles = num_missiles
        self.total_cost = self.get_cost()

        for enemy in self.seed:
            self.current_level += 1
            if enemy == "R":
                if num_knives > 0:
                    num_knives -= 1
                elif num_guns > 0:
                    num_guns -= 1
                elif num_missiles > 0:
                    num_missiles -= 1
                else:
                    self.game_status = "lost"
                    break
            elif enemy == "G":
                if num_guns > 0:
                    num_guns -= 1
                elif num_missiles > 0:
                    num_missiles -= 1
                else:
                    self.game_status = "lost"
                    break
            elif enemy == "D":
                if num_missiles > 0:
                    num_missiles -= 1
                else:
                    self.game_status = "lost"
                    break

            self.reward += 10

        if self.game_status == "won":
            self.reward += 10_000
            self.reward -= self.total_cost
        return self.reward

    def print_stats(self, game_num=None):
        print()
        print("current game: ", game_num, "current weights of enemies: ",
              "Weights of enemies: ", self.percentages, "Game seed: ", self.seed,
              "Price of weapons: ", self.weapon_costs, "number of rounds in a game: ", self.num_levels,
              "levels beaten: ", self.current_level, "number of rats: ", self.num_rats, "number of goblins",
              self.num_goblins,
              "number of dragons: ", self.num_dragons, "number of knives: ", self.initial_num_knives,
              "number of guns: ", self.initial_num_guns, "number of missiles", self.initial_num_missiles,
              "total price: ", self.get_cost(), "reward: ", self.reward)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    if mode == 1:
        game = Game()
        game.play(44, 29, 27)
        game.print_stats()

    if mode == 2:
        # Define the state and action dimensions
        state_dim = 7
        action_dim = 3  # Number of actions: [num_knives, num_guns, num_missiles]

        # Initialize Q-network
        q_network = QNetwork(state_dim, action_dim)
        optimizer = optim.Adam(q_network.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Q-learning parameters
        gamma = 0.999999  # Discount factor
        epsilon = 0.2  # Epsilon-greedy exploration parameter
        num_games = 250_000

        record_reward = 0  # Variable to store the previous reward

        for _ in range(num_games):
            # Initialize game environment
            game = Game()

            state = torch.tensor(game.get_state(), dtype=torch.float32)

            # Compute Q-values for the current state
            q_values = q_network(state)

            # Choose an action using epsilon-greedy policy
            if random.random() < epsilon:
                action_values = [random.randint(0, game.num_levels),
                                 random.randint(0, game.num_levels),
                                 random.randint(0, game.num_levels)]
                epsilon -= 0.00001
            else:
                action_values = [int(q_values[0].item()),
                                 int(q_values[1].item()),
                                 int(q_values[2].item())]

            reward = game.play(action_values[0], action_values[1], action_values[2])

            # Compare current reward with previous reward
            if reward >= record_reward:
                # Compute the loss (MSE between Q-values and reward)
                loss = criterion(q_values, torch.tensor([reward, reward, reward], dtype=torch.float32))

                # Zero gradients, perform a backward pass, and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                record_reward = reward  # Update the previous reward

            if _ % 1000 == 0:
                game.print_stats(game_num=_)
