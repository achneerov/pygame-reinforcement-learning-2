import random
import torch
import torch.nn as nn
import torch.optim as optim

mode = "2"


class Game:
    def __init__(self):
        self.A = 0.4
        self.B = 0.7
        self.C = 0.9
        self.percentages = [("A", self.A), ("B", self.B), ("C", self.C)]
        self.num_levels = 100
        self.chars, self.weights = zip(*self.percentages)
        self.seed = ""

        for _ in range(self.num_levels):
            j = random.uniform(0, 1)
            enemy = ""
            if 0 <= j < self.A:
                enemy = "A"
            elif self.A <= j < self.B:
                enemy = "B"
            elif self.B <= j <= 1:
                enemy = "C"
            self.seed += enemy

        self.knife = 1
        self.gun = 5
        self.missile = 15
        self.weapon_costs = {"knife": self.knife, "gun": self.gun, "missile": self.missile}
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
            if enemy == "A":
                if num_knives > 0:
                    num_knives -= 1
                elif num_guns > 0:
                    num_guns -= 1
                elif num_missiles > 0:
                    num_missiles -= 1
                else:
                    self.game_status = "lost"
                    break
            elif enemy == "B":
                if num_guns > 0:
                    num_guns -= 1
                elif num_missiles > 0:
                    num_missiles -= 1
                else:
                    self.game_status = "lost"
                    break
            elif enemy == "C":
                if num_missiles > 0:
                    num_missiles -= 1
                else:
                    self.game_status = "lost"
                    break

            self.reward += 10

        if self.game_status == "won":
            self.reward += 10000
            self.reward -= self.total_cost
        return self.reward

    def print_stats(self, game_num=None):
        if game_num % 1000 == 0:
            print()
            print("current game: ", game_num, "current weights of enemies: ",
                  "Weights of enemies: ", self.percentages, "Game seed: ", self.seed,
                  "Price of weapons: ", self.weapon_costs, "number of rounds in a game: ", self.num_levels,
                  "levels beaten: ", self.current_level, "number of knives: ", self.initial_num_knives,
                  "number of guns: ", self.initial_num_guns, "number of missiles", self.initial_num_missiles,
                  "total price: ", self.get_cost(), "reward: ", self.reward)


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
        game = Game()
        print(game.percentages)
        print(game.num_levels)
        print(game.weapon_costs)

        num_knives = int(input("Enter the number of knives: "))
        num_guns = int(input("Enter the number of guns: "))
        num_missiles = int(input("Enter the number of missiles: "))

        game.play(num_knives, num_guns, num_missiles)

    if mode == "2":
        num_games = 250_000
        input_size = 7
        hidden_size = 256
        output_size = 3

        model = WeaponSelector(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.000001)
        criterion = nn.MSELoss()

        for game_num in range(num_games):
            game_instance = Game()
            input_features = torch.tensor([
                game_instance.A, game_instance.B, game_instance.C,
                game_instance.num_levels, game_instance.knife, game_instance.gun, game_instance.missile
            ], dtype=torch.float32)

            output = model(input_features)
            num_knives_pred, num_guns_pred, num_missiles_pred = output

            round_reward = game_instance.play(int(num_knives_pred.item()), int(num_guns_pred.item()),
                                              int(num_missiles_pred.item()))

            expected_reward_tensor = torch.tensor(round_reward, dtype=torch.float32)
            loss = criterion(output, expected_reward_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            game_instance.print_stats(game_num)
            game_instance.reset()
