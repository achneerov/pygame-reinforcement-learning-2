import random
import numpy as np

mode = "1"  # Set to train mode for this example

class Game:
    def __init__(self):
        self.common_initialization()

    def reset(self):
        self.common_initialization()

    def common_initialization(self):
        # Generate random values for A, B, and C
        self.A = random.uniform(0, 1)
        self.B = random.uniform(0, 1 - self.A)
        self.C = 1 - self.A - self.B

        # Create a list of tuples for percentages
        self.percentages = [("A", self.A), ("B", self.B), ("C", self.C)]

        # Generate random number of levels
        self.num_levels = random.randint(90, 110)

        # Extract characters and weights from percentages
        self.chars, self.weights = zip(*self.percentages)

        # Generate seed based on characters and weights
        self.seed = random.choices(self.chars, weights=self.weights, k=self.num_levels)

        # Initialize round to 0
        self.round = 0

        # Generate random weapon costs
        self.knife = random.uniform(0, 10)
        self.gun = random.uniform(3, 30)
        self.missile = random.uniform(10, 100)

        # Set weapon costs dictionary
        self.weapon_costs = {"knife": self.knife, "gun": self.gun, "missile": self.missile}

        # Create enemy types based on seed
        self.enemy_types = ''.join(self.seed)

    def get_cost(self, num_knives, num_guns, num_missiles):
        return self.weapon_costs["knife"] * num_knives + self.weapon_costs["gun"] * num_guns + self.weapon_costs["missile"] * num_missiles

    def play(self, num_knives, num_guns, num_missiles):
        # Calculate the total cost based on the weapons
        total_cost = self.get_cost(num_knives, num_guns, num_missiles)

        # Display the total cost
        print(f"Total cost for {num_knives} knives, {num_guns} guns, and {num_missiles} missiles: {total_cost}")

        # Loop through the seed to encounter enemies and use weapons accordingly
        for enemy in self.seed:
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
                    print("Out of weapons. Game over!")
                    return
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
                    print("Out of weapons. Game over!")
                    return
            elif enemy == "C":
                if num_missiles > 0:
                    print("Enemy type C encountered. Using a missile.")
                    num_missiles -= 1
                    print(f"Remaining weapons: Knives: {num_knives}, Guns: {num_guns}, Missiles: {num_missiles}")
                else:
                    print("Out of missiles. Game over!")
                    return

        # If all enemies are defeated, the player wins
        print("All enemies defeated. You win!")

        # Increment the round after playing
        self.round += 1






if __name__ == "__main__":
    if mode == "1":
        game = Game()
        print(game.percentages)
        print(game.num_levels)
        print(game.weapon_costs)

        num_knives = int(input("Enter the number of knives: "))
        num_guns = int(input("Enter the number of guns: "))
        num_missiles = int(input("Enter the number of missiles: "))

        print(game.get_cost(num_knives, num_guns, num_missiles))
        game.play(num_knives, num_guns, num_missiles)


