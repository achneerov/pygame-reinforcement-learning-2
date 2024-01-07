import random


class SnakeGameAi:
    def __init__(self):
        self.board = [
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
        ]
        self.snake_head_row = None
        self.snake_head_col = None
        self.place_snake()
        self.place_food()

    def place_snake(self):
        # Calculate the center of the board
        rows = len(self.board)
        cols = len(self.board[0])

        center_row = rows // 2
        center_col = cols // 2

        # Place the snake head at the center
        self.board[center_row][center_col] = 'H'
        self.snake_head_row = center_row
        self.snake_head_col = center_col

        return self.board

    def place_food(self):
        flag = False
        while not flag:
            i = random.randint(0, 19)
            j = random.randint(0, 19)

            if self.board[i][j] == '.':
                self.board[i][j] = 'F'
                flag = True

    def get_input(self):
        direction = input("Enter direction (W/A/S/D): ").upper()
        if direction not in ['W', 'A', 'S', 'D']:
            print("Invalid input! Enter W, A, S, or D.")
            return
        if direction == 'W':
            self.move_snake(-1, 0)  # Move up
        elif direction == 'A':
            self.move_snake(0, -1)  # Move left
        elif direction == 'S':
            self.move_snake(1, 0)  # Move down
        elif direction == 'D':
            self.move_snake(0, 1)  # Move right

    def move_snake(self, delta_row, delta_col):
        # Calculate new position
        new_row = self.snake_head_row + delta_row
        new_col = self.snake_head_col + delta_col

        # Check if the new position is valid
        if self.board[new_row][new_col] == 'B':
            print("You hit the boundary! Game Over.")
            return False
        elif self.board[new_row][new_col] == '.':
            # Update the board
            self.board[self.snake_head_row][self.snake_head_col] = '.'
            self.board[new_row][new_col] = 'H'
            # Update the snake head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col
            # Print the updated board
            for row in self.board:
                print(' '.join(row))
            return True
        elif self.board[new_row][new_col] == 'F':
            # Update the board and grow the snake
            self.board[self.snake_head_row][self.snake_head_col] = '.'
            self.board[new_row][new_col] = 'H'

            # Update the snake head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col

            # Place new food on the board
            self.place_food()

            # Print the updated board
            for row in self.board:
                print(' '.join(row))

            return True
        else:
            print("You hit yourself! Game Over.")
            return False

    def play(self):
        while True:
            self.get_input()


if __name__ == '__main__':
    game = SnakeGameAi()
    game.play()
