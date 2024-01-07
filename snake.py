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
        self.snake_segments = [(self.snake_head_row, self.snake_head_col)]
        self.place_snake()
        self.place_food()

    def update_snake_segments(self):
        board_copy = [row[:] for row in self.board]
        for segment in self.snake_segments:
            row, col = segment
            board_copy[row][col] = 'H' if segment == self.snake_segments[0] else 'S'  # 'H' for head, 'S' for body
        self.board = board_copy

    def print_board(self):

        # Print the board
        for row in self.board:
            print(' '.join(row))

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

        # Initialize snake_segments after placing the snake head
        self.snake_segments = [(self.snake_head_row, self.snake_head_col)]

        return self.board

    def place_food(self):
        flag = False
        counter = 0
        while not flag:
            i = random.randint(0, 19)  # maybe make tighter later / depends on dimensions too.
            j = random.randint(0, 19)
            if self.board[i][j] == '.':
                self.board[i][j] = 'F'
                flag = True
            else:
                counter += 1
            if counter >= 10_000:
                print("you won!")  # to be improved

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
        self.update_snake_segments()

    def move_snake(self, delta_row, delta_col):
        # Calculate new position
        new_row = self.snake_head_row + delta_row
        new_col = self.snake_head_col + delta_col

        # Check if the new position hits the boundary
        if new_row < 0 or new_row >= len(self.board) or new_col < 0 or new_col >= len(self.board[0]):
            print("You hit the boundary! Game Over.")
            return False

        # Check if the new position overlaps with the snake's body
        if (new_row, new_col) in self.snake_segments:
            print("You hit yourself! Game Over.")
            return False

        if self.board[new_row][new_col] == '.':
            # Clear the previous positions of the snake segments on the board
            for segment in self.snake_segments:
                row, col = segment
                self.board[row][col] = '.'

            # Move the snake and update its segments
            self.snake_segments = [(new_row, new_col)] + self.snake_segments[:-1]
            self.board[new_row][new_col] = 'H'  # Update the new head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col
            return True
        elif self.board[new_row][new_col] == 'F':
            # Clear the previous positions of the snake segments on the board
            for segment in self.snake_segments:
                row, col = segment
                self.board[row][col] = '.'

            # Eat food, grow the snake, and update segments
            self.snake_segments = [(new_row, new_col)] + self.snake_segments
            self.board[new_row][new_col] = 'H'  # Update the new head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col
            self.place_food()
            return True

    def play(self):
        while True:
            self.print_board()
            self.get_input()


if __name__ == '__main__':
    game = SnakeGameAi()
    game.play()
