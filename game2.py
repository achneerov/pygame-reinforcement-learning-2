#game2.py

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
        self.reward = 0
        self.score = 0
        self.collision = False
        self.snake_head_row = None
        self.snake_head_col = None
        self.snake_segments = [(self.snake_head_row, self.snake_head_col)]
        self._place_snake()
        self._place_food()

    def reset(self):
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
        self.reward = 0
        self.score = 0
        self.collision = False
        self.snake_head_row = None
        self.snake_head_col = None
        self.snake_segments = [(self.snake_head_row, self.snake_head_col)]
        self._place_snake()
        self._place_food()

    def get_board(self):
        return self.board

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

    def _place_snake(self):
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

    def _place_food(self):
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

    def play_step(self, move=None):  # [0,0,0,0] form
        # Define your knowledge base (KB) here or load it from wherever you need
        kb_moves = [
            [0, 1, 0, 0],  # Up
            [0, 0, 1, 0],  # Left
            [0, 0, 0, 1],  # Right
            [1, 0, 0, 0]   # Down
        ]

        self.collision = False
        self.reward = 0

        if move is None:
            # Take input from the keyboard
            user_input = input("Enter 'W' for up, 'A' for left, 'S' for down, or 'D' for right: ").upper()

            if user_input == 'W':
                move = kb_moves[0]
            elif user_input == 'A':
                move = kb_moves[1]
            elif user_input == 'D':
                move = kb_moves[2]
            elif user_input == 'S':  # Option for moving down
                move = kb_moves[3]
            else:
                print("Invalid input. Defaulting to no move.")
                move = [0, 0, 0, 0]  # Default move (no move)


        print(move)
        if move == [0, 1, 0, 0]:  # Up
            self.move_snake(-1, 0)
        elif move == [0, 0, 1, 0]:  # Left
            self.move_snake(0, -1)
        elif move == [0, 0, 0, 1]:  # Right
            self.move_snake(0, 1)
        elif move == [1, 0, 0, 0]:  # Down
            self.move_snake(1, 0)

        self.update_snake_segments()

        # Check if the game is over due to a collision
        done = self.collision

        # Return the next state (board), reward, and done flag
        return self.get_board(), self.reward, done


    def move_snake(self, delta_row, delta_col):
        # Calculate new position
        new_row = self.snake_head_row + delta_row
        new_col = self.snake_head_col + delta_col

        if new_row < 0 or new_row >= len(self.board) or new_col < 0 or new_col >= len(self.board[0]):
            print("hit the wall")
            self.reward = -10  # Set the negative reward for hitting the wall
            self.collision = True  # Trigger collision flag
            return

        # Check if the new position hits the boundary
        if self.board[new_row][new_col] == "B":
            print("hit the wall")
            for segment in self.snake_segments:
                row, col = segment
                self.board[row][col] = '.'
            self.snake_segments = [(new_row, new_col)] + self.snake_segments[:-1]
            self.board[new_row][new_col] = 'H'  # Update the new head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col
            self.reward = -10
            self.collision = True

        # Check if the new position overlaps with the snake's body
        if self.board[new_row][new_col] == "S":
            print("hit yourself")
            for segment in self.snake_segments:
                row, col = segment
                self.board[row][col] = '.'
            self.snake_segments = [(new_row, new_col)] + self.snake_segments[:-1]
            self.board[new_row][new_col] = 'H'  # Update the new head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col
            self.reward = -10
            self.collision = True

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
        elif self.board[new_row][new_col] == 'F':
            # Clear the previous positions of the snake segments on the board
            for segment in self.snake_segments:
                row, col = segment
                self.board[row][col] = '.'
            self.score += + 1
            self.reward = 10
            # Eat food, grow the snake, and update segments
            self.snake_segments = [(new_row, new_col)] + self.snake_segments
            self.board[new_row][new_col] = 'H'  # Update the new head position
            self.snake_head_row = new_row
            self.snake_head_col = new_col
            self._place_food()

    def play(self):
        playing = True
        while playing:
            self.print_board()
            self.play_step()
            if self.collision:
                playing = False
            self.print_board()


if __name__ == '__main__':
    game = SnakeGameAi()
    game.play()
