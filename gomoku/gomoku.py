import numpy as np
from constants import WIN_LENGTH, BOARD_SIZE


class Gomoku:
    def __init__(self, size=BOARD_SIZE, win_length=WIN_LENGTH):
        self.size = size
        self.win_length = win_length
        self.current_player = 1  # Start with player 1
        self.winner = None
        self.last_move = None
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.generate_board()

    def generate_board(self):
        # generates board with 3 pieces in random locations
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 2  # Start placing with player 2
        for _ in range(3):
            while True:
                x, y = np.random.randint(0, self.size, size=2)
                if self.board[x, y] == 0:
                    self.board[x, y] = self.current_player
                    break
            # Switch between players 1 and 2
            self.current_player = 1 if self.current_player == 2 else 2
        self.current_player = 1  # reset to player 1

    def clone(self):
        clone = Gomoku(self.size, self.win_length)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.winner = self.winner
        clone.last_move = self.last_move
        return clone

    def get_legal_moves(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    # TODO kasuta seda et järgmiseid liigutusi genereerida
    def moves_next_to_pieces(self):
        next_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    if any(0 <= x < self.size and 0 <= y < self.size and self.board[x, y] != 0
                           for x in (i - 1, i, i + 1) for y in (j - 1, j, j + 1)):
                        next_moves.append((i, j))
        return next_moves

    # Change player switching logic
    def apply_move(self, move):
        if self.board[move] != 0:
            raise ValueError("Illegal move")
        self.board[move] = self.current_player
        self.last_move = move
        if self.check_win(move):
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.winner = 0  # draw
        else:
            # Change this line
            self.current_player = 2 if self.current_player == 1 else 1

    def check_win(self, move):
        x, y = move
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            for sign in [-1, 1]:
                for step in range(1, self.win_length):
                    nx, ny = x + dx * step * sign, y + dy * step * sign
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_length:
                return True
        return False

    def is_terminal(self):
        return self.winner is not None

    def render(self):
        symbol = {1: 'X', 2: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbol[cell] for cell in row))
        print()
