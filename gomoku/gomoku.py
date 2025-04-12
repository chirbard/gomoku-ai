import numpy as np
from constants import WIN_LENGTH, BOARD_SIZE


class Gomoku:
    def __init__(self, size=BOARD_SIZE, win_length=WIN_LENGTH):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.winner = None
        self.last_move = None

    def clone(self):
        clone = Gomoku(self.size, self.win_length)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.winner = self.winner
        clone.last_move = self.last_move
        return clone

    def get_legal_moves(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

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
            self.current_player *= -1

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
        symbol = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbol[cell] for cell in row))
        print()


# game = Gomoku()
# game.render()
# game.apply_move((0, 0))
# game.apply_move((1, 1))
# game.render()
