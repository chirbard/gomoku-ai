import numpy as np
from constants import WIN_LENGTH, BOARD_SIZE


class Gomoku:
    """
    Class representing the Gomoku game board and logic.
    """

    def __init__(self, size=BOARD_SIZE, win_length=WIN_LENGTH, generate_board=True):
        self.size = size
        self.win_length = win_length
        self.current_player = 1
        self.winner = None
        self.last_move = None
        self.board = np.zeros((self.size, self.size), dtype=int)
        if generate_board:
            self.generate_board()

    def generate_board(self):
        """
        Call without arguments to generate a new board.

        Takes the initialized board, makes it empty and then places 3 pieces on it randomly.
        The pieces are two player 2 and on player 1 Pieces.
        """
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 2
        for _ in range(3):
            while True:
                x, y = np.random.randint(0, self.size, size=2)
                if self.board[x, y] == 0:
                    self.board[x, y] = self.current_player
                    break
            self.current_player = 1 if self.current_player == 2 else 2
        self.current_player = 1

    def clone(self):
        """
        Call without arguments to get a deep copy of the whole class.

        Used in Monte Carlo Tree Search to simulate multiple games in parallel.
        """
        clone = Gomoku(self.size, self.win_length)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.winner = self.winner
        clone.last_move = self.last_move
        return clone

    def get_legal_moves(self):
        """
        Return a list of all coordinates of empty cells on the board.
        """
        # TODO Unused function
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def moves_next_to_pieces(self):
        """
        Return a list of all empty cells on the board that are next to a piece of either player.

        Used in Monte Carlo Tree Search to find all possible moves to evaluate.
        """
        next_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    if any(0 <= x < self.size and 0 <= y < self.size and self.board[x, y] != 0
                           for x in (i - 1, i, i + 1) for y in (j - 1, j, j + 1)):
                        next_moves.append((i, j))
        return next_moves

    def apply_move(self, move):
        """
        Pass the new move coordinates as a tuple (x, y) to the function.
        The function will check if the move is valid and apply it to the board.
        Also will check if its a winning move and switch player.
        """
        if self.board[move] != 0:
            raise ValueError("Illegal move")
        self.board[move] = self.current_player
        self.last_move = move
        if self.check_win(move):
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.winner = 0
        else:
            self.current_player = 2 if self.current_player == 1 else 1

    def check_win(self, move):
        """
        Returns true if the input move is a winning move.

        Will check take into account self.win_length and check all 4 directions:
        horizontal, vertical, diagonal right and diagonal left.
        """
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
        """
        Returns true if the game has ended.
        """
        return self.winner is not None

    def render(self):
        """
        Call without arguments to print the current board state.
        """
        symbol = {1: 'X', 2: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbol[cell] for cell in row))
        print()
