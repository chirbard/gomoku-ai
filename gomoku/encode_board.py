import torch
from gomoku import Gomoku
import numpy as np


def encode_board(game: Gomoku):
    """
    Converts a Gomoku board to a 2-channel PyTorch tensor.

    Channel 0: current player's stones (1)
    Channel 1: opponent's stones (1)

    Returns:
        A torch.FloatTensor of shape (2, 6, 6)
    """
    current_player = game.current_player
    board = game.board

    player_plane = (board == current_player).astype(float)
    opponent_plane = (board == -current_player).astype(float)

    stacked = np.stack([player_plane, opponent_plane])
    return torch.tensor(stacked, dtype=torch.float32)
