from gomoku import Gomoku
from encode_board import encode_board
import numpy as np
from mcts import MCTS
from constants import BOARD_SIZE


def play_self_game(net, board_size=BOARD_SIZE, sims=50, verbose=False):
    game = Gomoku(size=board_size)
    mcts = MCTS(net, num_simulations=sims)
    memory = []

    move_count = 0
    if verbose:
        print("===== Starting new self-play game =====")
        game.render()

    while not game.is_terminal():
        # Run MCTS to get improved policy
        pi_dict = mcts.get_action_probs(game, temperature=1.0)
        pi = np.zeros(board_size * board_size)
        for move, prob in pi_dict.items():
            idx = move[0] * board_size + move[1]
            pi[idx] = prob

        # Save training data
        state_tensor = encode_board(game)
        memory.append((state_tensor, pi, game.current_player))

        # Sample move from policy
        moves, probs = zip(*pi_dict.items())
        move = moves[np.random.choice(len(moves), p=probs)]
        game.apply_move(move)

        move_count += 1
        if verbose:
            print(
                f"Move {move_count}: Player {game.current_player} played at {move}")
            game.render()

    # Show game result if verbose
    if verbose:
        if game.winner == 0:
            print("Game ended in a draw")
        else:
            print(f"Game won by player {game.winner}")

    # Assign game result
    result = game.winner
    training_data = []

    # Calculate move efficiency factor (higher reward for shorter games)
    max_possible_moves = board_size * board_size
    speed_factor = max(0.5, 1.0 - (move_count / max_possible_moves))

    if verbose:
        print(f"Game length: {move_count} moves")
        print(f"Speed factor: {speed_factor:.2f}")

    for state_tensor, pi, player in memory:
        if result == 0:  # Draw
            reward = 0
        elif result == player:  # Win
            # Increase reward for faster wins
            reward = 1.0 * (1.0 + speed_factor)
        else:  # Loss
            reward = -1.0

        training_data.append((state_tensor, pi, reward))

    return training_data
