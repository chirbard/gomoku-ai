from gomoku import Gomoku
from encode_board import encode_board
from gomokunet import GomokuNet
from mcts import MCTS
from playselfgame import play_self_game
from train import train
import torch

# Step 1: Init
net = GomokuNet()
net.load_state_dict(torch.load("gomoku_net.pt"))  # optional if you trained it
net.eval()

# Step 2: New game
game = Gomoku(size=6)

# Step 3: Setup MCTS
mcts = MCTS(net, num_simulations=50)

# Game loop
while not game.is_terminal():
    # Human move
    game.render()
    print("Your turn! Enter your move as 'row,col':")
    try:
        human_move = tuple(map(int, input().strip().split(',')))
        # if not game.is_valid_move(human_move):
        #     print("Invalid move. Try again.")
        #     continue
        game.apply_move(human_move)
    except ValueError:
        print("Invalid input format. Please enter row and column as 'row,col'.")
        continue

    if game.is_terminal():
        break

    # AI move
    # 0 = always pick best move
    probs = mcts.get_action_probs(game, temperature=0)
    ai_move = max(probs.items(), key=lambda x: x[1])[0]
    game.apply_move(ai_move)
    print("AI's move:", ai_move)

# Game over
game.render()
if game.winner is None:
    print("It's a draw!")
else:
    print(f"Winner: {'You' if game.winner == 1 else 'AI'}")
