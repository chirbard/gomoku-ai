from gomoku import Gomoku
from encode_board import encode_board
from gomokunet import GomokuNet
from mcts import MCTS
from train import train
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = GomokuNet().to(device)
net.load_state_dict(torch.load("gomoku_net.pt"))
net.eval()
game = Gomoku()
mcts = MCTS(net, num_simulations=50)

while not game.is_terminal():
    game.render()
    print("Your turn! Enter your move as 'row,col':")
    try:
        human_move = tuple(map(int, input().strip().split(',')))
        game.apply_move(human_move)
    except ValueError:
        print("Invalid input format. Please enter row and column as 'row,col'.")
        continue

    if game.is_terminal():
        break

    probs = mcts.get_action_probs(game, temperature=0)
    ai_move = max(probs.items(), key=lambda x: x[1])[0]
    game.apply_move(ai_move)
    print("AI's move:", ai_move)

game.render()
if game.winner is None:
    print("It's a draw!")
else:
    print(f"Winner: {'You' if game.winner == 1 else 'AI'}")
