from gomoku import Gomoku
from encode_board import encode_board
from gomokunet import GomokuNet
from mcts import MCTS
from playselfgame import play_self_game
from train import train
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# game = Gomoku()
# net = GomokuNet()
# mcts = MCTS(net, num_simulations=50)

# probs = mcts.get_action_probs(game, temperature=1.0)
# for move, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
#     print(f"Move {move}: {prob:.2f}")

# net = GomokuNet()
net = GomokuNet().to(device)

for iteration in range(10):
    print(f"=== Self-play iteration {iteration + 1} ===")
    training_data = []
    for _ in range(5):  # 5 self-play games per iteration
        game_data = play_self_game(net)
        training_data.extend(game_data)

    print(f"Collected {len(training_data)} training samples.")
    train(net, training_data, epochs=5)


# Save
torch.save(net.state_dict(), "gomoku_net.pt")

# Load
net = GomokuNet()
net.load_state_dict(torch.load("gomoku_net.pt"))
net.eval()
