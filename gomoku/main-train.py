from gomoku import Gomoku
from encode_board import encode_board
from gomokunet import GomokuNet
from mcts import MCTS
from playselfgame import play_self_game
from train import train
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net = GomokuNet()
net = GomokuNet().to(device)
net.load_state_dict(torch.load("gomoku_net.pt"))
ITERATIONS = 1
GAMES_PER_ITERATION = 1
EPOCHS = 30
SIMULATIONS = 200
BATCH_SIZE = 1024

# | What You Increase           | What It Improves                      | Time Cost
# ----------------------------------------------------------------------------------
# | Iteration count             | Long-term learning, generalization    | High
# | Game count per iteration    | More diverse data per iteration       | Medium
# | Epochs per iteration        | Better fitting to current data        | Low
# | Simulation count per game   | Better move selection per game        | High
# | Training batch size         | Faster epochs                         | Medium

start_time = time.time()
for iteration in range(ITERATIONS):
    print(f"=== Self-play iteration {iteration + 1} ===")
    training_data = []
    for _ in range(GAMES_PER_ITERATION):  # 5 self-play games per iteration
        game_data = play_self_game(net, sims=SIMULATIONS, verbose=True)
        training_data.extend(game_data)

    print(f"Collected {len(training_data)} training samples.")
    train(net, training_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Iteration {iteration + 1} took {elapsed_time:.2f} seconds.")
    start_time = end_time
    if iteration % 5 == 0:
        # Save the model every 5 iterations
        torch.save(net.state_dict(), f"checkpoints/gomoku_net_{iteration}.pt")
        print(f"Model saved at iteration {iteration}.")


# Save
# torch.save(net.state_dict(), "gomoku_net.pt")

# Load
net = GomokuNet()
net.load_state_dict(torch.load("gomoku_net.pt"))
net.eval()
