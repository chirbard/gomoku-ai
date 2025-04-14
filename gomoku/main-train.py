from gomoku import Gomoku
from encode_board import encode_board
from gomokunet import GomokuNet
from mcts import MCTS
from playselfgame import play_self_game
from train import train
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = GomokuNet().to(device)


ITERATIONS = 200
GAMES_PER_ITERATION = 20
EPOCHS = 150
SIMULATIONS = 200
BATCH_SIZE = 1024
VERBOSE = False  # Do not run with more than 1 game per iteration
LOAD_MODEL = False  # Load model to continue training
SAVE_MODEL = True  # Save model after training
FILE_NAME = "gomoku_net.pt"  # Model file name

# | What You Increase           | What It Improves                      | Time Cost
# ----------------------------------------------------------------------------------
# | Iteration count             | Long-term learning, generalization    | High
# | Game count per iteration    | More diverse data per iteration       | Medium
# | Epochs per iteration        | Better fitting to current data        | Low
# | Simulation count per game   | Better move selection per game        | High
# | Training batch size         | Faster epochs                         | Medium

if LOAD_MODEL:
    net.load_state_dict(torch.load(FILE_NAME))
    print("Model loaded.")

start_time = time.time()
for iteration in range(ITERATIONS):
    print(f"=== Self-play iteration {iteration + 1} ===")
    training_data = []
    for _ in range(GAMES_PER_ITERATION):
        game_data = play_self_game(net, sims=SIMULATIONS, verbose=VERBOSE)
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
if SAVE_MODEL:
    torch.save(net.state_dict(), FILE_NAME)
    print("Model saved.")

# Load
net = GomokuNet()
net.load_state_dict(torch.load("gomoku_net.pt"))
net.eval()
