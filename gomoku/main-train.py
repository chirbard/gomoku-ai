from gomoku import Gomoku
from encode_board import encode_board
from gomokunet import GomokuNet
from mcts import MCTS
from playselfgame import play_self_game
from train import train
import time
import torch
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = GomokuNet().to(device)

USE_WANDB = True  # Set to False to disable wandb
if USE_WANDB:
    import wandb
    wandb.login()  # You'll need to authenticate first time
    wandb.init(
        project="gomoku-training",  # Your project name
        config={
            "iterations": 200,
            "games_per_iteration": 20,
            "epochs": 150,
            "simulations": 200,
            "batch_size": 1024,
            "board_size": 15,  # From constants.py
            "win_length": 5,    # From constants.py
            "architecture": "GomokuNet"
        }
    )

ITERATIONS = wandb.config.iterations if USE_WANDB else 200
GAMES_PER_ITERATION = wandb.config.games_per_iteration if USE_WANDB else 20
EPOCHS = wandb.config.epochs if USE_WANDB else 150
SIMULATIONS = wandb.config.simulations if USE_WANDB else 200
BATCH_SIZE = wandb.config.batch_size if USE_WANDB else 1024
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

os.makedirs('checkpoints', exist_ok=True)
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
    # train(net, training_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    loss_metrics = train(net, training_data, epochs=EPOCHS,
                         batch_size=BATCH_SIZE, return_metrics=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Iteration {iteration + 1} took {elapsed_time:.2f} seconds.")

    if USE_WANDB:
        metrics = {
            "iteration": iteration + 1,
            "elapsed_time": elapsed_time,
            "samples_collected": len(training_data),
            "avg_value_loss": loss_metrics.get("avg_value_loss", 0),
            "avg_policy_loss": loss_metrics.get("avg_policy_loss", 0),
            "avg_total_loss": loss_metrics.get("avg_total_loss", 0),
        }
        wandb.log(metrics)

    start_time = end_time
    if iteration % 5 == 0:
        # Save the model every 5 iterations
        checkpoint_path = f"checkpoints/gomoku_net_{iteration}.pt"
        torch.save(net.state_dict(), checkpoint_path)
        print(f"Model saved at iteration {iteration}.")

        if USE_WANDB:
            wandb.save(checkpoint_path)


# Save final model
if SAVE_MODEL:
    torch.save(net.state_dict(), FILE_NAME)
    print("Model saved.")
    if USE_WANDB:
        wandb.save(FILE_NAME)

# Finish wandb run
if USE_WANDB:
    wandb.finish()
