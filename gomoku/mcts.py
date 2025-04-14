import math
from collections import defaultdict
import numpy as np
import torch
from encode_board import encode_board
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=100):
        """
        ## c_puct
        How much to explore vs exploit.
        Higher values mean more exploration. (2, 3, 5)
        Lower values mean more exploitation. (0.5, 1)

        ## action_value_estimates
        - Type: Dictionary
        - Key: (game_state_hash, move)
        - Value: The average value (expected reward) of taking a specific move from a specific game state.
        - Purpose: Tracks the quality of each move based on the outcomes of simulations. It is updated during backpropagation.

        ## visit_counts_for_state_action_pairs
        - Type: Dictionary
        - Key: (game_state_hash, move)
        - Value: The number of times a specific move has been taken from a specific game state during simulations.
        - Purpose: Tracks how often each move has been explored. This is used in the UCB formula to balance exploration and exploitation.

        ## policy_probabilities
        - Type: Dictionary
        - Key: game_state_hash
        - Value: A dictionary mapping each legal move to its probability, as predicted by the neural network.
        - Purpose: Guides the search by providing prior probabilities for each move in a given state. 
                   These probabilities are used in the UCB formula to encourage exploration of moves with high prior probabilities.

        ## visit_counts_for_states
        - Type: defaultdict(int)
        - Key: game_state_hash
        - Value: The total number of times a specific game state has been visited during simulations.
        - Purpose: Tracks how often a state has been visited. This is used in the UCB formula to calculate the exploration term for moves.
        """

        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations

        self.action_value_estimates_Q = {}
        self.visit_counts_for_state_action_pairs_N = {}
        self.policy_probabilities_P = {}
        self.visit_counts_for_states_Ns = defaultdict(int)

    def get_action_probs(self, game, temperature=1.0):
        """
        Creates simulations of the game. With the simulations it will update a Q-table.

        Then returns a dictionary with all possible moves and probabilities how good the move is predicted to be.
        """
        for _ in range(self.num_simulations):
            self._simulate(game.clone())

        game_state_hash = self._get_game_state_hash(game)
        counts = [self.visit_counts_for_state_action_pairs_N.get((game_state_hash, move), 0)
                  for move in self._get_legal_moves(game)]
        if temperature == 0:
            best = np.argmax(counts)
            probs = [1 if i == best else 0 for i in range(len(counts))]
        else:
            counts = np.array(counts) ** (1 / temperature)
            probs = counts / counts.sum()

        return dict(zip(self._get_legal_moves(game), probs))

    def _simulate(self, game):
        """
        Pass in the current game state. This will simulate a game from this state.
        This will update the class variables with data from the simulation.
        This will return the value of the game state.

        It will call itself recursively until the game is over or a new unseen state is reached.
        """
        game_state_hash = self._get_game_state_hash(game)
        if game.is_terminal():
            winner = game.winner
            return 0 if winner == 0 else (1 if winner == game.current_player else -1)

        legal_moves = self._get_legal_moves(game)
        if game_state_hash not in self.policy_probabilities_P:
            return self._explore_unseen_state(game, game_state_hash, legal_moves)

        best_move = self._select_best_move(game_state_hash, legal_moves)
        game.apply_move(best_move)
        value = self._simulate(game)
        self._back_propagate(game_state_hash, best_move, value)

        return -value

    def _explore_unseen_state(self, game, game_state_hash, legal_moves):
        """
        Creates a tensor of the game state. Passes it to the GPU.
        Then uses the neural network to get the policy and value of the game state.

        Now we turn the generated log_policy into a numpy array.
        The neural network doesn't know which moves are valid, so it gives a probability to each and every cell.
        We mask out all the cells that are not valid moves.
        Then we add up all the probabilities of the valid moves and divide each valid move by the total probability.
        By doing this we normalize the probabilities between 0 and 1.

        p is now a dictionary of valid moves and their probabilities.

        We create a N and Q value for each valid move.
        """
        neural_network_input_tensor = encode_board(
            game).unsqueeze(0).to(device)
        with torch.no_grad():
            log_policy, value = self.net(neural_network_input_tensor)
        numpy_policy_matrix = torch.exp(log_policy).squeeze(0).cpu().numpy()

        p = {move: numpy_policy_matrix[move[0]*game.size + move[1]]
             for move in legal_moves}
        total_p = sum(p.values()) + 1e-8
        p = {move: val / total_p for move, val in p.items()}

        self.policy_probabilities_P[game_state_hash] = p
        for move in legal_moves:
            self.visit_counts_for_state_action_pairs_N[(
                game_state_hash, move)] = 0
            self.action_value_estimates_Q[(game_state_hash, move)] = 0
        return value.item()

    def _select_best_move(self, game_state_hash, legal_moves):
        """
        Selects the best move based on the UCB formula.
        UCB = Q + c * P * sqrt(Ns + 1e-8) / (1 + Ns)

        where Q is the action value estimate, P is the prior probability, Ns is the visit count for the state.
        c is the exploration parameter (c_puct).
        """
        best_score = -float('inf')
        best_move = None
        for move in legal_moves:
            q = self.action_value_estimates_Q[(game_state_hash, move)]
            u = self.c_puct * \
                self.policy_probabilities_P[game_state_hash][move] * math.sqrt(self.visit_counts_for_states_Ns[game_state_hash] + 1e-8) / \
                (1 +
                 self.visit_counts_for_state_action_pairs_N[(game_state_hash, move)])
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _back_propagate(self, game_state_hash, move, value):
        """
        Updates the N and Ns values. Adds 1.

        Updates the Q value. Adds the value of the move divided by the number of times the move has been visited.
        This is needed make this Q-state more valuable, but not kill exploration.
        """
        self.visit_counts_for_state_action_pairs_N[(
            game_state_hash, move)] += 1
        self.visit_counts_for_states_Ns[game_state_hash] += 1
        self.action_value_estimates_Q[(game_state_hash, move)] += (value -
                                                                   self.action_value_estimates_Q[(game_state_hash, move)]) / self.visit_counts_for_state_action_pairs_N[(game_state_hash, move)]

    def _get_legal_moves(self, game):
        """
        Use the method from the game class to get all possible next moves.
        """
        return game.moves_next_to_pieces()

    def _get_game_state_hash(self, game):
        """
        Returns a unique hash of the game board and the current player.

        We store this in a Q-table to remeber if we have already visited this state.
        If the state is visited, then the neural network should know how to act in this state.
        """
        player_id = 0 if game.current_player == 2 else 1
        return game.board.tobytes() + bytes([player_id])
