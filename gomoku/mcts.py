import math
from collections import defaultdict
import numpy as np
import torch
from encode_board import encode_board
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=100):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations

        self.Q = {}  # Q(s,a)
        self.N = {}  # N(s,a)
        self.P = {}  # P(s,a)
        self.Ns = defaultdict(int)  # N(s)

    def get_action_probs(self, game, temperature=1.0):
        for _ in range(self.num_simulations):
            self._simulate(game.clone())

        s = self._stringify(game)
        counts = [self.N.get((s, a), 0) for a in self._legal_moves(game)]
        if temperature == 0:
            best = np.argmax(counts)
            probs = [1 if i == best else 0 for i in range(len(counts))]
        else:
            counts = np.array(counts) ** (1 / temperature)
            probs = counts / counts.sum()

        return dict(zip(self._legal_moves(game), probs))

    def _simulate(self, game):
        s = self._stringify(game)
        if game.is_terminal():
            winner = game.winner
            return 0 if winner == 0 else (1 if winner == game.current_player else -1)

        legal_moves = self._legal_moves(game)
        if s not in self.P:
            # Leaf node: use neural network to evaluate
            input_tensor = encode_board(game).unsqueeze(0).to(device)
            with torch.no_grad():
                log_policy, value = self.net(input_tensor)
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()

            # Mask invalid moves
            p = {move: policy[move[0]*game.size + move[1]]
                 for move in legal_moves}
            total_p = sum(p.values()) + 1e-8
            p = {move: val / total_p for move, val in p.items()}

            self.P[s] = p
            for a in legal_moves:
                self.N[(s, a)] = 0
                self.Q[(s, a)] = 0
            return value.item()

        # Select move with highest UCB
        best_score = -float('inf')
        best_move = None
        for a in legal_moves:
            q = self.Q[(s, a)]
            u = self.c_puct * \
                self.P[s][a] * math.sqrt(self.Ns[s] + 1e-8) / \
                (1 + self.N[(s, a)])
            score = q + u
            if score > best_score:
                best_score = score
                best_move = a

        game.apply_move(best_move)
        v = self._simulate(game)

        # Backprop
        a = best_move
        self.N[(s, a)] += 1
        self.Ns[s] += 1
        self.Q[(s, a)] += (v - self.Q[(s, a)]) / self.N[(s, a)]
        return -v  # Because next player played

    def _legal_moves(self, game):
        return game.moves_next_to_pieces()

    def _stringify(self, game):
        player_id = 0 if game.current_player == 2 else 1
        return game.board.tobytes() + bytes([player_id])
