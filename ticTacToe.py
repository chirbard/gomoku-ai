import numpy as np
import pickle
import os
import torch
import time

BOARD_ROWS = 3
BOARD_COLS = 3
SEQUENCE_LENGTH = 3


class State:
    def __init__(self, p1, p2):
        self.board = torch.zeros((BOARD_ROWS, BOARD_COLS), device='cuda')
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self, sequence_length=SEQUENCE_LENGTH):
        # row
        for i in range(BOARD_ROWS):
            # Ensure we have enough elements to check
            for j in range(BOARD_COLS - sequence_length + 1):
                if torch.sum(self.board[i, j:j+sequence_length]) == sequence_length:
                    self.isEnd = True
                    return 1
                if torch.sum(self.board[i, j:j+sequence_length]) == -sequence_length:
                    self.isEnd = True
                    return -1

        # col
        for i in range(BOARD_COLS):
            # Ensure we have enough elements to check
            for j in range(BOARD_ROWS - sequence_length + 1):
                if torch.sum(self.board[j:j+sequence_length, i]) == sequence_length:
                    self.isEnd = True
                    return 1
                if torch.sum(self.board[j:j+sequence_length, i]) == -sequence_length:
                    self.isEnd = True
                    return -1

        # diagonal (top-left to bottom-right)
        for i in range(BOARD_ROWS - sequence_length + 1):
            for j in range(BOARD_COLS - sequence_length + 1):
                diag_sum1 = torch.sum(torch.stack(
                    [self.board[i+k, j+k] for k in range(sequence_length)]))
                if diag_sum1 == sequence_length:
                    self.isEnd = True
                    return 1
                if diag_sum1 == -sequence_length:
                    self.isEnd = True
                    return -1

        # diagonal (top-right to bottom-left)
        for i in range(BOARD_ROWS - sequence_length + 1):
            for j in range(sequence_length - 1, BOARD_COLS):
                diag_sum2 = torch.sum(torch.stack(
                    [self.board[i+k, j-k] for k in range(sequence_length)]))
                if diag_sum2 == sequence_length:
                    self.isEnd = True
                    return 1
                if diag_sum2 == -sequence_length:
                    self.isEnd = True
                    return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self, number_of_actions):
        result = self.winner()

        max_actions = BOARD_ROWS * BOARD_COLS
        reward_multiplier = max(0.1, 1 - (number_of_actions / max_actions))
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1 * reward_multiplier)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1 * reward_multiplier)
        else:
            self.p1.feedReward(0.1 * reward_multiplier)
            self.p2.feedReward(0.5 * reward_multiplier)

    # board reset
    def reset(self):
        self.board = torch.zeros((BOARD_ROWS, BOARD_COLS), device='cuda')
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        start_time = time.time()
        for i in range(rounds):
            # print("Rounds {}".format(i))
            if i % 10 == 0:
                end_time = time.time()
                print("Rounds {} Time: {}".format(i, end_time - start_time))
                start_time = end_time
            if i % 100 == 0:
                # create a checkpoint
                print("Checkpoint... {}".format(i))
                self.p1.saveCheckpoint(i)
                self.p2.saveCheckpoint(i)
                print("exploration rate p1: {}, p2: {}".format(
                    self.p1.exp_rate, self.p2.exp_rate))

            number_of_actions = 0
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(
                    positions, self.board, self.playerSymbol)
                number_of_actions += 1
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                # check board status if it is end
                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward(number_of_actions)
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(
                        positions, self.board, self.playerSymbol)
                    number_of_actions += 1
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward(number_of_actions)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

            self.p1.decayExploration()
            self.p2.decayExploration()

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(
                positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            line_count = BOARD_COLS * 3 + BOARD_COLS + 1
            print('-' * line_count)
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-' * line_count)


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_rate = 1 - 10e-5
        self.min_exp_rate = 0.01
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    '''
    Exploration rate decay.
    As the training progresses, the exploration rate decreases, allowing the agent to exploit its learned policy more.
    '''

    def decayExploration(self):
        self.exp_rate = max(self.min_exp_rate, self.exp_rate * self.decay_rate)

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if torch.rand(1).item() <= self.exp_rate:
            # take random action
            idx = torch.randint(0, len(positions), (1,)).item()
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                if not isinstance(current_board, torch.Tensor):
                    current_board = torch.tensor(current_board, device='cuda')
                next_board = current_board.clone()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(
                    next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * \
                (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        try:
            fw = open('policy_' + str(self.name) + '.pkl', 'wb')
            pickle.dump(self.states_value, fw)
            fw.close()
        except FileNotFoundError:
            with open('policy_' + str(self.name) + '.pkl', 'wb') as fw:
                pickle.dump(self.states_value, fw)

    def saveCheckpoint(self, generation):
        os.makedirs('checkpoints', exist_ok=True)
        file_path = os.path.join(
            'checkpoints', 'policy_' + str(self.name) + '_' + str(generation) + '.pkl')
        with open(file_path, 'wb') as fw:
            pickle.dump(self.states_value, fw)

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action
            print("Invalid position!")

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    # ------- training --------
    # p1 = Player("p1")
    # p2 = Player("p2")

    # st = State(p1, p2)
    # print("training...")
    # st.play(50000)
    # p1.savePolicy()
    # p2.savePolicy()
    # print("training done!")
    # ------- traning end --------

    # ------- continue training --------
    # p1 = Player("p1")
    # p1.loadPolicy("policy_p1.pkl")
    # p2 = Player("p2")
    # p2.loadPolicy("policy_p2.pkl")
    # st = State(p1, p2)
    # print("Continuing training...")
    # st.play(10000)
    # p1.savePolicy()
    # p2.savePolicy()
    # print("Continuing training done!")
    # ------- continue training end --------

    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1.pkl")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play2()
