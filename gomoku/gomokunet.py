import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import BOARD_SIZE


class GomokuNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super().__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        final_size = board_size // 4 if board_size >= 8 else 1

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_fc = nn.Linear(
            2 * final_size * final_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * final_size * final_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board):
        """
        board shape: (batch_size, 2, 6, 6)
        """
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        if self.board_size >= 8:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        if self.board_size >= 8:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))

        x = F.dropout(x, p=0.2, training=self.training)

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)  # Log probabilities

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # Output between -1 and 1

        return policy, value


# print out the model architecture
# net = GomokuNet()
# print(net)
# print out the model summary
