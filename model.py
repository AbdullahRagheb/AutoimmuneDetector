import torch
import torch.nn as nn

class MultiLabelNN(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.out(x), dim=1)
