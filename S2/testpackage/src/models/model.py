from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Linear(28 * 28, 200, dtype=float)
        self.relu0 = nn.ReLU()
        self.hidden0 = nn.Linear(200, 100, dtype=float)
        self.relu1 = nn.ReLU()
        self.hidden1 = nn.Linear(100, 50, dtype=float)
        self.relu2 = nn.ReLU()
        self.out_layer = nn.Linear(50, 10, dtype=float)

    def forward(self, out):
        out = self.in_layer(out)
        out = self.hidden0(self.relu0(out))
        out = self.hidden1(self.relu1(out))
        out = self.out_layer(self.relu2(out))
        return out
