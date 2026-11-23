import torch
from torch import nn


class RecognitionGRU(nn.Module):
    def __init__(self, hidden=128, classes=12):
        super().__init__()
        self.gru = nn.GRU(6, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, classes))

    def forward(self, x, lengths):
        outputs, _ = self.gru(x)
        return self.head(outputs[torch.arange(len(x), device=x.device), lengths - 1])


class RecognitionMLP(nn.Module):
    def __init__(self, hidden=128, classes=12):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1024, hidden), nn.ReLU(), nn.Dropout(.1),
                                    nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(.1),
                                    nn.Linear(hidden // 2, classes))

    def forward(self, image, lengths=None):
        return self.layers(image.reshape(len(image), 1024))


def make_model(family, hidden, classes=12):
    return (RecognitionGRU if family == 'gru' else RecognitionMLP)(hidden, classes)
