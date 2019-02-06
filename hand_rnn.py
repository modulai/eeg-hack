import torch.nn as nn


class HandRNN(nn.Module):
    def __init__(self,
                 input_size=32,
                 hidden_size=50,
                 num_layers=1,
                 events=6):
        super(HandRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.events = events

        # lstm
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        # Prediction layer
        self.pred = nn.Linear(self.hidden_size, self.events)

        # Normalise output to probability
        self.out = nn.Sigmoid()

    def forward(self, x, device):

        o, (_, _) = self.lstm(x)
        # Assume only one event
        y1 = self.pred(o)
        y = self.out(y1)
        return y
