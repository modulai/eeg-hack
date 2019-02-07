import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class HandRNN(nn.Module):

    def pack(self, x, lengths):
        return rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True)

    def unpack(self, x, l):
        p = rnn_utils.PackedSequence(x, l)
        return rnn_utils.pad_packed_sequence(
            p, batch_first=True)

    def __init__(self,
                 dummy_input,
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

        self.traced_packer = torch.jit.trace(self.pack,
                                             dummy_input)
        o = self.traced_packer(*dummy_input)

        self.traced_unpacker = torch.jit.trace(self.unpack,
                                               o)

    def forward(self, x, lengths):
        px = self.traced_packer(x, lengths)
        o = self.lstm(rnn_utils.PackedSequence(*px))
#        print(o)
        px = self.traced_unpacker(o[0].data, o[0].batch_sizes)
        # Assume only one event
        y1 = self.pred(px[0])
        y = self.out(y1)
        return y
