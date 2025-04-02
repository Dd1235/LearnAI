import torch
import torch.nn as nn

# character level languauge model


class CharGRU(nn.Module):
    # total number of unique characters, dimensionality of hidden state, num of stacked gru layers
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size
        )  # for each character index, returns a vector of hidden size, so look up table vocab size x hidden size
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    # x: input tensor (batch_size, sequence length), torch.LongTensor

    def forward(self, x, hidden):
        x = self.embed(x)  # converts to (batch_size, sequence length, hidden size)
        out, hidden = self.gru(
            x, hidden
        )  # out: (batch_size, sequence_length, hidden_size), hidden: (num_layers, batch_size, hidden_size)
        out = self.fc(
            out.reshape(-1, self.hidden_size)
        )  # (batch_size × seq_len, hidden_size)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
