import torch
import numpy as np


class TimeEncode(torch.nn.Module):
    """
    Time Encoding module as proposed by TGAT.
    Encodes time information into a high-dimensional representation using a learnable linear layer.
    """

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        # Linear layer for time encoding
        self.w = torch.nn.Linear(1, dimension)
        # Initialize weights with inverse exponential scaling
        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
            .float().reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        """
        Forward pass for time encoding.
        :param t: Tensor of shape [batch_size, seq_len]
        :return: Encoded time tensor of shape [batch_size, seq_len, dimension]
        """
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)
        # Apply linear layer and cosine activation
        output = torch.cos(self.w(t))

        return output
