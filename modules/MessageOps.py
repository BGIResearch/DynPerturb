# Message function using a two-layer MLP
class MLPMessageFunction:
    def __init__(self, raw_message_dim, hidden_dim, device):
        import torch.nn as nn

        self.linear1 = nn.Linear(raw_message_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.device = device

    def __call__(self, raw_messages):

        # Pass raw messages through MLP
        return self.linear2(self.relu(self.linear1(raw_messages)))


# Message function that returns the input as is
class IdentityMessageFunction:
    def __init__(self, raw_message_dim, hidden_dim, device):
        pass

    def __call__(self, raw_messages):
        # No transformation
        return raw_messages


# Factory for message function


def get_message_function(module_type, raw_message_dim, hidden_dim, device):
    """
    Returns a message function instance based on the type.
    """
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dim, hidden_dim, device)
    elif module_type == "identity":
        return IdentityMessageFunction(raw_message_dim, hidden_dim, device)
    else:
        raise ValueError(f"Unknown message function type: {module_type}")


# Aggregator that returns the last message
class LastMessageAggregator:
    def __init__(self):
        pass

    def aggregate(self, messages):
        # Return the last message in the list
        return messages[-1]


# Aggregator that computes the mean of messages
class MeanMessageAggregator:
    def __init__(self):
        import torch

        self.torch = torch

    def aggregate(self, messages):
        # Compute mean along the first dimension
        return self.torch.mean(messages, dim=0)


# Factory for message aggregator


def get_message_aggregator(aggregator_type):
    """
    Returns a message aggregator instance based on the type.
    """
    if aggregator_type == "last":
        return LastMessageAggregator()
    elif aggregator_type == "mean":
        return MeanMessageAggregator()
    else:
        raise ValueError(f"Unknown message aggregator type: {aggregator_type}")
