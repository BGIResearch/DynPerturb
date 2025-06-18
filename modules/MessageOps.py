class MLPMessageFunction:
    def __init__(self, raw_message_dim, hidden_dim, device):
        import torch.nn as nn
        import torch
        self.linear1 = nn.Linear(raw_message_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.device = device

    def __call__(self, raw_messages):
        import torch
        return self.linear2(self.relu(self.linear1(raw_messages)))


class IdentityMessageFunction:
    def __init__(self, raw_message_dim, hidden_dim, device):
        pass

    def __call__(self, raw_messages):
        return raw_messages


def get_message_function(module_type, raw_message_dim, hidden_dim, device):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dim, hidden_dim, device)
    elif module_type == "identity":
        return IdentityMessageFunction(raw_message_dim, hidden_dim, device)
    else:
        raise ValueError(f"Unknown message function type: {module_type}")


class LastMessageAggregator:
    def __init__(self):
        pass

    def aggregate(self, messages):
        return messages[-1]


class MeanMessageAggregator:
    def __init__(self):
        import torch
        self.torch = torch

    def aggregate(self, messages):
        return self.torch.mean(messages, dim=0)


def get_message_aggregator(aggregator_type):
    if aggregator_type == "last":
        return LastMessageAggregator()
    elif aggregator_type == "mean":
        return MeanMessageAggregator()
    else:
        raise ValueError(f"Unknown message aggregator type: {aggregator_type}")