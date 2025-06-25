from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
    """
    Abstract class for the message aggregator module. Given a batch of node ids and
    corresponding messages, aggregates messages with the same node id.
    """
    def __init__(self, device):
        super(MessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages):
        """
        Given a list of node ids and a list of messages of the same length, aggregate different
        messages for the same id using one of the possible strategies.
        :param node_ids: A list of node ids of length batch_size
        :param messages: A tensor of shape [batch_size, message_length]
        :param timestamps: A tensor of shape [batch_size]
        :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
        """
        pass

    def group_by_id(self, node_ids, messages, timestamps):
        """
        Group messages by node id, pairing each message with its timestamp.
        :param node_ids: List of node ids
        :param messages: List or tensor of messages
        :param timestamps: List or tensor of timestamps
        :return: Dictionary mapping node id to list of (message, timestamp) tuples
        """
        node_id_to_messages = defaultdict(list)
        for i, node_id in enumerate(node_ids):
            node_id_to_messages[node_id].append((messages[i], timestamps[i]))
        return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(LastMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """
        Only keep the last message for each node.
        :param node_ids: List of node ids
        :param messages: Dictionary mapping node id to list of (message, timestamp)
        :return: List of node ids to update, stacked messages, stacked timestamps
        """
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        to_update_node_ids = []
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])
        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
        return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(MeanMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """
        Aggregate messages for each node by taking the mean of all messages for that node.
        :param node_ids: List of node ids
        :param messages: Dictionary mapping node id to list of (message, timestamp)
        :return: List of node ids to update, stacked mean messages, stacked timestamps
        """
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        to_update_node_ids = []
        n_messages = 0
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                n_messages += len(messages[node_id])
                to_update_node_ids.append(node_id)
                unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
                unique_timestamps.append(messages[node_id][-1][1])
        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
        return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device):
    """
    Factory function to get the appropriate message aggregator.
    :param aggregator_type: Type of aggregator ('last' or 'mean')
    :param device: Device to use
    :return: MessageAggregator instance
    """
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    else:
        raise ValueError("Message aggregator {} not implemented".format(aggregator_type))

from torch import nn


class MessageFunction(nn.Module):
    """
    Module which computes the message for a given interaction.
    """
    def compute_message(self, raw_messages):
        """
        Compute the message from raw input messages. Should be implemented by subclasses.
        :param raw_messages: Input tensor
        :return: Output tensor
        """
        return None


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension, message_dimension):
        super(MLPMessageFunction, self).__init__()
        # Multi-layer perceptron for message transformation
        self.mlp = self.layers = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        )

    def compute_message(self, raw_messages):
        """
        Compute the message using an MLP transformation.
        :param raw_messages: Input tensor
        :return: Output tensor
        """
        messages = self.mlp(raw_messages)
        return messages


class IdentityMessageFunction(MessageFunction):
    def compute_message(self, raw_messages):
        """
        Return the raw messages without any transformation.
        :param raw_messages: Input tensor
        :return: Output tensor (same as input)
        """
        return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension):
    """
    Factory function to get the appropriate message function module.
    :param module_type: Type of message function ('mlp' or 'identity')
    :param raw_message_dimension: Input dimension
    :param message_dimension: Output dimension
    :return: MessageFunction instance
    """
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":
        return IdentityMessageFunction()


