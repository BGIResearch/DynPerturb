import torch
import torch.nn as nn
import numpy as np


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, input_dimension, device):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.memory_dimension = memory_dimension
        self.memory = torch.zeros((self.n_nodes, self.memory_dimension), dtype=torch.float, device=device)
        self.last_update = torch.zeros(self.n_nodes, dtype=torch.float, device=device)

    def forward(self):
        return self.memory

    def get_memory(self, node_ids):
        return self.memory[node_ids, :]

    def set_memory(self, node_ids, values):
        self.memory[node_ids, :] = values

    def update_last_update(self, node_ids, timestamps):
        self.last_update[node_ids] = timestamps

    def get_last_update(self, node_ids):
        return self.last_update[node_ids]

    def backup_memory(self):
        return self.memory.clone(), self.last_update.clone()

    def restore_memory(self, memory_backup):
        self.memory, self.last_update = memory_backup


class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory_dimension, message_dimension):
        super(GRUMemoryUpdater, self).__init__()
        self.memory_updater = nn.GRUCell(input_size=message_dimension, hidden_size=memory_dimension)

    def forward(self, memory, messages):
        return self.memory_updater(messages, memory)


def get_memory_updater(memory_dimension, message_dimension, module_type="gru"):
    if module_type == "gru":
        return GRUMemoryUpdater(memory_dimension, message_dimension)
    else:
        raise NotImplementedError(f"Memory updater type '{module_type}' not implemented")