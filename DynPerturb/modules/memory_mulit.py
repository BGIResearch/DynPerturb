import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    self.memory = torch.zeros((self.n_nodes, self.memory_dimension), device=self.device)
    self.last_update = torch.zeros(self.n_nodes, device=self.device)
    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend([(m[0].detach(), m[1]) for m in node_id_to_messages[node]])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :].clone().detach()

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values.detach()

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs].clone().detach()

  def backup_memory(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone().detach(), x[1]) for x in v]
    return self.memory.clone().detach(), self.last_update.clone().detach(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory = memory_backup[0].clone().detach()
    self.last_update = memory_backup[1].clone().detach()
    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone().detach(), x[1]) for x in v]

  def detach_memory(self):
    self.memory = self.memory.detach()
    self.last_update = self.last_update.detach()
    for k, v in self.messages.items():
      new_node_messages = [(msg[0].detach(), msg[1]) for msg in v]
      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
