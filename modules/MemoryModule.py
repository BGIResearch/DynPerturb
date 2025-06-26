import torch
from torch import nn

from collections import defaultdict



class Memory(nn.Module):
    def __init__(
        self,
        n_nodes,
        memory_dimension,
        input_dimension,
        message_dimension=None,
        device="cpu",
        combination_method="sum",
        mode="link_prediction",
    ):
        super(Memory, self).__init__()
        self.mode = mode
        if mode not in ["link_prediction", "node_classification"]:
            raise ValueError("Invalid mode. Choose either 'link_prediction' or 'node_classification'.")
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
        # Treat memory as parameter so that it is saved and loaded together with the model
        if self.mode == "link_prediction":
            self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),requires_grad=False,)
            self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device), requires_grad=False)
        else:
            self.memory = torch.zeros(
                (self.n_nodes, self.memory_dimension), device=self.device
            )
            self.last_update = torch.zeros(self.n_nodes, device=self.device)

        self.messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages):
        """
        Store raw messages for each node. In link prediction mode, messages are stored directly.
        In node classification mode, messages are detached from the computation graph.
        """
        if self.mode == "link_prediction":
            for node in nodes:
                self.messages[node].extend(node_id_to_messages[node])
        else:
            for node in nodes:
                self.messages[node].extend(
                    [(m[0].detach(), m[1]) for m in node_id_to_messages[node]]
                )

    def get_memory(self, node_idxs):
        """
        Get the memory for the given node indices.
        In link prediction mode, returns the memory directly.
        In node classification mode, returns a detached copy.
        """
        if self.mode == "link_prediction":
            return self.memory[node_idxs, :]
        else:
            return self.memory[node_idxs, :].clone().detach()

    def set_memory(self, node_idxs, values):
        """
        Set the memory for the given node indices.
        In link prediction mode, sets the memory directly.
        In node classification mode, sets a detached copy.
        """
        if self.mode == "link_prediction":
            self.memory[node_idxs, :] = values
        else:
            self.memory[node_idxs, :] = values.detach()

    def get_last_update(self, node_idxs):
        """
        Get the last update timestamps for the given node indices.
        In link prediction mode, returns the timestamps directly.
        In node classification mode, returns a detached copy.
        """
        if self.mode == "link_prediction":
            return self.last_update[node_idxs]
        else:
            return self.last_update[node_idxs].clone().detach()

    def backup_memory(self):
        """
        Backup the current memory, last update, and messages.
        Returns a tuple containing the memory, last update, and messages.
        """
        if self.mode == "link_prediction":
            messages_clone = {}
            for k, v in self.messages.items():
                messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

            return (self.memory.data.clone(),self.last_update.data.clone(),messages_clone)
        else:
            messages_clone = {}
            for k, v in self.messages.items():
                messages_clone[k] = [(x[0].clone().detach(), x[1]) for x in v]
            return (self.memory.clone().detach(),self.last_update.clone().detach(),messages_clone,)

    def restore_memory(self, memory_backup):
        """
        Restore the memory, last update, and messages from a backup.
        """
        if self.mode == "link_prediction":
            self.memory.data, self.last_update.data = (memory_backup[0].clone(),memory_backup[1].clone(),)

            self.messages = defaultdict(list)
            for k, v in memory_backup[2].items():
                self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]
        else:
            self.memory = memory_backup[0].clone().detach()
            self.last_update = memory_backup[1].clone().detach()
            self.messages = defaultdict(list)
            for k, v in memory_backup[2].items():
                self.messages[k] = [(x[0].clone().detach(), x[1]) for x in v]

    def detach_memory(self):
        """
        Detach memory and all stored messages from the computation graph.
        This is important to avoid memory leaks in PyTorch when using memory modules.
        """
        if self.mode == "link_prediction":
            self.memory.detach_()

            # Detach all stored messages for each node
            for k, v in self.messages.items():
                new_node_messages = []
                for message in v:
                    new_node_messages.append((message[0].detach(), message[1]))
                self.messages[k] = new_node_messages
        else:
            self.memory = self.memory.detach()
            self.last_update = self.last_update.detach()
            # Detach all stored messages for each node
            for k, v in self.messages.items():
                new_node_messages = [(msg[0].detach(), msg[1]) for msg in v]
                self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        """
        Clear all stored messages for the given nodes.
        """
        for node in nodes:
            self.messages[node] = []


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Abstract method for updating memory. Should be implemented by subclasses.
        """
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device, mode):
        """
        Sequence-based memory updater for temporal graph models.
        """
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device
        self.mode = mode

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Update memory for a batch of unique nodes and messages at given timestamps.
        Handles both link prediction and node classification modes.
        """
        if len(unique_node_ids) <= 0:
            return
        if self.mode == "link_prediction":
            assert ((self.memory.get_last_update(unique_node_ids) <= timestamps) .all() .item()
            ), ("Trying to " "update memory to time in the past")

            memory = self.memory.get_memory(unique_node_ids)
            self.memory.last_update[unique_node_ids] = timestamps

            updated_memory = self.memory_updater(unique_messages, memory)

            self.memory.set_memory(unique_node_ids, updated_memory)
        else:
            for i, node_id in enumerate(unique_node_ids):
                last_update_timestamp = self.memory.get_last_update(node_id)
                current_timestamp = timestamps[i]
                if last_update_timestamp > current_timestamp:
                    # Skip update if timestamp is in the past
                    continue
                # Update memory for this node
                memory = self.memory.get_memory(node_id)
                self.memory.last_update[node_id] = current_timestamp

                updated_memory = self.memory_updater(unique_messages[i], memory)
                self.memory.set_memory(node_id, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Return updated memory and last_update for a batch of unique nodes and messages at given timestamps.
        Does not modify the actual memory, just returns the updated values.
        """
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        if self.mode == "link_prediction":
            assert ((self.memory.get_last_update(unique_node_ids) <= timestamps) .all() .item()), ("Trying to " "update memory to time in the past")

            updated_memory = self.memory.memory.data.clone()
            updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

            updated_last_update = self.memory.last_update.data.clone()
            updated_last_update[unique_node_ids] = timestamps

            return updated_memory, updated_last_update
        else:
            updated_memory = self.memory.memory.data.clone()
            updated_last_update = self.memory.last_update.data.clone()

            # Iterate each node and update if timestamp is valid
            for i, node_id in enumerate(unique_node_ids):
                last_update_timestamp = self.memory.get_last_update(node_id)
                current_timestamp = timestamps[i]

                if last_update_timestamp > current_timestamp:
                    continue

                updated_memory[node_id] = self.memory_updater(unique_messages[i], updated_memory[node_id])
                updated_last_update[node_id] = current_timestamp

            return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device,mode):
        """
        GRU-based memory updater for temporal graph models.
        """
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device,mode)

        self.memory_updater = nn.GRUCell(input_size=message_dimension, hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device,mode):
        """
        RNN-based memory updater for temporal graph models.
        """
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device,mode)

        self.memory_updater = nn.RNNCell(input_size=message_dimension, hidden_size=memory_dimension)


class LSTMMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device,mode):
        """
        LSTM-based memory updater for temporal graph models.
        """
        super(LSTMMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device,mode)
        self.memory_updater = nn.LSTMCell(input_size=message_dimension, hidden_size=memory_dimension)
        # Create a memory cell with the same shape as memory
        self.memory_cell = torch.zeros_like(self.memory.memory.data)
        self.memory_cell = self.memory_cell.to(device)

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Update memory and cell state for a batch of unique nodes and messages at given timestamps using LSTM.
        """
        if len(unique_node_ids) <= 0:
            return

        for i, node_id in enumerate(unique_node_ids):
            last_update_timestamp = self.memory.get_last_update(node_id)
            current_timestamp = timestamps[i]

            if last_update_timestamp > current_timestamp:
                continue

            # Get current hidden and cell states
            h_prev = self.memory.get_memory(node_id)
            c_prev = self.memory_cell[node_id]

            h_new, c_new = self.memory_updater(unique_messages[i], (h_prev, c_prev))

            self.memory.set_memory(node_id, h_new)
            self.memory_cell[node_id] = c_new
            self.memory.last_update[node_id] = current_timestamp

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Return updated memory, cell state, and last_update for a batch of unique nodes and messages at given timestamps using LSTM.
        """
        updated_memory = self.memory.memory.data.clone()
        updated_cell = self.memory_cell.clone()
        updated_last_update = self.memory.last_update.data.clone()

        for i, node_id in enumerate(unique_node_ids):
            last_update_timestamp = self.memory.get_last_update(node_id)
            current_timestamp = timestamps[i]

            if last_update_timestamp > current_timestamp:
                continue

            h_prev = updated_memory[node_id]
            c_prev = updated_cell[node_id]

            h_new, c_new = self.memory_updater(unique_messages[i], (h_prev, c_prev))

            updated_memory[node_id] = h_new
            updated_cell[node_id] = c_new
            updated_last_update[node_id] = current_timestamp

        return updated_memory, updated_last_update


def get_memory_updater(
    module_type,
    memory,
    message_dimension,
    memory_dimension,
    device,
    mode="link_prediction"
):
    """
    Factory function to get the appropriate memory updater module.
    """
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device, mode)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device, mode)
    elif module_type == "lstm":
        return LSTMMemoryUpdater(memory, message_dimension, memory_dimension, device, mode)
