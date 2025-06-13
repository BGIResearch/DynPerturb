from torch import nn
import torch


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        # 遍历每个节点，逐一检查时间戳
        for i, node_id in enumerate(unique_node_ids):
            last_update_timestamp = self.memory.get_last_update(node_id)
            current_timestamp = timestamps[i]

            # print(f"Updating memory for node: {node_id}")
            # print(f"Last update timestamp: {last_update_timestamp}")
            # print(f"Current timestamp: {current_timestamp}")

            if last_update_timestamp > current_timestamp:
                # print(f"Skipping memory update for node {node_id} because timestamp is in the past")
                continue  # 跳过此节点

            # 如果时间戳匹配，更新该节点的内存
            memory = self.memory.get_memory(node_id)
            self.memory.last_update[node_id] = current_timestamp

            updated_memory = self.memory_updater(unique_messages[i], memory)
            self.memory.set_memory(node_id, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        updated_memory = self.memory.memory.data.clone()
        updated_last_update = self.memory.last_update.data.clone()

        # 遍历每个节点，逐一检查时间戳
        for i, node_id in enumerate(unique_node_ids):
            last_update_timestamp = self.memory.get_last_update(node_id)
            current_timestamp = timestamps[i]

            # print(f"Updating memory for node: {node_id}")
            # print(f"Last update timestamp: {last_update_timestamp}")
            # print(f"Current timestamp: {current_timestamp}")

            if last_update_timestamp > current_timestamp:
                # print(f"Skipping memory update for node {node_id} because timestamp is in the past")
                continue  # 跳过此节点

            # 更新内存
            updated_memory[node_id] = self.memory_updater(unique_messages[i], updated_memory[node_id])
            updated_last_update[node_id] = current_timestamp

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class LSTMMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(LSTMMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.LSTMCell(input_size=message_dimension, hidden_size=memory_dimension)
        self.memory_cell = torch.zeros_like(self.memory.memory.data)  # 与 memory 同 shape
        self.memory_cell = self.memory_cell.to(device)

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        for i, node_id in enumerate(unique_node_ids):
            last_update_timestamp = self.memory.get_last_update(node_id)
            current_timestamp = timestamps[i]

            if last_update_timestamp > current_timestamp:
                continue

            # 当前 h 和 c 状态
            h_prev = self.memory.get_memory(node_id)
            c_prev = self.memory_cell[node_id]

            h_new, c_new = self.memory_updater(unique_messages[i], (h_prev, c_prev))

            self.memory.set_memory(node_id, h_new)
            self.memory_cell[node_id] = c_new
            self.memory.last_update[node_id] = current_timestamp

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
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



def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "lstm":
        return LSTMMemoryUpdater(memory, message_dimension, memory_dimension, device)


