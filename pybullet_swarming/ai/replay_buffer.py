import torch
import numpy as np
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """
    Replay Buffer implementation of a Torch dataset
    """

    def __init__(self, mem_size):
        self.memory = []
        self.mem_size = int(mem_size)
        self.counter = 0

    def __len__(self):
        return min(self.mem_size, self.counter)

    def __getitem__(self, idx):
        data = []
        for item in self.memory:
            data.append(item[idx])

        return (*data,)

    def push(self, data):
        if self.counter == 0:
            self.memory = []
            for i in range(len(data)):
                self.memory.append(
                    np.zeros((self.mem_size, *data[i].shape), dtype=np.float32)
                )

        assert len(data) == len(
            self.memory
        ), "data length not similar to memory buffer length"

        i = self.counter % self.mem_size
        for j in range(len(data)):
            self.memory[j][i] = data[j]

        self.counter += 1

    @property
    def is_full(self):
        return self.counter >= self.mem_size
