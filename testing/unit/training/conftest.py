import pytest
import torch
from torch.utils.data import Dataset
from torch import nn

class MockDataset(Dataset):
    def __init__(self, length=5, N=1, T=100, fixed_shape=(4, 3, 8, 8)):
        self.N = N
        self.T = T
        self.fixed_shape = fixed_shape
        self.length = length

    def __getitem__(self, idx):
        return torch.rand((self.N, self.T, *self.fixed_shape))

    def __iter__(self):
        return iter([self[i] for i in range(self.length)])

    def __len__(self):
        return self.length

@pytest.fixture
def mock_dataset():
    return MockDataset()