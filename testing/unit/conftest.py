from src.dataloading.dataset import MatFile, MatDataset
from src.config import Config

import torch.nn as nn
import torch

train_dir = Config.DIRS.MAT.TRAIN

import pytest

@pytest.fixture
def mat_file():
    return MatFile(train_dir + '/t12.2022.04.28.mat')

@pytest.fixture
def mat_dataset():
    return MatDataset(train_dir)

class MockDataset:
    def __init__(self):
        self.n_batches = 5

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch = []
        for _ in range(10):
            # random T
            T = torch.randint(10, 100, (1,))
            batch.append(torch.rand(1, T, 4, 3, 8, 8))
        return {
            'signals': batch,
            'text': ['text'] * 10
        }

@pytest.fixture
def mock_dataset():
    return MockDataset()

class MockTICAModel(nn.Module):
    def __init__(self):
        super(MockTICAModel, self).__init__()
        # takes in a tensor of shape (N, T, 4, 3, 8, 8)
        # returns a tensor of shape (N, T, 41)
        self.linear = nn.Linear(4 * 3 * 8 * 8, 41)

    def forward(self, x):
        if isinstance(x, list):
            return [self._forward_tensor(xi) for xi in x]
        elif isinstance(x, torch.Tensor):
            return self._forward_tensor(x)

    def _forward_tensor(self, x):
        N = x.shape[0]
        T = x.shape[1]
        # reshape x to (N * T, 4 * 3 * 8 * 8)
        x = x.reshape(N * T, 4 * 3 * 8 * 8)
        # pass through linear layer
        x = self.linear(x)
        # reshape x to (N, T, 41)
        x = x.reshape(N, T, 41)
        return x

@pytest.fixture
def mock_tica_model():
    return MockTICAModel()