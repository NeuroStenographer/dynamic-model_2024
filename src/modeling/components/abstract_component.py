from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractModelComponent(ABC, nn.Module):
    def __init__(self):
        super(AbstractModelComponent, self).__init__()

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @property
    @abstractmethod
    def is_output_layer(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass