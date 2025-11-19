from src.modeling.components import AbstractModelComponent
import torch.nn as nn

class FrameProjector(AbstractModelComponent):
    """A block that takes in a contrastive frame tensor of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, FRAME_EMBEDDING_SIZE) and shape (N, T, F), and outputs a batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, FRAME_PROJECTION_SIZE) and shape (N, T, FPR).
    """
    def __init__(self, FRAME_EMBEDDING_SIZE, FRAME_PROJECTION_SIZE):
        super(FrameProjector, self).__init__()
        """Initialize the FrameProjector with a linear layer."""
        self.FRAME_EMBEDDING_SIZE = FRAME_EMBEDDING_SIZE
        self.FRAME_PROJECTION_SIZE = FRAME_PROJECTION_SIZE
        self.linear1 = nn.Linear(in_features=FRAME_EMBEDDING_SIZE, out_features=FRAME_PROJECTION_SIZE)
        self.bn = nn.BatchNorm1d(FRAME_PROJECTION_SIZE)
        self.relu = nn.ReLU()

    def forward(self, x, tokens=None, kv_cache=None):
        """Forward pass of the FrameProjector.

        Args:
            x (torch.Tensor): A batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, FRAME_EMBEDDING_SIZE) and shape (N, T, F).

        Returns:
            torch.Tensor: A batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, FRAME_PROJECTION_SIZE) and shape (N, T, PR).
        """
        # Flatten for linear layer: (N, T, F) -> (N*T, F)
        print(f"FrameProjector Input Shape: {x.shape}")
        N, T, F = x.size()
        x = x.view(-1, F)
        # Projection: (N*T, F) -> (N*T, PR)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        # Reshape back to original: (N*T, PR) -> (N, T, PR)
        x = x.view(N, T, -1)
        print(f"Frame Projector Output Shape: {x.shape}")
        return x

    @property
    def input_shape(self):
        return (0,0,self.FRAME_EMBEDDING_SIZE)

    @property
    def output_shape(self):
        return (0,0,self.FRAME_PROJECTION_SIZE)

    @property
    def is_output_layer(self):
        return True

    @property
    def name(self):
        return "FrameProjector"

