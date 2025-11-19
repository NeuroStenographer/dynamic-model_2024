from src.modeling.components import AbstractModelComponent
import torch.nn as nn

class FrameEncoder(AbstractModelComponent):
    """A block that takes in a batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, N_RGB_CHANNELS, X, Y) and shape (N, T, 3, 224, 224), and outputs a batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, FRAME_EMBEDDING_SIZE) and shape (N, T, F).
    """
    def __init__(self, FRAME_EMBEDDING_SIZE, image_encoder, weights=None):
        super(FrameEncoder, self).__init__()
        """Initialize the FrameEncoder with a ResNet50 model and a linear layer."""
        self.FRAME_EMBEDDING_SIZE = FRAME_EMBEDDING_SIZE
        self.image_encoder = image_encoder(weights=weights)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, FRAME_EMBEDDING_SIZE)

    def forward(self, x, tokens=None, kv_cache=None):
        """Forward pass of the FrameEncoder.

        Args:
            x (torch.Tensor): A batch of dimensions (BATCH_SIZE, MAX_SENTENCE_TIME, N_CHANNELS, X, Y) and shape (N, T, 3, 224, 224).

        Returns:
            torch.Tensor: A batch of dimensions (batch_size, max_sentence_time, frame_embedding_size) and shape (N, T, F).
        """
        # Pass the tensor x through the resnet50 model, (N, T, 3, 224, 224) -> (N, T, 2048) -> (N, T, F)
        N, T, C, H, W = x.size()
        # Reshape the tensor x to (N*T, C, H, W)
        x = x.view(N*T, C, H, W)
        x = self.image_encoder(x)
        # Reshape the tensor x to (N, T, F)
        x = x.view(N, T, -1)
        return x

    @property
    def input_shape(self):
        return (3, 224, 224)

    @property
    def output_shape(self):
        return (0, 0, self.FRAME_EMBEDDING_SIZE)

    @property
    def is_output_layer(self):
        return False

    @property
    def name(self):
        return "FrameEncoder"
