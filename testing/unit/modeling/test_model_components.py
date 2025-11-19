import pytest
import torch
from src.modeling.components import (FrameIntakeBlock,
                                   FrameEncoder,
                                   FrameProjector,
                                   TemporalProjector)

from torchvision.models import resnet18, ResNet18_Weights

class TestModelComponents:

    # define constants
    T = 10
    N = 10

    # test FrameIntakeBlock
    # Category 1: happy path
    # ------------------------------
    # Test 1.1: input of shape (N, T, 4, 3, 8, 8) -> (N, T, 3, 224, 224)
    def test_frame_frame_intake_forward_shape(self):
        frame_frame_intake = FrameIntakeBlock()
        input_batch = torch.rand((self.N, self.T, 4, 3, 8, 8))
        output = frame_frame_intake(input_batch)
        assert output.shape == (self.N, self.T, 3, 224, 224)

    # test FrameEncoder
    # Category 1: happy path
    # ------------------------------
    # Test 1.1: input of shape (N, T, 3, 224, 224) -> (N, T, 2048)
    @pytest.mark.parametrize('FRAME_EMBEDDING_SIZE', [16, 32, 64, 128, 256, 512, 1024, 2048])
    def test_frame_encoder_forward_shape(self, FRAME_EMBEDDING_SIZE):
        frame_encoder = FrameEncoder(FRAME_EMBEDDING_SIZE=FRAME_EMBEDDING_SIZE,
                                     image_encoder=resnet18,
                                     weights=ResNet18_Weights.DEFAULT)
        input_batch = torch.rand((self.N, self.T, 3, 224, 224))
        output = frame_encoder(input_batch)
        assert output.shape == (self.N, self.T, FRAME_EMBEDDING_SIZE)

    # test FrameProjector
    # Category 1: happy path
    # ------------------------------
    # Test 1.1: input of shape (N, T, F) -> (N, T, PR)
    @pytest.mark.parametrize('FRAME_EMBEDDING_SIZE', [16, 32, 64, 128, 256, 512, 1024, 2048])
    @pytest.mark.parametrize('FRAME_PROJECTION_SIZE', [10, 45, 100, 1024])
    def test_frame_projector_forward_shape(self, FRAME_EMBEDDING_SIZE, FRAME_PROJECTION_SIZE):
        frame_projector = FrameProjector(FRAME_EMBEDDING_SIZE=FRAME_EMBEDDING_SIZE, FRAME_PROJECTION_SIZE=FRAME_PROJECTION_SIZE)
        input_batch = torch.rand((self.N, self.T, FRAME_EMBEDDING_SIZE))
        output = frame_projector(input_batch)
        assert output.shape == (self.N, self.T, FRAME_PROJECTION_SIZE)

    # test TemporalEncoder
    # Category 1: happy path
    # ------------------------------
    # Test 1.1: input of shape (N, T, F) -> (N, T, TF)
    @pytest.mark.parametrize('FRAME_EMBEDDING_SIZE', [16, 32, 64, 128, 256, 512, 1024, 2048])
    @pytest.mark.parametrize('TEMPORAL_EMBEDDING_SIZE', [16, 32, 64, 128, 256, 512, 1024, 2048])
    def test_temporal_encoder_forward_shape(self, FRAME_EMBEDDING_SIZE, TEMPORAL_EMBEDDING_SIZE):
        temporal_encoder = TemporalEncoder(FRAME_EMBEDDING_SIZE=FRAME_EMBEDDING_SIZE, TEMPORAL_EMBEDDING_SIZE=TEMPORAL_EMBEDDING_SIZE)
        input_batch = torch.rand((self.N, self.T, FRAME_EMBEDDING_SIZE))
        output = temporal_encoder(input_batch)
        assert output.shape == (self.N, self.T, TEMPORAL_EMBEDDING_SIZE)

    # test TemporalProjector
    # Category 1: happy path
    # ------------------------------
    # Test 1.1: input of shape (N, T, TF) -> (N, T, TPR)
    @pytest.mark.parametrize('TEMPORAL_EMBEDDING_SIZE', [16, 32, 64, 128, 256, 512, 1024, 2048])
    @pytest.mark.parametrize('TEMPORAL_PROJECTION_SIZE', [10, 45, 100, 1024])
    def test_temporal_projector_forward_shape(self, TEMPORAL_EMBEDDING_SIZE, TEMPORAL_PROJECTION_SIZE):
        temporal_projector = TemporalProjector(TEMPORAL_EMBEDDING_SIZE=TEMPORAL_EMBEDDING_SIZE, TEMPORAL_PROJECTION_SIZE=TEMPORAL_PROJECTION_SIZE)
        input_batch = torch.rand((self.N, self.T, TEMPORAL_EMBEDDING_SIZE))
        output = temporal_projector(input_batch)
        assert output.shape == (self.N, self.T, TEMPORAL_PROJECTION_SIZE)

    # test TemporalDecoder
    # Category 1: happy path
    # ------------------------------
    # Test 1.1: input of shape (N, T, TPR) -> (N, T, TF)
    @pytest.mark.parametrize('TEMPORAL_EMBEDDING_SIZE', [16, 32, 64, 128, 256, 512, 1024, 2048])
    @pytest.mark.parametrize('N_PHONEMES', [10, 45, 100, 1024])
    def test_temporal_decoder_forward_shape(self, TEMPORAL_EMBEDDING_SIZE, N_PHONEMES):
        temporal_decoder = TemporalDecoder(TEMPORAL_EMBEDDING_SIZE=TEMPORAL_EMBEDDING_SIZE, N_PHONEMES=N_PHONEMES)
        input_batch = torch.rand((self.N, self.T, TEMPORAL_EMBEDDING_SIZE))
        output = temporal_decoder(input_batch)
        assert output.shape == (self.N, self.T, N_PHONEMES)
