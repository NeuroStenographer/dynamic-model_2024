from src.modeling import assemble_model
from src.dependency_injection import SHARED_COMPONENTS
from src.config import Config
from src.utils.memory_utils import get_current_memory_usage


import torch
import pytest

class TestDynamicModel:

    C = Config["MODEL_PARAMETERS"]["N_PHONEMES"]
    FP = Config["MODEL_PARAMETERS"]["FRAME_PROJECTION_SIZE"]

    @pytest.mark.parametrize('N', [1, 10])
    @pytest.mark.parametrize('T', [5, 10, 20])
    def test_ctc_decoder_assembly_and_io_shape(self, N, T):
        ctc_component_names = ['frame_intake', 'frame_encoder', 'temporal_encoder', 'temporal_decoder']
        ctc_decoder = assemble_model(SHARED_COMPONENTS=SHARED_COMPONENTS,
                                     REQUIRED_COMPONENT_NAMES=ctc_component_names)
        input_batch = torch.rand((N, T, 4, 3, 8, 8))
        output = ctc_decoder(input_batch)
        assert output.shape == (N, T, self.C)

    @pytest.mark.parametrize('N', [1, 10])
    @pytest.mark.parametrize('T', [5, 10, 20])
    def test_frame_simclr_assembly_and_io_shape(self, N, T):
        frame_simclr_component_names = ['frame_intake', 'frame_encoder', 'frame_projector1']
        frame_simclr = assemble_model(SHARED_COMPONENTS=SHARED_COMPONENTS,
                                      REQUIRED_COMPONENT_NAMES=frame_simclr_component_names)
        input_batch = torch.rand((N, T, 4, 3, 8, 8))
        output = frame_simclr(input_batch)
        assert output.shape == (N, T, self.FP)

    @pytest.mark.parametrize('N', [1, 10])
    @pytest.mark.parametrize('T', [5, 10, 20])
    def test_frame_temporal_ica_assembly_and_io_shape(self, N, T):
        frame_temporal_ica_component_names = ['frame_intake', 'frame_encoder', 'frame_projector2']
        frame_temporal_ica = assemble_model(SHARED_COMPONENTS=SHARED_COMPONENTS,
                                            REQUIRED_COMPONENT_NAMES=frame_temporal_ica_component_names)
        input_batch = torch.rand((N, T, 4, 3, 8, 8))
        output = frame_temporal_ica(input_batch)
        assert output.shape == (N, T, self.FP)