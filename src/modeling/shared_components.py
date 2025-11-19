from src.modeling.components import *
from src.config import Config
from torchvision.models import resnet18, ResNet18_Weights

PARAMS = Config["MODEL_PARAMETERS"]

SHARED_COMPONENTS = {
    "frame_intake": FrameIntakeBlock(),
    "frame_encoder": FrameEncoder(
        FRAME_EMBEDDING_SIZE=PARAMS["FRAME_EMBEDDING_SIZE"],
        image_encoder=resnet18,
        weights=ResNet18_Weights.DEFAULT
        ),
    "frame_projector1": FrameProjector(
        FRAME_EMBEDDING_SIZE=PARAMS["FRAME_EMBEDDING_SIZE"],
        FRAME_PROJECTION_SIZE=PARAMS["FRAME_PROJECTION_SIZE"]
        ),
    "frame_projector2": FrameProjector(
        FRAME_EMBEDDING_SIZE=PARAMS["FRAME_EMBEDDING_SIZE"],
        FRAME_PROJECTION_SIZE=PARAMS["FRAME_PROJECTION_SIZE"]
        ),
    "phoneme_decoder": PhonemeCTCDecoder(
        frame_embedding_size = PARAMS["FRAME_EMBEDDING_SIZE"],
        fixed_neural_sequence_length = PARAMS["FIXED_NEURAL_SEQUENCE_LENGTH"],
        n_temporal_encoder_state = PARAMS["N_TEMPORAL_ENCODER_STATE"],
        n_temporal_encoder_head = PARAMS["N_TEMPORAL_ENCODER_HEAD"],
        n_temporal_encoder_layer = PARAMS["N_TEMPORAL_ENCODER_LAYER"],
        n_phonemes = PARAMS["N_PHONEMES"],
        n_decoder_state = PARAMS["N_DECODER_STATE"]
        ),
    "token_decoder": TokenCTCDecoder(
        n_phonemes = PARAMS["N_PHONEMES"],
        fixed_neural_sequence_length = PARAMS["FIXED_NEURAL_SEQUENCE_LENGTH"],
        n_temporal_encoder_state = PARAMS["N_TEMPORAL_ENCODER_STATE"],
        n_temporal_encoder_head = PARAMS["N_TEMPORAL_ENCODER_HEAD"],
        n_temporal_encoder_layer = PARAMS["N_TEMPORAL_ENCODER_LAYER"],
        n_tokens = PARAMS["N_TOKENS"],
        n_decoder_state = PARAMS["N_DECODER_STATE"]
        )
}

