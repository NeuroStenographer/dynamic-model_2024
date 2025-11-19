from src.runs.temporal_ica_run import temporal_ica_run
from src.runs.simclr_run import simclr_run
# from src.runs.phoneme_cross_entropy_run import phoneme_cross_entropy_run
from src.runs.ctc_runs import decoder_ctc_run
from src.runs.ctc_runs import post_training_analysis

__all__ = ['temporal_ica_run', 'simclr_run',  'decoder_ctc_run', 'post_training_analysis']