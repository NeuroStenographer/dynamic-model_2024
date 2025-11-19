import functools
import gc
import wandb
import torch
from pathlib import Path
from src.config import Config
from src.runs.utils import init_device, run_training_epochs
from src.dataloading.dataset import get_dataset
from src.modeling.dynamic_model import assemble_model
from src.inference.ctc_inference import simple_ctc_inference
from training.trainer import Trainer
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import StepLR
from src.training.losses.word_error_rate import execute_ctc_tts_pipeline

@functools.lru_cache(maxsize=None)
def decoder_ctc_run(n_epochs=1, n_batches='all', n_passes=1, batch_size=2, parts_from=None, use_gpu=False):
    gc.collect()
    device = init_device(use_gpu)
    wandb_run = wandb.init(project="decoder-ctc", config=Config)
    print(f"Initialized wandb run: {wandb_run.project}/{wandb_run.name} ({wandb_run.id})")
    print("Loading Dataset")
    dataset = get_dataset(batch_size=batch_size, n_batches=n_batches, get_phonemes=True)
    print("Loading Model")
    model = assemble_model(parts_from).to(device)
    print("Setting up Training")
    trainer = setup_simple_ctc_trainer(model, wandb_run)
    print("Starting Training...")
    run_training_epochs(n_epochs, dataset, trainer, wandb_run, n_passes=n_passes)
    entity = wandb_run.entity
    project = wandb_run.project
    run_id = wandb_run.id
    wandb_run.finish()
    return entity, project, run_id

def setup_simple_ctc_trainer(model, wandb_run):
    loss_fn = CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    inference_fn = simple_ctc_inference
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["TRAINING"]["LEARNING_RATE"])
    scheduler = StepLR(
        optimizer=optimizer,
        step_size = Config["TRAINING"]["SCHEDULER_STEP_SIZE"],
        gamma=Config["TRAINING"]["SCHEDULER_FACTOR"])
    checkpoint_dir = Path(Config["DIRS"]["OUTPUT"]["CHECKPOINTS"])
    run_checkpoint_dir = checkpoint_dir / wandb_run.entity / wandb_run.project / wandb_run.id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        inference_fn=inference_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=run_checkpoint_dir)

def post_training_analysis(model, dataset, labels_map, use_tts=False):
    """
    Performs post-training analysis including decoding phonemes to text, TTS synthesis, and WER calculation.

    Args:
        model (torch.nn.Module): The trained CTC model.
        dataset (Dataset): The dataset used for evaluation, providing input features and reference texts.
        labels_map (dict): Mapping from model output indices to characters.
        use_tts (bool): Flag to enable TTS synthesis for verification.
    """
    # Iterate over the dataset to get input features and reference texts
    for input_features, reference_text in dataset:
        # Execute the CTC to TTS pipeline and calculate WER
        decoded_text, synthesized_speech, error_rate = execute_ctc_tts_pipeline(
            model, input_features, labels_map, use_tts=use_tts)

        # Log or print the results for analysis
        print(f"Decoded Text: {decoded_text}")
        print(f"Word Error Rate: {error_rate:.2%}")
        # Optionally, handle synthesized_speech (e.g., play audio, save to file)

# Example usage within decoder_ctc_run or as a separate step
# You would need to ensure labels_map is defined and accessible
# post_training_analysis(model, dataset, labels_map, use_tts=True)
