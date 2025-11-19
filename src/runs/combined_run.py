import functools
import gc
import wandb
import torch
from pathlib import Path
from src.config import Config
from src.runs.utils import init_device, run_training_epochs
from src.dataloading.dataset import get_dataset
from src.modeling.dynamic_model import assemble_model
from src.inference import simple_ctc_inference, simclr_inference, temporal_ica_inference
from training.trainer import Trainer
from torch.nn import CTCLoss
from src.training.losses import InfoNCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

@functools.lru_cache(maxsize=None)
def combined_run(
    n_epochs=1,
    n_batches='all',
    batch_size=2,
    ctc_parts_from=None,
    simclr_parts_from=None,
    temporal_ica_parts_from=None,
    use_gpu=False
    ):
    gc.collect()
    device = init_device(use_gpu)
    wandb_run = wandb.init(project="combined", config=Config)
    print(f"Initialized wandb run: {wandb_run.project}/{wandb_run.name} ({wandb_run.id})")
    print("Loading Dataset")
    dataset = get_dataset(batch_size=batch_size, n_batches=n_batches, get_phonemes=True)
    print("Loading Model")
    ctc_model = assemble_model(ctc_parts_from).to(device)
    print("Setting up Training")
    ctc_trainer = setup_simple_ctc_trainer(ctc_model, wandb_run)
    simclr_trainer = setup_simclr_trainer(ctc_model, wandb_run)
    temporal_ica_trainer = setup_temporal_ica_trainer(ctc_model, Config, wandb_run)
    print("Starting Training...")
    run_training_epochs(n_epochs, dataset, trainer, wandb_run)
    entity = wandb_run.entity
    project = wandb_run.project
    run_id = wandb_run.id
    wandb_run.finish()
    return entity, project, run_id

def run_training_epochs(n_epochs, dataset, trainers, wandb_run):
    bar = tqdm(total=len(dataset)*n_epochs)
    for epoch in range(n_epochs):
        agg_loss = 0
        for i, batch in enumerate(dataset):
            for j, (trainer_name, trainer) in enumerate(trainers.items()):
                loss, lr = trainer.train_batch(
                    batch=batch,
                    wandb_run=wandb_run)
                bar.update(1)
                if i+1 % 20 == 0 and j == len(trainers) - 1:
                    agg_loss /= 20
                    wandb_run.log({"batch_loss": agg_loss})
                    trainer.scheduler.step(agg_loss)
                    agg_loss = 0
                    trainer.save_model(epoch_n=epoch, batch_n=i+1)
                else:
                    agg_loss += loss
                bar.set_description(f"Trainer: {trainer_name} | Loss: {loss:.4f} | LR: {lr:.4f} | EP: {epoch+1}/{n_epochs}")
        wandb_run.log({"epoch_loss": agg_loss})
        trainer.save_model(epoch_n=epoch, batch_n=len(dataset))

def setup_simple_ctc_trainer(model, wandb_run):
    loss_fn = CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    inference_fn = simple_ctc_inference
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["TRAINING"]["LEARNING_RATE"])
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=Config["TRAINING"]["SCHEDULER_FACTOR"],
        patience=Config["TRAINING"]["SCHEDULER_PATIENCE"],
        verbose=True)
    checkpoint_dir = Path(Config["DIRS"]["OUTPUT"]["CHECKPOINTS"])
    run_checkpoint_dir = checkpoint_dir / wandb_run.entity / wandb_run.project / wandb_run.id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        inference_fn=inference_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=run_checkpoint_dir,
        scheduler_step_size=Config["TRAINING"]["SCHEDULER_STEP_SIZE"])

def setup_simclr_trainer(model, wandb_run):
    loss_fn = InfoNCELoss(negative_mode='paired')
    inference_fn = simclr_inference
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["TRAINING"]["LEARNING_RATE"])
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=Config["TRAINING"]["SCHEDULER_FACTOR"],
        patience=Config["TRAINING"]["SCHEDULER_PATIENCE"],
        verbose=True)
    checkpoint_dir = Path(Config["DIRS"]["OUTPUT"]["CHECKPOINTS"])
    run_checkpoint_dir = checkpoint_dir / wandb_run.entity / wandb_run.project / wandb_run.id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        inference_fn=inference_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=run_checkpoint_dir,
        scheduler_step_size=Config["TRAINING"]["SCHEDULER_STEP_SIZE"])

def setup_temporal_ica_trainer(model, Config, wandb_run):
    loss_fn = InfoNCELoss(negative_mode="paired")
    inference_fn = temporal_ica_inference
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["TRAINING"]["LEARNING_RATE"])
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=Config["TRAINING"]["SCHEDULER_FACTOR"],
        patience=Config["TRAINING"]["SCHEDULER_PATIENCE"],
        verbose=True)
    checkpoint_dir = Path(Config["DIRS"]["OUTPUT"]["CHECKPOINTS"])
    run_checkpoint_dir = checkpoint_dir / wandb_run.entity / wandb_run.project / wandb_run.id # names are not guarenteed to be unique; but they are easier to find in the dashboard
    run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        inference_fn=inference_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=run_checkpoint_dir,
        scheduler_step_size=Config["TRAINING"]["SCHEDULER_STEP_SIZE"])
