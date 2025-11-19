import functools
import gc
import wandb
import torch
from pathlib import Path
from src.config import Config
from src.runs.utils import init_device, run_training_epochs
from src.dataloading.dataset import get_dataset
from src.modeling.dynamic_model import assemble_model
from src.inference.contrastive_inference import simclr_inference
from training.trainer import Trainer
from training.losses.info_nce_loss import InfoNCELoss
from torch.optim.lr_scheduler import StepLR

@functools.lru_cache(maxsize=None)
def simclr_run(n_epochs=1, n_batches='all', sim_func='cosine', parts_from=None, use_gpu=False):
    gc.collect()
    device = init_device(use_gpu)
    wandb_run = wandb.init(project="simclr", config=Config)
    print(f"Initialized wandb run: {wandb_run.project}/{wandb_run.name} ({wandb_run.id})")
    print("Loading Dataset")
    dataset = get_dataset(batch_size=1, n_batches=n_batches)
    print("Loading Model")
    model = assemble_model(parts_from).to(device)
    print("Setting up Training")
    trainer = setup_simclr_trainer(model, wandb_run, sim_func=sim_func)
    print("Starting Training...")
    run_training_epochs(n_epochs, dataset, trainer, wandb_run)
    entity = wandb_run.entity
    project = wandb_run.project
    run_id = wandb_run.id
    wandb_run.finish()
    return entity, project, run_id

def setup_simclr_trainer(model, wandb_run, sim_func='cosine'):
    loss_fn = InfoNCELoss(negative_mode='paired', sim_func=sim_func)
    inference_fn = simclr_inference
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
