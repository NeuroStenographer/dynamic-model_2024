from tqdm import tqdm
import torch
import math
import time # for measuring the time each batch cycle takes

def run_training_epochs(n_epochs, dataset, trainer, wandb_run, agg_n=20, n_passes=1):
    bar = tqdm(total=len(dataset)*n_epochs)
    for epoch in range(n_epochs):
        agg_loss = 0
        agg_batch_cycle_time = 0
        for i, batch in enumerate(dataset):
            for j in range(n_passes):
                start_time = time.time()
                loss, lr = trainer.train_batch(
                    batch=batch)
                # if loss (int type) is nan
                lr = trainer.optimizer.param_groups[0]['lr']
                batch_cycle_time = time.time() - start_time
                if wandb_run is not None:
                    log_entry = {
                        'loss': loss,
                        'learning_rate': lr,
                        'batch_cycle_time':batch_cycle_time}
                    wandb_run.log(log_entry)
                bar.update(1)
                if (i+1) % agg_n == 0:
                    agg_loss /= agg_n
                    agg_batch_cycle_time /= agg_n
                    wandb_run.log({
                        f"agg{agg_n}_loss": agg_loss,
                        f"agg{agg_n}_batch_cycle_time": agg_batch_cycle_time}
                    )
                    agg_loss = 0
                    trainer.save_model(epoch_n=epoch, batch_n=i+1)
                else:
                    agg_loss += loss
                bar.set_description(f"Loss: {loss:.4f} | LR: {lr:.4f} | EP: {epoch+1}/{n_epochs}")
        wandb_run.log({"epoch_loss": agg_loss})
        trainer.save_model(epoch_n=epoch, batch_n=len(dataset))

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

def init_device(use_gpu):
    with ClearCache():
        cuda_torch_installed = True if torch.version.cuda is not None else False
        print(f"CUDA Torch Support: {cuda_torch_installed}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        print(f"Use GPU?: {use_gpu}")
        device = torch.device('cuda' if use_gpu and cuda_available else 'cpu')
        print(f"Device: {device}")