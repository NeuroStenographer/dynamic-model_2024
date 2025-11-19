from tqdm import tqdm
import traceback
import torch


class Trainer:
    """A general training module that can be used for any model.
    """
    def __init__(self, model, inference_fn, loss_fn, optimizer, scheduler, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.inference_fn = inference_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, epoch_n, batch_n):
        self.model.save(self.checkpoint_dir, epoch_n, batch_n)

    def train_batch(self, batch, wandb_run=None):
        self.model.train()
        self.optimizer.zero_grad()
        try:
            loss = self.inference_fn(self.model, self.loss_fn, batch)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_result = loss.item()
            curr_lr = self._get_lr()
            del loss  # delete the loss tensor
        except Exception as e:
            raise RuntimeError(f'Error during training: {e}\n{traceback.format_exc()}')
        torch.cuda.empty_cache()  # free up GPU memory
        return loss_result, curr_lr

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
