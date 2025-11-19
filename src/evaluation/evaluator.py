import torch

class Evaluator:
    def __init__(self, dataset, model, eval_fn, inference_fn):
        self.dataset = dataset
        self.model = model
        self.inference_fn = inference_fn
        self.eval_fn = eval_fn

def evaluate(self, wandb_run):
    self.model.eval()
    total_loss = 0
    for batch_kwargs in self.dataset:
        loss = self.inference_fn(self.model, self.eval_fn, **batch_kwargs)
        total_loss += loss.item()
        log_entry = {'loss': loss.item()}
        wandb_run.log(log_entry)
    return total_loss / len(self.dataset)
