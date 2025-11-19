from src.inference.contrastive_inference import temporal_ica_inference_with_block_neg_sample
from src.modeling import assemble_model
from src.training.losses import InfoNCELoss
from src.training import Trainer

import torch
import pytest

@pytest.mark.skip()
def test_temporal_ica_inference(mock_dataset, mock_tica_model):
    randint = torch.randint(0, 5, (1,))
    batch = mock_dataset[randint]
    model = mock_tica_model
    loss_fn = InfoNCELoss()
    loss = temporal_ica_inference_with_block_neg_sample(model, loss_fn, batch)

def test_temporal_ica_trainer(mock_dataset, mock_tica_model):
    optimizer = torch.optim.Adam(mock_tica_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    trainer = Trainer(dataset=mock_dataset,
                      model=mock_tica_model,
                      loss_fn=InfoNCELoss(),
                      inference_fn=temporal_ica_inference_with_block_neg_sample,
                      optimizer=optimizer,
                      scheduler=scheduler)
    trainer.train_epoch()