from src.training import TemporalICATrainer, InfoNCELoss
from src.modeling.dynamic_model import DynamicModel
from src.dependency_injection import SHARED_COMPONENTS

import pytest
import torch

class TestTemporalICATrainer:

    def test_train_epoch(self, mock_dataset):
        temporal_ica_loss = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode='paired')
        tica_component_names = ['frame_intake', 'frame_encoder', 'temporal_encoder', 'temporal_decoder']
        tica_components = [SHARED_COMPONENTS[name] for name in tica_component_names]
        tica_model = DynamicModel(components=tica_components)
        optimizer = torch.optim.Adam(tica_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        trainer = TemporalICATrainer(
            dataset = mock_dataset,
            model=tica_model,
            loss_fn=temporal_ica_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            gap = 10
            )
        loss = trainer.train_epoch()
        print(loss)

    def test_train_epoch_with_gap_too_large(self, mock_dataset):
        temporal_ica_loss = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode='paired')
        tica_component_names = ['frame_intake', 'frame_encoder', 'temporal_encoder', 'temporal_decoder']
        tica_components = [SHARED_COMPONENTS[name] for name in tica_component_names]
        tica_model = DynamicModel(components=tica_components)
        optimizer = torch.optim.Adam(tica_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        trainer = TemporalICATrainer(
            dataset = mock_dataset,
            model=tica_model,
            loss_fn=temporal_ica_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            gap = 1000
            )
        with pytest.raises(ValueError):
            loss = trainer.train_epoch()
