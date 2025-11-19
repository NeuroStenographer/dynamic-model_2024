from src.training import InfoNCELoss
import torch
import pytest

class TestInfoNCELoss:
    # InfoNCELoss happy path
    # ------------------------------
    # Category 1: Test that the loss function returns an expected value.
    # Test 1.1: negative_mode = None;
    # Test 1.2: negative_mode = 'unpaired'; query & positive & negative keys of shape (NUM_Frames, FEATURE_DIMS)
    # Test 1.3: negative_mode = 'paired'; query & positive key are of shape (NUM_FRAMES, FEATURE_DIMS), negative key is of shape (NUM_FRAMES, NUM_PAIRS, FEATURE_DIMS)


    temperature_options = [0.5, 1.0, 10.0]

    reduction_options = ['mean', 'sum']

    NUM_FRAMES_options = [2, 3, 4]

    NUM_PAIRS_options = [2, 3, 4]

    FEATURE_DIMS_options = [2, 3, 4]

    def gen_query_and_keys(self, NUM_FRAMES, FEATURE_DIMS, NUM_NEG_PAIRS=None):
        query = torch.rand((NUM_FRAMES, FEATURE_DIMS))
        # copy the query to the positive key
        positive_key = query.clone()
        if NUM_NEG_PAIRS is None:
            negative_key = torch.rand((NUM_FRAMES, FEATURE_DIMS))
        else:
            negative_key = torch.rand((NUM_FRAMES, NUM_NEG_PAIRS, FEATURE_DIMS))
        return query, positive_key, negative_key

    @pytest.mark.parametrize('temperature', temperature_options)
    @pytest.mark.parametrize('reduction', reduction_options)
    @pytest.mark.parametrize('NUM_FRAMES', NUM_FRAMES_options)
    @pytest.mark.parametrize('FEATURE_DIMS', FEATURE_DIMS_options)
    def test_info_nce_negative_mode_none(self, temperature, reduction, NUM_FRAMES, FEATURE_DIMS):
        query, positive_key, _ = self.gen_query_and_keys(NUM_FRAMES, FEATURE_DIMS)
        loss_fn = InfoNCELoss(temperature=temperature, reduction=reduction, negative_mode=None)
        loss = loss_fn(query, positive_key)
        assert loss.shape == torch.Size([])

    @pytest.mark.parametrize('temperature', temperature_options)
    @pytest.mark.parametrize('reduction', reduction_options)
    @pytest.mark.parametrize('NUM_FRAMES', NUM_FRAMES_options)
    @pytest.mark.parametrize('FEATURE_DIMS', FEATURE_DIMS_options)
    def test_info_nce_negative_mode_unpaired(self, temperature, reduction, NUM_FRAMES, FEATURE_DIMS):
        query, positive_key, negative_key = self.gen_query_and_keys(NUM_FRAMES, FEATURE_DIMS)
        loss_fn = InfoNCELoss(temperature=temperature, reduction=reduction, negative_mode='unpaired')
        loss = loss_fn(query, positive_key, negative_key)
        assert loss.shape == torch.Size([])

    @pytest.mark.parametrize('temperature', temperature_options)
    @pytest.mark.parametrize('reduction', reduction_options)
    @pytest.mark.parametrize('NUM_FRAMES', NUM_FRAMES_options)
    @pytest.mark.parametrize('FEATURE_DIMS', FEATURE_DIMS_options)
    @pytest.mark.parametrize('NUM_PAIRS', NUM_PAIRS_options)
    def test_info_nce_negative_mode_paired(self, temperature, reduction, NUM_FRAMES, FEATURE_DIMS, NUM_PAIRS):
        query, positive_key, negative_key = self.gen_query_and_keys(NUM_FRAMES, FEATURE_DIMS, NUM_PAIRS)
        loss_fn = InfoNCELoss(temperature=temperature, reduction=reduction, negative_mode='paired')
        loss = loss_fn(query, positive_key, negative_key)
        assert loss.shape == torch.Size([])

    # InfoNCELoss unhappy path
    # ------------------------------
    # Category 1: wrong input dimensionality.
    # Test 1.1: query.dim() != 2
    # Test 1.2: positive_key.dim() != 2
    # Test 1.3: negative_keys.dim() != 2 when negative_mode == 'unpaired'
    # Test 1.4: negative_keys.dim() != 3 when negative_mode == 'paired'
    # Category 2: mismatched number of samples.
    # Test 2.1: len(query) != len(positive_key)
    # Test 2.2: len(query) != len(negative_keys) when negative_mode == 'paired'
    # Category 3: mismatched number of components.
    # Test 3.1: query.shape[-1] != positive_key.shape[-1]
    # Test 3.2: query.shape[-1] != negative_keys.shape[-1]


    # InfoNCELoss negative_mode = None
    def test_info_nce_mismatched_query_positive_key_shapes_neg_mode_none(self):
        query = torch.rand((1, 2, 3))
        positive_key = torch.rand((2, 3))
        with pytest.raises(ValueError):
            loss_fn = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode=None)
            loss = loss_fn(query, positive_key)

    def test_info_nce_mismatched_query_positive_key_shapes_other_neg_modes_unpaired(self):
        query = torch.rand((1, 2, 3))
        positive_key = torch.rand((2, 3))
        negative_key = torch.rand((2, 3))
        with pytest.raises(ValueError):
            loss_fn = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode='unpaired')
            loss = loss_fn(query, positive_key, negative_key)

    def test_info_nce_mismatched_query_positive_key_shapes_other_neg_modes_paired(self):
        query = torch.rand((1, 2, 3))
        positive_key = torch.rand((2, 3))
        negative_key = torch.rand((2, 4, 3))
        with pytest.raises(ValueError):
            loss_fn = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode='paired')
            loss = loss_fn(query, positive_key, negative_key)

    # InfoNCELoss annoying paths; check that the mathematical formulas are correct
    # ---------------------------------
    # Category 1: Check that the loss is less when the positive pairs are the same vs when they are random.

    def test_paired_loss_decreases_with_similarity(self):
        F = 10
        for N in range(1,11):
            for M in range(1,11):
                mock_query = torch.rand((N, F))
            mock_const_positive_key = mock_query.clone()
            mock_rand_positive_key = torch.rand((N, F))
            mock_paired_negative_key = torch.rand((N, N, F))
            loss_fn = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode='paired')
            loss_const = loss_fn(mock_query, mock_const_positive_key, mock_paired_negative_key)
            loss_rand = loss_fn(mock_query, mock_rand_positive_key, mock_paired_negative_key)
            assert loss_const < loss_rand

    def test_paired_loss_decreases_with_similarity(self):
        F = 10
        for N in range(1,11):
            mock_query = torch.rand((N, F))
            mock_const_positive_key = mock_query.clone()
            mock_rand_positive_key = torch.rand((N, F))
            mock_paired_negative_key = torch.rand((N, N, F))
            loss_fn = InfoNCELoss(temperature=0.1, reduction='mean', negative_mode='paired')
            loss_const = loss_fn(mock_query, mock_const_positive_key, mock_paired_negative_key)
            loss_rand = loss_fn(mock_query, mock_rand_positive_key, mock_paired_negative_key)
            assert loss_const < loss_rand




class TestCTCLoss:
    # ctc happy path
    # ------------------------------
    # Category 1:


    pass