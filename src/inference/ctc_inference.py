import torch
from torch.nn.functional import log_softmax

def simple_ctc_inference(model, loss_fn, batch):
    """Inference function for CTC models.
    """
    signals = batch['signals']
    signal_seq_lens = batch['signal_lengths']
    phoneme_ids = batch['phoneme_ids']
    phoneme_lengths = batch['phoneme_lengths']
    # forward pass
    output = model({'input': signals})
    # reshape output (N, T, C) -> (T, N, C)
    logits = output["PhonemeCTCDecoder"].transpose(0, 1)

    # calculate loss
    loss = loss_fn(logits, phoneme_ids, signal_seq_lens, phoneme_lengths)

    return loss

def compound_ctc_inference(model, loss_fn, batch):
    """Inference function for CTC models.
    """
    signals = batch['signals']
    signal_seq_lens = batch['signal_lengths']
    phoneme_ids = batch['phoneme_ids']
    phoneme_lengths = batch['phoneme_lengths']
    token_ids = batch['token_ids']
    token_lengths = batch['token_lengths']
    # forward pass
    output = model({'input': signals})
    # reshape output (N, T, C) -> (T, N, C)
    phoneme_logits = output['PhonemeCTCDecoder'].transpose(0, 1)
    token_logits = output['TokenCTCDecoder'].transpose(0, 1)

    # calculate loss
    phoneme_loss = loss_fn(phoneme_logits, phoneme_ids, signal_seq_lens, phoneme_lengths)
    token_loss = loss_fn(token_logits, token_ids, signal_seq_lens, token_lengths)

    return (phoneme_loss + token_loss)/2


