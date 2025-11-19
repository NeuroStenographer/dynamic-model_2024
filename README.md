

# NOTES

- Do not use a batch_size > 1 for contrastive leraning. Dataset returns padded arrays.
- Do not use a batch_size of 1 for CTC inference. Dataset returns un-padded arrays.