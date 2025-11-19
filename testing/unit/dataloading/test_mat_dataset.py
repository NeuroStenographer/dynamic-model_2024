
class TestMatDataset:
    """Tests for MatFile."""

    def test_mat_dataset(self, mat_file):
        """Test MatFile."""
        print(mat_file.n_sentences)
        print(mat_file[5:10][0]['text'])
        print(mat_file[5]['text'])

    def test_mat_partition(self, mat_dataset):
        """Test MatDataset."""
        from tqdm import tqdm
        block_lens = []
        with tqdm(total=len(mat_dataset)) as pbar:
            for i in range(len(mat_dataset)):
                signals = mat_dataset[i]['signals']
                print(f"\n{i}: {[int(s.shape[1]) for s in signals]}")
        print(block_lens)
        print(max(block_lens))
        print(min(block_lens))

