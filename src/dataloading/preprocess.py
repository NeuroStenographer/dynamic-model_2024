from src.dataloading.dataset import DaskMatDataset

import h5py

def preprocess_mat_data(data_path, save_path):
    ds = DaskMatDataset(data_path)
    with h5py.File(save_path, 'w') as f:
        for i in range(len(ds)):
            block = ds.get_batch(i)
            n = len(block['text'])
            print(f'Processing block {i}/{len(ds)}')
            for j in range(n):
                print(f'.', end='')
                f.create_group(f'{i}_{j}')
                # add signals to group
                f.create_dataset(f'{i}_{j}/signals',
                                 data=block['signals'][j])
                f.create_dataset(f'{i}_{j}/phoneme_vectors',
                                 data=block['phoneme_one_hot'][j])
                # add metadata to group
                group = f[f'{i}_{j}']
                group.attrs['date'] = block['date'][j]
                group.attrs['text'] = block['text'][j]
                group.attrs['phonemes'] = block['phonemes'][j]

if __name__ == '__main__':
    preprocess_mat_data("K:/ke/sta/data/Willett&EtAl2023/data/train", "./data/train.hdf5")





